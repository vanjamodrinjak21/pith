//! `pith ui`: a terminal explorer for one project's index. Stack (directory -> file -> chunk), Map
//! (2-D PCA of the vectors), RAG (what `search_code` would send Claude) and Index (a live
//! reindex). Reads the database like `pith mcp`: read-only, no writer lock. Only the Index view
//! writes, through `index_project_with_progress`, which takes the lock for its final write. A
//! reindex from elsewhere is picked up on `r`, not automatically.

mod model;
mod state;
mod view;

use crate::budget::{context, DEFAULT_K, MAX_PER_FILE};
use crate::commands::{
    clean, gitignore_warning, index_project_with_progress, open_project, render_error,
    search_project, unix_now,
};
use crate::embed::Embedder;
use crate::storage::{Db, StoredChunk};
use crate::{Error, Result};
use crossterm::event::{self, Event, KeyEventKind};
use model::{Map, Pca, Tree, MAP_CAP};
use state::{Action, Indexing, State, View};
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

/// The embedder is dropped after this long without a query (as in `pith mcp`).
const IDLE_UNLOAD: Duration = Duration::from_secs(5 * 60);

/// One loaded index (or none yet) and everything derived from it.
pub(crate) struct Data {
    root: PathBuf,
    db: Option<Db>,
    tree: Tree,
    map: Map,
}

impl Data {
    fn load(start: &Path) -> Result<Data> {
        let (root, db) = open_project(start)?;
        let n = db.index().len();
        let tree = Tree::build((0..n as u32).filter_map(|i| db.chunk(i).map(|c| c.path.as_str())));
        let map = Map::build(db.index().quantized(), db.index().dim(), n, MAP_CAP);
        Ok(Data {
            root,
            db: Some(db),
            tree,
            map,
        })
    }

    /// No index yet: only the Index view has anything to do.
    fn empty(root: PathBuf) -> Data {
        Data {
            root,
            db: None,
            tree: Tree::build([]),
            map: Map::default(),
        }
    }

    fn chunk(&self, id: u32) -> Option<&StoredChunk> {
        self.db.as_ref()?.chunk(id)
    }
}

/// Restores the terminal when dropped, i.e. on every return path. ratatui's init also installs
/// a panic hook that restores it before the panic message prints.
struct Restore;

impl Drop for Restore {
    fn drop(&mut self) {
        ratatui::restore();
    }
}

/// Status-line text for an error; the model hint matches `pith mcp`.
fn message(e: &Error) -> String {
    match e {
        Error::ModelMissing(_) => "embedding model is not installed; run `pith init`".into(),
        Error::Cancelled => "indexing cancelled; the previous index is unchanged".into(),
        e => render_error(e),
    }
}

/// From the indexing worker to the UI thread.
enum Msg {
    Progress(Indexing),
    Done(std::result::Result<String, String>),
}

/// A running index worker: one thread, cancelled through the flag, reporting over the channel.
struct Worker {
    cancel: Arc<AtomicBool>,
    rx: mpsc::Receiver<Msg>,
    thread: std::thread::JoinHandle<()>,
}

/// Indexes `root`, sending progress; returns the summary line.
fn index_run(
    root: &Path,
    mut basis: Option<(Pca, f32)>,
    stop: &AtomicBool,
    tx: &mpsc::Sender<Msg>,
) -> Result<String> {
    let embedder = crate::load_default_embedder()?;
    let mut ix = Indexing::new(Instant::now());
    let mut seen = 0usize;
    let r = index_project_with_progress(root, embedder.as_ref(), false, &mut |p| {
        if basis.is_none() && !p.vectors.is_empty() {
            let flat: Vec<f32> = p.vectors.iter().flatten().copied().collect();
            let ids: Vec<usize> = (0..p.vectors.len()).collect();
            basis = Pca::fit(&flat, p.vectors[0].len(), &ids).map(|b| (b, 1.0));
        }
        // The same stride as the final map, so the run never streams more than MAP_CAP points.
        let stride = p.chunks.div_ceil(MAP_CAP).max(1);
        for (j, v) in p.vectors.iter().enumerate() {
            if (seen + j).is_multiple_of(stride) {
                if let Some(pt) = basis.as_ref().and_then(|(b, s)| b.project_vector(v, *s)) {
                    ix.points.push(pt);
                }
            }
        }
        seen += p.vectors.len();
        (ix.files, ix.chunks) = (p.files, p.chunks);
        (ix.files_done, ix.chunks_done) = (p.files_done, p.chunks_done);
        ix.current = clean(p.current);
        if ix.skipped.len() != p.skipped.len() {
            ix.skipped = p
                .skipped
                .iter()
                .map(|(path, why)| (clean(path), format!("{why:?}")))
                .collect();
        }
        // shortcut: sends a full copy of the run each time (points included); fine up to
        // MAP_CAP points at one message per embedded group.
        let _ = tx.send(Msg::Progress(ix.clone()));
        !stop.load(Ordering::Relaxed)
    });
    let warning = gitignore_warning(root);
    let r = r?;
    let secs = ix.secs();
    let done = format!(
        "indexed {} chunks from {} files in {secs:.1}s",
        r.chunks, r.files
    );
    Ok(match warning {
        Some(w) => format!("{done}; {w}"),
        None => done,
    })
}

impl Worker {
    /// Projects sampled vectors with `basis` (the current map's PCA, in int8 units of `scale`);
    /// without one, the basis is fitted on the first embedded group. The map is recomputed from
    /// the new index when the run ends.
    fn start(root: PathBuf, basis: Option<(Pca, f32)>) -> Worker {
        let cancel = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel();
        let stop = cancel.clone();
        let thread = std::thread::spawn(move || {
            let r = index_run(&root, basis, &stop, &tx);
            let _ = tx.send(Msg::Done(r.map_err(|e| message(&e))));
        });
        Worker { cancel, rx, thread }
    }

    /// Cancels and waits; the worker stops after the group it is embedding.
    fn stop(self) {
        self.cancel.store(true, Ordering::Relaxed);
        drop(self.rx);
        let _ = self.thread.join();
    }
}

/// A fresh state for `data`, keeping the view and the indexing panel.
fn reset(state: &mut State, data: &Data) {
    let (view, indexing) = (state.view, state.indexing.take());
    *state = State::new(&data.tree, &data.map);
    (state.view, state.indexing) = (view, indexing);
}

pub fn run(project: Option<PathBuf>, index: bool) -> Result<()> {
    if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
        return Err(Error::Invalid(
            "pith ui needs an interactive terminal".into(),
        ));
    }
    let start = match project {
        Some(p) => p,
        None => std::env::current_dir()?,
    };
    let mut data = match Data::load(&start) {
        Err(Error::NoIndex(_)) if index && start.is_dir() => Data::empty(start),
        r => r?,
    };
    let mut state = State::new(&data.tree, &data.map);
    let mut terminal = ratatui::try_init()?;
    let _restore = Restore;
    let mut embedder: Option<(Box<dyn Embedder>, Instant)> = None;
    let mut worker: Option<Worker> = None;
    let mut pending = index.then_some(Action::StartIndex);
    let mut dirty = true;
    loop {
        if dirty {
            terminal.draw(|f| view::draw(f, &state, &data))?;
            dirty = false;
        }
        if embedder
            .as_ref()
            .is_some_and(|(_, t)| t.elapsed() >= IDLE_UNLOAD)
        {
            embedder = None;
        }
        // Worker messages: the UI thread only draws them.
        while let Some(m) = worker.as_ref().and_then(|w| w.rx.try_recv().ok()) {
            dirty = true;
            match m {
                Msg::Progress(p) => state.indexing = Some(p),
                Msg::Done(r) => {
                    if let Some(w) = worker.take() {
                        let _ = w.thread.join();
                    }
                    if let Some(i) = state.indexing.as_mut() {
                        i.finish();
                    }
                    state.status = match r.map(|done| (done, Data::load(&data.root))) {
                        Ok((done, Ok(d))) => {
                            data = d;
                            reset(&mut state, &data);
                            done
                        }
                        Ok((_, Err(e))) => message(&e),
                        Err(e) => e,
                    };
                }
            }
        }
        let action = match pending.take() {
            Some(a) => a,
            None => {
                let wait = if worker.is_some() { 100 } else { 250 };
                if !event::poll(Duration::from_millis(wait))? {
                    // Keeps the elapsed time and rate moving while indexing.
                    dirty |= worker.is_some();
                    continue;
                }
                match event::read()? {
                    Event::Key(k) if k.kind == KeyEventKind::Press => {
                        state.update(k, &data.tree, &data.map)
                    }
                    Event::Resize(..) => Action::None,
                    _ => continue,
                }
            }
        };
        dirty = true;
        match action {
            Action::None => {}
            Action::Quit => {
                if let Some(w) = worker.take() {
                    w.stop();
                }
                return Ok(());
            }
            Action::StartIndex => {
                if worker.is_none() {
                    let scale = data.db.as_ref().map(|d| d.index().scale());
                    worker = Some(Worker::start(
                        data.root.clone(),
                        data.map.pca.clone().zip(scale),
                    ));
                    state.view = View::Index;
                    state.indexing = Some(Indexing::new(Instant::now()));
                    state.status = "indexing...".into();
                }
            }
            Action::CancelIndex => {
                if let Some(w) = &worker {
                    w.cancel.store(true, Ordering::Relaxed);
                    state.status = "cancelling...".into();
                }
            }
            Action::Reload => match Data::load(&data.root) {
                Ok(d) => {
                    data = d;
                    reset(&mut state, &data);
                    state.status = "reloaded".into();
                }
                Err(e) => state.status = message(&e),
            },
            Action::Search(q) => {
                // Loading the model and embedding block for a moment: say so first.
                state.status = "searching...".into();
                terminal.draw(|f| view::draw(f, &state, &data))?;
                state.status = match search(&mut embedder, &data, &mut state, &q) {
                    Ok(msg) => msg.into(),
                    Err(e) => message(&e),
                };
            }
        }
    }
}

/// Runs `q` the way `search_code` does and returns a status message, loading the embedder on
/// first use; projects the query and the candidates onto the map.
fn search(
    embedder: &mut Option<(Box<dyn Embedder>, Instant)>,
    d: &Data,
    s: &mut State,
    q: &str,
) -> Result<&'static str> {
    let db =
        d.db.as_ref()
            .ok_or_else(|| Error::NoIndex(d.root.display().to_string()))?;
    if embedder.is_none() {
        *embedder = Some((crate::load_default_embedder()?, Instant::now()));
    }
    let (e, used) = embedder.as_mut().expect("set above");
    *used = Instant::now();
    // Over-fetched exactly like search_code, so the dropped hits are the ones Claude never sees.
    s.hits = search_project(&d.root, db, e.as_ref(), q, DEFAULT_K * MAX_PER_FILE)?;
    let meta = db.meta();
    s.context = Some(context(
        &s.hits,
        meta.git_head.as_deref(),
        meta.indexed_at_unix,
        unix_now(),
        DEFAULT_K,
    ));
    (s.hit, s.scroll) = (0, 0);
    let index = db.index();
    let dim = index.dim();
    let pca = d.map.pca.as_ref();
    s.hit_points = s
        .hits
        .iter()
        .map(|h| {
            let row = index
                .quantized()
                .get(h.chunk_id as usize * dim..)?
                .get(..dim)?;
            pca?.project_row(row)
        })
        .collect();
    // shortcut: embeds the query a second time for its map position (a few ms); search_project
    // does not return the vector.
    s.query_point = match pca {
        Some(p) => p.project_vector(&e.embed_query(q)?, index.scale()),
        None => None,
    };
    Ok(if s.hits.is_empty() { "no results" } else { "" })
}
