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
    clean, gitignore_warning, gitignore_warning_for, index_project_with_progress, open_project,
    render_error, search_with, unix_now, SearchMode, KEYWORD_ONLY,
};
use crate::embed::Embedder;
use crate::storage::{Db, StoredChunk};
use crate::{Error, Result};
use crossterm::event::{self, Event, KeyEventKind};
use model::{sample_ids, Map, Pca, Tree, COLOURS, MAP_CAP};
use state::{Action, Delta, Indexing, State, View};
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
    /// Empty until the projection thread delivers it (`pending`).
    map: Map,
    /// Map points bucketed by colour group (top-level entries, then one "other" bucket), built
    /// once when the map arrives instead of every frame.
    groups: Vec<Vec<(f64, f64)>>,
    pending: Option<mpsc::Receiver<Map>>,
    /// The projection thread died without a map.
    map_failed: bool,
}

impl Data {
    /// Opens the index and builds the tree; the map (PCA, ~0.4 s at 20,000 x 384 in release) is
    /// computed on a thread over a copy of only the sampled rows, see [`Data::poll_map`].
    fn load(start: &Path) -> Result<Data> {
        let (root, db) = open_project(start)?;
        let n = db.index().len();
        let tree = Tree::build((0..n as u32).filter_map(|i| db.chunk(i).map(|c| c.path.as_str())));
        let (dim, q) = (db.index().dim(), db.index().quantized());
        let rows: Vec<i8> = sample_ids(n, MAP_CAP)
            .filter_map(|i| q.get(i * dim..(i + 1) * dim))
            .flatten()
            .copied()
            .collect();
        let (tx, rx) = mpsc::channel();
        std::thread::Builder::new()
            .name(MAP_THREAD.into())
            .spawn(move || {
                let m = rows.len().checked_div(dim).unwrap_or(0);
                let _ = tx.send(Map::build(&rows, dim, m, MAP_CAP).with_total(n));
            })?;
        Ok(Data {
            root,
            db: Some(db),
            tree,
            map: Map::default(),
            groups: Vec::new(),
            pending: Some(rx),
            map_failed: false,
        })
    }

    /// No index yet: only the Index view has anything to do.
    fn empty(root: PathBuf) -> Data {
        Data {
            root,
            db: None,
            tree: Tree::build([]),
            map: Map::default(),
            groups: Vec::new(),
            pending: None,
            map_failed: false,
        }
    }

    /// Takes the map once it is ready; true when it just arrived. `wait` blocks for it.
    fn poll_map(&mut self, wait: bool) -> bool {
        let Some(rx) = &self.pending else {
            return false;
        };
        let r = if wait {
            rx.recv().map_err(|_| mpsc::TryRecvError::Disconnected)
        } else {
            rx.try_recv()
        };
        let map = match r {
            Ok(m) => m,
            Err(mpsc::TryRecvError::Empty) => return false,
            Err(mpsc::TryRecvError::Disconnected) => {
                self.pending = None;
                self.map_failed = true;
                return true;
            }
        };
        let mut groups = vec![Vec::new(); COLOURS + 1];
        for (i, p) in map.points.iter().enumerate() {
            let g = self
                .chunk(map.id(i))
                .and_then(|c| self.tree.group(&c.path))
                .map_or(COLOURS, |g| g.min(COLOURS));
            groups[g].push((p[0] as f64, p[1] as f64));
        }
        self.groups = groups;
        self.map = map;
        self.pending = None;
        true
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

/// Name of the indexing thread; the UI's panic hook leaves the terminal alone for it.
const WORKER: &str = "pith-index";

/// Name of the map projection thread; a panic there just leaves the map empty.
const MAP_THREAD: &str = "pith-map";

/// Loads the embedder inside the worker (a test seam: tests pass a fake or a panicking one).
type Loader = fn() -> Result<Box<dyn Embedder>>;

/// The last worker panic message, recorded by the UI's panic hook for the status line.
static WORKER_PANIC: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

/// From the indexing worker to the UI thread.
enum Msg {
    Progress(Delta),
    Done(std::result::Result<String, String>),
}

/// A running index worker: one thread, cancelled through the flag, reporting over the channel.
/// Dropping it sets the flag and detaches the thread (see `detach`).
struct Worker {
    cancel: Arc<AtomicBool>,
    rx: mpsc::Receiver<Msg>,
    #[cfg_attr(not(test), allow(dead_code))] // only tests join it; the UI never waits
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

/// Sends the run's result exactly once; if the worker unwinds from a panic, Drop reports it.
struct Reporter {
    tx: mpsc::Sender<Msg>,
    sent: bool,
}

impl Reporter {
    fn done(&mut self, r: std::result::Result<String, String>) {
        self.sent = true;
        let _ = self.tx.send(Msg::Done(r));
    }
}

impl Drop for Reporter {
    fn drop(&mut self) {
        if !self.sent {
            let why = WORKER_PANIC
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take()
                .unwrap_or_else(|| "unknown error".into());
            let _ = self.tx.send(Msg::Done(Err(format!(
                "indexing failed (internal error: {}); the previous index is unchanged",
                clean(&why)
            ))));
        }
    }
}

/// Indexes `root`, sending progress; returns the summary line.
fn index_run(
    root: &Path,
    mut basis: Option<(Pca, f32)>,
    stop: &AtomicBool,
    tx: &mpsc::Sender<Msg>,
    load: Loader,
) -> Result<String> {
    let started = Instant::now();
    let embedder = load()?;
    let mut seen = 0usize;
    let mut skipped = 0usize;
    let r = index_project_with_progress(root, embedder.as_ref(), false, &mut |p| {
        if stop.load(Ordering::Relaxed) {
            return false;
        }
        if basis.is_none() && !p.vectors.is_empty() {
            let flat: Vec<f32> = p.vectors.iter().flatten().copied().collect();
            let ids: Vec<usize> = (0..p.vectors.len()).collect();
            basis = Pca::fit(&flat, p.vectors[0].len(), &ids).map(|b| (b, 1.0));
        }
        // The same stride as the final map, so the run never streams more than MAP_CAP points.
        let stride = p.chunks.div_ceil(MAP_CAP).max(1);
        let mut d = Delta {
            files: p.files,
            chunks: p.chunks,
            files_done: p.files_done,
            chunks_done: p.chunks_done,
            current: clean(p.current),
            ..Delta::default()
        };
        for (j, v) in p.vectors.iter().enumerate() {
            if (seen + j).is_multiple_of(stride) {
                if let Some(pt) = basis.as_ref().and_then(|(b, s)| b.project_vector(v, *s)) {
                    d.points.push((pt[0] as f64, pt[1] as f64));
                }
            }
        }
        seen += p.vectors.len();
        if skipped != p.skipped.len() || d.chunks_done == 0 {
            skipped = p.skipped.len();
            let list = p
                .skipped
                .iter()
                .map(|(path, why)| (clean(path), format!("{why:?}")));
            d.skipped = Some(Arc::new(list.collect()));
        }
        let _ = tx.send(Msg::Progress(d));
        // Checked again here: a cancel during this group's embedding stops before the next.
        !stop.load(Ordering::Relaxed)
    });
    // A cancelled run leaves .pith alone and lists only the lock file it may have created.
    let warning = match &r {
        Err(Error::Cancelled) => gitignore_warning_for(root, false),
        _ => gitignore_warning(root),
    };
    let r = r?;
    let done = r.summary(started.elapsed().as_secs_f64());
    Ok(match warning {
        Some(w) => format!("{done}; {w}"),
        None => done,
    })
}

impl Worker {
    /// Projects sampled vectors with `basis` (the current map's PCA, in int8 units of `scale`);
    /// without one, the basis is fitted on the first embedded group. The map is recomputed from
    /// the new index when the run ends.
    fn start(root: PathBuf, basis: Option<(Pca, f32)>, load: Loader) -> std::io::Result<Worker> {
        let cancel = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel();
        let stop = cancel.clone();
        let thread = std::thread::Builder::new()
            .name(WORKER.into())
            .spawn(move || {
                let mut rep = Reporter {
                    tx: tx.clone(),
                    sent: false,
                };
                let r = index_run(&root, basis, &stop, &tx, load);
                rep.done(r.map_err(|e| message(&e)));
            })?;
        Ok(Worker {
            cancel,
            rx,
            thread: Some(thread),
        })
    }

    /// Cancels without waiting (quit). The worker sees the flag at its next progress call: after
    /// the group it is embedding (or the walk), or at the final check just before the rename,
    /// so a cancelled run never replaces `.pith` and its temp file is removed. Only a quit
    /// that lands after that final check lets the (atomic) rename finish, or the process exit
    /// interrupts it before the rename, leaving the old file.
    fn detach(self) {
        drop(self);
    }
}

/// Starts a run unless one is active.
fn start_index(worker: &mut Option<Worker>, state: &mut State, data: &Data, load: Loader) {
    if worker.is_some() {
        return;
    }
    let scale = data.db.as_ref().map(|d| d.index().scale());
    state.view = View::Index;
    match Worker::start(data.root.clone(), data.map.pca.clone().zip(scale), load) {
        Ok(w) => {
            *worker = Some(w);
            state.indexing = Some(Indexing::new(Instant::now()));
            state.status = "indexing...".into();
        }
        Err(e) => state.status = message(&e.into()),
    }
}

/// Applies the worker's messages to the UI; true if anything changed. A worker that vanished
/// without reporting counts as a failed run.
fn drain(worker: &mut Option<Worker>, state: &mut State, data: &mut Data) -> bool {
    let mut changed = false;
    while let Some(w) = worker.as_ref() {
        let m = match w.rx.try_recv() {
            Ok(m) => m,
            Err(mpsc::TryRecvError::Empty) => break,
            Err(mpsc::TryRecvError::Disconnected) => Msg::Done(Err(
                "indexing stopped unexpectedly; the previous index is unchanged".into(),
            )),
        };
        changed = true;
        match m {
            Msg::Progress(d) => {
                if let Some(i) = state.indexing.as_mut() {
                    i.apply(d);
                }
            }
            Msg::Done(r) => {
                *worker = None; // finished: dropping only sets an unused flag
                if let Some(i) = state.indexing.as_mut() {
                    i.finish();
                }
                state.status = match r.map(|done| (done, Data::load(&data.root))) {
                    Ok((done, Ok(d))) => {
                        *data = d;
                        reset(state, data);
                        done
                    }
                    Ok((_, Err(e))) => message(&e),
                    Err(e) => e,
                };
            }
        }
    }
    changed
}

/// A fresh state for `data`, keeping the view and the indexing panel.
fn reset(state: &mut State, data: &Data) {
    let (view, indexing) = (state.view, state.indexing.take());
    *state = State::new(&data.tree, &data.map);
    (state.view, state.indexing) = (view, indexing);
}

type Hook = Arc<dyn Fn(&std::panic::PanicHookInfo<'_>) + Sync + Send>;

/// Puts back the panic hook that was installed before `pith ui` started, when dropped.
struct HookGuard(Option<Hook>);

impl Drop for HookGuard {
    fn drop(&mut self) {
        if let Some(prev) = self.0.take() {
            std::panic::set_hook(Box::new(move |info| prev(info)));
        }
    }
}

/// Takes the current hook out (so ratatui's init chains to it) and returns a guard that
/// reinstalls it.
fn save_panic_hook() -> HookGuard {
    let prev: Hook = Arc::from(std::panic::take_hook());
    let chained = prev.clone();
    std::panic::set_hook(Box::new(move |info| chained(info)));
    HookGuard(Some(prev))
}

/// Restores the terminal on panics of the UI thread. A worker panic only records its message
/// (its Reporter then shows it in the status line) and a map-thread panic leaves the map empty,
/// so neither writes over the screen. Wraps the hook current at the call (ratatui's).
fn install_panic_hook() {
    let outer = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let name = std::thread::current().name().map(str::to_owned);
        if name.as_deref() == Some(MAP_THREAD) {
            return;
        }
        if name.as_deref() == Some(WORKER) {
            let msg = info
                .payload()
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| info.payload().downcast_ref::<String>().cloned());
            *WORKER_PANIC.lock().unwrap_or_else(|e| e.into_inner()) = msg;
            return;
        }
        outer(info)
    }));
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
    // Before init, so a half-done init is undone too (restore is idempotent). Dropped in
    // reverse order on return: the terminal is restored, then the original panic hook.
    let _hook = save_panic_hook();
    let _restore = Restore;
    let mut terminal = ratatui::try_init()?;
    install_panic_hook();
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
        dirty |= drain(&mut worker, &mut state, &mut data);
        if data.poll_map(false) {
            (state.cursor, state.near) = {
                let fresh = State::new(&data.tree, &data.map);
                (fresh.cursor, fresh.near)
            };
            dirty = true;
        }
        let action = match pending.take() {
            Some(a) => a,
            None => {
                let busy = worker.is_some() || data.pending.is_some();
                let wait = if busy { 100 } else { 250 };
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
                    w.detach();
                }
                return Ok(());
            }
            Action::StartIndex => {
                start_index(&mut worker, &mut state, &data, crate::load_default_embedder)
            }
            Action::CancelIndex => {
                if let Some(w) = &worker {
                    w.cancel.store(true, Ordering::Relaxed);
                    state.status = "stopping...".into();
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
                state.status = match search_lazy(&mut embedder, &data, &mut state, &q) {
                    Ok(msg) => msg,
                    Err(e) => message(&e),
                };
            }
        }
    }
}

/// Loads the embedder on first use, then runs [`search`].
fn search_lazy(
    embedder: &mut Option<(Box<dyn Embedder>, Instant)>,
    d: &Data,
    s: &mut State,
    q: &str,
) -> Result<String> {
    clear_results(s); // also when loading the model fails
    if embedder.is_none() {
        match crate::load_optional_embedder()? {
            Some(e) => *embedder = Some((e, Instant::now())),
            None => return search(None, d, s, q),
        }
    }
    let (e, used) = embedder.as_mut().expect("set above");
    *used = Instant::now();
    search(Some(e.as_ref()), d, s, q)
}

/// Runs `q` the way `search_code` does and returns a status message; projects the query and
/// the candidates onto the map. On error the previous results are already cleared.
fn clear_results(s: &mut State) {
    (s.hits, s.context, s.hit_points, s.query_point) = (Vec::new(), None, Vec::new(), None);
    (s.hit, s.scroll) = (0, 0);
}

/// Without an embedder the search is keyword-only and says so.
fn search(e: Option<&dyn Embedder>, d: &Data, s: &mut State, q: &str) -> Result<String> {
    clear_results(s);
    let db =
        d.db.as_ref()
            .ok_or_else(|| Error::NoIndex(d.root.display().to_string()))?;
    // Over-fetched exactly like search_code, so the dropped hits are the ones Claude never sees.
    let n = DEFAULT_K * MAX_PER_FILE;
    let hits = search_with(&d.root, db, e, q, n, SearchMode::Hybrid)?;
    let index = db.index();
    let pca = d.map.pca.as_ref();
    // shortcut: embeds the query a second time for its map position (a few ms); search_project
    // does not return the vector.
    let query_point = match (pca, e) {
        (Some(p), Some(e)) => p.project_vector(&e.embed_query(q)?, index.scale()),
        _ => None,
    };
    let dim = index.dim();
    s.hit_points = hits
        .iter()
        .map(|h| {
            let row = index
                .quantized()
                .get(h.chunk_id as usize * dim..)?
                .get(..dim)?;
            pca?.project_row(row)
        })
        .collect();
    let notice = e.is_none().then_some(KEYWORD_ONLY);
    let meta = db.meta();
    s.context = Some(context(
        &hits,
        meta.git_head.as_deref(),
        meta.indexed_at_unix,
        unix_now(),
        DEFAULT_K,
        notice,
    ));
    s.query_point = query_point;
    s.hits = hits;
    Ok(match (s.hits.is_empty(), notice) {
        (true, Some(n)) => format!("no results; {n}"),
        (true, None) => "no results".into(),
        (false, n) => n.unwrap_or_default().into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::FakeEmbedder;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    fn fake() -> Result<Box<dyn Embedder>> {
        Ok(Box::new(FakeEmbedder))
    }

    fn boom() -> Result<Box<dyn Embedder>> {
        panic!("boom in the worker")
    }

    /// FakeEmbedder that takes 200 ms per group, so a run is still going when the test acts.
    struct Slow;
    impl Embedder for Slow {
        fn model_id(&self) -> &str {
            FakeEmbedder.model_id()
        }
        fn dim(&self) -> usize {
            FakeEmbedder.dim()
        }
        fn embed_passages(&self, t: &[String]) -> Result<Vec<Vec<f32>>> {
            std::thread::sleep(Duration::from_millis(200));
            FakeEmbedder.embed_passages(t)
        }
        fn embed_query(&self, t: &str) -> Result<Vec<f32>> {
            FakeEmbedder.embed_query(t)
        }
    }

    fn slow() -> Result<Box<dyn Embedder>> {
        Ok(Box::new(Slow))
    }

    fn project(files: usize) -> tempfile::TempDir {
        let d = tempfile::tempdir().unwrap();
        for i in 0..files {
            let text: String = (0..40).map(|j| format!("fn f{i}_{j}() {{}}\n\n")).collect();
            std::fs::write(d.path().join(format!("f{i}.rs")), text.repeat(4)).unwrap();
        }
        d
    }

    /// Drains until the worker is gone (10 s limit).
    fn finish(w: &mut Option<Worker>, s: &mut State, d: &mut Data) {
        let t = Instant::now();
        while w.is_some() {
            assert!(
                t.elapsed() < Duration::from_secs(10),
                "worker never finished"
            );
            drain(w, s, d);
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn worker_runs_end_to_end_and_the_map_is_recomputed() {
        let dir = project(6);
        let mut d = Data::empty(dir.path().to_path_buf());
        let mut s = State::new(&d.tree, &d.map);
        let mut w = None;
        start_index(&mut w, &mut s, &d, fake);
        assert!(w.is_some() && s.view == View::Index);
        start_index(&mut w, &mut s, &d, fake); // a second start is ignored
        finish(&mut w, &mut s, &mut d);
        assert!(d.map.points.is_empty() && d.poll_map(true));
        let ix = s.indexing.as_ref().unwrap();
        assert!(
            !ix.running && ix.chunks > 32 && ix.chunks_done == ix.chunks,
            "{ix:?}"
        );
        assert_eq!(ix.files_done, 6);
        assert_eq!(
            ix.points.len(),
            ix.chunks,
            "every point streamed below the cap"
        );
        assert!(s.status.starts_with("indexed "), "{}", s.status);
        assert_eq!(
            d.map.points.len(),
            ix.chunks,
            "map recomputed from the new index"
        );
        assert_eq!(
            d.groups.iter().map(Vec::len).sum::<usize>(),
            d.map.points.len()
        );
        assert_eq!(s.view, View::Index, "view kept across the reload");
    }

    #[test]
    fn worker_panic_is_reported_and_indexing_works_again() {
        // The real hooks, as `run` installs them (restored when `_hook` drops).
        let _hook = save_panic_hook();
        install_panic_hook();
        let dir = project(1);
        let mut d = Data::empty(dir.path().to_path_buf());
        let mut s = State::new(&d.tree, &d.map);
        let mut w = None;
        start_index(&mut w, &mut s, &d, boom);
        finish(&mut w, &mut s, &mut d);
        assert!(!s.indexing.as_ref().unwrap().running);
        assert!(
            s.status
                .contains("indexing failed (internal error: boom in the worker)"),
            "{}",
            s.status
        );
        assert!(!dir.path().join(".pith").exists());
        let i = KeyEvent::new(KeyCode::Char('i'), KeyModifiers::NONE);
        assert_eq!(s.update(i, &d.tree, &d.map), Action::StartIndex);
        start_index(&mut w, &mut s, &d, fake);
        finish(&mut w, &mut s, &mut d);
        assert!(s.status.starts_with("indexed "), "{}", s.status);
    }

    #[test]
    fn a_vanished_worker_counts_as_failed() {
        let dir = project(1);
        let mut d = Data::empty(dir.path().to_path_buf());
        let mut s = State::new(&d.tree, &d.map);
        s.indexing = Some(Indexing::new(Instant::now()));
        let (tx, rx) = mpsc::channel();
        drop(tx);
        let mut w = Some(Worker {
            cancel: Arc::default(),
            rx,
            thread: None,
        });
        assert!(drain(&mut w, &mut s, &mut d));
        assert!(w.is_none() && !s.indexing.as_ref().unwrap().running);
        assert!(s.status.contains("stopped unexpectedly"), "{}", s.status);
    }

    #[test]
    fn cancel_and_quit_while_indexing_leave_the_index_untouched() {
        let dir = project(10);
        crate::index_project(dir.path(), &FakeEmbedder, false).unwrap();
        std::fs::write(dir.path().join("new.rs"), "fn new() {}\n").unwrap();
        let db = dir.path().join(".pith");
        let before = std::fs::read(&db).unwrap();
        // Lists the lock but not .pith: a cancelled run must not add .pith.
        let ignore = dir.path().join(".gitignore");
        std::fs::write(&ignore, "target/\n.pith.lock\n").unwrap();
        let unchanged = || {
            assert_eq!(std::fs::read(&db).unwrap(), before);
            assert_eq!(
                std::fs::read_to_string(&ignore).unwrap(),
                "target/\n.pith.lock\n"
            );
            assert_eq!(
                std::fs::read_dir(dir.path()).unwrap().count(),
                10 + 1 + 3,
                "no temp file"
            );
        };
        let mut d = Data::load(dir.path()).unwrap();
        d.poll_map(true);
        let mut s = State::new(&d.tree, &d.map);
        // Cancel: the run stops and says so.
        let mut w = None;
        start_index(&mut w, &mut s, &d, slow);
        std::thread::sleep(Duration::from_millis(80));
        w.as_ref().unwrap().cancel.store(true, Ordering::Relaxed);
        finish(&mut w, &mut s, &mut d);
        assert!(s.status.contains("cancelled"), "{}", s.status);
        unchanged();
        // Quit: detach returns at once; the thread then ends by itself without writing.
        start_index(&mut w, &mut s, &d, slow);
        std::thread::sleep(Duration::from_millis(80));
        let mut worker = w.take().unwrap();
        let thread = worker.thread.take().unwrap();
        let t = Instant::now();
        worker.detach();
        assert!(
            t.elapsed() < Duration::from_millis(20),
            "quit must not block"
        );
        thread.join().unwrap(); // the done signal: the worker has exited
        unchanged();
    }

    #[test]
    fn a_cancelled_first_run_lists_only_the_lock() {
        let dir = project(1);
        let ignore = dir.path().join(".gitignore");
        std::fs::write(&ignore, "target/\n").unwrap();
        let mut d = Data::empty(dir.path().to_path_buf());
        let mut s = State::new(&d.tree, &d.map);
        let mut w = None;
        start_index(&mut w, &mut s, &d, slow);
        w.as_ref().unwrap().cancel.store(true, Ordering::Relaxed);
        finish(&mut w, &mut s, &mut d);
        assert!(s.status.contains("cancelled"), "{}", s.status);
        assert!(!dir.path().join(".pith").exists());
        assert_eq!(
            std::fs::read_to_string(&ignore).unwrap(),
            "target/\n.pith.lock\n"
        );
    }

    #[test]
    fn poll_map_delivers_once_and_reports_a_dead_thread() {
        let dir = project(2);
        crate::index_project(dir.path(), &FakeEmbedder, false).unwrap();
        let mut d = Data::load(dir.path()).unwrap();
        let t = Instant::now();
        while !d.poll_map(false) {
            assert!(t.elapsed() < Duration::from_secs(10));
        }
        assert!(!d.map.points.is_empty() && !d.map_failed && d.pending.is_none());
        assert!(!d.poll_map(false), "nothing more to take");
        // A sender dropped without a map: failed, not "nothing to map".
        let (tx, rx) = mpsc::channel::<Map>();
        drop(tx);
        d.pending = Some(rx);
        assert!(d.poll_map(false) && d.map_failed && d.pending.is_none());
        // Still waiting: no change.
        let (_tx, rx) = mpsc::channel::<Map>();
        d.pending = Some(rx);
        d.map_failed = false;
        assert!(!d.poll_map(false) && d.pending.is_some());
    }

    #[test]
    fn search_clears_old_results_first() {
        let dir = project(2);
        crate::index_project(dir.path(), &FakeEmbedder, false).unwrap();
        let mut d = Data::load(dir.path()).unwrap();
        d.poll_map(true);
        let mut s = State::new(&d.tree, &d.map);
        search(Some(&FakeEmbedder), &d, &mut s, "fn f0_1() {}").unwrap();
        assert!(!s.hits.is_empty() && s.query_point.is_some() && s.context.is_some());
        s.hit = 3;
        /// Same model id, wrong dimension: search_project refuses it.
        struct Wrong;
        impl Embedder for Wrong {
            fn model_id(&self) -> &str {
                "fake-384"
            }
            fn dim(&self) -> usize {
                3
            }
            fn embed_passages(&self, _: &[String]) -> Result<Vec<Vec<f32>>> {
                unreachable!()
            }
            fn embed_query(&self, _: &str) -> Result<Vec<f32>> {
                Ok(vec![1.0, 0.0, 0.0])
            }
        }
        assert!(search(Some(&Wrong), &d, &mut s, "x").is_err());
        assert!(s.hits.is_empty() && s.context.is_none() && s.query_point.is_none());
        assert!(s.hit_points.is_empty() && s.hit == 0);
    }

    #[test]
    fn search_without_a_model_is_keyword_only_and_says_so() {
        let dir = project(2);
        crate::index_project(dir.path(), &FakeEmbedder, false).unwrap();
        let mut d = Data::load(dir.path()).unwrap();
        d.poll_map(true);
        let mut s = State::new(&d.tree, &d.map);
        assert_eq!(search(None, &d, &mut s, "f0_1").unwrap(), KEYWORD_ONLY);
        assert!(!s.hits.is_empty() && s.query_point.is_none());
        assert!(s.context.as_ref().unwrap().text.contains(KEYWORD_ONLY));
        assert_eq!(
            search(None, &d, &mut s, "zzzz").unwrap(),
            format!("no results; {KEYWORD_ONLY}")
        );
    }

    #[test]
    #[ignore = "timing; run with --release -- --ignored --nocapture"]
    fn map_build_time_at_the_cap() {
        let mut x = 1u32;
        let q: Vec<i8> = (0..MAP_CAP * 384)
            .map(|_| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (x >> 24) as i8
            })
            .collect();
        let t = Instant::now();
        let m = Map::build(&q, 384, MAP_CAP, MAP_CAP);
        eprintln!("Map::build 20000 x 384: {:?}", t.elapsed());
        assert_eq!(m.points.len(), MAP_CAP);
    }
}
