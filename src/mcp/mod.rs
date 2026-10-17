//! `pith mcp`: a read-only stdio MCP server for one project. Tools: `search_code`, `get_chunk`,
//! `stats`. It never indexes, never writes, and never takes the writer lock. Stdout carries only
//! the protocol; nothing else in this process may print there.

mod budget;
mod tools;

pub use budget::fit_budget;

use crate::embed::Embedder;
use crate::storage::{locate_project, Db};
use crate::{Error, Result};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime};

/// The embedder is dropped after this long without a query.
const IDLE_UNLOAD: Duration = Duration::from_secs(5 * 60);

type Loaded = Box<dyn Embedder + Send>;

/// Database identity: a reindex replaces the file, so path, size or mtime change.
type DbKey = (PathBuf, u64, Option<SystemTime>);

pub(crate) struct State {
    start: PathBuf,
    db: Mutex<Option<(DbKey, PathBuf, Db)>>,
    embedder: Mutex<Option<(Loaded, Instant)>>,
}

/// Locks even if a previous holder panicked; the guarded data stays consistent (replace-only).
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl State {
    /// Runs `f` with the project root and its database, reopening it when the file changed.
    /// The project is located again on every call, so indexing later needs no restart.
    pub(crate) fn with_db<T>(&self, f: impl FnOnce(&Path, &Db) -> Result<T>) -> Result<T> {
        let path = locate_project(&self.start)
            .ok_or_else(|| Error::NoIndex(self.start.display().to_string()))?;
        let m = std::fs::metadata(&path)?;
        let key = (path.clone(), m.len(), m.modified().ok());
        let mut g = lock(&self.db);
        if g.as_ref().is_none_or(|(k, _, _)| *k != key) {
            *g = None; // drop the old mapping before opening the new file
            let db = Db::open(&path, None)?;
            let root = path.parent().unwrap_or(Path::new(".")).to_path_buf();
            *g = Some((key, root, db));
        }
        let (_, root, db) = g.as_ref().expect("set above");
        f(root, db)
    }

    /// Runs `f` with the embedder, loading it on first use.
    pub(crate) fn with_embedder<T>(&self, f: impl FnOnce(&dyn Embedder) -> Result<T>) -> Result<T> {
        let mut g = lock(&self.embedder);
        if g.is_none() {
            *g = Some((load_embedder()?, Instant::now()));
        }
        let (e, used) = g.as_mut().expect("set above");
        *used = Instant::now();
        f(e.as_ref())
    }

    fn unload_if_idle(&self) {
        let mut g = lock(&self.embedder);
        if g.as_ref().is_some_and(|(_, t)| t.elapsed() >= IDLE_UNLOAD) {
            *g = None;
        }
    }
}

fn load_embedder() -> Result<Loaded> {
    // Test hook (documented): integration tests drive the real binary without the ONNX model.
    if std::env::var_os("PITH_TEST_FAKE_EMBEDDER").is_some_and(|v| v == "1") {
        return Ok(Box::new(crate::embed::FakeEmbedder));
    }
    #[cfg(feature = "onnx")]
    return Ok(Box::new(crate::embed::OnnxEmbedder::load(
        &crate::embed::models_dir()?,
    )?));
    #[cfg(not(feature = "onnx"))]
    return crate::load_default_embedder().map(|_| unreachable!("always errors without onnx"));
}

/// Serves MCP over stdio until the client closes stdin. The project is `project` or the current
/// directory; if no index is found there, each tool reports it (nothing is guessed).
pub fn run(project: Option<PathBuf>) -> Result<()> {
    let start = match project {
        Some(p) => p,
        None => std::env::current_dir()?,
    };
    if !start.is_dir() {
        return Err(Error::Invalid(format!(
            "{} is not a directory",
            start.display()
        )));
    }
    let state = Arc::new(State {
        start,
        db: Mutex::new(None),
        embedder: Mutex::new(None),
    });
    let weak = Arc::downgrade(&state);
    std::thread::spawn(move || loop {
        std::thread::sleep(Duration::from_secs(30));
        match weak.upgrade() {
            Some(s) => s.unload_if_idle(),
            None => return,
        }
    });
    tokio::runtime::Builder::new_current_thread()
        .build()?
        .block_on(tools::serve(state))
}
