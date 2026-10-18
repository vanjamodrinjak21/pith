//! `pith mcp`: a read-only stdio MCP server for one project. Tools: `search_code`, `get_chunk`,
//! `stats`. It never indexes, never writes, and never takes the writer lock. Stdout carries only
//! the protocol; nothing else in this process may print there.

mod budget;
mod tools;

use crate::embed::Embedder;
use crate::storage::{locate_project, Db};
use crate::{Error, Result};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime};

/// The embedder is dropped after this long without a query.
const IDLE_UNLOAD: Duration = Duration::from_secs(5 * 60);

/// Longest accepted stdin line (one JSON-RPC message). rmcp's stdio reader has no bound, so
/// [`LineCap`] ends the session on a longer line instead of buffering it without limit.
const MAX_LINE: usize = 1 << 20;

type Loaded = Box<dyn Embedder + Send>;

/// Database identity: a reindex renames a new file into place, so the inode (unix) changes even
/// when size and mtime match. Elsewhere (dev, ino) is (0, 0) and path, size, mtime decide.
type DbKey = (PathBuf, u64, Option<SystemTime>, (u64, u64));

#[cfg(unix)]
fn file_id(m: &std::fs::Metadata) -> (u64, u64) {
    use std::os::unix::fs::MetadataExt;
    (m.dev(), m.ino())
}

#[cfg(not(unix))]
fn file_id(_: &std::fs::Metadata) -> (u64, u64) {
    (0, 0)
}

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
    fn new(start: PathBuf) -> State {
        State {
            start,
            db: Mutex::new(None),
            embedder: Mutex::new(None),
        }
    }

    /// Runs `f` with the project root and its database, reopening it when the file changed.
    /// The project is located again on every call, so indexing later needs no restart.
    pub(crate) fn with_db<T>(&self, f: impl FnOnce(&Path, &Db) -> Result<T>) -> Result<T> {
        let path = locate_project(&self.start)
            .ok_or_else(|| Error::NoIndex(self.start.display().to_string()))?;
        let m = std::fs::metadata(&path)?;
        let key = (path.clone(), m.len(), m.modified().ok(), file_id(&m));
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

    /// Drops the embedder when it was last used at least `idle` before `now`.
    fn unload_if_idle(&self, now: Instant, idle: Duration) {
        let mut g = lock(&self.embedder);
        if g.as_ref()
            .is_some_and(|(_, t)| now.saturating_duration_since(*t) >= idle)
        {
            *g = None;
        }
    }
}

fn load_embedder() -> Result<Loaded> {
    // Test hook, compiled only with the non-default `test-hooks` feature in debug builds: integration tests
    // drive the real binary without the ONNX model.
    #[cfg(all(feature = "test-hooks", debug_assertions))]
    if std::env::var_os("PITH_TEST_FAKE_EMBEDDER").is_some_and(|v| v == "1") {
        return Ok(Box::new(crate::embed::FakeEmbedder));
    }
    #[cfg(feature = "onnx")]
    return Ok(Box::new(crate::embed::OnnxEmbedder::load(
        &crate::embed::models_dir()?,
    )?));
    #[cfg(not(feature = "onnx"))]
    return Err(Error::Invalid(crate::commands::NO_ONNX.into()));
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
    let state = Arc::new(State::new(start));
    let weak = Arc::downgrade(&state);
    std::thread::spawn(move || loop {
        std::thread::sleep(Duration::from_secs(30));
        match weak.upgrade() {
            Some(s) => s.unload_if_idle(Instant::now(), IDLE_UNLOAD),
            None => return,
        }
    });
    tokio::runtime::Builder::new_current_thread()
        .enable_time() // rmcp's service loop uses timers; without this closing stdin panics
        .build()?
        .block_on(tools::serve(state, MAX_LINE))
}

/// AsyncRead wrapper that fails once more than `max` bytes arrive without a newline.
pub(crate) struct LineCap<R> {
    inner: R,
    max: usize,
    run: usize,
    tripped: Arc<std::sync::atomic::AtomicBool>,
}

impl<R> LineCap<R> {
    pub(crate) fn new(inner: R, max: usize) -> Self {
        LineCap {
            inner,
            max,
            run: 0,
            tripped: Arc::default(),
        }
    }

    /// Set once an over-long line has been refused, so the caller can exit with an error status.
    pub(crate) fn tripped(&self) -> Arc<std::sync::atomic::AtomicBool> {
        self.tripped.clone()
    }
}

impl<R: tokio::io::AsyncRead + Unpin> tokio::io::AsyncRead for LineCap<R> {
    fn poll_read(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        let before = buf.filled().len();
        let r = std::pin::Pin::new(&mut self.inner).poll_read(cx, buf);
        if let std::task::Poll::Ready(Ok(())) = r {
            let new = &buf.filled()[before..];
            let mut run = self.run;
            for (i, seg) in new.split(|&b| b == b'\n').enumerate() {
                run = if i == 0 { run + seg.len() } else { seg.len() };
                if run > self.max {
                    self.tripped
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                    buf.set_filled(before);
                    return std::task::Poll::Ready(Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "MCP message longer than the line limit",
                    )));
                }
            }
            self.run = run;
        }
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::FakeEmbedder;

    #[test]
    fn unload_if_idle_respects_the_boundary() {
        let s = State::new(PathBuf::from("."));
        let t0 = Instant::now();
        *lock(&s.embedder) = Some((Box::new(FakeEmbedder), t0));
        s.unload_if_idle(t0 + IDLE_UNLOAD - Duration::from_nanos(1), IDLE_UNLOAD);
        assert!(
            lock(&s.embedder).is_some(),
            "1ns short of the limit keeps it"
        );
        s.unload_if_idle(t0 + IDLE_UNLOAD, IDLE_UNLOAD);
        assert!(lock(&s.embedder).is_none(), "exactly the limit unloads");
    }

    #[test]
    fn reindex_is_picked_up_without_restart() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join("a.txt"), "one\n").unwrap();
        crate::index_project(d.path(), &FakeEmbedder, false).unwrap();
        let s = State::new(d.path().to_path_buf());
        assert_eq!(s.with_db(|_, db| Ok(db.index().len())).unwrap(), 1);
        std::fs::write(d.path().join("b.txt"), "two\n").unwrap();
        crate::index_project(d.path(), &FakeEmbedder, false).unwrap();
        assert_eq!(s.with_db(|_, db| Ok(db.index().len())).unwrap(), 2);
    }

    #[cfg(unix)]
    #[test]
    fn reindex_with_same_size_and_mtime_is_picked_up_by_inode() {
        let d = tempfile::tempdir().unwrap();
        let db = d.path().join(".pith");
        let text = |s: &State| {
            s.with_db(|_, db| Ok(db.chunk(0).unwrap().text.clone()))
                .unwrap()
        };
        std::fs::write(d.path().join("a.txt"), "one\n").unwrap();
        crate::index_project(d.path(), &FakeEmbedder, false).unwrap();
        let before = std::fs::metadata(&db).unwrap();
        let s = State::new(d.path().to_path_buf());
        assert_eq!(text(&s), "one");
        std::fs::write(d.path().join("a.txt"), "two\n").unwrap();
        crate::index_project(d.path(), &FakeEmbedder, false).unwrap();
        std::fs::File::options()
            .write(true)
            .open(&db)
            .unwrap()
            .set_modified(before.modified().unwrap())
            .unwrap();
        let after = std::fs::metadata(&db).unwrap();
        assert_eq!(
            (after.len(), after.modified().ok()),
            (before.len(), before.modified().ok())
        );
        assert_eq!(text(&s), "two");
    }

    #[test]
    fn line_cap_rejects_only_overlong_lines() {
        use tokio::io::AsyncReadExt;
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let read = |data: Vec<u8>| {
            rt.block_on(async {
                let mut out = Vec::new();
                LineCap::new(&data[..], 8)
                    .read_to_end(&mut out)
                    .await
                    .map(|_| out)
            })
        };
        assert!(read(b"12345678\n12345678\n".to_vec()).is_ok());
        assert!(read(b"123456789\n".to_vec()).is_err());
        // The run of bytes without a newline carries over between reads.
        let chunked = |a: &'static [u8], b: &'static [u8]| {
            rt.block_on(async {
                let mut out = Vec::new();
                LineCap::new(a.chain(b), 8)
                    .read_to_end(&mut out)
                    .await
                    .map(|_| out)
            })
        };
        assert!(chunked(b"12345", b"6789\n").is_err());
        assert!(chunked(b"1234", b"5678\n1").is_ok());
        assert!(chunked(b"12\n345678", b"123").is_err());
    }
}
