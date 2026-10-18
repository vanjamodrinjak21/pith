//! Library behind the `pith` subcommands: index a project, search it, report stats.
//! Network access never happens here (only `embed::pull` touches the network).

use crate::embed::Embedder;
use crate::ingest::{collect, CollectReport, SkipReason};
use crate::storage::{locate_project, Db, IndexMeta, StoredChunk, WriterLock};
use crate::{index::VectorIndex, Error, Result};
use std::collections::{HashMap, HashSet};
use std::io::IsTerminal;
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Chunks per `embed_passages` call. Progress granularity only: the ONNX embedder runs its own
/// batch size (default 1, see `OnnxConfig`) inside each call, so vectors do not depend on this.
const EMBED_GROUP: usize = 32;
const MAX_K: usize = 100;
const DB_NAME: &str = ".pith";

#[derive(Debug)]
pub struct IndexReport {
    pub chunks: usize,
    pub files: usize,
    /// Chunks whose vector was taken from the previous index, and chunks embedded.
    pub reused: usize,
    pub embedded: usize,
    /// Nothing changed, so the previous index was left as it was (not rewritten).
    pub up_to_date: bool,
    /// Files that produced no chunks, with the reason.
    pub skipped: Vec<(String, SkipReason)>,
}

impl IndexReport {
    /// One-line summary for the CLI and the TUI.
    pub fn summary(&self, secs: f64) -> String {
        let (c, f) = (self.chunks, self.files);
        if self.up_to_date {
            format!("index is up to date ({c} chunks from {f} files) in {secs:.1}s")
        } else {
            format!(
                "indexed {c} chunks from {f} files ({} reused, {} embedded) in {secs:.1}s",
                self.reused, self.embedded
            )
        }
    }
}

/// Indexing progress, passed to the callback of [`index_project_with_progress`].
#[derive(Debug)]
pub struct Progress<'a> {
    /// Files and chunks found by the walk.
    pub files: usize,
    pub chunks: usize,
    /// Files and chunks done so far (reused from the previous index or embedded).
    pub files_done: usize,
    pub chunks_done: usize,
    /// Path of the last chunk done; empty before the first.
    pub current: &'a str,
    pub skipped: &'a [(String, SkipReason)],
    /// Vectors done since the previous call (reused or embedded), in chunk order.
    pub vectors: &'a [Vec<f32>],
}

#[derive(Debug, Clone, PartialEq)]
pub struct SearchHit {
    pub chunk_id: u32,
    pub score: f32,
    pub path: String,
    pub start_line: u32,
    pub end_line: u32,
    pub text: String,
    /// The file is gone, unsafe to resolve, or was modified after the index was built.
    pub source_changed: bool,
}

#[derive(Debug)]
pub struct Stats {
    pub chunks: usize,
    pub files: usize,
    pub model_id: String,
    pub db_bytes: u64,
    pub indexed_at_unix: u64,
    pub age_secs: u64,
    pub git_head: Option<String>,
}

/// Seconds since the Unix epoch.
#[doc(hidden)]
pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// The default ONNX embedder, loaded offline from the models directory.
#[cfg(feature = "onnx")]
pub fn load_default_embedder() -> Result<Box<dyn Embedder>> {
    let dir = crate::embed::models_dir()?;
    Ok(Box::new(crate::embed::OnnxEmbedder::load(&dir)?))
}

/// Error text when the build has no embedding support.
pub(crate) const NO_ONNX: &str =
    "this build of pith has no embedding model support (built without the `onnx` feature)";

#[cfg(not(feature = "onnx"))]
pub fn load_default_embedder() -> Result<Box<dyn Embedder>> {
    Err(Error::Invalid(NO_ONNX.into()))
}

/// Opens the existing index for reuse (None when there is none). Refuses to overwrite an index
/// built with another model, or one that cannot be read.
fn open_existing(db_path: &Path, embedder: &dyn Embedder) -> Result<Option<Db>> {
    match db_path.symlink_metadata() {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.into()),
        Ok(_) => {}
    }
    match Db::open(db_path, Some((embedder.model_id(), embedder.dim()))) {
        Ok(db) => Ok(Some(db)),
        // The real cause (permissions, ...) must not be dressed up as corruption.
        Err(e @ (Error::Io(_) | Error::ModelMismatch { .. } | Error::UnsupportedFormat(_))) => {
            Err(e)
        }
        Err(e) => Err(Error::CorruptDb(format!(
            "existing .pith is unreadable ({e}); run `pith index --reindex` to rebuild it"
        ))),
    }
}

/// `git rev-parse HEAD` in `root`, only when a `.git` exists at or above it. Any failure is None.
fn git_head(root: &Path) -> Option<String> {
    let root = root.canonicalize().ok()?;
    if !root.ancestors().any(|d| d.join(".git").exists()) {
        return None;
    }
    let out = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(&root)
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    let head = String::from_utf8(out.stdout).ok()?.trim().to_string();
    let ok = out.status.success()
        && matches!(head.len(), 40 | 64)
        && head.bytes().all(|b| b.is_ascii_hexdigit());
    ok.then_some(head)
}

/// Appends `.pith` and `.pith.lock` to an existing `<root>/.gitignore` when not already listed.
/// Never creates the file and never rewrites existing content. The file is opened without
/// following a symlink and checked as a regular file on the open handle; all reads and the
/// append go through that handle.
fn update_gitignore(root: &Path, with_db: bool) -> Result<()> {
    use std::io::{Read, Write};
    let path = root.join(".gitignore");
    let open = |write: bool| {
        let mut o = std::fs::OpenOptions::new();
        if write {
            o.append(true);
        } else {
            o.read(true);
        }
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::custom_flags(
            &mut o,
            (rustix::fs::OFlags::NOFOLLOW | rustix::fs::OFlags::NONBLOCK).bits() as i32,
        );
        o.open(&path)
    };
    // Read-only first: only a regular file is ever reopened for writing.
    let mut f = match open(false) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e.into()),
    };
    let meta = f.metadata()?;
    if !meta.is_file() {
        return Err(Error::Invalid(".gitignore is not a regular file".into()));
    }
    let mut bytes = Vec::new();
    f.read_to_end(&mut bytes)?;
    let text = String::from_utf8_lossy(&bytes);
    let listed = |name: &str| {
        text.lines()
            .map(str::trim)
            .any(|l| l == name || l.strip_prefix('/') == Some(name))
    };
    let mut add = String::new();
    let names: &[&str] = if with_db {
        &[DB_NAME, ".pith.lock"]
    } else {
        &[".pith.lock"]
    };
    for name in names.iter().copied() {
        if !listed(name) {
            add.push_str(name);
            add.push('\n');
        }
    }
    if add.is_empty() {
        return Ok(());
    }
    if !bytes.is_empty() && !bytes.ends_with(b"\n") {
        add.insert(0, '\n');
    }
    let mut w = open(true)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let m = w.metadata()?;
        if !m.is_file() || (m.dev(), m.ino()) != (meta.dev(), meta.ino()) {
            return Err(Error::Invalid(
                ".gitignore changed while it was being updated".into(),
            ));
        }
    }
    w.write_all(add.as_bytes())?;
    Ok(())
}

/// Walks `root`, embeds every chunk and atomically writes `<root>/.pith`. Without `reindex`, a
/// chunk whose text is unchanged takes its vector from the existing index instead of being
/// embedded again, nothing is written when no chunk changed, and an existing index built with a
/// different model (or unreadable) is an error. The writer lock is
/// taken only for the final write, so a second concurrent writer fails with `Error::Locked`
/// after it has done its embedding work. Progress goes to stderr, only when stderr is a TTY.
/// Also lists `.pith` and `.pith.lock` in an existing `.gitignore` (a failure is a warning).
pub fn index_project(root: &Path, embedder: &dyn Embedder, reindex: bool) -> Result<IndexReport> {
    let tty = std::io::stderr().is_terminal();
    let mut shown = false;
    let r = index_project_with_progress(root, embedder, reindex, &mut |p| {
        if tty && p.chunks > 0 && p.chunks_done > 0 {
            eprint!("\rembedding {}/{}", p.chunks_done, p.chunks);
            shown = true;
        }
        true
    });
    if r.is_ok() && shown {
        eprint!("\r\x1b[K");
    }
    // The lock file exists after any attempt (even a failed one), so keep it ignored on every exit.
    if root.is_dir() {
        if let Some(w) = gitignore_warning(root) {
            eprintln!("{w}");
        }
    }
    r
}

/// [`index_project`] without any output and without the `.gitignore` update. `progress` is
/// called once after the walk, after every group (reused or embedded), and once more (with no
/// new vectors) just before the new database replaces the old one. When the index is up to date
/// nothing is written and that last call is not made. Returning false cancels with
/// `Error::Cancelled`: the temp file is removed and the previous index stays as it was. The
/// walk and the group being embedded are not interrupted; a cancel takes effect at the next call.
pub fn index_project_with_progress(
    root: &Path,
    embedder: &dyn Embedder,
    reindex: bool,
    progress: &mut dyn FnMut(&Progress) -> bool,
) -> Result<IndexReport> {
    if !root.metadata()?.is_dir() {
        return Err(Error::Invalid(format!(
            "{} is not a directory",
            root.display()
        )));
    }
    index_inner(root, embedder, reindex, progress)
}

/// Best-effort `.gitignore` update; returns the warning text on failure.
pub(crate) fn gitignore_warning(root: &Path) -> Option<String> {
    // `.pith` is listed only once it exists; a failed first run leaves just the lock file.
    gitignore_warning_for(root, root.join(DB_NAME).symlink_metadata().is_ok())
}

/// [`gitignore_warning`], listing `.pith` only when `with_db` (a cancelled run lists just the lock).
pub(crate) fn gitignore_warning_for(root: &Path, with_db: bool) -> Option<String> {
    update_gitignore(root, with_db).err().map(|e| {
        format!(
            "pith: warning: could not update .gitignore: {}",
            clean(&e.to_string())
        )
    })
}

fn index_inner(
    root: &Path,
    embedder: &dyn Embedder,
    reindex: bool,
    progress: &mut dyn FnMut(&Progress) -> bool,
) -> Result<IndexReport> {
    let db_path = root.join(DB_NAME);
    // Fail fast if another writer is active, before the expensive embedding. Racy by nature: a
    // writer may start after this check, in which case `Db::write` still returns Locked.
    drop(WriterLock::acquire(&db_path)?);
    let old = if reindex {
        None
    } else {
        open_existing(&db_path, embedder)?
    };
    // Text hash -> id in the old index. Equal text gives an equal vector (batch size 1).
    let reuse: HashMap<[u8; 32], u32> = old.as_ref().map_or_else(HashMap::new, |db| {
        (0..db.index().len() as u32)
            .filter_map(|i| db.text_hash(i).map(|h| (*h, i)))
            .collect()
    });
    // Taken before the walk: any file edited after this second is flagged by `source_changed`.
    let indexed_at_unix = unix_now();
    let CollectReport { chunks, skipped } = collect(root)?;
    let total = chunks.len();
    let files = chunks
        .iter()
        .map(|c| c.path.as_str())
        .collect::<HashSet<_>>()
        .len();
    let base = Progress {
        files,
        chunks: total,
        files_done: 0,
        chunks_done: 0,
        current: "",
        skipped: &skipped,
        vectors: &[],
    };
    if !progress(&base) {
        return Err(Error::Cancelled);
    }
    let dim = embedder.dim();
    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(total);
    let (mut files_done, mut embedded) = (0, 0);
    // Text hash -> position in `vectors`, for chunks done earlier in this run.
    let mut fresh: HashMap<[u8; 32], usize> = HashMap::new();
    for group in chunks.chunks(EMBED_GROUP) {
        let start = vectors.len();
        // Where each chunk's vector comes from: the old index, an earlier chunk of this run, or
        // `texts[k]`. Identical new texts are embedded once.
        enum From {
            Old(u32),
            Run(usize),
            New(usize),
        }
        let mut texts: Vec<String> = Vec::new();
        let mut in_group: HashMap<[u8; 32], usize> = HashMap::new();
        let from: Vec<(From, [u8; 32])> = group
            .iter()
            .map(|c| {
                let h = *blake3::hash(c.text.as_bytes()).as_bytes();
                let f = if let Some(&i) = reuse.get(&h) {
                    From::Old(i)
                } else if let Some(&j) = fresh.get(&h) {
                    From::Run(j)
                } else {
                    From::New(*in_group.entry(h).or_insert_with(|| {
                        texts.push(c.text.clone());
                        texts.len() - 1
                    }))
                };
                (f, h)
            })
            .collect();
        let new = if texts.is_empty() {
            Vec::new()
        } else {
            embedder.embed_passages(&texts)?
        };
        if new.len() != texts.len() {
            return Err(Error::Embed(format!(
                "embedder returned {} vectors for {} chunks",
                new.len(),
                texts.len()
            )));
        }
        embedded += texts.len();
        for (f, h) in from {
            let v = match (f, &old) {
                (From::Old(i), Some(db)) => db.index().full()[i as usize * dim..][..dim].to_vec(),
                (From::Run(j), _) => vectors[j].clone(),
                (From::New(k), _) => new[k].clone(),
                (From::Old(_), None) => unreachable!("reuse is empty without an old index"),
            };
            fresh.entry(h).or_insert(vectors.len());
            vectors.push(v);
        }
        // Chunks are sorted by path: a file is done once a later chunk has another path.
        files_done = if vectors.len() == total {
            files
        } else {
            files_done
                + chunks[start..=vectors.len()]
                    .windows(2)
                    .filter(|w| w[0].path != w[1].path)
                    .count()
        };
        let last = &group[group.len() - 1].path;
        let p = Progress {
            files_done,
            chunks_done: vectors.len(),
            current: last,
            vectors: &vectors[start..],
            ..base
        };
        if !progress(&p) {
            return Err(Error::Cancelled);
        }
    }
    let stored: Vec<StoredChunk> = chunks
        .into_iter()
        .map(|c| StoredChunk {
            path: c.path,
            start_line: c.start_line,
            end_line: c.end_line,
            text: c.text,
        })
        .collect();
    let meta = IndexMeta {
        indexed_at_unix,
        git_head: git_head(root),
    };
    // Same chunks in the same order, all reused (so the same vectors), same git head, and no file
    // modified since the old index time: keep the old file, so readers that watch it do not
    // reload for nothing. A touched file still gets a write, which moves `indexed_at` forward
    // so `source_changed` clears.
    let up_to_date = old.as_ref().is_some_and(|db| {
        let at = db.meta().indexed_at_unix;
        embedded == 0
            && db.chunks() == stored.as_slice()
            && db.meta().git_head == meta.git_head
            && !stored
                .iter()
                .map(|c| c.path.as_str())
                .collect::<HashSet<_>>()
                .into_iter()
                .any(|p| changed_since(root, p, at))
    });
    // Unmapped before the write: on Windows a mapped file cannot be replaced.
    drop(reuse);
    drop(old);
    let report = |skipped| IndexReport {
        chunks: total,
        files,
        reused: total - embedded,
        embedded,
        up_to_date,
        skipped,
    };
    if up_to_date {
        return Ok(report(skipped));
    }
    let index = if vectors.is_empty() {
        VectorIndex::from_parts(embedder.dim(), vec![], vec![])?
    } else {
        VectorIndex::build(vectors)?
    };
    if index.dim() != embedder.dim() {
        return Err(Error::DimMismatch {
            expected: embedder.dim(),
            got: index.dim(),
        });
    }
    let last = stored.last().map(|c| c.path.clone()).unwrap_or_default();
    // Last chance to cancel: after this the rename is the only step left.
    let last_check = || {
        let p = Progress {
            files_done: files,
            chunks_done: total,
            current: &last,
            vectors: &[],
            ..base
        };
        if progress(&p) {
            Ok(())
        } else {
            Err(Error::Cancelled)
        }
    };
    Db::write_checked(
        &db_path,
        embedder.model_id(),
        stored,
        &index,
        meta,
        last_check,
    )?;
    Ok(report(skipped))
}

/// Finds the project's `.pith` from `start` (see `locate_project`) and opens it. Returns the
/// project root (the directory holding `.pith`) and the database.
pub fn open_project(start: &Path) -> Result<(PathBuf, Db)> {
    let path = locate_project(start).ok_or_else(|| Error::NoIndex(start.display().to_string()))?;
    let db = Db::open(&path, None)?;
    let root = path.parent().unwrap_or(Path::new(".")).to_path_buf();
    Ok((root, db))
}

/// Resolves a stored relative path under `root`, refusing absolute paths, `..`, and any symlink
/// along the way (the indexer never follows them). None means "do not touch the filesystem".
pub(crate) fn safe_join(root: &Path, rel: &str) -> Option<PathBuf> {
    if rel.is_empty() || rel.contains('\0') || (cfg!(windows) && rel.contains('\\')) {
        return None;
    }
    let mut cur = root.to_path_buf();
    for c in Path::new(rel).components() {
        let Component::Normal(part) = c else {
            return None;
        };
        cur.push(part);
        if cur
            .symlink_metadata()
            .is_ok_and(|m| m.file_type().is_symlink())
        {
            return None;
        }
    }
    Some(cur)
}

/// True when the stored file is missing, unsafe to resolve, not a regular file, or its mtime is
/// later (whole seconds) than the index time. The database stores no per-file mtime or size, so
/// this is the sound check available without a format change; an edit within the same second as
/// the index start is not detected.
fn changed_since(root: &Path, rel: &str, indexed_at_unix: u64) -> bool {
    let Some(p) = safe_join(root, rel) else {
        return true;
    };
    let Ok(m) = p.symlink_metadata() else {
        return true;
    };
    if !m.is_file() {
        return true;
    }
    m.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .is_none_or(|d| d.as_secs() > indexed_at_unix)
}

/// How [`search_with`] ranks chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SearchMode {
    /// Cosine similarity of embeddings only.
    Vector,
    /// BM25 over identifier-aware tokens only; needs no embedding model.
    Keyword,
    /// Reciprocal rank fusion of the vector and keyword rankings.
    #[default]
    Hybrid,
}

impl SearchMode {
    pub const ALL: [SearchMode; 3] = [SearchMode::Vector, SearchMode::Keyword, SearchMode::Hybrid];

    pub fn name(self) -> &'static str {
        match self {
            SearchMode::Vector => "vector",
            SearchMode::Keyword => "keyword",
            SearchMode::Hybrid => "hybrid",
        }
    }
}

impl std::str::FromStr for SearchMode {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, String> {
        SearchMode::ALL
            .into_iter()
            .find(|m| m.name() == s)
            .ok_or_else(|| format!("unknown mode {s:?}; use vector, keyword or hybrid"))
    }
}

/// Candidates taken from each ranking before fusion.
const FUSE_N: usize = 50;
/// The reciprocal rank fusion constant: a hit at rank r adds 1 / (RRF_K + r).
const RRF_K: f32 = 60.0;

/// Shown when hybrid search runs without a model and falls back to keyword search.
pub const KEYWORD_ONLY: &str =
    "no embedding model: keyword search only (run `pith init` for semantic search)";

/// True when `e` means no embedding model is available: not installed, or not built in.
pub fn is_no_model(e: &Error) -> bool {
    matches!(e, Error::ModelMissing(_)) || matches!(e, Error::Invalid(m) if m == NO_ONNX)
}

/// [`load_default_embedder`], or None when no model is available (see [`is_no_model`]).
pub fn load_optional_embedder() -> Result<Option<Box<dyn Embedder>>> {
    match load_default_embedder() {
        Ok(e) => Ok(Some(e)),
        Err(e) if is_no_model(&e) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Top-`k` chunks for `query` by hybrid search (see [`search_with`]).
pub fn search_project(
    root: &Path,
    db: &Db,
    embedder: &dyn Embedder,
    query: &str,
    k: usize,
) -> Result<Vec<SearchHit>> {
    search_with(root, db, Some(embedder), query, k, SearchMode::Hybrid)
}

/// Top-`k` chunks for `query` (k must be at least 1; capped at 100). `root` is the project
/// root, used only to compute `source_changed`. Errors with `ModelMismatch` if the embedder is
/// not the one that built the database. Without an embedder, `Hybrid` is keyword-only and
/// `Vector` is an error.
///
/// `score` is the cosine similarity whenever an embedder is given (also for a hybrid hit found
/// only by keyword); keyword-only results carry the BM25 score divided by the top one, so the
/// first is 1.0. Hybrid results are ordered by fused rank, not by `score`.
pub fn search_with(
    root: &Path,
    db: &Db,
    embedder: Option<&dyn Embedder>,
    query: &str,
    k: usize,
    mode: SearchMode,
) -> Result<Vec<SearchHit>> {
    if query.trim().is_empty() {
        return Err(Error::Invalid("query must not be empty".into()));
    }
    if k == 0 {
        return Err(Error::Invalid("k must be at least 1".into()));
    }
    let k = k.min(MAX_K);
    let embedder = if mode == SearchMode::Keyword {
        None
    } else {
        embedder
    };
    let q = match embedder {
        Some(e) => {
            if db.model_id() != e.model_id() || db.index().dim() != e.dim() {
                return Err(Error::ModelMismatch {
                    expected: e.model_id().into(),
                    expected_dim: e.dim(),
                    found: db.model_id().into(),
                    found_dim: db.index().dim(),
                });
            }
            Some(e.embed_query(query)?)
        }
        None if mode == SearchMode::Vector => {
            return Err(Error::ModelMissing(
                "vector search needs the embedding model".into(),
            ))
        }
        None => None,
    };
    let ranked: Vec<(u32, f32)> = match &q {
        Some(q) if mode == SearchMode::Vector => db
            .index()
            .try_search(q, k)?
            .into_iter()
            .map(|h| (h.id, h.score))
            .collect(),
        Some(q) => {
            let n = k.max(FUSE_N);
            let vector: Vec<u32> = db.index().try_search(q, n)?.iter().map(|h| h.id).collect();
            let keyword: Vec<u32> = db.keywords().search(query, n).iter().map(|h| h.0).collect();
            let full = db.index().full();
            let dim = db.index().dim();
            fuse(&vector, &keyword, k)
                .into_iter()
                .map(|id| {
                    let row = &full[id as usize * dim..][..dim];
                    (id, row.iter().zip(q).map(|(a, b)| a * b).sum())
                })
                .collect()
        }
        None => {
            let mut hits = db.keywords().search(query, k);
            let top = hits.first().map_or(1.0, |h| h.1);
            hits.iter_mut().for_each(|h| h.1 /= top);
            hits
        }
    };
    let at = db.meta().indexed_at_unix;
    Ok(ranked
        .into_iter()
        .filter_map(|(id, score)| {
            let c = db.chunk(id)?;
            Some(SearchHit {
                chunk_id: id,
                score,
                path: c.path.clone(),
                start_line: c.start_line,
                end_line: c.end_line,
                text: c.text.clone(),
                source_changed: changed_since(root, &c.path, at),
            })
        })
        .collect())
}

/// Reciprocal rank fusion of two rankings (best first): the top `k` ids by the sum of
/// 1 / (RRF_K + rank) over the rankings that hold them, ties by ascending id.
fn fuse(a: &[u32], b: &[u32], k: usize) -> Vec<u32> {
    let mut score: HashMap<u32, f32> = HashMap::new();
    for list in [a, b] {
        for (r, &id) in list.iter().enumerate() {
            *score.entry(id).or_default() += 1.0 / (RRF_K + r as f32 + 1.0);
        }
    }
    let mut ids: Vec<(u32, f32)> = score.into_iter().collect();
    ids.sort_by(|x, y| y.1.total_cmp(&x.1).then(x.0.cmp(&y.0)));
    ids.into_iter().take(k).map(|(id, _)| id).collect()
}

/// Summary of an open database; needs no embedding model. Strings that come from the file are
/// sanitised: a git head that is not 40/64 lowercase hex becomes "(invalid)".
pub fn stats_project(root: &Path, db: &Db) -> Result<Stats> {
    let chunks = db.index().len();
    let files = (0..chunks as u32)
        .filter_map(|i| db.chunk(i).map(|c| c.path.as_str()))
        .collect::<HashSet<_>>()
        .len();
    let at = db.meta().indexed_at_unix;
    Ok(Stats {
        chunks,
        files,
        model_id: clean(db.model_id()),
        db_bytes: std::fs::metadata(root.join(DB_NAME))?.len(),
        indexed_at_unix: at,
        age_secs: unix_now().saturating_sub(at),
        git_head: db.meta().git_head.as_deref().map(|h| {
            if valid_head(h) {
                h.to_string()
            } else {
                "(invalid)".to_string()
            }
        }),
    })
}

/// A recorded git head is shown only when it is 40/64 lowercase hex.
pub(crate) fn valid_head(h: &str) -> bool {
    matches!(h.len(), 40 | 64) && h.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

pub(crate) fn age(secs: u64) -> String {
    match secs {
        0..=59 => format!("{secs} seconds ago"),
        60..=3599 => format!("{} minutes ago", secs / 60),
        3600..=86399 => format!("{} hours ago", secs / 3600),
        _ => format!("{} days ago", secs / 86400),
    }
}

/// First 3 lines, each cut to 100 chars, indented, control characters scrubbed.
pub(crate) fn snippet(text: &str) -> String {
    text.lines()
        .take(3)
        .map(|l| format!("    {}", clean(&l.chars().take(100).collect::<String>())))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Plain-text search output: rank, score, location, snippet, then a staleness footer.
pub fn render_search(hits: &[SearchHit], indexed_at_unix: u64, now_unix: u64) -> String {
    let mut out = String::new();
    for (i, h) in hits.iter().enumerate() {
        let mark = if h.source_changed {
            "  [source changed]"
        } else {
            ""
        };
        out.push_str(&format!(
            "{}. {:.2}  {}:{}-{}{mark}\n{}\n",
            i + 1,
            h.score,
            clean(&h.path),
            h.start_line,
            h.end_line,
            snippet(&h.text)
        ));
    }
    let changed = hits.iter().filter(|h| h.source_changed).count();
    out.push_str(&format!(
        "indexed {}; {changed} of {} results have changed sources\n",
        age(now_unix.saturating_sub(indexed_at_unix)),
        hits.len()
    ));
    out
}

pub fn render_stats(s: &Stats) -> String {
    let mut out = format!(
        "chunks:  {}\nfiles:   {}\nmodel:   {}\ndb size: {} bytes\nindexed: {}\n",
        s.chunks,
        s.files,
        s.model_id,
        s.db_bytes,
        age(s.age_secs)
    );
    if let Some(h) = &s.git_head {
        out.push_str(&format!("git head: {h}\n"));
    }
    out
}

/// The text after `pith: ` for a failed command; sanitised because errors embed paths.
pub fn render_error(e: &Error) -> String {
    match e {
        Error::Locked => {
            "another pith process is writing this index; wait for it to finish and retry".into()
        }
        e => clean(&e.to_string()),
    }
}

/// `[Y/n]` answer: empty, `y` or `yes` (any case) is yes; everything else is no.
pub fn is_yes(line: &str) -> bool {
    matches!(line.trim().to_ascii_lowercase().as_str(), "" | "y" | "yes")
}

/// Compact stderr summary of skipped files: count per reason plus up to 5 example paths.
/// Never includes file contents. Paths are stripped of control characters.
pub fn skip_summary(skipped: &[(String, SkipReason)]) -> String {
    let label = |r: &SkipReason| match r {
        SkipReason::Denied => "denied (secrets list)",
        SkipReason::Symlink => "symlink",
        SkipReason::Binary => "binary",
        SkipReason::NotUtf8 => "not UTF-8",
        SkipReason::SpecialFile => "special file",
        SkipReason::Secret => "private key",
        SkipReason::TooLarge => "too large",
        SkipReason::Empty => "empty",
        SkipReason::Unreadable(_) => "unreadable",
    };
    let mut groups: Vec<(&str, Vec<&str>)> = Vec::new();
    for (path, r) in skipped {
        let l = label(r);
        match groups.iter_mut().find(|(g, _)| *g == l) {
            Some((_, v)) => v.push(path),
            None => groups.push((l, vec![path])),
        }
    }
    groups
        .iter()
        .map(|(l, v)| {
            let ex: Vec<String> = v.iter().take(5).map(|p| clean(p)).collect();
            let more = if v.len() > 5 {
                format!(", +{} more", v.len() - 5)
            } else {
                String::new()
            };
            format!("skipped {} {l}: {}{more}", v.len(), ex.join(", "))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Replaces control characters (tab becomes a space) so file-derived text cannot drive the
/// terminal. Apply to every string that comes from the database or the indexed files.
pub(crate) fn clean(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '\t' => ' ',
            c if c.is_control() => '?',
            // Invisible/bidi-override characters that can spoof what a line says (not ZWJ/ZWNJ).
            '\u{200B}'
            | '\u{200E}'
            | '\u{200F}'
            | '\u{2028}'..='\u{202E}'
            | '\u{2060}'..='\u{2064}'
            | '\u{2066}'..='\u{2069}'
            | '\u{061C}'
            | '\u{FEFF}' => '?',
            c => c,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_is_monotonic_and_ends_at_the_report_totals() {
        let d = tempfile::tempdir().unwrap();
        for i in 0..5 {
            let text: String = (0..30).map(|j| format!("fn f{i}_{j}() {{}}\n\n")).collect();
            std::fs::write(d.path().join(format!("f{i}.rs")), text.repeat(8)).unwrap();
        }
        let mut seen: Vec<(usize, usize, usize, usize, usize)> = Vec::new();
        let r =
            index_project_with_progress(d.path(), &crate::embed::FakeEmbedder, false, &mut |p| {
                seen.push((
                    p.files,
                    p.chunks,
                    p.files_done,
                    p.chunks_done,
                    p.vectors.len(),
                ));
                true
            })
            .unwrap();
        assert!(seen.len() > 2, "{seen:?}");
        assert!(
            seen.windows(2)
                .all(|w| w[0].2 <= w[1].2 && w[0].3 <= w[1].3),
            "{seen:?}"
        );
        assert!(seen.iter().all(|s| (s.0, s.1) == (r.files, r.chunks)));
        assert_eq!(seen.iter().map(|s| s.4).sum::<usize>(), r.chunks);
        let last = seen.last().unwrap();
        assert_eq!((last.2, last.3), (r.files, r.chunks));
    }

    #[test]
    fn cancel_just_before_the_rename_writes_nothing() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join("a.txt"), "one\n").unwrap();
        index_project(d.path(), &crate::embed::FakeEmbedder, false).unwrap();
        let before = std::fs::read(d.path().join(DB_NAME)).unwrap();
        std::fs::write(d.path().join("b.txt"), "two\n").unwrap();
        let mut last = false;
        let r =
            index_project_with_progress(d.path(), &crate::embed::FakeEmbedder, false, &mut |p| {
                // The final call: everything embedded, nothing new, the temp file already synced.
                last = p.chunks_done == p.chunks && p.vectors.is_empty() && p.chunks > 0;
                !last
            });
        assert!(last && matches!(r, Err(Error::Cancelled)), "{r:?}");
        assert_eq!(std::fs::read(d.path().join(DB_NAME)).unwrap(), before);
        let mut names: Vec<_> = std::fs::read_dir(d.path())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        names.sort();
        assert_eq!(
            names,
            [".pith", ".pith.lock", "a.txt", "b.txt"],
            "temp file removed"
        );
    }

    #[test]
    fn cancel_leaves_the_previous_index_intact() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join("a.txt"), "one\n").unwrap();
        index_project(d.path(), &crate::embed::FakeEmbedder, false).unwrap();
        let before = std::fs::read(d.path().join(DB_NAME)).unwrap();
        std::fs::write(d.path().join("b.txt"), "two\n").unwrap();
        for stop_after in [0, 1] {
            let mut calls = 0;
            let r = index_project_with_progress(
                d.path(),
                &crate::embed::FakeEmbedder,
                false,
                &mut |_| {
                    calls += 1;
                    calls <= stop_after
                },
            );
            assert!(matches!(r, Err(Error::Cancelled)), "{r:?}");
        }
        assert_eq!(std::fs::read(d.path().join(DB_NAME)).unwrap(), before);
        let names: Vec<_> = std::fs::read_dir(d.path())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(names.len(), 4, "no temp file left: {names:?}");
    }

    #[test]
    fn safe_join_rejects_escapes_and_symlinks() {
        let d = tempfile::tempdir().unwrap();
        let r = d.path();
        std::fs::create_dir(r.join("a")).unwrap();
        assert_eq!(safe_join(r, "a/b.txt"), Some(r.join("a/b.txt")));
        for bad in ["", "/etc/passwd", "../x", "a/../../x", "./a", "a\0b"] {
            assert_eq!(safe_join(r, bad), None, "{bad:?}");
        }
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink("/etc", r.join("link")).unwrap();
            assert_eq!(safe_join(r, "link/passwd"), None);
        }
    }

    #[test]
    fn age_and_snippet_format() {
        assert_eq!(age(5), "5 seconds ago");
        assert_eq!(age(120), "2 minutes ago");
        assert_eq!(age(7200), "2 hours ago");
        assert_eq!(age(3 * 86400), "3 days ago");
        let s = snippet("a\tb\n\x1b[31mred\nc\nd");
        assert_eq!(s, "    a b\n    ?[31mred\n    c");
        assert!(snippet(&"x".repeat(500)).len() == 104);
    }

    #[test]
    fn clean_scrubs_spoofing_characters() {
        for c in [
            '\u{200B}', '\u{061C}', '\u{2060}', '\u{2061}', '\u{2062}', '\u{2063}', '\u{2064}',
            '\u{200E}', '\u{200F}', '\u{2028}', '\u{2029}', '\u{202A}', '\u{202B}', '\u{202C}',
            '\u{202D}', '\u{202E}', '\u{2066}', '\u{2067}', '\u{2068}', '\u{2069}', '\u{FEFF}',
            '\x1b', '\u{7f}', '\u{85}',
        ] {
            assert_eq!(clean(&format!("a{c}b")), "a?b", "{:x}", c as u32);
        }
        // ZWJ/ZWNJ are load-bearing in emoji and Persian/Hindi text.
        let keep = "\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467} \u{645}\u{200C}\u{6CC}";
        assert_eq!(clean(keep), keep);
        assert_eq!(clean("a\tb h\u{e9}llo"), "a b h\u{e9}llo");
    }

    #[test]
    fn symlinked_gitignore_yields_a_warning_and_is_untouched() {
        #[cfg(unix)]
        {
            let d = tempfile::tempdir().unwrap();
            let target = d.path().join("real");
            std::fs::write(&target, "x\n").unwrap();
            let root = d.path().join("p");
            std::fs::create_dir(&root).unwrap();
            std::os::unix::fs::symlink(&target, root.join(".gitignore")).unwrap();
            let w = gitignore_warning(&root).expect("warning");
            assert!(
                w.starts_with("pith: warning: could not update .gitignore"),
                "{w}"
            );
            assert_eq!(std::fs::read_to_string(&target).unwrap(), "x\n");
        }
    }

    #[test]
    fn fusion_is_deterministic_and_ties_break_by_id() {
        // 7 is first in one list and absent from the other; 3 and 5 are in both.
        assert_eq!(fuse(&[7, 3, 5], &[3, 5, 9], 10), [3, 5, 7, 9]);
        // Mirrored positions score equally: the lower id wins.
        assert_eq!(fuse(&[4, 2], &[2, 4], 10), [2, 4]);
        assert_eq!(fuse(&[8], &[6], 10), [6, 8]);
        assert_eq!(fuse(&[1, 2, 3], &[], 2), [1, 2]);
        assert!(fuse(&[], &[], 5).is_empty());
    }

    #[test]
    fn modes_parse_by_name() {
        for m in SearchMode::ALL {
            assert_eq!(m.name().parse::<SearchMode>(), Ok(m));
        }
        assert!("semantic".parse::<SearchMode>().is_err());
        assert_eq!(SearchMode::default(), SearchMode::Hybrid);
    }

    #[test]
    fn yes_answers() {
        assert!(is_yes("") && is_yes("Y\n") && is_yes(" yes "));
        assert!(!is_yes("n") && !is_yes("yep") && !is_yes("no"));
    }

    #[test]
    fn skip_summary_groups_and_caps_examples() {
        let s: Vec<_> = (0..7)
            .map(|i| (format!("f{i}.bin"), SkipReason::Binary))
            .chain([(".env".to_string(), SkipReason::Denied)])
            .collect();
        let t = skip_summary(&s);
        assert!(t.contains("skipped 7 binary: f0.bin, f1.bin, f2.bin, f3.bin, f4.bin, +2 more"));
        assert!(t.contains("skipped 1 denied (secrets list): .env"));
    }
}
