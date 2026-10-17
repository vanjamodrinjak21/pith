//! Library behind the `pith` subcommands: index a project, search it, report stats.
//! Network access never happens here (only `embed::pull` touches the network).

use crate::embed::Embedder;
use crate::ingest::{collect, CollectReport, SkipReason};
use crate::storage::{locate_project, Db, IndexMeta, StoredChunk, WriterLock};
use crate::{index::VectorIndex, Error, Result};
use std::collections::HashSet;
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
    /// Files that produced no chunks, with the reason.
    pub skipped: Vec<(String, SkipReason)>,
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

fn now_unix() -> u64 {
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

#[cfg(not(feature = "onnx"))]
pub fn load_default_embedder() -> Result<Box<dyn Embedder>> {
    Err(Error::Invalid(
        "this build of pith has no embedding model support (built without the `onnx` feature)"
            .into(),
    ))
}

/// Refuses to overwrite an index built with another model, or one that cannot be read.
fn check_existing(db_path: &Path, embedder: &dyn Embedder) -> Result<()> {
    match db_path.symlink_metadata() {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e.into()),
        Ok(_) => {}
    }
    match Db::open(db_path, Some((embedder.model_id(), embedder.dim()))) {
        Ok(_) => Ok(()),
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
fn update_gitignore(root: &Path) -> Result<()> {
    use std::io::{Read, Write};
    let mut o = std::fs::OpenOptions::new();
    o.read(true).append(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::custom_flags(
        &mut o,
        (rustix::fs::OFlags::NOFOLLOW | rustix::fs::OFlags::NONBLOCK).bits() as i32,
    );
    let mut f = match o.open(root.join(".gitignore")) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e.into()),
    };
    if !f.metadata()?.is_file() {
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
    for name in [DB_NAME, ".pith.lock"] {
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
    f.write_all(add.as_bytes())?;
    Ok(())
}

/// Walks `root`, embeds every chunk and atomically writes `<root>/.pith`. Without `reindex`, an
/// existing index built with a different model (or unreadable) is an error. The writer lock is
/// taken only for the final write, so a second concurrent writer fails with `Error::Locked`
/// after it has done its embedding work. Progress goes to stderr, only when stderr is a TTY.
pub fn index_project(root: &Path, embedder: &dyn Embedder, reindex: bool) -> Result<IndexReport> {
    let db_path = root.join(DB_NAME);
    // Fail fast if another writer is active, before the expensive embedding. Racy by nature: a
    // writer may start after this check, in which case `Db::write` still returns Locked.
    drop(WriterLock::acquire(&db_path)?);
    if !reindex {
        check_existing(&db_path, embedder)?;
    }
    // Taken before the walk: any file edited after this second is flagged by `source_changed`.
    let indexed_at_unix = now_unix();
    let CollectReport { chunks, skipped } = collect(root)?;
    let total = chunks.len();
    let progress = total > 0 && std::io::stderr().is_terminal();
    let mut vectors = Vec::with_capacity(total);
    for group in chunks.chunks(EMBED_GROUP) {
        let texts: Vec<String> = group.iter().map(|c| c.text.clone()).collect();
        let v = embedder.embed_passages(&texts)?;
        if v.len() != texts.len() {
            return Err(Error::Embed(format!(
                "embedder returned {} vectors for {} chunks",
                v.len(),
                texts.len()
            )));
        }
        vectors.extend(v);
        if progress {
            eprint!("\rembedding {}/{total}", vectors.len());
        }
    }
    if progress {
        eprint!("\r\x1b[K");
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
    let files = chunks
        .iter()
        .map(|c| c.path.as_str())
        .collect::<HashSet<_>>()
        .len();
    let stored = chunks
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
    Db::write(&db_path, embedder.model_id(), stored, &index, meta)?;
    // The index is already written; a .gitignore problem must not fail the run.
    if let Err(e) = update_gitignore(root) {
        eprintln!("pith: warning: could not update .gitignore: {e}");
    }
    Ok(IndexReport {
        chunks: total,
        files,
        skipped,
    })
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
fn safe_join(root: &Path, rel: &str) -> Option<PathBuf> {
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

/// Top-`k` chunks for `query` (k must be at least 1; capped at 100). `root` is the project root, used only to
/// compute `source_changed`. Errors with `ModelMismatch` if the embedder is not the one that
/// built the database.
pub fn search_project(
    root: &Path,
    db: &Db,
    embedder: &dyn Embedder,
    query: &str,
    k: usize,
) -> Result<Vec<SearchHit>> {
    if query.trim().is_empty() {
        return Err(Error::Invalid("query must not be empty".into()));
    }
    if k == 0 {
        return Err(Error::Invalid("k must be at least 1".into()));
    }
    if db.model_id() != embedder.model_id() || db.index().dim() != embedder.dim() {
        return Err(Error::ModelMismatch {
            expected: embedder.model_id().into(),
            expected_dim: embedder.dim(),
            found: db.model_id().into(),
            found_dim: db.index().dim(),
        });
    }
    let q = embedder.embed_query(query)?;
    let hits = db.index().try_search(&q, k.min(MAX_K))?;
    let at = db.meta().indexed_at_unix;
    Ok(hits
        .into_iter()
        .filter_map(|h| {
            let c = db.chunk(h.id)?;
            Some(SearchHit {
                chunk_id: h.id,
                score: h.score,
                path: c.path.clone(),
                start_line: c.start_line,
                end_line: c.end_line,
                text: c.text.clone(),
                source_changed: changed_since(root, &c.path, at),
            })
        })
        .collect())
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
        age_secs: now_unix().saturating_sub(at),
        git_head: db.meta().git_head.as_deref().map(|h| {
            let ok = matches!(h.len(), 40 | 64)
                && h.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'));
            if ok {
                h.to_string()
            } else {
                "(invalid)".to_string()
            }
        }),
    })
}

pub fn age(secs: u64) -> String {
    match secs {
        0..=59 => format!("{secs} seconds ago"),
        60..=3599 => format!("{} minutes ago", secs / 60),
        3600..=86399 => format!("{} hours ago", secs / 3600),
        _ => format!("{} days ago", secs / 86400),
    }
}

/// First 3 lines, each cut to 100 chars, indented, control characters scrubbed.
pub fn snippet(text: &str) -> String {
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

/// `[Y/n]` answer: empty, `y` or `yes` (any case) is yes; everything else is no.
pub fn is_yes(line: &str) -> bool {
    matches!(line.trim().to_ascii_lowercase().as_str(), "" | "y" | "yes")
}

pub fn now() -> u64 {
    now_unix()
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
            c => c,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

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
