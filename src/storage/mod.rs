//! Single-file database (layout in `format.rs`). Writes replace the whole file
//! atomically (unique temp file, fsync, rename, fsync dir) under the writer lock;
//! Pith never modifies a database file in place, so a reader sees a complete old or
//! new file. Vectors stay in a shared read-only mapping for the life of the index.
//!
//! Unsupported: another process modifying or truncating a database file in place
//! while it is open. Modification after `open` is not detected; truncation makes
//! reads of the mapping raise SIGBUS (inherent to mmap; accepted trade-off).
//!
//! Windows (best-effort): replacing a file that a reader has mapped fails there, so
//! a write can fail while any `Db` or `VectorIndex` from that file is alive.

#![deny(unsafe_op_in_unsafe_fn)]

mod format;
mod locate;
mod lock;

pub use locate::{locate_project, locate_project_within};
#[doc(hidden)]
pub use lock::WriterLock;

use crate::index::VectorIndex;
use crate::{Error, Result};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredChunk {
    pub path: String,
    pub start_line: u32,
    pub end_line: u32,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexMeta {
    pub indexed_at_unix: u64,
    pub git_head: Option<String>,
}

pub struct Db {
    model_id: String,
    meta: IndexMeta,
    chunks: Vec<StoredChunk>,
    hashes: Vec<[u8; 32]>,
    index: VectorIndex,
}

/// Read-only shared mapping of the whole file. The only `unsafe` in storage.
fn map_readonly(file: &File) -> Result<memmap2::Mmap> {
    // SAFETY: memmap2 requires that the file is not modified or truncated while
    // mapped. Pith guarantees this for its own writes: a database file is never
    // modified in place, only replaced by renaming a new file over its path, which
    // leaves the mapped inode untouched. Nothing protects against another process
    // writing or truncating the same inode; that is unsupported (see module docs):
    // a write is not detected after `open` validated the file, and a truncation
    // makes later reads raise SIGBUS.
    Ok(unsafe { memmap2::Mmap::map(file)? })
}

impl Db {
    /// Opens and fully validates the file. `expect` is (model id, dim); a mismatch
    /// is `Error::ModelMismatch`. Never panics on corrupt input. The vectors stay
    /// mapped; the mapping lives as long as this `Db` or its index.
    pub fn open(path: &Path, expect: Option<(&str, usize)>) -> Result<Db> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        // Too short to be valid; also avoids mapping an empty file.
        if len < format::MIN_LEN as u64 {
            return Err(Error::CorruptDb(format!("file is {len} bytes, too short")));
        }
        let d = format::decode(Arc::new(map_readonly(&file)?))?;
        if let Some((model, dim)) = expect {
            if d.model_id != model || d.index.dim() != dim {
                return Err(Error::ModelMismatch {
                    expected: model.into(),
                    expected_dim: dim,
                    found: d.model_id,
                    found_dim: d.index.dim(),
                });
            }
        }
        Ok(Db {
            model_id: d.model_id,
            meta: d.meta,
            chunks: d.chunks,
            hashes: d.hashes,
            index: d.index,
        })
    }

    /// Atomically replaces `path`. Chunk `i` belongs to vector `i`. For an empty
    /// database pass `VectorIndex::from_parts(model_dim, vec![], vec![])` so the
    /// dimension is recorded; a dimension of 0 is `Error::Invalid`.
    ///
    /// On any error before the rename, `path` is untouched and the temp file is
    /// removed. The rename is the commit point: if only the directory fsync after it
    /// fails, the new file is already in place (but may not survive power loss) and
    /// `Err` is returned. Either way `path` holds a complete old or new database.
    pub fn write(
        path: &Path,
        model_id: &str,
        chunks: Vec<StoredChunk>,
        index: &VectorIndex,
        meta: IndexMeta,
    ) -> Result<()> {
        Self::write_checked(path, model_id, chunks, index, meta, || Ok(()))
    }

    /// [`write`](Self::write), calling `before_rename` once the temp file is synced; an error
    /// from it aborts the write (temp file removed, `path` untouched) and is returned.
    pub(crate) fn write_checked(
        path: &Path,
        model_id: &str,
        chunks: Vec<StoredChunk>,
        index: &VectorIndex,
        meta: IndexMeta,
        before_rename: impl FnOnce() -> Result<()>,
    ) -> Result<()> {
        if let Some(m) = format::oversized(&chunks) {
            return Err(Error::Invalid(m));
        }
        write_inner(path, model_id, &chunks, index, &meta, before_rename)
    }

    pub fn chunk(&self, id: u32) -> Option<&StoredChunk> {
        self.chunks.get(id as usize)
    }

    /// All chunks, in id order.
    pub fn chunks(&self) -> &[StoredChunk] {
        &self.chunks
    }

    /// blake3 of chunk `id`'s text: the exact input its vector was embedded from.
    pub fn text_hash(&self, id: u32) -> Option<&[u8; 32]> {
        self.hashes.get(id as usize)
    }

    pub fn index(&self) -> &VectorIndex {
        &self.index
    }

    /// Takes the index out; it keeps the mapping alive after the `Db` is gone.
    pub fn into_index(self) -> VectorIndex {
        self.index
    }

    pub fn meta(&self) -> &IndexMeta {
        &self.meta
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }
}

const TEMP_RAND: usize = 6;

/// Removes temp files left by a killed writer: `<prefix>` + exactly `TEMP_RAND`
/// alphanumerics. Only called with the writer lock held, so none belongs to a live
/// writer. A matching symlink is removed itself, never followed.
fn remove_stale_temps(dir: &Path, prefix: &str) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(rest) = name.to_str().and_then(|n| n.strip_prefix(prefix)) else {
            continue;
        };
        if !prefix.is_empty()
            && rest.len() == TEMP_RAND
            && rest.bytes().all(|b| b.is_ascii_alphanumeric())
            && !entry.file_type()?.is_dir()
        {
            std::fs::remove_file(entry.path())?;
        }
    }
    Ok(())
}

/// `before_rename` runs after the temp file is synced; an error aborts the write.
fn write_inner(
    path: &Path,
    model_id: &str,
    chunks: &[StoredChunk],
    index: &VectorIndex,
    meta: &IndexMeta,
    before_rename: impl FnOnce() -> Result<()>,
) -> Result<()> {
    let _lock = WriterLock::acquire(path)?;
    let dir = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut prefix = path.file_name().unwrap_or_default().to_os_string();
    prefix.push(".tmp");
    remove_stale_temps(dir, prefix.to_str().unwrap_or_default())?;
    // Random name created with O_EXCL, so a planted file or symlink is never
    // followed. Deleted on drop, i.e. on every early return. tempfile creates it
    // with mode 0600, which the database keeps after the rename: it holds source
    // code, so it is deliberately readable by its owner only.
    let tmp = tempfile::Builder::new()
        .prefix(&prefix)
        .rand_bytes(TEMP_RAND)
        .tempfile_in(dir)?;
    format::encode(BufWriter::new(tmp.as_file()), model_id, chunks, index, meta)?;
    tmp.as_file().sync_all()?;
    before_rename()?;
    tmp.persist(path).map_err(|e| Error::Io(e.error))?;
    #[cfg(unix)]
    File::open(dir)?.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn oversized_chunks_are_refused_on_write_and_rejected_on_open() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("db");
        let index = VectorIndex::build(vec![vec![0.6, 0.8]]).unwrap();
        let meta = IndexMeta {
            indexed_at_unix: 0,
            git_head: None,
        };
        for (path, text) in [
            ("a".repeat(format::MAX_PATH_BYTES + 1), "t".to_string()),
            ("a".to_string(), "t".repeat(format::MAX_TEXT_BYTES + 1)),
        ] {
            let chunks = vec![StoredChunk {
                path,
                start_line: 1,
                end_line: 1,
                text,
            }];
            let e = Db::write(&p, "m", chunks.clone(), &index, meta.clone()).unwrap_err();
            assert!(matches!(e, Error::Invalid(_)), "{e:?}");
            // Bypass the write check: decode must reject it as corrupt, not panic.
            let mut f = File::create(&p).unwrap();
            format::encode(&mut f, "m", &chunks, &index, &meta).unwrap();
            drop(f);
            let Err(e) = Db::open(&p, None) else {
                panic!("opened")
            };
            assert!(
                matches!(e, Error::CorruptDb(ref m) if m.contains("bytes (limit")),
                "{e:?}"
            );
        }
        // At the limits it round-trips.
        let chunks = vec![StoredChunk {
            path: "a".repeat(format::MAX_PATH_BYTES),
            start_line: 1,
            end_line: 1,
            text: "t".repeat(format::MAX_TEXT_BYTES),
        }];
        Db::write(&p, "m", chunks, &index, meta).unwrap();
        assert!(Db::open(&p, None).is_ok());
    }

    #[test]
    fn map_readonly_matches_file_and_survives_replace_by_rename() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("f");
        let data: Vec<u8> = (0..10_000u32).map(|i| (i * 7) as u8).collect();
        fs::write(&p, &data).unwrap();
        let map = map_readonly(&File::open(&p).unwrap()).unwrap();
        assert_eq!(&map[..], &data[..]);
        // Replacing the path the way writers do leaves the old mapping intact.
        let other = dir.path().join("g");
        fs::write(&other, b"new").unwrap();
        fs::rename(&other, &p).unwrap();
        assert_eq!(&map[..], &data[..]);
    }

    #[test]
    fn failed_write_leaves_previous_file_intact() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".pith");
        let one = |dim| VectorIndex::build(vec![vec![0.5; dim]]).unwrap();
        let c = StoredChunk {
            path: "a".into(),
            start_line: 1,
            end_line: 2,
            text: "t".into(),
        };
        let meta = IndexMeta {
            indexed_at_unix: 1,
            git_head: None,
        };
        Db::write(&path, "old", vec![c.clone()], &one(4), meta.clone()).unwrap();
        let before = fs::read(&path).unwrap();
        let r = write_inner(&path, "new", &[c], &one(4), &meta, || {
            Err(Error::Io(std::io::Error::other("injected")))
        });
        assert!(matches!(r, Err(Error::Io(e)) if e.to_string() == "injected"));
        assert_eq!(fs::read(&path).unwrap(), before);
        assert_eq!(Db::open(&path, None).unwrap().model_id(), "old");
        let mut names: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        names.sort();
        assert_eq!(names, [".pith", ".pith.lock"], "temp file left behind");
    }
}
