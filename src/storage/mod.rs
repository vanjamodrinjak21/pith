//! Single-file database (layout in `format.rs`). Writes replace the whole file
//! atomically (temp file, fsync, rename, fsync dir) under the writer lock; the file
//! is never modified in place, so readers always see a complete old or new file.

#![deny(unsafe_op_in_unsafe_fn)]

mod format;
mod locate;
mod lock;

pub use locate::locate_project;
pub use lock::WriterLock;

use crate::index::VectorIndex;
use crate::{Error, Result};
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

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
    index: VectorIndex,
}

/// Read-only mapping of the whole file. The only `unsafe` in storage.
fn map_readonly(file: &File) -> Result<memmap2::Mmap> {
    // SAFETY: Pith never modifies a database file in place: writers build a new file
    // and rename it over the old path, which leaves this inode (and so this mapping)
    // untouched. The mapping is read-only and is dropped before `Db::open` returns
    // (validated data is copied out). The remaining hazard is an external process
    // truncating or writing the same inode during that short window: a write can
    // only change bytes (caught by the checksums or, if after validation, copied as
    // is), while a truncation can raise SIGBUS. That cannot be prevented with a
    // shared mapping and is documented as unsupported.
    Ok(unsafe { memmap2::Mmap::map(file)? })
}

impl Db {
    /// Opens and fully validates the file. `expect` is (model id, dim); a mismatch
    /// is `Error::ModelMismatch`. Never panics on corrupt input.
    pub fn open(path: &Path, expect: Option<(&str, usize)>) -> Result<Db> {
        let file = File::open(path)?;
        // Mapping an empty file fails on some platforms; let decode report it.
        let d = if file.metadata()?.len() < format::HEADER_LEN as u64 {
            format::decode(&fs::read(path)?)?
        } else {
            format::decode(&map_readonly(&file)?)?
        };
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
            index: d.index,
        })
    }

    /// Atomically replaces `path`. Chunk `i` belongs to vector `i`. For an empty
    /// database pass `VectorIndex::from_parts(model_dim, vec![], vec![])` so the
    /// dimension is recorded.
    pub fn write(
        path: &Path,
        model_id: &str,
        chunks: Vec<StoredChunk>,
        index: &VectorIndex,
        meta: IndexMeta,
    ) -> Result<()> {
        Self::write_with_hook(path, model_id, chunks, index, meta, || Ok(()))
    }

    /// `write` with a hook run after the temp file is synced and before the rename;
    /// an error from it aborts the write. For failure-injection tests.
    #[doc(hidden)]
    pub fn write_with_hook(
        path: &Path,
        model_id: &str,
        chunks: Vec<StoredChunk>,
        index: &VectorIndex,
        meta: IndexMeta,
        before_rename: impl FnOnce() -> Result<()>,
    ) -> Result<()> {
        let bytes = format::encode(model_id, &chunks, index, &meta)?;
        let _lock = WriterLock::acquire(path)?;
        let mut tmp_name = path.file_name().unwrap_or_default().to_os_string();
        tmp_name.push(".tmp");
        let tmp = path.with_file_name(tmp_name);
        // The lock makes us the only writer, so a fixed temp name is safe; a stale one
        // from a killed writer is simply overwritten.
        let result = (|| {
            let mut f = File::create(&tmp)?;
            f.write_all(&bytes)?;
            f.sync_all()?;
            before_rename()?;
            fs::rename(&tmp, path)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&tmp);
            return result;
        }
        #[cfg(unix)]
        {
            let dir = path.parent().filter(|p| !p.as_os_str().is_empty());
            File::open(dir.unwrap_or(Path::new(".")))?.sync_all()?;
        }
        Ok(())
    }

    pub fn chunk(&self, id: u32) -> Option<&StoredChunk> {
        self.chunks.get(id as usize)
    }

    pub fn index(&self) -> &VectorIndex {
        &self.index
    }

    pub fn meta(&self) -> &IndexMeta {
        &self.meta
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
