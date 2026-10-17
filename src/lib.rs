//! Pith: lightweight embeddable vector database. Work in progress.

/// Longest chunk the chunker emits, in chars; the storage table bound derives from it.
pub(crate) const CHUNK_MAX_CHARS: usize = 1_400;

pub mod embed;
pub mod error;
pub mod index;
pub mod ingest;
pub mod mcp;
pub mod storage;

pub use error::{Error, Result};
