//! Pith: a local single-file vector database with built-in ONNX embeddings, for searching a codebase by meaning.

/// Longest chunk the chunker emits, in chars; the storage table bound derives from it.
pub(crate) const CHUNK_MAX_CHARS: usize = 1_400;

mod budget;
pub mod commands;
pub mod embed;
pub mod error;
#[doc(hidden)] // the engine of `pith eval`; not a stable library API
pub mod eval;
pub mod index;
pub mod ingest;
#[cfg(feature = "mcp")]
pub mod mcp;
pub mod storage;
#[cfg(feature = "tui")]
pub mod tui;

pub use commands::{
    index_project, index_project_with_progress, is_no_model, is_yes, load_default_embedder,
    load_optional_embedder, open_project, render_error, render_search, render_stats,
    search_project, search_with, skip_summary, stats_project, IndexReport, Progress, SearchHit,
    SearchMode, Stats, KEYWORD_ONLY,
};
pub use error::{Error, Result};
