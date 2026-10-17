//! Pith: lightweight embeddable vector database. Work in progress.

pub mod embed;
pub mod error;
pub mod index;
pub mod ingest;
pub mod mcp;
pub mod storage;

pub use error::{Error, Result};
