//! Ingest: walk a project, drop secrets and junk, window files into chunks.

mod chunk;
mod deny;
mod walk;

pub use deny::is_denied;
pub use walk::{collect, MAX_FILE_BYTES};

/// One window of a source file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceChunk {
    /// Relative to the project root, `/` separators.
    pub path: String,
    /// 1-based, inclusive.
    pub start_line: u32,
    /// 1-based, inclusive.
    pub end_line: u32,
    pub text: String,
}

/// Why a file produced no chunks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkipReason {
    /// Matched the secrets deny list; never opened.
    Denied,
    /// Symlink (file or directory); never followed.
    Symlink,
    /// NUL byte in the first 8 KiB.
    Binary,
    NotUtf8,
    /// Larger than [`MAX_FILE_BYTES`].
    TooLarge,
    Empty,
    /// Could not be read; carries the OS error.
    Unreadable(String),
}

/// Result of [`collect`]: chunks sorted by path then start line, plus every skipped file.
#[derive(Debug, Default)]
pub struct CollectReport {
    pub chunks: Vec<SourceChunk>,
    /// `(relative path, reason)`, sorted by path.
    pub skipped: Vec<(String, SkipReason)>,
}
