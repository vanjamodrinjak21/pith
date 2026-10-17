//! Ingest: walk a project, drop secrets and junk, window files into chunks.
//!
//! The walk honours `.gitignore`, `.ignore`, `.git/info/exclude` and a project-local
//! `.pithignore` (never the global gitignore, never parent directories). Dot-files and
//! dot-directories are walked on purpose so the secrets deny list, not the walker's
//! hidden-file filter, decides what is excluded; `.git` itself is never entered.
//!
//! Limits, stated plainly. The PEM content check is header-only: raw base64 or DER keys
//! without a header are not detected by content (DER is binary and skipped as such).
//! Unicode-confusable file names are not normalized; they are different names from the
//! real secret names, so they are not a bypass. On non-unix targets (best effort) files
//! are opened without `O_NOFOLLOW`, without a file-identity check and without the
//! resolved-path check; only the walk-time symlink skip applies, so the race guarantee
//! documented on [`collect`] is weaker there.

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
    /// FIFO, socket or device; never opened.
    SpecialFile,
    /// Contains a PEM private-key header.
    Secret,
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
