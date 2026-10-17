/// Crate-wide error type; later tasks add variants.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("not implemented: {0}")]
    NotImplemented(&'static str),
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimMismatch { expected: usize, got: usize },
    #[error("vector has zero length")]
    EmptyVector,
    #[error("vector contains NaN or infinity")]
    NonFinite,
    #[error("corrupt index: {0}")]
    CorruptIndex(String),
    #[error("database was built with model {found} ({found_dim}-d), expected {expected} ({expected_dim}-d); run `pith index --reindex`")]
    ModelMismatch {
        expected: String,
        expected_dim: usize,
        found: String,
        found_dim: usize,
    },
    #[error("no Pith index found at or above {0}; run `pith index` first")]
    NoIndex(String),
    #[error("another writer holds the database lock")]
    Locked,
    #[error("unsupported database format version {0}; run `pith index --reindex`")]
    UnsupportedFormat(u32),
    #[error("corrupt database: {0}")]
    CorruptDb(String),
    #[error("invalid input: {0}")]
    Invalid(String),
    #[error("embedding model not found: {0}; run `pith model pull` to download it")]
    ModelMissing(String),
    #[error("model download failed: {0}")]
    Download(String),
    #[error("model file failed verification: {0}")]
    Integrity(String),
    #[error("embedding failed: {0}")]
    Embed(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
