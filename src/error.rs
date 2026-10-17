/// Crate-wide error type; later tasks add variants.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("not implemented: {0}")]
    NotImplemented(&'static str),
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimMismatch { expected: usize, got: usize },
    #[error("vector contains NaN or infinity")]
    NonFinite,
    #[error("corrupt index: {0}")]
    CorruptIndex(String),
}

pub type Result<T> = std::result::Result<T, Error>;
