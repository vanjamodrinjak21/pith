//! Text embeddings: the [`Embedder`] trait, the ONNX implementation (feature `onnx`), a
//! deterministic [`FakeEmbedder`] for tests, and verified model delivery ([`pull`]).
//!
//! Network access happens only inside [`pull`]; loading and embedding are fully offline.

mod manifest;
#[cfg(feature = "onnx")]
mod onnx;
mod pull;

pub use manifest::{Manifest, ModelFile, DEFAULT_MODEL};
#[cfg(feature = "onnx")]
pub use onnx::{OnnxConfig, OnnxEmbedder};
pub use pull::{pull, pull_files};

use crate::{Error, Result};
use std::path::PathBuf;

/// Produces L2-normalized, finite vectors of length [`Embedder::dim`].
pub trait Embedder {
    fn model_id(&self) -> &str;
    fn dim(&self) -> usize;
    fn embed_passages(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    fn embed_query(&self, text: &str) -> Result<Vec<f32>>;
}

/// `$PITH_HOME/models`, else `$HOME/.pith/models`. Pure: creating it is `pull`'s job.
pub fn models_dir() -> PathBuf {
    match std::env::var_os("PITH_HOME") {
        Some(h) if !h.is_empty() => PathBuf::from(h).join("models"),
        _ => PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".pith/models"),
    }
}

/// Scales `v` to unit length; rejects NaN/inf and all-zero vectors.
fn normalize(v: &mut [f32]) -> Result<()> {
    if !v.iter().all(|x| x.is_finite()) {
        return Err(Error::NonFinite);
    }
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n == 0.0 {
        return Err(Error::EmptyVector);
    }
    v.iter_mut().for_each(|x| *x /= n);
    Ok(())
}

/// CLS pooling over a row-major `[batch, seq, dim]` hidden-state tensor, then L2 normalization.
pub fn cls_pool_normalize(
    hidden: &[f32],
    batch: usize,
    seq: usize,
    dim: usize,
) -> Result<Vec<Vec<f32>>> {
    if seq == 0 || hidden.len() != batch * seq * dim {
        return Err(Error::Embed(format!(
            "hidden state has {} values, expected {batch}x{seq}x{dim}",
            hidden.len()
        )));
    }
    (0..batch)
        .map(|b| {
            let mut v = hidden[b * seq * dim..][..dim].to_vec();
            normalize(&mut v)?;
            Ok(v)
        })
        .collect()
}

/// Deterministic hash-based 384-d embedder for tests. Always compiled (tiny, no deps beyond
/// blake3) so integration tests and `--no-default-features` builds can use it.
/// Queries embed exactly like passages, so a query equal to a passage scores 1.0.
#[derive(Debug, Clone, Copy, Default)]
pub struct FakeEmbedder;

impl FakeEmbedder {
    fn one(text: &str) -> Vec<f32> {
        let mut bytes = [0u8; 384 * 4];
        blake3::Hasher::new()
            .update(text.as_bytes())
            .finalize_xof()
            .fill(&mut bytes);
        let mut v: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32 / u32::MAX as f32 - 0.5)
            .collect();
        normalize(&mut v).expect("hash output is finite and nonzero");
        v
    }
}

impl Embedder for FakeEmbedder {
    fn model_id(&self) -> &str {
        "fake-384"
    }
    fn dim(&self) -> usize {
        384
    }
    fn embed_passages(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| Self::one(t)).collect())
    }
    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        Ok(Self::one(text))
    }
}
