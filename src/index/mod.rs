//! Cosine index: an int8 SIMD scan selects the top `max(CANDIDATE_MULT * k, 32)` candidates,
//! which are then rescored with f32 dot products. Not strictly exact: a true top-k vector
//! could be lost in the int8 pass (recall is checked in tests/index_reference.rs).

pub(crate) mod bm25;
pub(crate) mod kernel;
mod quant;

use crate::{Error, Result};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::ops::Range;
use std::sync::Arc;

/// Candidates kept per requested hit during the int8 scan.
const CANDIDATE_MULT: usize = 4;

/// Largest supported vector dimension.
pub const MAX_DIM: usize = 65_536;

fn check_parts(dim: usize, full: &[f32]) -> Result<()> {
    if dim > MAX_DIM {
        return Err(Error::Invalid(format!("dimension {dim} exceeds {MAX_DIM}")));
    }
    if full.iter().any(|x| !x.is_finite()) {
        return Err(Error::NonFinite);
    }
    if dim > 0 && full.len() / dim > u32::MAX as usize {
        return Err(Error::CorruptIndex("more than u32::MAX rows".into()));
    }
    Ok(())
}

/// Checks shape, finiteness and that `quantized == quantize(full)`; returns the scale.
/// Compares element by element so no second copy of the vectors is allocated.
fn validate(dim: usize, quantized: &[i8], full: &[f32]) -> Result<f32> {
    let ok = quantized.len() == full.len()
        && (dim == 0 && full.is_empty() || dim > 0 && full.len().is_multiple_of(dim));
    if !ok {
        return Err(Error::CorruptIndex(format!(
            "dim {dim}, {} quantized, {} full values",
            quantized.len(),
            full.len()
        )));
    }
    check_parts(dim, full)?;
    let scale = quant::scale_for(full);
    if full
        .iter()
        .zip(quantized)
        .any(|(&x, &q)| quant::quantize_one(x, scale) != q)
    {
        return Err(Error::CorruptIndex(
            "quantized data does not match full vectors".into(),
        ));
    }
    Ok(scale)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    pub id: u32,
    pub score: f32,
}

/// Vector storage: owned, or a view into a shared read-only file mapping.
enum Store {
    Owned {
        quantized: Vec<i8>,
        full: Vec<f32>,
    },
    Mapped {
        map: Arc<memmap2::Mmap>,
        quantized: Range<usize>,
        full: Range<usize>,
    },
}

pub struct VectorIndex {
    dim: usize,
    store: Store,
    /// Query quantization scale; matches the stored data's scale.
    scale: f32,
}

impl VectorIndex {
    /// Vectors must already be L2-normalized and share one dimension.
    pub fn build(vectors: Vec<Vec<f32>>) -> Result<Self> {
        let dim = vectors.first().map_or(0, Vec::len);
        if let Some(v) = vectors.iter().find(|v| v.len() != dim) {
            return Err(Error::DimMismatch {
                expected: dim,
                got: v.len(),
            });
        }
        if dim == 0 && !vectors.is_empty() {
            return Err(Error::EmptyVector);
        }
        let full: Vec<f32> = vectors.into_iter().flatten().collect();
        check_parts(dim, &full)?;
        let scale = quant::scale_for(&full);
        Ok(Self {
            dim,
            store: Store::Owned {
                quantized: quant::quantize(&full, scale),
                full,
            },
            scale,
        })
    }

    pub fn from_parts(dim: usize, quantized: Vec<i8>, full: Vec<f32>) -> Result<Self> {
        let scale = validate(dim, &quantized, &full)?;
        Ok(Self {
            dim,
            store: Store::Owned { quantized, full },
            scale,
        })
    }

    /// A zero-copy index over `map[quantized]` (int8) and `map[full]` (f32, native
    /// endian, 4-byte aligned). Validated exactly like [`from_parts`](Self::from_parts).
    pub(crate) fn from_mapped(
        dim: usize,
        map: Arc<memmap2::Mmap>,
        quantized: Range<usize>,
        full: Range<usize>,
    ) -> Result<Self> {
        let bad = |what: &str| {
            Error::CorruptIndex(format!("mapped {what} section out of bounds or misaligned"))
        };
        let q = map.get(quantized.clone()).ok_or_else(|| bad("int8"))?;
        let f = map.get(full.clone()).ok_or_else(|| bad("f32"))?;
        let f: &[f32] = bytemuck::try_cast_slice(f).map_err(|_| bad("f32"))?;
        let scale = validate(dim, bytemuck::cast_slice(q), f)?;
        Ok(Self {
            dim,
            store: Store::Mapped {
                map,
                quantized,
                full,
            },
            scale,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.full().len().checked_div(self.dim).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The int8 quantization scale: `quantized[i] ≈ full[i] / scale`.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn quantized(&self) -> &[i8] {
        match &self.store {
            Store::Owned { quantized, .. } => quantized,
            Store::Mapped { map, quantized, .. } => bytemuck::cast_slice(&map[quantized.clone()]),
        }
    }

    pub fn full(&self) -> &[f32] {
        match &self.store {
            Store::Owned { full, .. } => full,
            // Alignment and length were checked in `from_mapped`.
            Store::Mapped { map, full, .. } => bytemuck::cast_slice(&map[full.clone()]),
        }
    }

    /// Like [`try_search`](Self::try_search), but returns empty on a dimension
    /// mismatch or a non-finite query. The query must be L2-normalized so the
    /// dot product equals cosine (an unnormalized query only scales scores).
    pub fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        self.try_search(query, k).unwrap_or_default()
    }

    /// Top `min(k, len)` hits by cosine, descending (ties by ascending id).
    /// The query must be L2-normalized so the dot product equals cosine; an
    /// unnormalized query only scales the scores and does not change the ranking.
    /// The query is validated before `k == 0` or emptiness short-circuits; the one
    /// exception is a dimensionless index (`dim() == 0`, built from no vectors),
    /// which has nothing to validate against and returns `Ok(vec![])`.
    pub fn try_search(&self, query: &[f32], k: usize) -> Result<Vec<Hit>> {
        if self.dim == 0 {
            return Ok(Vec::new());
        }
        if query.len() != self.dim {
            return Err(Error::DimMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if query.iter().any(|x| !x.is_finite()) {
            return Err(Error::NonFinite);
        }
        if self.is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        let q8 = quant::quantize(query, self.scale);
        let cap = k.saturating_mul(CANDIDATE_MULT).max(32).min(self.len());
        // Min-heap of the best `cap` int8 scores.
        let mut heap: BinaryHeap<Reverse<(i32, u32)>> = BinaryHeap::with_capacity(cap + 1);
        kernel::dot_i8_rows(&q8, self.quantized(), |id, s| {
            if heap.len() < cap {
                heap.push(Reverse((s, id as u32)));
            } else if s > heap.peek().unwrap().0 .0 {
                heap.pop();
                heap.push(Reverse((s, id as u32)));
            }
        });
        let full = self.full();
        let mut hits: Vec<Hit> = heap
            .into_iter()
            .map(|Reverse((_, id))| {
                let row = &full[id as usize * self.dim..][..self.dim];
                Hit {
                    id,
                    score: row.iter().zip(query).map(|(a, b)| a * b).sum(),
                }
            })
            .collect();
        hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.id.cmp(&b.id)));
        hits.truncate(k);
        Ok(hits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimension_over_the_limit_is_an_error() {
        let mut v = vec![0.0f32; MAX_DIM + 1];
        v[0] = 1.0;
        assert!(matches!(
            VectorIndex::build(vec![v]),
            Err(Error::Invalid(_))
        ));
        assert!(matches!(
            VectorIndex::from_parts(MAX_DIM + 1, vec![], vec![]),
            Err(Error::Invalid(_))
        ));
        assert!(VectorIndex::from_parts(MAX_DIM, vec![], vec![]).is_ok());
    }
}
