//! Exact cosine index: int8 SIMD scan to a candidate set, then f32 rescoring.

pub mod kernel;
mod quant;

use crate::{Error, Result};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    pub id: u32,
    pub score: f32,
}

pub struct VectorIndex {
    dim: usize,
    quantized: Vec<i8>,
    full: Vec<f32>,
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
        let full: Vec<f32> = vectors.into_iter().flatten().collect();
        let scale = quant::scale_for(&full);
        Ok(Self {
            dim,
            quantized: quant::quantize(&full, scale),
            full,
            scale,
        })
    }

    pub fn from_parts(dim: usize, quantized: Vec<i8>, full: Vec<f32>) -> Result<Self> {
        let ok = quantized.len() == full.len()
            && (dim == 0 && full.is_empty() || dim > 0 && full.len().is_multiple_of(dim));
        if !ok {
            return Err(Error::CorruptIndex(format!(
                "dim {dim}, {} quantized, {} full values",
                quantized.len(),
                full.len()
            )));
        }
        let scale = quant::scale_for(&full);
        Ok(Self {
            dim,
            quantized,
            full,
            scale,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        if self.dim == 0 {
            0
        } else {
            self.full.len() / self.dim
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn quantized(&self) -> &[i8] {
        &self.quantized
    }

    pub fn full(&self) -> &[f32] {
        &self.full
    }

    /// Top `min(k, len)` hits by cosine, descending. Query must match `dim`.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        if self.is_empty() || k == 0 || query.len() != self.dim {
            return Vec::new();
        }
        let q8 = quant::quantize(query, self.scale);
        let cap = (4 * k).max(32);
        // Min-heap of the best `cap` int8 scores.
        let mut heap: BinaryHeap<Reverse<(i32, u32)>> = BinaryHeap::with_capacity(cap + 1);
        for (id, row) in self.quantized.chunks_exact(self.dim).enumerate() {
            let s = kernel::dot_i8(&q8, row);
            if heap.len() < cap {
                heap.push(Reverse((s, id as u32)));
            } else if s > heap.peek().unwrap().0 .0 {
                heap.pop();
                heap.push(Reverse((s, id as u32)));
            }
        }
        let mut hits: Vec<Hit> = heap
            .into_iter()
            .map(|Reverse((_, id))| {
                let row = &self.full[id as usize * self.dim..][..self.dim];
                Hit {
                    id,
                    score: row.iter().zip(query).map(|(a, b)| a * b).sum(),
                }
            })
            .collect();
        hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.id.cmp(&b.id)));
        hits.truncate(k);
        hits
    }
}
