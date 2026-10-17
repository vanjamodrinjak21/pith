use pithdb::index::{kernel, VectorIndex};

struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
    fn unit(&mut self, dim: usize) -> Vec<f32> {
        let v: Vec<f32> = (0..dim).map(|_| self.next_f32()).collect();
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.into_iter().map(|x| x / n).collect()
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[test]
fn index_search_matches_naive_f32_top_k() {
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let data: Vec<Vec<f32>> = (0..2000).map(|_| rng.unit(384)).collect();
    let idx = VectorIndex::build(data.clone()).unwrap();
    assert_eq!((idx.len(), idx.dim()), (2000, 384));
    for _ in 0..50 {
        let q = rng.unit(384);
        let mut naive: Vec<(u32, f32)> = data
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u32, dot(&q, v)))
            .collect();
        naive.sort_by(|a, b| b.1.total_cmp(&a.1));
        let got: Vec<u32> = idx.search(&q, 10).iter().map(|h| h.id).collect();
        let want: Vec<u32> = naive[..10].iter().map(|h| h.0).collect();
        assert_eq!(got, want);
    }
}

#[test]
fn index_k_larger_than_len_returns_all() {
    let mut rng = Rng(7);
    let idx = VectorIndex::build((0..5).map(|_| rng.unit(8)).collect()).unwrap();
    let hits = idx.search(&rng.unit(8), 100);
    assert_eq!(hits.len(), 5);
    assert!(hits.windows(2).all(|w| w[0].score >= w[1].score));
}

#[test]
fn index_empty_index_returns_empty() {
    let idx = VectorIndex::build(vec![]).unwrap();
    assert!(idx.is_empty());
    assert!(idx.search(&[1.0, 0.0], 5).is_empty());
}

#[test]
fn index_ragged_dims_rejected() {
    assert!(VectorIndex::build(vec![vec![1.0, 0.0], vec![1.0]]).is_err());
}

#[test]
fn index_from_parts_roundtrip() {
    let mut rng = Rng(11);
    let idx = VectorIndex::build((0..20).map(|_| rng.unit(16)).collect()).unwrap();
    let back = VectorIndex::from_parts(16, idx.quantized().to_vec(), idx.full().to_vec()).unwrap();
    let q = rng.unit(16);
    assert_eq!(idx.search(&q, 3)[0].id, back.search(&q, 3)[0].id);
    assert!(VectorIndex::from_parts(16, vec![0; 15], vec![0.0; 16]).is_err());
}

#[test]
fn index_simd_kernel_equals_scalar_kernel() {
    let mut s: u64 = 42;
    for len in [0usize, 1, 7, 15, 16, 31, 32, 33, 100, 384, 385] {
        for _ in 0..20 {
            let mut gen = || {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 56) as u8 as i8
            };
            let a: Vec<i8> = (0..len).map(|_| gen()).collect();
            let b: Vec<i8> = (0..len).map(|_| gen()).collect();
            assert_eq!(
                kernel::dot_i8(&a, &b),
                kernel::dot_i8_scalar(&a, &b),
                "len {len}"
            );
        }
    }
    let a = vec![-128i8; 384];
    assert_eq!(kernel::dot_i8(&a, &a), kernel::dot_i8_scalar(&a, &a));
}
