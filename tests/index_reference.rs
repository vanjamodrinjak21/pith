use pithdb::index::VectorIndex;
use pithdb::Error;

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
fn search_matches_naive_f32_top_k() {
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
fn k_larger_than_len_returns_all() {
    let mut rng = Rng(7);
    let idx = VectorIndex::build((0..5).map(|_| rng.unit(8)).collect()).unwrap();
    let hits = idx.search(&rng.unit(8), 100);
    assert_eq!(hits.len(), 5);
    assert!(hits.windows(2).all(|w| w[0].score >= w[1].score));
}

#[test]
fn empty_index_returns_empty() {
    let idx = VectorIndex::build(vec![]).unwrap();
    assert!(idx.is_empty());
    assert!(idx.search(&[1.0, 0.0], 5).is_empty());
}

#[test]
fn ragged_dims_rejected() {
    assert!(VectorIndex::build(vec![vec![1.0, 0.0], vec![1.0]]).is_err());
}

#[test]
fn from_parts_roundtrip() {
    let mut rng = Rng(11);
    let idx = VectorIndex::build((0..20).map(|_| rng.unit(16)).collect()).unwrap();
    let back = VectorIndex::from_parts(16, idx.quantized().to_vec(), idx.full().to_vec()).unwrap();
    let q = rng.unit(16);
    assert_eq!(idx.search(&q, 5), back.search(&q, 5));
    assert!(VectorIndex::from_parts(16, vec![0; 15], vec![0.0; 16]).is_err());
}

#[test]
fn huge_k_is_bounded() {
    let mut rng = Rng(3);
    let idx = VectorIndex::build((0..10).map(|_| rng.unit(8)).collect()).unwrap();
    assert_eq!(idx.search(&rng.unit(8), usize::MAX).len(), 10);
}

#[test]
fn k_zero_returns_empty() {
    let mut rng = Rng(5);
    let idx = VectorIndex::build((0..10).map(|_| rng.unit(8)).collect()).unwrap();
    assert!(idx.search(&rng.unit(8), 0).is_empty());
}

#[test]
fn ties_ordered_by_id() {
    let v = vec![1.0f32, 0.0];
    let idx = VectorIndex::build(vec![v.clone(), vec![0.0, 1.0], v.clone(), v.clone()]).unwrap();
    let ids: Vec<u32> = idx.search(&v, 4).iter().map(|h| h.id).collect();
    assert_eq!(ids, vec![0, 2, 3, 1]);
}

#[test]
fn non_finite_rejected() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(matches!(
            VectorIndex::build(vec![vec![1.0, 0.0], vec![bad, 0.0]]),
            Err(Error::NonFinite)
        ));
        assert!(matches!(
            VectorIndex::from_parts(2, vec![0, 0], vec![bad, 0.0]),
            Err(Error::NonFinite)
        ));
    }
}

#[test]
fn zero_length_rows_rejected() {
    assert!(matches!(
        VectorIndex::build(vec![vec![], vec![]]),
        Err(Error::EmptyVector)
    ));
}

#[test]
fn try_search_errors() {
    let idx = VectorIndex::build(vec![vec![1.0, 0.0]]).unwrap();
    assert!(matches!(
        idx.try_search(&[1.0, 0.0, 0.0], 1),
        Err(Error::DimMismatch {
            expected: 2,
            got: 3
        })
    ));
    assert!(matches!(
        idx.try_search(&[f32::NAN, 0.0], 1),
        Err(Error::NonFinite)
    ));
    assert!(idx.search(&[1.0], 1).is_empty());
    assert!(idx.search(&[f32::NAN, 0.0], 1).is_empty());
    assert_eq!(idx.try_search(&[1.0, 0.0], 1).unwrap()[0].id, 0);
}

/// Clustered data with a few outlier-heavy dims, which stresses the shared int8 scale.
#[test]
fn recall_on_anisotropic_clustered_data() {
    let dim = 384;
    let mut rng = Rng(0xDEADBEEF);
    let centers: Vec<Vec<f32>> = (0..16).map(|_| rng.unit(dim)).collect();
    let sample = |rng: &mut Rng, c: usize| {
        let mut v: Vec<f32> = centers[c]
            .iter()
            .map(|x| x + 0.05 * rng.next_f32())
            .collect();
        for d in [3, 77, 200, 301] {
            v[d] *= 12.0;
        }
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.into_iter().map(|x| x / n).collect::<Vec<f32>>()
    };
    let data: Vec<Vec<f32>> = (0..5000).map(|i| sample(&mut rng, i % 16)).collect();
    let idx = VectorIndex::build(data.clone()).unwrap();
    let mut found = 0;
    for qi in 0..50 {
        let q = sample(&mut rng, qi % 16);
        let mut naive: Vec<(u32, f32)> = data
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u32, dot(&q, v)))
            .collect();
        naive.sort_by(|a, b| b.1.total_cmp(&a.1));
        let got: Vec<u32> = idx.search(&q, 10).iter().map(|h| h.id).collect();
        found += naive[..10].iter().filter(|h| got.contains(&h.0)).count();
    }
    let recall = found as f64 / 500.0;
    println!("recall@10 = {recall}");
    assert!(recall >= 0.99, "recall@10 = {recall}");
}

#[test]
fn from_parts_rejects_flipped_quantized_byte() {
    let mut rng = Rng(13);
    let idx = VectorIndex::build((0..4).map(|_| rng.unit(16)).collect()).unwrap();
    let mut q = idx.quantized().to_vec();
    q[5] = q[5].wrapping_add(1);
    assert!(matches!(
        VectorIndex::from_parts(16, q, idx.full().to_vec()),
        Err(Error::CorruptIndex(_))
    ));
}

#[test]
fn all_zero_vector_is_indexed_with_zero_score() {
    let idx = VectorIndex::build(vec![vec![0.0; 4], vec![1.0, 0.0, 0.0, 0.0]]).unwrap();
    let hits = idx.search(&[1.0, 0.0, 0.0, 0.0], 2);
    assert_eq!(hits[0].id, 1);
    assert_eq!(hits[1], pithdb::index::Hit { id: 0, score: 0.0 });
}

#[test]
fn dim_not_multiple_of_16_matches_naive() {
    let mut rng = Rng(17);
    let data: Vec<Vec<f32>> = (0..300).map(|_| rng.unit(100)).collect();
    let idx = VectorIndex::build(data.clone()).unwrap();
    for _ in 0..10 {
        let q = rng.unit(100);
        let mut naive: Vec<(u32, f32)> = data
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u32, dot(&q, v)))
            .collect();
        naive.sort_by(|a, b| b.1.total_cmp(&a.1));
        let got: Vec<u32> = idx.search(&q, 5).iter().map(|h| h.id).collect();
        assert_eq!(got, naive[..5].iter().map(|h| h.0).collect::<Vec<_>>());
    }
}

#[test]
fn bad_query_errors_even_when_k_zero_or_index_empty() {
    let idx = VectorIndex::build(vec![vec![1.0, 0.0]]).unwrap();
    assert!(matches!(
        idx.try_search(&[1.0], 0),
        Err(Error::DimMismatch { .. })
    ));
    assert!(matches!(
        idx.try_search(&[f32::NAN, 0.0], 0),
        Err(Error::NonFinite)
    ));
    let empty = VectorIndex::from_parts(2, vec![], vec![]).unwrap();
    assert!(matches!(
        empty.try_search(&[1.0], 3),
        Err(Error::DimMismatch { .. })
    ));
    assert!(matches!(
        empty.try_search(&[f32::INFINITY, 0.0], 3),
        Err(Error::NonFinite)
    ));
    assert!(empty.try_search(&[1.0, 0.0], 3).unwrap().is_empty());
    // dim 0 = no dimension yet: any query yields an empty result.
    let blank = VectorIndex::build(vec![]).unwrap();
    assert!(blank.try_search(&[1.0, 2.0], 3).unwrap().is_empty());
}
