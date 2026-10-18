//! Brute-force int8-scan + f32-rescore search over 100k x 384 random unit vectors, k = 10.
//! Embedding is excluded. Besides criterion's mean, prints p50/p95/p99 over 2,000 distinct queries.
use criterion::{criterion_group, criterion_main, Criterion};
use pithdb::index::VectorIndex;
use std::hint::black_box;
use std::time::Instant;

const N: usize = 100_000;
const DIM: usize = 384;

struct Rng(u64);
impl Rng {
    fn unit_vec(&mut self) -> Vec<f32> {
        let mut v: Vec<f32> = (0..DIM)
            .map(|_| {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                (self.0 >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect();
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter_mut().for_each(|x| *x /= n);
        v
    }
}

fn search(c: &mut Criterion) {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let index = VectorIndex::build((0..N).map(|_| rng.unit_vec()).collect()).unwrap();
    let queries: Vec<Vec<f32>> = (0..2_000).map(|_| rng.unit_vec()).collect();

    let mut times: Vec<f64> = queries
        .iter()
        .map(|q| {
            let t = Instant::now();
            black_box(index.search(black_box(q), 10));
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    times.sort_by(f64::total_cmp);
    let pct = |p: f64| times[((times.len() as f64 * p) as usize).min(times.len() - 1)];
    println!(
        "search_100k_x_384 latency over {} queries: p50 {:.2} ms, p95 {:.2} ms, p99 {:.2} ms (gate: p95 < 15 ms)",
        times.len(),
        pct(0.50),
        pct(0.95),
        pct(0.99)
    );

    let mut i = 0;
    c.bench_function("search_100k_x_384", |b| {
        b.iter(|| {
            i = (i + 1) % queries.len();
            black_box(index.search(black_box(&queries[i]), 10))
        })
    });
}

criterion_group!(benches, search);
criterion_main!(benches);
