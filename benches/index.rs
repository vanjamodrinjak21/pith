//! Embedding throughput by batch size and thread count. Needs the model already installed
//! (`pith init`); this bench never downloads it. Note: vectors depend slightly on batch size.
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use pithdb::embed::{models_dir, Embedder, OnnxConfig, OnnxEmbedder};
use std::hint::black_box;

const CHUNKS: usize = 64;

fn throughput(c: &mut Criterion) {
    let dir = match models_dir() {
        Ok(d) => d,
        Err(e) => return eprintln!("skipping embedding bench: {e}"),
    };
    let texts: Vec<String> = (0..CHUNKS)
        .map(|i| {
            format!("fn handler_{i}(req: &Request) -> Result<Response> {{ validate(req)?; }} ")
                .repeat(10)
        })
        .collect();
    let mut g = c.benchmark_group("embed_64_chunks_830_chars");
    g.sample_size(10)
        .throughput(Throughput::Elements(CHUNKS as u64));
    for batch_size in [8, 16, 32, 64] {
        for threads in 1..=8 {
            let cfg = OnnxConfig {
                batch_size,
                threads,
            };
            let e = match OnnxEmbedder::load_with(&dir, cfg) {
                Ok(e) => e,
                Err(e) => return eprintln!("skipping embedding bench: {e}"),
            };
            g.bench_function(format!("batch{batch_size}_threads{threads}"), |b| {
                b.iter(|| black_box(e.embed_passages(&texts).unwrap()))
            });
        }
    }
    g.finish();
}

criterion_group!(benches, throughput);
criterion_main!(benches);
