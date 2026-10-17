//! Heap used by `Db::open`: vectors must stay in the mapping, not on the heap.
//! `PITH_HEAP_N=100000 cargo test --release --test storage_heap -- --nocapture` for the 100k x 384 figure.

use pithdb::index::VectorIndex;
use pithdb::storage::{Db, IndexMeta, StoredChunk};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let n = LIVE.fetch_add(l.size(), Relaxed) + l.size();
        PEAK.fetch_max(n, Relaxed);
        // SAFETY: forwards the caller's contract to the system allocator.
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size(), Relaxed);
        // SAFETY: as above.
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static A: Counting = Counting;

#[test]
fn open_keeps_vectors_off_the_heap() {
    let n: usize = std::env::var("PITH_HEAP_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);
    let dim = 384;
    let mut s = 0x9e3779b97f4a7c15u64;
    let full: Vec<f32> = (0..n * dim)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        })
        .collect();
    let index = VectorIndex::build(full.chunks(dim).map(<[f32]>::to_vec).collect()).unwrap();
    drop(full);
    let chunks = (0..n)
        .map(|i| StoredChunk {
            path: format!("f{i}"),
            start_line: 1,
            end_line: 2,
            text: "x".into(),
        })
        .collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(".pith");
    Db::write(
        &path,
        "m",
        chunks,
        &index,
        IndexMeta {
            indexed_at_unix: 0,
            git_head: None,
        },
    )
    .unwrap();
    drop(index);

    let base = LIVE.load(Relaxed);
    PEAK.store(base, Relaxed);
    let db = Db::open(&path, None).unwrap();
    let retained = LIVE.load(Relaxed) - base;
    let peak = PEAK.load(Relaxed) - base;
    let vector_bytes = n * dim * 5;
    println!(
        "n={n} dim={dim} vector_bytes={vector_bytes} retained_heap={retained} peak_heap={peak}"
    );
    assert_eq!(db.index().len(), n);
    assert!(
        retained < vector_bytes / 4,
        "retained {retained} of {vector_bytes} vector bytes"
    );
    assert!(
        peak < vector_bytes / 4,
        "peak {peak} of {vector_bytes} vector bytes"
    );
}
