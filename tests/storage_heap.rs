//! Memory used by `Db::open` + a full search on 100k x 384: vectors must stay in
//! the file mapping, not be copied to private memory.
//! Run: `cargo test --release --test storage_heap -- --ignored --nocapture`.
//!
//! Plain RSS (`ps -o rss=`) is printed but cannot be asserted on: it also counts
//! clean file-backed pages of the mapping, which open's validation touches in full.
//! The assertion uses private memory instead: "Physical footprint" from
//! `vmmap --summary` on macOS, `RssAnon` from /proc/self/status on Linux.

use pithdb::index::VectorIndex;
use pithdb::storage::{Db, IndexMeta, StoredChunk};
use std::process::Command;

const N: usize = 100_000;
const DIM: usize = 384;

/// Parses "Key: 1234 kB" (Linux) or "Key: 190.5M" / "2016K" (vmmap) as KiB.
fn kib_after(text: &str, key: &str) -> Option<u64> {
    let v = text.lines().find(|l| l.trim_start().starts_with(key))?;
    let v = v.split_once(':')?.1.trim().trim_end_matches("kB").trim();
    let (num, mult) = match v.chars().last()? {
        'G' => (&v[..v.len() - 1], 1024.0 * 1024.0),
        'M' => (&v[..v.len() - 1], 1024.0),
        'K' => (&v[..v.len() - 1], 1.0),
        _ => (v, 1.0),
    };
    Some((num.parse::<f64>().ok()? * mult) as u64)
}

fn rss_kib() -> u64 {
    let out = Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .unwrap();
    String::from_utf8(out.stdout)
        .unwrap()
        .trim()
        .parse()
        .unwrap()
}

fn private_kib() -> u64 {
    if cfg!(target_os = "linux") {
        let s = std::fs::read_to_string("/proc/self/status").unwrap();
        kib_after(&s, "RssAnon:").unwrap()
    } else {
        let out = Command::new("vmmap")
            .args(["--summary", &std::process::id().to_string()])
            .output()
            .unwrap();
        kib_after(&String::from_utf8_lossy(&out.stdout), "Physical footprint:").unwrap()
    }
}

/// Builds the database in a child process (this test binary re-run with
/// `PITH_HEAP_BUILD` set), so memory freed by building is not reused by `open`
/// in the measured process and cannot hide a copy.
fn build_in_child(path: &std::path::Path) {
    let st = Command::new(std::env::current_exe().unwrap())
        .args([
            "--ignored",
            "--exact",
            "open_keeps_vectors_out_of_private_memory",
        ])
        .env("PITH_HEAP_BUILD", path)
        .status()
        .unwrap();
    assert!(st.success());
}

fn build(path: &std::path::Path) {
    let mut s = 0x9e3779b97f4a7c15u64;
    let full: Vec<f32> = (0..N * DIM)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        })
        .collect();
    let index = VectorIndex::build(full.chunks(DIM).map(<[f32]>::to_vec).collect()).unwrap();
    drop(full);
    let chunks = (0..N)
        .map(|i| StoredChunk {
            path: format!("f{i}"),
            start_line: 1,
            end_line: 2,
            text: "x".into(),
        })
        .collect();
    let meta = IndexMeta {
        indexed_at_unix: 0,
        git_head: None,
    };
    Db::write(path, "m", chunks, &index, meta).unwrap();
}

#[test]
#[ignore = "slow: builds a 100k x 384 database; run with --release -- --ignored"]
fn open_keeps_vectors_out_of_private_memory() {
    if let Some(p) = std::env::var_os("PITH_HEAP_BUILD") {
        return build(std::path::Path::new(&p));
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(".pith");
    build_in_child(&path);
    let (rss0, priv0) = (rss_kib(), private_kib());
    let db = Db::open(&path, None).unwrap();
    let q = vec![1.0 / (DIM as f32).sqrt(); DIM];
    assert_eq!(db.index().try_search(&q, 10).unwrap().len(), 10);
    let (rss1, priv1) = (rss_kib(), private_kib());

    let owned_kib = (N * DIM * 5 / 1024) as u64; // i8 + f32 copy: 187,500 KiB
    let grew = priv1.saturating_sub(priv0);
    println!(
        "owned copy would be {owned_kib} KiB; rss {rss0} -> {rss1} KiB (+{}); private {priv0} -> {priv1} KiB (+{grew})",
        rss1.saturating_sub(rss0)
    );
    // Bound: chunk table (~100k small Strings, ~6 MiB measured) plus allocator
    // slack stays far below a quarter of the owned copy; a copy adds >= 187,500 KiB.
    assert!(grew < owned_kib / 4, "private memory grew {grew} KiB");
}
