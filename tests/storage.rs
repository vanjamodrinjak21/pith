use pithdb::index::VectorIndex;
use pithdb::storage::{locate_project, Db, IndexMeta, StoredChunk, WriterLock};
use pithdb::Error;
use proptest::prelude::*;
use std::fs;
use std::path::Path;

// Offsets from the layout documented in src/storage/format.rs.
const VERSION_AT: usize = 8;
const CHUNKS_LEN_AT: usize = 24;
const HEADER_HASHED: usize = 128;

fn chunk(i: u32) -> StoredChunk {
    StoredChunk {
        path: format!("src/file_{i}.rs"),
        start_line: i * 10 + 1,
        end_line: i * 10 + 9,
        text: format!("fn f{i}() {{ /* body {i} */ }}\n").repeat(3),
    }
}

fn sample(n: u32, dim: usize) -> (Vec<StoredChunk>, VectorIndex) {
    let vecs: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            (0..dim)
                .map(|d| ((i as usize * 31 + d * 7) % 13) as f32 / 13.0 - 0.4)
                .collect()
        })
        .collect();
    let index = if n == 0 {
        VectorIndex::from_parts(dim, vec![], vec![]).unwrap()
    } else {
        VectorIndex::build(vecs).unwrap()
    };
    ((0..n).map(chunk).collect(), index)
}

fn meta() -> IndexMeta {
    IndexMeta {
        indexed_at_unix: 1_760_000_000,
        git_head: Some("abc123".into()),
    }
}

fn write_sample(path: &Path, n: u32) {
    let (chunks, index) = sample(n, 8);
    Db::write(path, "bge-small", chunks, &index, meta()).unwrap();
}

/// Recompute header and whole-file hashes so only the semantic edit remains.
fn reseal(bytes: &mut [u8]) {
    let h = *blake3::hash(&bytes[..HEADER_HASHED]).as_bytes();
    bytes[HEADER_HASHED..HEADER_HASHED + 32].copy_from_slice(&h);
    let end = bytes.len() - 32;
    let h = *blake3::hash(&bytes[..end]).as_bytes();
    bytes[end..].copy_from_slice(&h);
}

#[test]
fn roundtrip_preserves_chunks_vectors_meta() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(".pith");
    let (chunks, index) = sample(5, 8);
    Db::write(&path, "bge-small", chunks.clone(), &index, meta()).unwrap();
    let db = Db::open(&path, Some(("bge-small", 8))).unwrap();
    assert_eq!(db.model_id(), "bge-small");
    assert_eq!(db.meta(), &meta());
    for (i, c) in chunks.iter().enumerate() {
        assert_eq!(db.chunk(i as u32), Some(c));
    }
    assert_eq!(db.chunk(5), None);
    assert_eq!(db.index().quantized(), index.quantized());
    assert_eq!(db.index().full(), index.full());
    let q = &index.full()[8..16];
    assert_eq!(
        db.index().try_search(q, 3).unwrap(),
        index.try_search(q, 3).unwrap()
    );
}

#[test]
fn empty_database_roundtrips() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(".pith");
    let (chunks, index) = sample(0, 384);
    let m = IndexMeta {
        indexed_at_unix: 0,
        git_head: None,
    };
    Db::write(&path, "bge-small", chunks, &index, m.clone()).unwrap();
    let db = Db::open(&path, Some(("bge-small", 384))).unwrap();
    assert_eq!(db.index().len(), 0);
    assert_eq!(db.index().dim(), 384);
    assert_eq!(db.chunk(0), None);
    assert_eq!(db.meta(), &m);
}

#[test]
fn bit_flip_anywhere_is_detected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(".pith");
    write_sample(&path, 3);
    let good = fs::read(&path).unwrap();
    let bad = dir.path().join("bad");
    // Every byte, which is a superset of 20 random ones.
    for i in 0..good.len() {
        let mut b = good.clone();
        b[i] ^= 1 << (i % 8);
        fs::write(&bad, &b).unwrap();
        assert!(
            Db::open(&bad, None).is_err(),
            "flip at byte {i} not detected"
        );
    }
}

#[test]
fn truncated_file_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(".pith");
    write_sample(&path, 3);
    let good = fs::read(&path).unwrap();
    let bad = dir.path().join("bad");
    // Every length, so header interior and every section boundary are covered.
    for len in 0..good.len() {
        fs::write(&bad, &good[..len]).unwrap();
        assert!(
            Db::open(&bad, None).is_err(),
            "truncation to {len} accepted"
        );
    }
}

#[test]
fn checksum_valid_length_past_eof_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(".pith");
    write_sample(&path, 3);
    for bogus in [u64::MAX, u64::MAX / 2, 1 << 40, 0] {
        let mut b = fs::read(&path).unwrap();
        b[CHUNKS_LEN_AT..CHUNKS_LEN_AT + 8].copy_from_slice(&bogus.to_le_bytes());
        reseal(&mut b);
        let bad = dir.path().join("bad");
        fs::write(&bad, &b).unwrap();
        assert!(
            matches!(Db::open(&bad, None), Err(Error::CorruptDb(_))),
            "len {bogus}"
        );
    }
}

#[test]
fn model_mismatch_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(".pith");
    write_sample(&path, 2);
    assert!(matches!(
        Db::open(&path, Some(("other", 8))),
        Err(Error::ModelMismatch { .. })
    ));
    assert!(matches!(
        Db::open(&path, Some(("bge-small", 384))),
        Err(Error::ModelMismatch { .. })
    ));
}

#[test]
fn unknown_format_version_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(".pith");
    write_sample(&path, 2);
    let mut b = fs::read(&path).unwrap();
    b[VERSION_AT..VERSION_AT + 4].copy_from_slice(&99u32.to_le_bytes());
    fs::write(&path, &b).unwrap();
    assert!(matches!(
        Db::open(&path, None),
        Err(Error::UnsupportedFormat(99))
    ));
}

#[test]
fn second_writer_fails_fast() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(".pith");
    let held = WriterLock::acquire(&path).unwrap();
    assert!(dir.path().join(".pith.lock").exists());
    let (chunks, index) = sample(1, 8);
    let t = std::time::Instant::now();
    let r = Db::write(&path, "m", chunks, &index, meta());
    assert!(matches!(r, Err(Error::Locked)));
    assert!(t.elapsed() < std::time::Duration::from_secs(1));
    drop(held);
    write_sample(&path, 1);
}

#[test]
fn failed_write_leaves_previous_file_intact() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(".pith");
    write_sample(&path, 2);
    let before = fs::read(&path).unwrap();
    let (chunks, index) = sample(4, 8);
    let r = Db::write_with_hook(&path, "new", chunks, &index, meta(), || {
        Err(Error::Io(std::io::Error::other("injected")))
    });
    assert!(r.is_err());
    assert_eq!(fs::read(&path).unwrap(), before);
    assert_eq!(Db::open(&path, None).unwrap().model_id(), "bge-small");
    let names: Vec<_> = fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert_eq!(names.len(), 2, "temp file left behind: {names:?}"); // .pith + .pith.lock
}

#[test]
fn write_rejects_chunk_count_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let (chunks, index) = sample(3, 8);
    let r = Db::write(
        &dir.path().join(".pith"),
        "m",
        chunks[..2].to_vec(),
        &index,
        meta(),
    );
    assert!(r.is_err());
}

#[test]
fn locate_project_walks_up_and_returns_none_when_absent() {
    let dir = tempfile::tempdir().unwrap();
    let deep = dir.path().join("a/b/c");
    fs::create_dir_all(&deep).unwrap();
    assert_eq!(locate_project(&deep), None);
    fs::write(dir.path().join("a/.pith"), b"x").unwrap();
    assert_eq!(locate_project(&deep), Some(dir.path().join("a/.pith")));
    // A directory named .pith is not a database.
    fs::create_dir(dir.path().join("a/b/.pith")).unwrap();
    assert_eq!(locate_project(&deep), Some(dir.path().join("a/.pith")));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]
    #[test]
    fn proptest_roundtrip_arbitrary_chunks(
        dim in 1usize..6,
        raw in prop::collection::vec((".{0,40}", any::<u32>(), any::<u32>(), "(?s).{0,200}"), 0..8),
        seed in any::<u64>(),
        head in prop::option::of("[0-9a-f]{0,40}"),
    ) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".pith");
        let chunks: Vec<StoredChunk> = raw.into_iter()
            .map(|(path, start_line, end_line, text)| StoredChunk { path, start_line, end_line, text })
            .collect();
        let mut s = seed | 1;
        let vecs: Vec<Vec<f32>> = (0..chunks.len()).map(|_| (0..dim).map(|_| {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        }).collect()).collect();
        let index = if vecs.is_empty() {
            VectorIndex::from_parts(dim, vec![], vec![]).unwrap()
        } else {
            VectorIndex::build(vecs).unwrap()
        };
        let m = IndexMeta { indexed_at_unix: seed, git_head: head };
        Db::write(&path, "model", chunks.clone(), &index, m.clone()).unwrap();
        let db = Db::open(&path, Some(("model", dim))).unwrap();
        prop_assert_eq!(db.meta(), &m);
        for (i, c) in chunks.iter().enumerate() {
            prop_assert_eq!(db.chunk(i as u32), Some(c));
        }
        prop_assert_eq!(db.index().full(), index.full());
        prop_assert_eq!(db.index().quantized(), index.quantized());
    }
}
