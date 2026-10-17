use pithdb::index::VectorIndex;
use pithdb::storage::{locate_project_within, Db, IndexMeta, StoredChunk, WriterLock};
use pithdb::Error;
use proptest::prelude::*;
use std::fs;
use std::path::Path;

// Offsets from the layout documented in src/storage/format.rs.
const VERSION_AT: usize = 8;
const RESERVED_AT: usize = 20;
const CHUNKS_LEN_AT: usize = 24;
const RAW_LEN_AT: usize = 32;
const HASHES_AT: usize = 40;
const HEADER_HASHED: usize = 136;
const HEADER_LEN: usize = 168;

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

fn reseal_header(b: &mut [u8]) {
    let h = *blake3::hash(&b[..HEADER_HASHED]).as_bytes();
    b[HEADER_HASHED..HEADER_LEN].copy_from_slice(&h);
}

fn reseal_file(b: &mut [u8]) {
    let end = b.len() - 32;
    let h = *blake3::hash(&b[..end]).as_bytes();
    b[end..].copy_from_slice(&h);
}

/// Recompute header and whole-file hashes so only the semantic edit remains.
fn reseal(b: &mut [u8]) {
    reseal_header(b);
    reseal_file(b);
}

fn align64(n: usize) -> usize {
    n.div_ceil(64) * 64
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

/// Section byte ranges of a valid file: [chunks, int8, f32].
fn sections(b: &[u8]) -> [std::ops::Range<usize>; 3] {
    let chunks_len = u64_at(b, CHUNKS_LEN_AT) as usize;
    let q = align64(HEADER_LEN + chunks_len);
    let at = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap()) as usize;
    let n = at(12) * at(16); // dim * count values
    let f = align64(q + n);
    [HEADER_LEN..HEADER_LEN + chunks_len, q..q + n, f..f + 4 * n]
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

/// Raw chunk table with an empty git head flag of `flag`, `entries` chunks and `tail`.
fn table(flag: u8, entries: u32, tail: &[u8]) -> Vec<u8> {
    let mut t = Vec::new();
    put_str(&mut t, "m");
    t.extend_from_slice(&7u64.to_le_bytes());
    t.push(flag);
    for i in 0..entries {
        put_str(&mut t, "p");
        t.extend_from_slice(&i.to_le_bytes());
        t.extend_from_slice(&i.to_le_bytes());
        put_str(&mut t, "text");
    }
    t.extend_from_slice(tail);
    t
}

/// A version-1 file with all-zero vectors (valid: quantize(0) == 0), built
/// independently of the writer, so fields the writer never produces can be set
/// while every checksum is valid.
fn assemble(dim: u32, count: u32, packed: &[u8], raw_len: u64) -> Vec<u8> {
    let n = dim as usize * count as usize;
    let (q, f) = (vec![0u8; n], vec![0u8; 4 * n]);
    let mut b = Vec::new();
    b.extend_from_slice(b"PITHDB\0\0");
    for v in [1, dim, count, 0] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b.extend_from_slice(&(packed.len() as u64).to_le_bytes());
    b.extend_from_slice(&raw_len.to_le_bytes());
    for s in [packed, &q, &f] {
        b.extend_from_slice(blake3::hash(s).as_bytes());
    }
    b.extend_from_slice(&[0; 32]);
    reseal_header(&mut b);
    b.extend_from_slice(packed);
    b.resize(align64(b.len()), 0);
    b.extend_from_slice(&q);
    b.resize(align64(b.len()), 0);
    b.extend_from_slice(&f);
    b.extend_from_slice(&[0; 32]);
    reseal_file(&mut b);
    b
}

fn assemble_table(dim: u32, count: u32, raw: &[u8]) -> Vec<u8> {
    assemble(
        dim,
        count,
        &zstd::bulk::compress(raw, 3).unwrap(),
        raw.len() as u64,
    )
}

fn open_bytes(b: &[u8]) -> Result<Db, Error> {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("x.pith");
    fs::write(&p, b).unwrap();
    Db::open(&p, None)
}

fn corrupt_msg(b: &[u8]) -> String {
    match open_bytes(b) {
        Err(Error::CorruptDb(m)) => m,
        Err(e) => panic!("expected CorruptDb, got {e:?}"),
        Ok(_) => panic!("expected CorruptDb, got Ok"),
    }
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
}

#[test]
fn mapped_search_equals_owned_bit_for_bit() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(".pith");
    let (chunks, index) = sample(200, 24);
    Db::write(&path, "m", chunks, &index, meta()).unwrap();
    let db = Db::open(&path, None).unwrap();
    for row in 0..200 {
        let q = &index.full()[row * 24..][..24];
        for k in [1, 5, 50, 300] {
            let (a, b) = (
                db.index().try_search(q, k).unwrap(),
                index.try_search(q, k).unwrap(),
            );
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(&b) {
                assert_eq!((x.id, x.score.to_bits()), (y.id, y.score.to_bits()));
            }
        }
    }
    assert!(matches!(
        db.index().try_search(&[0.0; 3], 1),
        Err(Error::DimMismatch { .. })
    ));
}

#[test]
fn mapped_index_outlives_db_handle_and_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(".pith");
    let (chunks, index) = sample(10, 8);
    Db::write(&path, "m", chunks, &index, meta()).unwrap();
    let mapped = Db::open(&path, None).unwrap().into_index();
    // Replace and then delete the file: the mapping keeps the old inode alive.
    write_sample(&path, 2);
    fs::remove_file(&path).unwrap();
    assert_eq!(mapped.len(), 10);
    assert_eq!(mapped.full(), index.full());
    let q = &index.full()[16..24];
    assert_eq!(
        mapped.try_search(q, 3).unwrap(),
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
    assert_eq!(db.index().try_search(&[0.0; 384], 3).unwrap(), vec![]);
}

#[test]
fn write_refuses_dimension_zero() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(".pith");
    let r = Db::write(
        &path,
        "m",
        vec![],
        &VectorIndex::build(vec![]).unwrap(),
        meta(),
    );
    match r {
        Err(Error::Invalid(m)) => assert!(m.contains("from_parts(model_dim"), "{m}"),
        other => panic!("{other:?}"),
    }
    assert!(!path.exists());
}

#[test]
fn write_rejects_chunk_count_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(".pith");
    let (chunks, index) = sample(3, 8);
    let r = Db::write(&path, "m", chunks[..2].to_vec(), &index, meta());
    match r {
        Err(Error::Invalid(m)) => assert_eq!(m, "2 chunks but 3 vectors"),
        other => panic!("{other:?}"),
    }
    assert!(!path.exists());
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
        assert!(corrupt_msg(&b).contains("section lengths"), "len {bogus}");
    }
}

#[test]
fn each_section_checksum_is_checked() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(".pith");
    write_sample(&path, 3);
    let good = fs::read(&path).unwrap();
    for (i, r) in sections(&good).into_iter().enumerate() {
        assert!(!r.is_empty());
        let mut b = good.clone();
        b[r.start + r.len() / 2] ^= 0x10;
        reseal_file(&mut b); // only the whole-file hash: the section hash must catch it
        assert_eq!(corrupt_msg(&b), format!("section {i} checksum mismatch"));
    }
    // Sanity: the section ranges are right, so the hash at HASHES_AT matches.
    let r = &sections(&good)[1];
    assert_eq!(
        blake3::hash(&good[r.clone()]).as_bytes(),
        &good[HASHES_AT + 32..HASHES_AT + 64]
    );
}

#[test]
fn reserved_field_must_be_zero() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(".pith");
    write_sample(&path, 1);
    let mut b = fs::read(&path).unwrap();
    b[RESERVED_AT] = 1;
    reseal(&mut b);
    assert_eq!(corrupt_msg(&b), "reserved header field is not zero");
}

#[test]
fn crafted_valid_file_opens() {
    // Proves `assemble` matches the format, so the crafted rejections below are real.
    let db = open_bytes(&assemble_table(4, 2, &table(0, 2, &[]))).unwrap();
    assert_eq!(db.chunk(1).unwrap().text, "text");
    assert_eq!(db.index().len(), 2);
    let db = open_bytes(&assemble_table(4, 0, &table(0, 0, &[]))).unwrap();
    assert_eq!(db.model_id(), "m");
    assert_eq!(db.meta().indexed_at_unix, 7);
    assert_eq!(db.index().dim(), 4);
}

#[test]
fn huge_count_without_matching_data_rejected_by_layout() {
    // A huge count raises the table bound, but the file must then hold the vectors.
    let raw = table(0, 0, &[]);
    let packed = zstd::bulk::compress(&raw, 3).unwrap();
    let mut b = assemble(1, 0, &packed, 1 << 33);
    b[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
    reseal(&mut b);
    assert!(corrupt_msg(&b).contains("section lengths"));
}

#[test]
fn million_tiny_chunks_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(".pith");
    let n = 1_000_000;
    let chunks: Vec<StoredChunk> = (0..n)
        .map(|i| StoredChunk {
            path: String::new(),
            start_line: i,
            end_line: i,
            text: String::new(),
        })
        .collect();
    let full: Vec<f32> = (0..n).map(|i| (i % 7) as f32 / 7.0).collect();
    let index = VectorIndex::build(full.chunks(1).map(<[f32]>::to_vec).collect()).unwrap();
    Db::write(&path, "m", chunks, &index, meta()).unwrap();
    let db = Db::open(&path, Some(("m", 1))).unwrap();
    assert_eq!(db.index().len(), n as usize);
    assert_eq!(db.chunk(n - 1).unwrap().start_line, n - 1);
    assert_eq!(db.chunk(n), None);
}

fn single_chunk(text: String) -> (Vec<StoredChunk>, VectorIndex) {
    let c = StoredChunk {
        path: "p".into(),
        start_line: 1,
        end_line: 1,
        text,
    };
    (vec![c], VectorIndex::build(vec![vec![1.0]]).unwrap())
}

#[test]
fn write_enforces_table_bound_exactly() {
    // count 1: bound = 9,696 + 65,536 = 75,232. Table = model "m" (5) + 8 + flag 1
    // + path "p" (5) + lines 8 + text (4 + L) = 31 + L. One text far over its
    // 1,400-char share is fine: the budget is per table, not per field.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(".pith");
    let no_head = IndexMeta {
        indexed_at_unix: 0,
        git_head: None,
    };
    let (chunks, index) = single_chunk("x".repeat(75_232 - 31));
    Db::write(&path, "m", chunks, &index, no_head.clone()).unwrap();
    assert_eq!(
        Db::open(&path, None).unwrap().chunk(0).unwrap().text.len(),
        75_201
    );
    let (chunks, index) = single_chunk("x".repeat(75_232 - 30));
    match Db::write(&path, "m", chunks, &index, no_head) {
        Err(Error::Invalid(m)) => {
            assert_eq!(
                m,
                "the indexed text is too large to store (0.1 MB for 1 chunks); exclude \
                 generated, vendored or duplicated files with .pithignore, or index a \
                 smaller path"
            )
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn highly_compressible_table_is_accepted() {
    // 100k identical 1,400-char chunks (~141 MB raw, ~10,864:1): a duplicated corpus
    // well over the old 1024:1 bound must still index, round-trip and open.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(".pith");
    let n = 100_000;
    let text = "a".repeat(1_400);
    let chunks: Vec<StoredChunk> = (0..n)
        .map(|_| StoredChunk {
            path: "src/same.rs".into(),
            start_line: 1,
            end_line: 1,
            text: text.clone(),
        })
        .collect();
    let index = VectorIndex::build(vec![vec![1.0]; n]).unwrap();
    Db::write(&path, "m", chunks, &index, meta()).unwrap();
    let b = fs::read(&path).unwrap();
    let ratio = u64_at(&b, RAW_LEN_AT) / u64_at(&b, CHUNKS_LEN_AT);
    assert!(ratio > 1024, "{ratio}");
    let db = Db::open(&path, Some(("m", 1))).unwrap();
    assert_eq!(db.index().len(), n);
    assert_eq!(db.chunk(n as u32 - 1).unwrap().text, text);
}

/// 1M identical empty chunks: a table that compresses far beyond 16384:1.
fn zero_entries() -> Vec<StoredChunk> {
    (0..1_000_000)
        .map(|_| StoredChunk {
            path: String::new(),
            start_line: 0,
            end_line: 0,
            text: String::new(),
        })
        .collect()
}

#[test]
fn table_over_ratio_bound_refused_at_write_and_read() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(".pith");
    let chunks = zero_entries();
    let index = VectorIndex::build(vec![vec![1.0]; chunks.len()]).unwrap();
    match Db::write(&path, "m", chunks, &index, meta()) {
        Err(Error::Invalid(m)) => {
            assert!(
                m.contains("too repetitive") && m.contains(".pithignore"),
                "{m}"
            )
        }
        other => panic!("{other:?}"),
    }
    assert!(!path.exists());
    // The same kind of table crafted into a file is refused on open.
    let mut raw = table(0, 0, &[]);
    raw.resize(raw.len() + 16 * 1_000_000, 0);
    let packed = zstd::bulk::compress(&raw, 19).unwrap();
    assert!(raw.len() as u64 > packed.len() as u64 * 16_384 + 65_536);
    let m = corrupt_msg(&assemble(1, 1_000_000, &packed, raw.len() as u64));
    assert!(m.contains("compressed bytes"), "{m}");
}

#[test]
fn crafted_table_errors_rejected() {
    assert_eq!(
        corrupt_msg(&assemble_table(4, 0, &table(2, 0, &[]))),
        "bad git_head flag"
    );
    assert_eq!(
        corrupt_msg(&assemble_table(4, 0, &table(0, 0, b"x"))),
        "trailing bytes in chunk table"
    );
    assert_eq!(
        corrupt_msg(&assemble_table(4, 0, &table(0, 0, &[])[..5])),
        "chunk table truncated"
    );
}

#[test]
fn dim_zero_with_chunks_rejected() {
    assert_eq!(
        corrupt_msg(&assemble_table(0, 1, &table(0, 1, &[]))),
        "dimension is 0"
    );
    assert_eq!(
        corrupt_msg(&assemble_table(0, 0, &table(0, 0, &[]))),
        "dimension is 0"
    );
}

#[test]
fn decompression_is_bounded_by_stated_raw_length() {
    let raw = table(0, 0, &[]);
    let packed = zstd::bulk::compress(&raw, 3).unwrap();
    // Small payload claiming a huge raw length: refused before decompressing.
    let m = corrupt_msg(&assemble(4, 0, &packed, 1 << 40));
    assert!(m.contains("over limit"), "{m}");
    // Over the count-derived bound by one byte (count 0 and count 1).
    let m = corrupt_msg(&assemble(4, 0, &packed, 64 * 1024 + 1));
    assert!(m.contains("over limit 65536 for 0 chunks"), "{m}");
    let m = corrupt_msg(&assemble(1, 1, &packed, 9_696 + 65_536 + 1));
    assert!(m.contains("over limit 75232 for 1 chunks"), "{m}");
    // 1 GiB claimed by a header whose count (200k, dim 1) allows it, with a tiny
    // payload: the ratio bound (compressed bytes * 16384 + 64 KiB) refuses it.
    let m = corrupt_msg(&assemble(1, 200_000, &packed, 1 << 30));
    let ratio_max = packed.len() as u64 * 16_384 + 65_536;
    assert!(
        m.contains(&format!(
            "over limit {ratio_max} for {} compressed bytes",
            packed.len()
        )),
        "{m}"
    );
    // At exactly the ratio bound it gets to decompression and fails there cleanly.
    let m = corrupt_msg(&assemble(1, 200_000, &packed, ratio_max));
    assert!(
        m.contains(&format!("{} bytes, header says {ratio_max}", raw.len())),
        "{m}"
    );
    // Claims more than it holds.
    let m = corrupt_msg(&assemble(4, 0, &packed, raw.len() as u64 + 100));
    assert!(m.contains("header says"), "{m}");
    // A bomb: 64 MiB of zeros in a tiny payload, stated as 16 bytes. As a normal
    // frame its declared window is too large and it is refused before decoding.
    let bomb = zstd::bulk::compress(&vec![0u8; 64 << 20], 19).unwrap();
    assert!(bomb.len() < 64 << 10);
    let m = corrupt_msg(&assemble(4, 0, &bomb, 16));
    assert!(m.contains("too much memory"), "{m}");
    // With the smallest window (1 KiB) it passes the window check; decoding then
    // stops after 17 bytes instead of inflating it.
    let mut enc = zstd::stream::write::Encoder::new(Vec::new(), 19).unwrap();
    enc.window_log(10).unwrap();
    enc.include_contentsize(false).unwrap();
    std::io::Write::write_all(&mut enc, &vec![0u8; 64 << 20]).unwrap();
    let bomb = enc.finish().unwrap();
    assert!(bomb.len() < 8 << 20, "{}", bomb.len()); // small blocks compress less
    let m = corrupt_msg(&assemble(4, 0, &bomb, 16));
    assert!(m.contains("17+ bytes, header says 16"), "{m}");
}

#[test]
fn raw_length_field_is_checked_in_real_files() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(".pith");
    write_sample(&path, 2);
    let mut b = fs::read(&path).unwrap();
    let raw = u64_at(&b, RAW_LEN_AT);
    b[RAW_LEN_AT..RAW_LEN_AT + 8].copy_from_slice(&(raw - 1).to_le_bytes());
    reseal(&mut b);
    assert!(corrupt_msg(&b).contains("header says"));
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

#[cfg(unix)]
#[test]
fn write_does_not_follow_planted_temp_symlink() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(".pith");
    let victim = dir.path().join("victim");
    fs::write(&victim, b"keep").unwrap();
    for name in [
        ".pith.tmp",
        ".pith.tmp0",
        ".pith.tmp.tmp",
        ".pith.tmpAbC123",
    ] {
        std::os::unix::fs::symlink(&victim, dir.path().join(name)).unwrap();
    }
    write_sample(&path, 2);
    assert_eq!(fs::read(&victim).unwrap(), b"keep");
    assert!(Db::open(&path, None).is_ok());
    assert!(fs::symlink_metadata(dir.path().join(".pith.tmp"))
        .unwrap()
        .is_symlink());
    // The one matching the stale-temp pattern was removed (the link, not the target).
    assert!(fs::symlink_metadata(dir.path().join(".pith.tmpAbC123")).is_err());
}

#[test]
fn write_removes_stale_temp_files_only() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(".pith");
    let stale = [".pith.tmpXy12Zq", ".pith.tmp000000"];
    let kept = [
        ".pith.tmp",
        ".pith.tmpshort",
        ".pith.tmp1234567",
        "notes.tmpAbC123",
        ".pith.tmp-a_b.c",
    ];
    for n in stale.iter().chain(&kept) {
        fs::write(dir.path().join(n), b"old").unwrap();
    }
    fs::create_dir(dir.path().join(".pith.tmpDIRDIR")).unwrap();
    write_sample(&path, 1);
    for n in stale {
        assert!(!dir.path().join(n).exists(), "{n} not removed");
    }
    for n in kept.iter().chain(&[".pith.tmpDIRDIR"]) {
        assert!(dir.path().join(n).exists(), "{n} removed");
    }
    // And the writer's own temp file is gone after a successful write.
    let left: Vec<_> = fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|n| {
            n.starts_with(".pith.tmp") && !kept.contains(&n.as_str()) && n != ".pith.tmpDIRDIR"
        })
        .collect();
    assert!(left.is_empty(), "{left:?}");
}

#[cfg(unix)]
#[test]
fn database_file_is_owner_only() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(".pith");
    write_sample(&path, 1);
    assert_eq!(
        fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o600
    );
}

#[test]
fn oversized_zstd_window_refused() {
    use std::io::Write;
    let raw = table(0, 0, &[]);
    // A frame that declares a 128 MiB window for a few bytes of content.
    let mut enc = zstd::stream::write::Encoder::new(Vec::new(), 3).unwrap();
    enc.window_log(27).unwrap();
    enc.include_contentsize(false).unwrap();
    enc.write_all(&raw).unwrap();
    let packed = enc.finish().unwrap();
    let m = corrupt_msg(&assemble(4, 0, &packed, raw.len() as u64));
    assert!(m.starts_with("chunk section:"), "{m}");
    // The same content with a normal frame opens.
    assert!(open_bytes(&assemble_table(4, 0, &raw)).is_ok());
}

mod locate {
    use super::*;

    /// root/home/proj/a/b, with `home` as the boundary.
    fn tree() -> (tempfile::TempDir, std::path::PathBuf, std::path::PathBuf) {
        let root = tempfile::tempdir().unwrap();
        let home = root.path().join("home");
        let deep = home.join("proj/a/b");
        fs::create_dir_all(&deep).unwrap();
        (root, home, deep)
    }

    #[test]
    fn locate_project_walks_up_and_returns_none_when_absent() {
        let (_root, home, deep) = tree();
        assert_eq!(locate_project_within(&deep, Some(&home)), None);
        fs::write(home.join("proj/.pith"), b"x").unwrap();
        assert_eq!(
            locate_project_within(&deep, Some(&home)),
            Some(home.join("proj/.pith"))
        );
        // A directory named .pith is not a database.
        fs::create_dir(home.join("proj/a/.pith")).unwrap();
        assert_eq!(
            locate_project_within(&deep, Some(&home)),
            Some(home.join("proj/.pith"))
        );
        // The start directory itself counts.
        fs::write(deep.join(".pith"), b"x").unwrap();
        assert_eq!(
            locate_project_within(&deep, Some(&home)),
            Some(deep.join(".pith"))
        );
    }

    #[test]
    fn never_walks_above_home() {
        let (root, home, deep) = tree();
        fs::write(root.path().join(".pith"), b"planted").unwrap();
        assert_eq!(locate_project_within(&deep, Some(&home)), None);
        fs::write(home.join(".pith"), b"x").unwrap();
        assert_eq!(
            locate_project_within(&deep, Some(&home)),
            Some(home.join(".pith"))
        );
    }

    #[test]
    #[cfg(unix)]
    fn dangling_git_symlink_stops_walk() {
        let (_root, home, deep) = tree();
        fs::write(home.join(".pith"), b"x").unwrap();
        std::os::unix::fs::symlink(home.join("missing"), home.join("proj/.git")).unwrap();
        assert_eq!(locate_project_within(&deep, Some(&home)), None);
    }

    #[test]
    fn stops_at_git_root() {
        let (_root, home, deep) = tree();
        fs::write(home.join(".pith"), b"x").unwrap();
        fs::create_dir(home.join("proj/.git")).unwrap();
        assert_eq!(locate_project_within(&deep, Some(&home)), None);
        fs::write(home.join("proj/.pith"), b"x").unwrap();
        assert_eq!(
            locate_project_within(&deep, Some(&home)),
            Some(home.join("proj/.pith"))
        );
    }

    #[test]
    fn outside_home_checks_only_start() {
        let (root, home, _deep) = tree();
        let start = root.path().join("elsewhere/x");
        fs::create_dir_all(&start).unwrap();
        fs::write(root.path().join("elsewhere/.pith"), b"x").unwrap();
        assert_eq!(locate_project_within(&start, Some(&home)), None);
        assert_eq!(locate_project_within(&start, None), None);
        fs::write(start.join(".pith"), b"x").unwrap();
        assert_eq!(
            locate_project_within(&start, Some(&home)),
            Some(start.join(".pith"))
        );
        assert_eq!(
            locate_project_within(&start, None),
            Some(start.join(".pith"))
        );
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_pith_is_skipped() {
        let (_root, home, deep) = tree();
        let real = home.join("real.pith");
        fs::write(&real, b"x").unwrap();
        std::os::unix::fs::symlink(&real, deep.join(".pith")).unwrap();
        assert_eq!(locate_project_within(&deep, Some(&home)), None);
    }
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
