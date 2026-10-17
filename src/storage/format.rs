//! On-disk layout of a Pith database file (format version 1, unstable until 1.0).
//!
//! All integers and floats are little-endian. Big-endian targets are refused at
//! compile time, because the f32 section is viewed in place as native `f32`.
//!
//! ```text
//! 0    magic            8   b"PITHDB\0\0"
//! 8    format_version   u32 = 1 (checked before anything else; unknown -> UnsupportedFormat)
//! 12   dim              u32 vector dimension (> 0; may have zero vectors)
//! 16   count            u32 number of chunks == number of vectors
//! 20   reserved         u32 must be 0
//! 24   chunks_len       u64 byte length of the compressed chunk section
//! 32   table_raw_len    u64 decompressed length of the chunk table (see check_table_len)
//! 40   hash_chunks      32  blake3 of the chunk section
//! 72   hash_quant       32  blake3 of the int8 section
//! 104  hash_f32         32  blake3 of the f32 section
//! 136  hash_header      32  blake3 of bytes [0, 136)
//! 168  chunk section    chunks_len bytes: zstd(model id, meta, chunk table)
//!      zero padding to the next multiple of 64
//! Q    int8 section     count * dim bytes (row-major)
//!      zero padding to the next multiple of 64
//! F    f32 section      count * dim * 4 bytes (row-major, f32 LE)
//! E    hash_file        32  blake3 of bytes [0, E)
//! ```
//!
//! There are no stored offsets: Q, F and E follow from the header with checked
//! arithmetic and must land exactly on the real file length, so a length field
//! pointing past EOF (even with valid checksums) or any truncation is rejected
//! before a byte of a section is read. Q and F are 64-byte aligned in the file and
//! the mapping base is page aligned, so the vector sections are used in place
//! (zero-copy) through a shared read-only mapping.
//!
//! What `decode` validates, in order: minimum length, magic, version, header hash,
//! reserved field, layout vs. real length, whole-file hash (so a flip of ANY byte,
//! padding included, fails), per-section hashes, the chunk table decompressed with a
//! hard bound (`table_raw_len`, itself capped by the
//! count, by `chunks_len * MAX_RATIO` and by a ceiling) and parsed with bounds checks (must
//! be consumed exactly, hold `count` entries, UTF-8 strings), `dim > 0`,
//! chunk count == vector count, and `VectorIndex::from_mapped` re-checks that the
//! int8 data equals `quantize(f32)` and that all floats are finite.
//!
//! Chunk table (inside zstd): str model_id, u64 indexed_at_unix, u8 has_git_head,
//! [str git_head], then `count` x (str path, u32 start_line, u32 end_line, str text);
//! `str` is u32 length + UTF-8 bytes.
//!
//! Unsupported (not detected after `open` validated the file): modifying a database
//! file in place, or truncating it, while it is open. Pith itself only ever replaces
//! the file by rename. An in-place write after validation would be seen as changed
//! vectors; a truncation makes reads of the mapping raise SIGBUS.

#[cfg(target_endian = "big")]
compile_error!("the Pith file format is little-endian and maps f32 data in place");

use super::{IndexMeta, StoredChunk};
use crate::index::VectorIndex;
use crate::{Error, Result};
use std::io::{Read, Write};
use std::sync::Arc;

pub const MAGIC: &[u8; 8] = b"PITHDB\0\0";
pub const VERSION: u32 = 1;
pub const HEADER_LEN: usize = 168;
const HASHED_HEADER: usize = 136;
const HASHES_AT: usize = 40;
/// Smallest valid file: header plus the trailing file hash.
pub const MIN_LEN: usize = HEADER_LEN + 32;
/// Per-chunk raw table budget, derived from the chunker's limit
/// (`crate::CHUNK_MAX_CHARS` = 1,400 chars, up to 4 UTF-8 bytes each) plus 4,096
/// bytes for the path and the 16 bytes of entry overhead:
/// 1,400 * 4 + 4,096 = 9,696 bytes.
///
/// This is a total budget per chunk (`count * PER_CHUNK_MAX` for the whole table),
/// not a per-field limit: a single path or text over its share is absorbed by the
/// headroom of other, typically much smaller, chunks. Only when the whole table
/// exceeds the bound does `Db::write` refuse it with `Error::Invalid`. The ratio
/// bound (`MAX_RATIO`) can also refuse a table with `Error::Invalid`.
///
/// `CHUNK_MAX_CHARS` is not enforced here: it will be enforced by the Task 4
/// chunker, which does not exist yet. Until then this budget is only an estimate.
pub const PER_CHUNK_MAX: u64 = crate::CHUNK_MAX_CHARS as u64 * 4 + 4_096;
/// Model id, meta and git head share this; also the additive slack of the ratio bound.
pub const TABLE_SLACK: u64 = 64 << 10;
/// Last-resort ceiling regardless of count: 4 GiB. The spec's scale is ~1M chunks;
/// a typical chunk is ~1,400 ASCII bytes + ~100 path + 16 overhead, so 1M chunks
/// is ~1.5e9 bytes < 4 GiB (4.29e9). Only pathological tables (e.g. 1M chunks of
/// all-4-byte text, 1M * 9,696 = 9.7e9) hit it, and the writer refuses those.
/// Note: 1M chunks of mostly-3-byte (CJK) text at the 1,400-char limit is ~4.3e9
/// (1,400 * 3 + ~100 + 16 per chunk) and can exceed this ceiling; `Db::write` then
/// refuses with the actionable "too large" error. Raising the ceiling is a
/// deliberate decision for later (it raises the worst-case allocation a crafted
/// file can force), not an oversight.
pub const TABLE_RAW_CEILING: u64 = 4 << 30;
/// Largest raw/compressed ratio a file may declare: defense in depth against a
/// crafted header claiming a huge raw table for a tiny payload.
///
/// Why 16384 and not tighter: duplicated corpora (vendored or generated copies)
/// compress very well (100k identical 1,400-char chunks reach ~10,864:1), and
/// `pith index` must not fail on them. The looser ratio is safe because it is not
/// the primary bound: the layout check ties `count` to vector bytes that must really
/// exist (`count * dim * 5 <= file size`), and `PER_CHUNK_MAX` bounds raw bytes per
/// chunk, so at a realistic dim (384) raw <= 9,696 / 1,920 * file size, about 5x the
/// file size. Only a crafted tiny-dim file is exposed to this looser bound, and it
/// still caps a 1 MB file at about 16 GB claimed, further clamped by the 4 GiB
/// ceiling and reserved only as bytes are really produced.
pub const MAX_RATIO: u64 = 16_384;
/// Decompression grows in steps of this size, each reserved fallibly.
const GROW_BLOCK: usize = 1 << 20;

/// Largest raw chunk table a file with `count` chunks may declare.
pub fn table_raw_max(count: u64) -> u64 {
    count
        .checked_mul(PER_CHUNK_MAX)
        .and_then(|n| n.checked_add(TABLE_SLACK))
        .map_or(TABLE_RAW_CEILING, |n| n.min(TABLE_RAW_CEILING))
}

/// Largest raw chunk table `compressed_len` bytes of zstd may declare.
pub fn table_ratio_max(compressed_len: u64) -> u64 {
    compressed_len
        .saturating_mul(MAX_RATIO)
        .saturating_add(TABLE_SLACK)
}

/// The one bound shared by writer and reader, so the writer cannot produce a file
/// that fails to open. Returns the reason on violation.
fn check_table_len(
    raw_len: u64,
    compressed_len: u64,
    count: u64,
) -> std::result::Result<(), String> {
    let max = table_raw_max(count);
    if raw_len > max {
        return Err(format!(
            "chunk table length {raw_len} over limit {max} for {count} chunks"
        ));
    }
    let max = table_ratio_max(compressed_len);
    if raw_len > max {
        return Err(format!(
            "chunk table length {raw_len} over limit {max} for {compressed_len} compressed bytes"
        ));
    }
    Ok(())
}

const ALIGN: usize = 64;
/// Up-front decompression reservation per compressed byte (text rarely exceeds it).
const RESERVE_RATIO: u64 = 64;
const ZSTD_LEVEL: i32 = 3;

pub struct Decoded {
    pub model_id: String,
    pub meta: IndexMeta,
    pub chunks: Vec<StoredChunk>,
    pub index: VectorIndex,
}

fn corrupt(msg: impl Into<String>) -> Error {
    Error::CorruptDb(msg.into())
}

fn align(n: usize) -> Option<usize> {
    n.checked_add(ALIGN - 1).map(|n| n & !(ALIGN - 1))
}

fn put_str(out: &mut Vec<u8>, s: &str) -> Result<()> {
    let len =
        u32::try_from(s.len()).map_err(|_| Error::Invalid("string longer than u32::MAX".into()))?;
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(s.as_bytes());
    Ok(())
}

/// Hashes everything written through it and pads to `ALIGN`.
struct HashingWriter<W> {
    inner: W,
    hasher: blake3::Hasher,
    pos: usize,
}

impl<W: Write> HashingWriter<W> {
    fn put(&mut self, b: &[u8]) -> Result<()> {
        self.inner.write_all(b)?;
        self.hasher.update(b);
        self.pos += b.len();
        Ok(())
    }
    fn pad(&mut self) -> Result<()> {
        let n = align(self.pos).expect("file offset fits usize") - self.pos;
        self.put(&[0u8; ALIGN][..n])
    }
}

/// Streams the file to `w`. The chunk table is buffered (raw, then compressed);
/// the vector sections are written straight from the index without a copy.
pub fn encode(
    w: impl Write,
    model_id: &str,
    chunks: &[StoredChunk],
    index: &VectorIndex,
    meta: &IndexMeta,
) -> Result<()> {
    if index.dim() == 0 {
        return Err(Error::Invalid(
            "index has dimension 0; build an empty index with \
             VectorIndex::from_parts(model_dim, vec![], vec![])"
                .into(),
        ));
    }
    if chunks.len() != index.len() {
        return Err(Error::Invalid(format!(
            "{} chunks but {} vectors",
            chunks.len(),
            index.len()
        )));
    }
    let too_big = || Error::Invalid("dimension or count exceeds u32::MAX".into());
    let dim = u32::try_from(index.dim()).map_err(|_| too_big())?;
    let count = u32::try_from(chunks.len()).map_err(|_| too_big())?;

    let mut table = Vec::new();
    put_str(&mut table, model_id)?;
    table.extend_from_slice(&meta.indexed_at_unix.to_le_bytes());
    match &meta.git_head {
        Some(h) => {
            table.push(1);
            put_str(&mut table, h)?;
        }
        None => table.push(0),
    }
    for c in chunks {
        put_str(&mut table, &c.path)?;
        table.extend_from_slice(&c.start_line.to_le_bytes());
        table.extend_from_slice(&c.end_line.to_le_bytes());
        put_str(&mut table, &c.text)?;
    }
    // Count check before compressing (passes the ratio check by construction);
    // after compressing, only the ratio check can still fail.
    const REMEDY: &str = "exclude generated, vendored or duplicated files with \
                          .pithignore, or index a smaller path";
    let mb = table.len() as f64 / 1e6;
    check_table_len(table.len() as u64, u64::MAX, count as u64).map_err(|_| {
        Error::Invalid(format!(
            "the indexed text is too large to store ({mb:.1} MB for {count} chunks); {REMEDY}"
        ))
    })?;
    let packed = zstd::bulk::compress(&table, ZSTD_LEVEL)?;
    check_table_len(table.len() as u64, packed.len() as u64, count as u64).map_err(|_| {
        Error::Invalid(format!(
            "the indexed text is too repetitive to store safely ({mb:.1} MB compresses \
             over {MAX_RATIO}:1); {REMEDY}"
        ))
    })?;
    let quant: &[u8] = bytemuck::cast_slice(index.quantized());
    let full: &[u8] = bytemuck::cast_slice(index.full()); // LE: big-endian is refused above

    let mut header = Vec::with_capacity(HEADER_LEN);
    header.extend_from_slice(MAGIC);
    header.extend_from_slice(&VERSION.to_le_bytes());
    header.extend_from_slice(&dim.to_le_bytes());
    header.extend_from_slice(&count.to_le_bytes());
    header.extend_from_slice(&0u32.to_le_bytes());
    header.extend_from_slice(&(packed.len() as u64).to_le_bytes());
    header.extend_from_slice(&(table.len() as u64).to_le_bytes());
    for s in [&packed[..], quant, full] {
        header.extend_from_slice(blake3::hash(s).as_bytes());
    }
    let h = blake3::hash(&header);
    header.extend_from_slice(h.as_bytes());
    debug_assert_eq!(header.len(), HEADER_LEN);
    drop(table);

    let mut w = HashingWriter {
        inner: w,
        hasher: blake3::Hasher::new(),
        pos: 0,
    };
    w.put(&header)?;
    w.put(&packed)?;
    w.pad()?;
    w.put(quant)?;
    w.pad()?;
    w.put(full)?;
    let h = w.hasher.finalize();
    w.inner.write_all(h.as_bytes())?;
    w.inner.flush()?;
    Ok(())
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().expect("4 bytes"))
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().expect("8 bytes"))
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if n > self.0.len() {
            return Err(corrupt("chunk table truncated"));
        }
        let (head, rest) = self.0.split_at(n);
        self.0 = rest;
        Ok(head)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("4 bytes"),
        ))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("8 bytes"),
        ))
    }
    fn str(&mut self) -> Result<String> {
        let n = self.u32()? as usize;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|_| corrupt("string is not UTF-8"))
    }
}

#[cfg(test)]
thread_local! {
    /// Test hook: (fail every reservation, largest capacity requested so far).
    static RESERVE_HOOK: std::cell::Cell<(bool, usize)> = const { std::cell::Cell::new((false, 0)) };
}

/// Fallibly reserves room for `additional` more bytes.
fn grow(out: &mut Vec<u8>, additional: usize) -> Result<()> {
    let total = out.len() + additional;
    let no_mem = || corrupt(format!("cannot reserve {total} bytes for chunk table"));
    #[cfg(test)]
    {
        let (fail, max) = RESERVE_HOOK.get();
        RESERVE_HOOK.set((fail, max.max(total)));
        if fail {
            return Err(no_mem());
        }
    }
    out.try_reserve(additional).map_err(|_| no_mem())
}

/// Decompresses exactly `raw_len` bytes; more or fewer is corruption. Up front it
/// reserves at most `compressed_len * RESERVE_RATIO + 4 KiB` (fallibly), so a lying
/// header cannot force a large allocation; the decoder refuses frames whose window
/// exceeds what `raw_len` needs.
fn decompress_bounded(src: &[u8], raw_len: u64, count: u64) -> Result<Vec<u8>> {
    check_table_len(raw_len, src.len() as u64, count).map_err(corrupt)?;
    let err = |e: std::io::Error| corrupt(format!("chunk section: {e}"));
    // Reserve in proportion to the input, not the header's claim, and fallibly;
    // the buffer grows past that only as bytes are really produced (<= raw_len + 1,
    // the +1 detecting overlong output).
    let reserve = (raw_len + 1).min(src.len() as u64 * RESERVE_RATIO + 4096);
    let mut out = Vec::new();
    grow(&mut out, reserve as usize)?;
    // A frame needs no window larger than its content; refuse frames declaring more.
    let window_log = (64 - raw_len.leading_zeros()).clamp(10, 31);
    let mut dec = zstd::stream::read::Decoder::new(src).map_err(err)?;
    dec.window_log_max(window_log).map_err(err)?;
    let limit = raw_len + 1;
    let mut dec = dec.take(limit);
    loop {
        let len = out.len();
        let want = GROW_BLOCK.min((limit - len as u64) as usize);
        if want == 0 {
            break;
        }
        if out.capacity() - len < want {
            grow(&mut out, want)?;
        }
        out.resize(len + want, 0); // within capacity: cannot allocate
        let n = dec.read(&mut out[len..]).map_err(err)?;
        out.truncate(len + n);
        if n == 0 {
            break;
        }
    }
    if out.len() as u64 != raw_len {
        return Err(corrupt(format!(
            "chunk table decompressed to {}{} bytes, header says {raw_len}",
            out.len(),
            if out.len() as u64 > raw_len { "+" } else { "" }
        )));
    }
    Ok(out)
}

pub fn decode(map: Arc<memmap2::Mmap>) -> Result<Decoded> {
    let b: &[u8] = &map;
    if b.len() < MIN_LEN {
        return Err(corrupt(format!("file is {} bytes, too short", b.len())));
    }
    if &b[..8] != MAGIC {
        return Err(corrupt("not a Pith database (bad magic)"));
    }
    let version = u32_at(b, 8);
    if version != VERSION {
        return Err(Error::UnsupportedFormat(version));
    }
    if blake3::hash(&b[..HASHED_HEADER]).as_bytes() != &b[HASHED_HEADER..HEADER_LEN] {
        return Err(corrupt("header checksum mismatch"));
    }
    let dim = u32_at(b, 12) as usize;
    let count = u32_at(b, 16) as usize;
    if u32_at(b, 20) != 0 {
        return Err(corrupt("reserved header field is not zero"));
    }
    let raw_len = u64_at(b, 32);

    let bad_layout = || corrupt("section lengths do not match file length");
    let chunks_len = usize::try_from(u64_at(b, 24)).map_err(|_| bad_layout())?;
    let n_vals = dim.checked_mul(count).ok_or_else(bad_layout)?;
    let q_off = HEADER_LEN
        .checked_add(chunks_len)
        .and_then(align)
        .ok_or_else(bad_layout)?;
    let f_off = q_off
        .checked_add(n_vals)
        .and_then(align)
        .ok_or_else(bad_layout)?;
    let e_off = n_vals
        .checked_mul(4)
        .and_then(|n| f_off.checked_add(n))
        .ok_or_else(bad_layout)?;
    if e_off.checked_add(32) != Some(b.len()) {
        return Err(bad_layout());
    }
    if blake3::hash(&b[..e_off]).as_bytes() != &b[e_off..] {
        return Err(corrupt("file checksum mismatch"));
    }
    let ranges = [
        HEADER_LEN..HEADER_LEN + chunks_len,
        q_off..q_off + n_vals,
        f_off..e_off,
    ];
    for (i, r) in ranges.iter().enumerate() {
        let at = HASHES_AT + 32 * i;
        if blake3::hash(&b[r.clone()]).as_bytes() != &b[at..at + 32] {
            return Err(corrupt(format!("section {i} checksum mismatch")));
        }
    }
    if dim == 0 {
        return Err(corrupt("dimension is 0"));
    }

    let table = decompress_bounded(&b[ranges[0].clone()], raw_len, count as u64)?;
    let mut r = Reader(&table);
    let model_id = r.str()?;
    let indexed_at_unix = r.u64()?;
    let git_head = match r.take(1)?[0] {
        0 => None,
        1 => Some(r.str()?),
        _ => return Err(corrupt("bad git_head flag")),
    };
    // Each entry is at least 16 bytes, which bounds the allocation by the real data.
    let mut chunks = Vec::with_capacity(count.min(r.0.len() / 16));
    for _ in 0..count {
        chunks.push(StoredChunk {
            path: r.str()?,
            start_line: r.u32()?,
            end_line: r.u32()?,
            text: r.str()?,
        });
    }
    if !r.0.is_empty() {
        return Err(corrupt("trailing bytes in chunk table"));
    }
    let [_, q, f] = ranges;
    let index =
        VectorIndex::from_mapped(dim, map, q, f).map_err(|e| corrupt(format!("vectors: {e}")))?;
    check_counts(chunks.len(), &index)?;
    Ok(Decoded {
        model_id,
        meta: IndexMeta {
            indexed_at_unix,
            git_head,
        },
        chunks,
        index,
    })
}

/// Defensive: unreachable from file bytes today, because with `dim > 0` the header
/// `count` fixes both the chunk count and the vector count. Kept so a future
/// layout change cannot silently pair chunks with the wrong vectors.
fn check_counts(chunks: usize, index: &VectorIndex) -> Result<()> {
    if index.len() != chunks {
        return Err(corrupt(format!(
            "{chunks} chunks but {} vectors",
            index.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packed() -> Vec<u8> {
        zstd::bulk::compress(b"tiny", 3).unwrap()
    }

    /// Runs `f` with the reserve hook set to `fail`; returns its result and the
    /// largest capacity requested.
    fn hooked<T>(fail: bool, f: impl FnOnce() -> T) -> (T, usize) {
        RESERVE_HOOK.set((fail, 0));
        let r = f();
        let (_, max) = RESERVE_HOOK.replace((false, 0));
        (r, max)
    }

    #[test]
    fn reservation_failure_is_clean_corruption() {
        let p = packed();
        let (r, _) = hooked(true, || decompress_bounded(&p, 4, 0));
        match r {
            Err(Error::CorruptDb(m)) => assert!(m.starts_with("cannot reserve 5 bytes"), "{m}"),
            other => panic!("{other:?}"),
        }
        let (r, _) = hooked(false, || decompress_bounded(&p, 4, 0));
        assert_eq!(r.unwrap(), b"tiny");
    }

    #[test]
    fn header_claiming_the_maximum_fails_without_large_reservation() {
        // Payload of 4 KiB of hash-scrambled bytes that decompresses to `content`; big enough
        // that for counts 0, 1 and 1000 the count-derived bound is the binding one.
        let content: Vec<u8> = (0..4096u32)
            .map(|i| (i.wrapping_mul(2_654_435_761) >> 24) as u8)
            .collect();
        let p = zstd::bulk::compress(&content, 3).unwrap();
        for count in [0u64, 1, 1_000] {
            let max = table_raw_max(count);
            assert!(
                max < table_ratio_max(p.len() as u64),
                "count bound must bind"
            );
            assert!(check_table_len(max, p.len() as u64, count).is_ok());
            assert!(check_table_len(max + 1, p.len() as u64, count).is_err());
            let (r, reserved) = hooked(false, || decompress_bounded(&p, max, count));
            match r {
                Err(Error::CorruptDb(m)) => {
                    let want = format!("{} bytes, header says {max}", content.len());
                    assert!(m.contains(&want), "{m}")
                }
                other => panic!("{other:?}"),
            }
            // Up front min(claim + 1, packed * 64 + 4 KiB); then at most one block
            // past the bytes produced, and never past the claim + 1.
            let bound = (max as usize + 1).min(content.len() + GROW_BLOCK);
            assert!(reserved <= bound, "{reserved} for claim {max}");
        }
        // Huge count: the ratio bound binds; a tiny payload reserves only a few KiB.
        let tiny = packed();
        let max = table_ratio_max(tiny.len() as u64);
        assert!(max < table_raw_max(u32::MAX as u64));
        let (r, reserved) = hooked(false, || decompress_bounded(&tiny, max, u32::MAX as u64));
        assert!(matches!(r, Err(Error::CorruptDb(m)) if m.contains("4 bytes, header says")));
        assert!(reserved <= 4 + GROW_BLOCK, "{reserved}");
    }

    #[test]
    fn ratio_bound_rejects_before_decompressing() {
        let p = packed();
        let over = table_ratio_max(p.len() as u64) + 1;
        let (r, reserved) = hooked(false, || decompress_bounded(&p, over, 1_000_000));
        match r {
            Err(Error::CorruptDb(m)) => assert!(
                m.contains(&format!(
                    "over limit {} for {} compressed bytes",
                    over - 1,
                    p.len()
                )),
                "{m}"
            ),
            other => panic!("{other:?}"),
        }
        assert_eq!(reserved, 0, "reserved before the bound check");
    }

    #[test]
    fn ceiling_is_4_gib() {
        assert_eq!(TABLE_RAW_CEILING, 4 << 30);
        assert_eq!(table_raw_max(u32::MAX as u64), 4 << 30);
        assert_eq!(PER_CHUNK_MAX, 9_696);
    }

    #[test]
    fn check_counts_rejects_mismatch() {
        let index = VectorIndex::build(vec![vec![0.6, 0.8]; 3]).unwrap();
        assert!(check_counts(3, &index).is_ok());
        for n in [0, 2, 4] {
            match check_counts(n, &index) {
                Err(Error::CorruptDb(m)) => assert_eq!(m, format!("{n} chunks but 3 vectors")),
                other => panic!("{other:?}"),
            }
        }
    }
}
