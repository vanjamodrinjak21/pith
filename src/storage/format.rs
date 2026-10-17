//! On-disk layout of a Pith database file (format version 1, unstable until 1.0).
//!
//! All integers are little-endian. Offsets in bytes.
//!
//! ```text
//! 0    magic            8   b"PITHDB\0\0"
//! 8    format_version   u32 = 1 (checked before anything else; unknown -> UnsupportedFormat)
//! 12   dim              u32 vector dimension (may be > 0 with zero vectors)
//! 16   count            u32 number of chunks == number of vectors
//! 20   reserved         u32 must be 0
//! 24   chunks_len       u64 byte length of the compressed chunk section
//! 32   hash_chunks      32  blake3 of the chunk section
//! 64   hash_quant       32  blake3 of the int8 section
//! 96   hash_f32         32  blake3 of the f32 section
//! 128  hash_header      32  blake3 of bytes [0, 128)
//! 160  chunk section    chunks_len bytes: zstd(model id, meta, chunk table)
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
//! before a byte of a section is read. Both vector sections start 64-byte aligned
//! (mmap base is page aligned), so they can later be used zero-copy.
//!
//! What `decode` validates, in order: minimum length, magic, version, header hash,
//! reserved field, layout vs. real length, whole-file hash (so a flip of ANY byte,
//! padding included, fails), per-section hashes, then the decompressed chunk table
//! is parsed with bounds checks and must be consumed exactly and hold `count`
//! entries, strings must be UTF-8, and `VectorIndex::from_parts` re-checks that the
//! int8 data matches the f32 data and that all floats are finite.
//!
//! Chunk table (inside zstd): str model_id, u64 indexed_at_unix, u8 has_git_head,
//! [str git_head], then `count` x (str path, u32 start_line, u32 end_line, str text);
//! `str` is u32 length + UTF-8 bytes.
//!
//! shortcut: zstd output size is not capped, so a checksum-valid crafted file could
//! decompress to a huge allocation; add a stored raw length cap if files ever come
//! from untrusted sources.

use super::{IndexMeta, StoredChunk};
use crate::index::VectorIndex;
use crate::{Error, Result};

pub const MAGIC: &[u8; 8] = b"PITHDB\0\0";
pub const VERSION: u32 = 1;
pub const HEADER_LEN: usize = 160;
const HASHED_HEADER: usize = 128;
const ALIGN: usize = 64;
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
    let len = u32::try_from(s.len()).map_err(|_| corrupt("string longer than u32::MAX"))?;
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(s.as_bytes());
    Ok(())
}

pub fn encode(
    model_id: &str,
    chunks: &[StoredChunk],
    index: &VectorIndex,
    meta: &IndexMeta,
) -> Result<Vec<u8>> {
    if chunks.len() != index.len() {
        return Err(corrupt(format!(
            "{} chunks but {} vectors",
            chunks.len(),
            index.len()
        )));
    }
    let too_big = || corrupt("dimension or count exceeds u32::MAX");
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
    let packed = zstd::bulk::compress(&table, ZSTD_LEVEL)?;
    let quant: Vec<u8> = index.quantized().iter().map(|&q| q as u8).collect();
    let full: Vec<u8> = index.full().iter().flat_map(|f| f.to_le_bytes()).collect();

    let mut out =
        Vec::with_capacity(HEADER_LEN + packed.len() + quant.len() + full.len() + 3 * ALIGN);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&dim.to_le_bytes());
    out.extend_from_slice(&count.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&(packed.len() as u64).to_le_bytes());
    for s in [&packed, &quant, &full] {
        out.extend_from_slice(blake3::hash(s).as_bytes());
    }
    let h = blake3::hash(&out);
    out.extend_from_slice(h.as_bytes());
    debug_assert_eq!(out.len(), HEADER_LEN);
    for s in [&packed, &quant] {
        out.extend_from_slice(s);
        out.resize(align(out.len()).ok_or_else(too_big)?, 0);
    }
    out.extend_from_slice(&full);
    let h = blake3::hash(&out);
    out.extend_from_slice(h.as_bytes());
    Ok(out)
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().expect("4 bytes"))
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

pub fn decode(b: &[u8]) -> Result<Decoded> {
    if b.len() < HEADER_LEN + 32 {
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
    let chunks_len = u64::from_le_bytes(b[24..32].try_into().expect("8 bytes"));

    let bad_layout = || corrupt("section lengths do not match file length");
    let chunks_len = usize::try_from(chunks_len).map_err(|_| bad_layout())?;
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
    let sections = [
        &b[HEADER_LEN..HEADER_LEN + chunks_len],
        &b[q_off..q_off + n_vals],
        &b[f_off..e_off],
    ];
    for (i, s) in sections.iter().enumerate() {
        let at = 32 + 32 * i;
        if blake3::hash(s).as_bytes() != &b[at..at + 32] {
            return Err(corrupt(format!("section {i} checksum mismatch")));
        }
    }

    let table = zstd::stream::decode_all(sections[0])
        .map_err(|e| corrupt(format!("chunk section: {e}")))?;
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
    let quant = sections[1].iter().map(|&x| x as i8).collect();
    let full = sections[2]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
        .collect();
    let index =
        VectorIndex::from_parts(dim, quant, full).map_err(|e| corrupt(format!("vectors: {e}")))?;
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
