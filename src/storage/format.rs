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
//! 32   table_raw_len    u64 decompressed length of the chunk table (<= MAX_TABLE_RAW)
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
//! hard bound (`table_raw_len`, itself capped) and parsed with bounds checks (must
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
/// Upper bound for the decompressed chunk table.
pub const MAX_TABLE_RAW: u64 = 1 << 30;
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

/// Streams the file to `w`. Only the compressed chunk table is buffered; vector
/// sections are written straight from the index.
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
    if table.len() as u64 > MAX_TABLE_RAW {
        return Err(Error::Invalid(format!(
            "chunk table is {} bytes, over the {MAX_TABLE_RAW} byte limit",
            table.len()
        )));
    }
    let packed = zstd::bulk::compress(&table, ZSTD_LEVEL)?;
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

/// Decompresses at most `raw_len` bytes; more or fewer is corruption. Memory grows
/// only with real output, so a small payload claiming a huge length cannot force
/// a large allocation up front.
fn decompress_bounded(src: &[u8], raw_len: u64) -> Result<Vec<u8>> {
    if raw_len > MAX_TABLE_RAW {
        return Err(corrupt(format!("chunk table length {raw_len} over limit")));
    }
    let err = |e: std::io::Error| corrupt(format!("chunk section: {e}"));
    let mut out = Vec::new();
    zstd::stream::read::Decoder::new(src)
        .map_err(err)?
        .take(raw_len + 1)
        .read_to_end(&mut out)
        .map_err(err)?;
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

    let table = decompress_bounded(&b[ranges[0].clone()], raw_len)?;
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
