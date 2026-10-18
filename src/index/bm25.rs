//! In-memory BM25 keyword index over chunk texts (k1 = 1.2, b = 0.75), built on first use and
//! never stored. The tokenizer is identifier-aware: `parseHttpURL2` yields `parsehttpurl2` plus
//! `parse`, `http`, `url`; `MAX_K` yields `max_k` and `max`. Tokens are lowercased, 2 to 64 chars,
//! with no stemming, no stop words and no Unicode normalization; a run of CJK text is one token.
//!
//! Memory is bounded: each chunk contributes at most [`MAX_CHUNK_TOKENS`] tokens, and every
//! posting and every distinct term (at most 64 chars) is charged one unit of [`MAX_UNITS`]. A
//! chunk that would start once the budget is spent is not keyword-indexed, and does not count
//! toward the document statistics.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

const K1: f32 = 1.2;
const B: f32 = 0.75;
const MIN_TOKEN: usize = 2;
const MAX_TOKEN: usize = 64;
pub(crate) const MAX_CHUNK_TOKENS: usize = 1024;
pub(crate) const MAX_UNITS: usize = 32 << 20;

/// FxHash-style hasher: much faster than SipHash on short keys. The keys come from the user's
/// own indexed files, so hash flooding is not a concern.
#[derive(Default, Clone, Copy)]
struct Fx(u64);

impl Hasher for Fx {
    fn write(&mut self, bytes: &[u8]) {
        for c in bytes.chunks(8) {
            let mut w = [0u8; 8];
            w[..c.len()].copy_from_slice(c);
            self.0 =
                (self.0.rotate_left(5) ^ u64::from_le_bytes(w)).wrapping_mul(0x517c_c1b7_2722_0a95);
        }
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<Fx>>;

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Calls `f` with every token of `text`, at most `max` of them.
fn tokens(text: &str, max: usize, mut f: impl FnMut(&str)) {
    let mut left = max;
    let mut buf = String::new();
    let mut emit = |t: &str, left: &mut usize| {
        let len = if t.is_ascii() {
            t.len()
        } else {
            t.chars().count()
        };
        if *left == 0 || !(MIN_TOKEN..=MAX_TOKEN).contains(&len) {
            return;
        }
        buf.clear();
        if t.is_ascii() {
            buf.push_str(t);
            buf.make_ascii_lowercase();
        } else {
            buf.extend(t.chars().flat_map(char::to_lowercase));
        }
        f(&buf);
        *left -= 1;
    };
    let mut rest = text;
    while left > 0 {
        let Some(start) = rest.find(is_word) else {
            return;
        };
        rest = &rest[start..];
        let end = rest.find(|c| !is_word(c)).unwrap_or(rest.len());
        let word = rest[..end].trim_matches('_');
        rest = &rest[end..];
        emit(word, &mut left);
        parts(word, |p| {
            // A word with one part is that part: emitted above already.
            if p.len() != word.len() {
                emit(p, &mut left);
            }
        });
    }
}

/// Calls `f` with the parts of an identifier, split at `_`, lower-to-upper case changes
/// (`parseHttp`), the last capital of an acronym (`HTTPServer`) and letter/digit changes (`f32`).
fn parts(word: &str, mut f: impl FnMut(&str)) {
    for seg in word.split('_').filter(|s| !s.is_empty()) {
        let mut start = 0;
        let mut it = seg.char_indices().peekable();
        let mut prev: Option<char> = None;
        while let Some((i, b)) = it.next() {
            if let Some(a) = prev {
                let next_lower = it.peek().is_some_and(|&(_, c)| c.is_lowercase());
                let cut = (a.is_lowercase() && b.is_uppercase())
                    || (a.is_uppercase() && b.is_uppercase() && next_lower)
                    || (a.is_numeric() != b.is_numeric());
                if cut {
                    f(&seg[start..i]);
                    start = i;
                }
            }
            prev = Some(b);
        }
        f(&seg[start..]);
    }
}

/// Every token of `text`, in order.
#[cfg(test)]
pub(crate) fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    tokens(text, MAX_CHUNK_TOKENS, |t| out.push(t.to_string()));
    out
}

pub(crate) struct Bm25 {
    terms: FxMap<String, u32>,
    /// Per term: (chunk id, term frequency), in ascending chunk id.
    postings: Vec<Vec<(u32, u32)>>,
    /// Token count per chunk; 0 for a chunk left out by the memory budget.
    lens: Vec<u32>,
    /// Chunks indexed, and their average token count.
    docs: usize,
    avg_len: f32,
}

impl Bm25 {
    /// Indexes `texts`; chunk ids are their positions.
    pub(crate) fn build<'a>(texts: impl IntoIterator<Item = &'a str>) -> Bm25 {
        Self::build_capped(texts, MAX_UNITS)
    }

    fn build_capped<'a>(texts: impl IntoIterator<Item = &'a str>, max_units: usize) -> Bm25 {
        let mut terms: FxMap<String, u32> = FxMap::default();
        let mut postings: Vec<Vec<(u32, u32)>> = Vec::new();
        let mut lens = Vec::new();
        let (mut units, mut docs, mut total_len) = (0usize, 0usize, 0u64);
        for (id, text) in texts.into_iter().enumerate() {
            let id = id as u32;
            let mut len = 0;
            if units < max_units {
                docs += 1;
                tokens(text, MAX_CHUNK_TOKENS, |t| {
                    let term = match terms.get(t) {
                        Some(&i) => i as usize,
                        None => {
                            terms.insert(t.to_string(), postings.len() as u32);
                            postings.push(Vec::new());
                            units += 1;
                            postings.len() - 1
                        }
                    };
                    // Chunks arrive in id order, so this chunk's posting is the last one if any.
                    match postings[term].last_mut() {
                        Some((last, n)) if *last == id => *n += 1,
                        _ => {
                            postings[term].push((id, 1));
                            units += 1;
                        }
                    }
                    len += 1;
                });
            }
            total_len += u64::from(len);
            lens.push(len);
        }
        // Growth slack would otherwise be up to half of all posting memory.
        postings.iter_mut().for_each(Vec::shrink_to_fit);
        let avg_len = if docs == 0 {
            0.0
        } else {
            (total_len as f64 / docs as f64) as f32
        };
        Bm25 {
            terms,
            postings,
            lens,
            docs,
            avg_len,
        }
    }

    /// Distinct terms and postings: the units charged against [`MAX_UNITS`].
    #[cfg(test)]
    fn units(&self) -> (usize, usize) {
        (self.terms.len(), self.postings.iter().map(Vec::len).sum())
    }

    /// Top `n` chunks for `query` by BM25 score, descending (ties by ascending id). Chunks that
    /// share no token with the query are not returned.
    pub(crate) fn search(&self, query: &str, n: usize) -> Vec<(u32, f32)> {
        let mut seen: Vec<u32> = Vec::new();
        tokens(query, MAX_CHUNK_TOKENS, |t| {
            if let Some(&i) = self.terms.get(t) {
                if !seen.contains(&i) {
                    seen.push(i);
                }
            }
        });
        let docs = self.docs as f32;
        let mut scores: FxMap<u32, f32> = FxMap::default();
        for term in seen {
            let list = &self.postings[term as usize];
            let df = list.len() as f32;
            let idf = (1.0 + (docs - df + 0.5) / (df + 0.5)).ln();
            for &(id, tf) in list {
                let (tf, len) = (tf as f32, self.lens[id as usize] as f32);
                let norm = K1 * (1.0 - B + B * len / self.avg_len.max(1.0));
                *scores.entry(id).or_default() += idf * tf * (K1 + 1.0) / (tf + norm);
            }
        }
        let mut out: Vec<(u32, f32)> = scores.into_iter().collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        out.truncate(n);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifiers_yield_the_whole_name_and_its_parts() {
        assert_eq!(
            tokenize("search_project"),
            ["search_project", "search", "project"]
        );
        assert_eq!(
            tokenize("parseHttpURL2"),
            ["parsehttpurl2", "parse", "http", "url"]
        );
        assert_eq!(tokenize("HTTPServer"), ["httpserver", "http", "server"]);
        assert_eq!(tokenize("u32 f32x4"), ["u32", "32", "f32x4", "32"]);
        assert_eq!(tokenize("__MAX_K__"), ["max_k", "max"]);
    }

    #[test]
    fn punctuation_short_and_long_tokens_are_dropped() {
        assert_eq!(tokenize("a == b; -> {} :: x"), Vec::<String>::new());
        assert_eq!(tokenize("Error::Locked"), ["error", "locked"]);
        let long = "y".repeat(65);
        assert_eq!(tokenize(&format!("{long} ok")), ["ok"]);
        assert_eq!(tokenize(&"z".repeat(64)), ["z".repeat(64)]);
    }

    #[test]
    fn unicode_is_lowercased_and_split_on_case() {
        assert_eq!(
            tokenize("Größe ÜberMaß"),
            ["größe", "übermaß", "über", "maß"]
        );
        assert_eq!(tokenize("数据库 検索"), ["数据库", "検索"]);
    }

    #[test]
    fn tokens_per_chunk_are_capped() {
        let text = "ab ".repeat(MAX_CHUNK_TOKENS * 2);
        assert_eq!(tokenize(&text).len(), MAX_CHUNK_TOKENS);
    }

    #[test]
    fn exact_identifier_outranks_near_synonyms() {
        let texts = [
            "fn acquire_lock() { lock the database writer }",
            "fn take_writer_lock() -> WriterLock { WriterLock::acquire(path) }",
            "fn open_reader() { read only, no lock taken here at all }",
            "fn unrelated() {}",
        ];
        let b = Bm25::build(texts);
        let hits = b.search("WriterLock", 10);
        assert_eq!(hits[0].0, 1, "{hits:?}");
        assert!(hits.iter().all(|h| h.0 != 3), "{hits:?}");
        assert!(b.search("zzz nothing", 10).is_empty());
        let hits = b.search("lock", 2);
        assert_eq!(hits.len(), 2);
        assert!(hits[0].1 >= hits[1].1);
    }

    #[test]
    fn chunks_past_the_budget_are_left_out_of_the_statistics() {
        let b = Bm25::build(["alpha beta", "gamma"]);
        assert_eq!((b.docs, b.lens.as_slice()), (2, &[2, 1][..]));
        assert_eq!(b.units(), (3, 3));
        // 4 units after the first chunk: the second starts, the third does not.
        let b = Bm25::build_capped(["alpha beta", "gamma", "gamma delta"], 5);
        assert_eq!((b.docs, b.lens.as_slice()), (2, &[2, 1, 0][..]));
        assert_eq!(b.search("delta", 5), []);
        assert!((b.avg_len - 1.5).abs() < 1e-6);
    }

    #[test]
    fn ties_break_by_ascending_id() {
        let b = Bm25::build(["same text", "same text", "same text"]);
        let ids: Vec<u32> = b.search("text", 10).iter().map(|h| h.0).collect();
        assert_eq!(ids, [0, 1, 2]);
    }

    #[test]
    #[ignore = "timing; run with --release -- --ignored --nocapture"]
    fn build_time_at_100k_chunks() {
        let mut x = 1u32;
        let mut word = || {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            format!("ident_{}", x >> 20)
        };
        // About 600 chars of code-like text per chunk, from a vocabulary of ~4k identifiers.
        let texts: Vec<String> = (0..100_000)
            .map(|_| {
                (0..30)
                    .map(|_| format!("let {} = {}(x);\n", word(), word()))
                    .collect()
            })
            .collect();
        let rss = || {
            let out = std::process::Command::new("ps")
                .args(["-o", "rss=", "-p", &std::process::id().to_string()])
                .output()
                .unwrap();
            String::from_utf8_lossy(&out.stdout)
                .trim()
                .parse::<u64>()
                .unwrap_or(0)
                / 1024
        };
        let before = rss();
        let t = std::time::Instant::now();
        let b = Bm25::build(texts.iter().map(String::as_str));
        let took = t.elapsed();
        let (terms, postings) = b.units();
        eprintln!(
            "Bm25::build 100k chunks: {took:?}; {terms} terms + {postings} postings of {MAX_UNITS} \
             units; RSS +{} MiB",
            rss().saturating_sub(before)
        );
        assert_eq!(
            b.docs, 100_000,
            "the budget must not cut a 100k-chunk index"
        );
        let t = std::time::Instant::now();
        let hits = b.search("ident_123 let", 50);
        eprintln!("Bm25::search: {:?}", t.elapsed());
        assert!(!hits.is_empty());
    }
}
