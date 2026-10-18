//! In-memory BM25 keyword index over chunk texts (k1 = 1.2, b = 0.75), built on first use and
//! never stored. The tokenizer is identifier-aware: `parseHttpURL2` yields `parsehttpurl2` plus
//! `parse`, `http`, `url`; `MAX_K` yields `max_k` and `max`. Tokens are lowercased, 2 to 64 chars,
//! with no stemming and no stop words.
//!
//! Memory is bounded: each chunk contributes at most [`MAX_CHUNK_TOKENS`] tokens, and once the
//! index holds [`MAX_POSTINGS`] postings (about 128 MiB) later chunks are not keyword-indexed.

use std::collections::HashMap;

const K1: f32 = 1.2;
const B: f32 = 0.75;
const MIN_TOKEN: usize = 2;
const MAX_TOKEN: usize = 64;
pub(crate) const MAX_CHUNK_TOKENS: usize = 1024;
pub(crate) const MAX_POSTINGS: usize = 16 << 20;

/// Calls `f` with every token of `text`, at most `max` of them.
fn tokens(text: &str, max: usize, mut f: impl FnMut(&str)) {
    let mut left = max;
    let mut buf = String::new();
    let mut emit = |chars: &[char], left: &mut usize| {
        if *left == 0 || !(MIN_TOKEN..=MAX_TOKEN).contains(&chars.len()) {
            return;
        }
        buf.clear();
        buf.extend(chars.iter().flat_map(|c| c.to_lowercase()));
        f(&buf);
        *left -= 1;
    };
    let mut run: Vec<char> = Vec::new();
    let mut chars = text.chars().peekable();
    while left > 0 {
        run.clear();
        while let Some(&c) = chars.peek() {
            chars.next();
            if c.is_alphanumeric() || c == '_' {
                run.push(c);
            } else if !run.is_empty() {
                break;
            }
        }
        if run.is_empty() {
            return;
        }
        let word = {
            let s = run.iter().position(|&c| c != '_').unwrap_or(run.len());
            let e = run.iter().rposition(|&c| c != '_').map_or(s, |e| e + 1);
            &run[s..e]
        };
        emit(word, &mut left);
        let parts = split_parts(word);
        if parts.len() > 1 {
            for p in parts {
                emit(p, &mut left);
            }
        }
    }
}

/// Splits an identifier at `_`, lower-to-upper case changes (`parseHttp`), the last capital of
/// an acronym (`HTTPServer`) and letter/digit changes (`f32`).
fn split_parts(word: &[char]) -> Vec<&[char]> {
    let mut parts = Vec::new();
    for seg in word.split(|&c| c == '_').filter(|s| !s.is_empty()) {
        let mut start = 0;
        for i in 1..seg.len() {
            let (a, b) = (seg[i - 1], seg[i]);
            let next_lower = seg.get(i + 1).is_some_and(|c| c.is_lowercase());
            let cut = (a.is_lowercase() && b.is_uppercase())
                || (a.is_uppercase() && b.is_uppercase() && next_lower)
                || (a.is_numeric() != b.is_numeric());
            if cut {
                parts.push(&seg[start..i]);
                start = i;
            }
        }
        parts.push(&seg[start..]);
    }
    parts
}

/// Every token of `text`, in order (tests and callers that want a list).
#[cfg(test)]
pub(crate) fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    tokens(text, MAX_CHUNK_TOKENS, |t| out.push(t.to_string()));
    out
}

pub(crate) struct Bm25 {
    terms: HashMap<String, u32>,
    /// Per term: (chunk id, term frequency), in ascending chunk id.
    postings: Vec<Vec<(u32, u32)>>,
    lens: Vec<u32>,
    avg_len: f32,
}

impl Bm25 {
    /// Indexes `texts`; chunk ids are their positions.
    pub(crate) fn build<'a>(texts: impl IntoIterator<Item = &'a str>) -> Bm25 {
        let mut terms: HashMap<String, u32> = HashMap::new();
        let mut postings: Vec<Vec<(u32, u32)>> = Vec::new();
        let mut lens = Vec::new();
        let mut total = 0usize;
        for (id, text) in texts.into_iter().enumerate() {
            let id = id as u32;
            let mut len = 0;
            if total < MAX_POSTINGS {
                tokens(text, MAX_CHUNK_TOKENS, |t| {
                    let term = match terms.get(t) {
                        Some(&i) => i as usize,
                        None => {
                            terms.insert(t.to_string(), postings.len() as u32);
                            postings.push(Vec::new());
                            postings.len() - 1
                        }
                    };
                    // Chunks arrive in id order, so this chunk's posting is the last one if any.
                    match postings[term].last_mut() {
                        Some((last, n)) if *last == id => *n += 1,
                        _ => {
                            postings[term].push((id, 1));
                            total += 1;
                        }
                    }
                    len += 1;
                });
            }
            lens.push(len);
        }
        let avg_len = if lens.is_empty() {
            0.0
        } else {
            lens.iter().map(|&l| l as f64).sum::<f64>() as f32 / lens.len() as f32
        };
        Bm25 {
            terms,
            postings,
            lens,
            avg_len,
        }
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
        let docs = self.lens.len() as f32;
        let mut scores: HashMap<u32, f32> = HashMap::new();
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
        let t = std::time::Instant::now();
        let b = Bm25::build(texts.iter().map(String::as_str));
        eprintln!("Bm25::build 100k chunks: {:?}", t.elapsed());
        let t = std::time::Instant::now();
        let hits = b.search("ident_123 let", 50);
        eprintln!("Bm25::search: {:?}", t.elapsed());
        assert!(!hits.is_empty());
    }
}
