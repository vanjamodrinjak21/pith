//! `pith eval`: retrieval quality on a set of hand-written cases. Each line of the cases file is
//! `{"query": "...", "expect": ["src/a.rs", "src/b.rs:10-40"]}`. An expectation is a path (any
//! chunk of that file) or `path:start-end` (a hit must overlap those lines). A case is found at
//! the first rank whose hit meets any of its expectations.

use crate::budget::{context, tokens, DEFAULT_K, MAX_PER_FILE};
use crate::commands::{clean, safe_join, search_with, unix_now, SearchMode, MAX_K};
use crate::embed::Embedder;
use crate::storage::Db;
use crate::{Error, Result, SearchHit};
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub struct Case {
    pub query: String,
    pub expect: Vec<Expect>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expect {
    File(String),
    Lines(String, u32, u32),
}

impl Expect {
    fn path(&self) -> &str {
        match self {
            Expect::File(p) | Expect::Lines(p, ..) => p,
        }
    }

    fn met_by(&self, h: &SearchHit) -> bool {
        match self {
            Expect::File(p) => h.path == *p,
            Expect::Lines(p, s, e) => h.path == *p && h.start_line <= *e && h.end_line >= *s,
        }
    }

    fn parse(s: &str) -> std::result::Result<Expect, String> {
        if s.is_empty() {
            return Err("empty expectation".into());
        }
        let range = s.rsplit_once(':').and_then(|(p, r)| {
            let (a, b) = r.split_once('-')?;
            Some((p, a.parse::<u32>().ok()?, b.parse::<u32>().ok()?))
        });
        match range {
            Some((p, a, b)) if p.is_empty() || a == 0 || a > b => {
                Err(format!("bad line range in {s:?}"))
            }
            Some((p, a, b)) => Ok(Expect::Lines(p.into(), a, b)),
            None => Ok(Expect::File(s.into())),
        }
    }
}

/// Parses a cases file; blank lines are skipped, anything else malformed is an error naming
/// its line number, and a file with no cases is an error.
pub fn parse_cases(text: &str) -> Result<Vec<Case>> {
    let cases = text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| parse_case(l).map_err(|e| Error::Invalid(format!("line {}: {e}", i + 1))))
        .collect::<Result<Vec<_>>>()?;
    if cases.is_empty() {
        return Err(Error::Invalid("the cases file has no cases".into()));
    }
    Ok(cases)
}

/// The k that searches actually use: 1 to 100.
pub fn effective_k(k: usize) -> usize {
    k.clamp(1, MAX_K)
}

/// The two-key subset of JSON that a case is: an object with a string `query` and a non-empty
/// array of strings `expect`.
fn parse_case(line: &str) -> std::result::Result<Case, String> {
    let mut p = Json(line.trim().chars().peekable());
    let (mut query, mut expect) = (None, None);
    p.eat('{')?;
    loop {
        let key = p.string()?;
        p.eat(':')?;
        match key.as_str() {
            "query" if query.is_none() => query = Some(p.string()?),
            "expect" if expect.is_none() => {
                let mut v = Vec::new();
                p.eat('[')?;
                if !p.try_eat(']') {
                    loop {
                        v.push(Expect::parse(&p.string()?)?);
                        if p.try_eat(']') {
                            break;
                        }
                        p.eat(',')?;
                    }
                }
                expect = Some(v);
            }
            k => return Err(format!("unexpected or repeated key {k:?}")),
        }
        if p.try_eat('}') {
            break;
        }
        p.eat(',')?;
    }
    p.end()?;
    let query = query.ok_or("missing \"query\"")?;
    let expect = expect.ok_or("missing \"expect\"")?;
    if query.trim().is_empty() || expect.is_empty() {
        return Err("\"query\" and \"expect\" must not be empty".into());
    }
    Ok(Case { query, expect })
}

struct Json<'a>(std::iter::Peekable<std::str::Chars<'a>>);

impl Json<'_> {
    fn skip_ws(&mut self) {
        while self.0.next_if(|c| c.is_whitespace()).is_some() {}
    }

    fn try_eat(&mut self, c: char) -> bool {
        self.skip_ws();
        self.0.next_if_eq(&c).is_some()
    }

    fn eat(&mut self, c: char) -> std::result::Result<(), String> {
        if self.try_eat(c) {
            Ok(())
        } else {
            Err(format!("expected '{c}'"))
        }
    }

    fn end(&mut self) -> std::result::Result<(), String> {
        self.skip_ws();
        match self.0.next() {
            None => Ok(()),
            Some(_) => Err("trailing characters".into()),
        }
    }

    fn string(&mut self) -> std::result::Result<String, String> {
        self.eat('"')?;
        let mut s = String::new();
        loop {
            match self.0.next().ok_or("unterminated string")? {
                '"' => return Ok(s),
                '\\' => s.push(match self.0.next().ok_or("unterminated string")? {
                    '"' => '"',
                    '\\' => '\\',
                    '/' => '/',
                    'n' => '\n',
                    't' => '\t',
                    'r' => '\r',
                    'u' => {
                        let hex: String = (0..4).filter_map(|_| self.0.next()).collect();
                        u32::from_str_radix(&hex, 16)
                            .ok()
                            .and_then(char::from_u32)
                            .ok_or("bad \\u escape (surrogate pairs are not supported)")?
                    }
                    c => return Err(format!("bad escape \\{c}")),
                }),
                c if c.is_control() => return Err("control character in string".into()),
                c => s.push(c),
            }
        }
    }
}

/// 1-based rank of the first hit that meets any expectation.
fn found_at(hits: &[SearchHit], expect: &[Expect]) -> Option<usize> {
    hits.iter()
        .position(|h| expect.iter().any(|e| e.met_by(h)))
        .map(|r| r + 1)
}

/// One mode's results over all cases.
#[derive(Debug, Clone, PartialEq)]
pub struct Score {
    pub mode: SearchMode,
    pub cases: usize,
    pub hit1: f64,
    pub hit3: f64,
    pub hitk: f64,
    pub mrr: f64,
    /// Average estimated tokens of the `search_code` response, and of the expected whole files.
    pub tokens: f64,
    pub file_tokens: f64,
    /// Cases not found in the top k: the query and the `path:start-end` of every hit.
    pub misses: Vec<(String, Vec<String>)>,
}

/// Metrics from the found-at rank of each case (None: not in the top k).
fn score(mode: SearchMode, ranks: &[Option<usize>]) -> Score {
    let n = ranks.len().max(1) as f64;
    let within = |r: usize| ranks.iter().filter(|x| x.is_some_and(|x| x <= r)).count() as f64 / n;
    Score {
        mode,
        cases: ranks.len(),
        hit1: within(1),
        hit3: within(3),
        hitk: within(usize::MAX),
        mrr: ranks.iter().flatten().map(|&r| 1.0 / r as f64).sum::<f64>() / n,
        tokens: 0.0,
        file_tokens: 0.0,
        misses: Vec::new(),
    }
}

/// Runs every case in `mode` with the top `k` hits. Token cost is measured on exactly what
/// `search_code` would send (default k, over-fetched and budgeted the same way).
pub fn run_eval(
    root: &Path,
    db: &Db,
    embedder: Option<&dyn Embedder>,
    cases: &[Case],
    k: usize,
    mode: SearchMode,
) -> Result<Score> {
    let (mut ranks, mut misses) = (Vec::new(), Vec::new());
    let (mut sent, mut whole) = (0usize, 0usize);
    for c in cases {
        let hits = search_with(root, db, embedder, &c.query, k, mode)?;
        let rank = found_at(&hits, &c.expect);
        if rank.is_none() {
            let got = hits
                .iter()
                .map(|h| format!("{}:{}-{}", clean(&h.path), h.start_line, h.end_line))
                .collect();
            misses.push((clean(&c.query), got));
        }
        ranks.push(rank);
        let budgeted = search_with(root, db, embedder, &c.query, DEFAULT_K * MAX_PER_FILE, mode)?;
        let m = db.meta();
        let at = m.indexed_at_unix;
        sent += context(
            &budgeted,
            m.git_head.as_deref(),
            at,
            unix_now(),
            DEFAULT_K,
            None,
        )
        .tokens;
        let mut paths: Vec<&str> = c.expect.iter().map(Expect::path).collect();
        paths.sort_unstable();
        paths.dedup();
        for p in paths {
            let text = safe_join(root, p)
                .and_then(|f| std::fs::read_to_string(f).ok())
                .ok_or_else(|| {
                    Error::Invalid(format!("expected file {} is not readable", clean(p)))
                })?;
            whole += tokens(text.chars().count());
        }
    }
    let n = cases.len().max(1) as f64;
    Ok(Score {
        tokens: sent as f64 / n,
        file_tokens: whole as f64 / n,
        misses,
        ..score(mode, &ranks)
    })
}

/// The comparison table (a row per score, then a row per `skipped` mode), plus every miss when
/// `verbose`.
pub fn render_eval(scores: &[Score], skipped: &[SearchMode], k: usize, verbose: bool) -> String {
    let cases = scores.first().map_or(0, |s| s.cases);
    let mut out = format!(
        "{cases} cases, k = {k}\n{:<8} {:>6} {:>6} {:>7} {:>6} {:>8} {:>8} {:>9}\n",
        "mode",
        "hit@1",
        "hit@3",
        format!("hit@{k}"),
        "MRR",
        "tokens",
        "files",
        "reduction"
    );
    for s in scores {
        out.push_str(&format!(
            "{:<8} {:>6.2} {:>6.2} {:>7.2} {:>6.2} {:>8.0} {:>8.0} {:>8.1}x\n",
            s.mode.name(),
            s.hit1,
            s.hit3,
            s.hitk,
            s.mrr,
            s.tokens,
            s.file_tokens,
            s.file_tokens / s.tokens.max(1.0)
        ));
    }
    for m in skipped {
        out.push_str(&format!("{:<8} skipped (no embedding model)\n", m.name()));
    }
    out.push_str(
        "tokens: average search_code response; files: average size of the expected whole files\n",
    );
    if verbose {
        for s in scores {
            for (q, got) in &s.misses {
                out.push_str(&format!(
                    "miss [{}] {q:?}: {}\n",
                    s.mode.name(),
                    got.join(", ")
                ));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(path: &str, start: u32, end: u32) -> SearchHit {
        SearchHit {
            chunk_id: 0,
            score: 0.0,
            path: path.into(),
            start_line: start,
            end_line: end,
            text: String::new(),
            source_changed: false,
        }
    }

    #[test]
    fn cases_parse_and_errors_name_the_line() {
        let text =
            "{\"query\": \"where is \\\"lock\\\"\", \"expect\": [\"a.rs\", \"b.rs:3-9\"]}\n\n \
                    {\"expect\":[\"c:d.rs\"],\"query\":\"\\u00e9\\n\"}\n";
        let c = parse_cases(text).unwrap();
        assert_eq!(c[0].query, "where is \"lock\"");
        assert_eq!(
            c[0].expect,
            [
                Expect::File("a.rs".into()),
                Expect::Lines("b.rs".into(), 3, 9)
            ]
        );
        assert_eq!(c[1].query, "\u{e9}\n");
        assert_eq!(c[1].expect, [Expect::File("c:d.rs".into())]);
        for (bad, why) in [
            ("{\"query\": \"q\"}", "missing"),
            ("{\"query\": \"q\", \"expect\": []}", "empty"),
            ("{\"query\": \"q\", \"expect\": [\"a:5-2\"]}", "range"),
            (
                "{\"query\": \"q\", \"expect\": [\"a\"], \"k\": \"1\"}",
                "key",
            ),
            ("{\"query\": \"q\", \"expect\": [\"a\"]} x", "trailing"),
            ("{\"query\": \"q", "unterminated"),
            ("[1]", "expected"),
        ] {
            let e = parse_cases(&format!("\n{bad}")).unwrap_err().to_string();
            assert!(e.contains("line 2:") && e.contains(why), "{bad}: {e}");
        }
    }

    #[test]
    fn line_ranges_must_overlap() {
        let e = Expect::Lines("a.rs".into(), 10, 20);
        assert!(e.met_by(&hit("a.rs", 20, 30)) && e.met_by(&hit("a.rs", 1, 10)));
        assert!(e.met_by(&hit("a.rs", 12, 14)) && e.met_by(&hit("a.rs", 1, 99)));
        assert!(!e.met_by(&hit("a.rs", 21, 30)) && !e.met_by(&hit("a.rs", 1, 9)));
        assert!(!e.met_by(&hit("b.rs", 10, 20)));
        assert!(Expect::File("a.rs".into()).met_by(&hit("a.rs", 400, 401)));
    }

    #[test]
    fn metrics_match_a_hand_computed_fixture() {
        let hits = [hit("x.rs", 1, 5), hit("a.rs", 1, 5), hit("b.rs", 1, 5)];
        assert_eq!(found_at(&hits, &[Expect::File("a.rs".into())]), Some(2));
        assert_eq!(found_at(&hits, &[Expect::File("z.rs".into())]), None);
        // Ranks 1, 2, 4, miss: hit@1 1/4, hit@3 2/4, hit@k 3/4, MRR (1 + 1/2 + 1/4) / 4.
        let s = score(SearchMode::Hybrid, &[Some(1), Some(2), Some(4), None]);
        assert_eq!((s.hit1, s.hit3, s.hitk), (0.25, 0.5, 0.75));
        assert_eq!(s.mrr, 1.75 / 4.0);
        assert!(parse_cases("\n  \n")
            .unwrap_err()
            .to_string()
            .contains("no cases"));
        assert_eq!(
            (effective_k(0), effective_k(7), effective_k(500)),
            (1, 7, 100)
        );
        let empty = score(SearchMode::Vector, &[]);
        assert_eq!((empty.hitk, empty.mrr), (0.0, 0.0));
    }
}
