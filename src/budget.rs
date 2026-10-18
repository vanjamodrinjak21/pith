//! What `search_code` sends to Claude: the token budget, per-file cap and snippet trim. Shared
//! by `pith mcp` and the RAG view of `pith ui`, so both show exactly the same context.

use crate::commands::{age, clean, valid_head};
use crate::SearchHit;
use std::collections::HashMap;

pub(crate) const MAX_TOKENS: usize = 4000;
pub(crate) const MAX_PER_FILE: usize = 3;
pub(crate) const SNIPPET_LINES: usize = 15;
pub(crate) const DEFAULT_K: usize = 8;

/// What the budget did with one ranked hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Fate {
    /// Sent; `trimmed_from` is the original line count when the snippet was cut.
    Kept { trimmed_from: Option<usize> },
    /// The file already had `max_per_file` hits.
    FileCap,
    /// It, or an earlier hit, would have pushed the response over the token cap.
    TokenCap,
    /// Fit, but more than `k` hits did.
    BeyondK,
}

/// One `search_code` response: its text, estimated tokens, and the fate of every ranked hit.
#[derive(Debug, Clone)]
#[cfg_attr(not(feature = "tui"), allow(dead_code))] // `pith mcp` sends only the text
pub(crate) struct Context {
    pub text: String,
    pub tokens: usize,
    pub fates: Vec<Fate>,
}

/// Builds the `search_code` response for `hits` (ranked, over-fetched), keeping at most `k`.
pub(crate) fn context(
    hits: &[SearchHit],
    git_head: Option<&str>,
    indexed_at: u64,
    now: u64,
    k: usize,
) -> Context {
    let header = staleness(git_head, indexed_at, now);
    let budget = MAX_TOKENS.saturating_sub(tokens(header.chars().count()));
    let (mut kept, mut fates) =
        fit_budget_explained(hits.to_vec(), budget, MAX_PER_FILE, SNIPPET_LINES);
    kept.truncate(k);
    fates
        .iter_mut()
        .filter(|f| matches!(f, Fate::Kept { .. }))
        .skip(k)
        .for_each(|f| *f = Fate::BeyondK);
    let text = render_response(&header, &kept);
    Context {
        tokens: tokens(text.chars().count()),
        text,
        fates,
    }
}

/// First line of every search_code response: index age, plus the git head only when valid.
pub(crate) fn staleness(git_head: Option<&str>, indexed_at: u64, now: u64) -> String {
    let mut out = format!("indexed {}", age(now.saturating_sub(indexed_at)));
    if let Some(h) = git_head.filter(|h| valid_head(h)) {
        out.push_str(&format!(" at git {h}"));
    }
    out.push('\n');
    out
}

pub(crate) fn render_response(header: &str, hits: &[SearchHit]) -> String {
    let mut out = header.to_string();
    out.extend(hits.iter().map(render_hit));
    out
}

/// Rough token estimate: chars / 4, rounded up.
pub(crate) fn tokens(chars: usize) -> usize {
    chars.div_ceil(4)
}

/// `clean` per line, keeping the newlines.
pub(crate) fn clean_block(text: &str) -> String {
    text.lines().map(clean).collect::<Vec<_>>().join("\n")
}

/// One hit exactly as `search_code` shows it, markers included.
pub(crate) fn render_hit(h: &SearchHit) -> String {
    let shown = h.text.lines().count() as u32;
    let full = h.end_line.saturating_sub(h.start_line) + 1;
    let mut out = format!(
        "\n{}:{}-{}  score {:.2}  chunk_id {}{}\n{}\n",
        clean(&h.path),
        h.start_line,
        h.end_line,
        h.score,
        h.chunk_id,
        if h.source_changed {
            "  [source changed since indexing]"
        } else {
            ""
        },
        clean_block(&h.text)
    );
    if shown < full {
        out.push_str(&format!(
            "[{} more lines: get_chunk {}]\n",
            full - shown,
            h.chunk_id
        ));
    }
    out
}

/// Keeps hits in rank order, at most `max_per_file` per path, each text cut to its first
/// `snippet_lines` lines, and stops at the first hit whose rendered form (see [`render_hit`])
/// would push the estimated total (chars / 4) over `max_tokens`.
#[cfg(test)]
pub(crate) fn fit_budget(
    hits: Vec<SearchHit>,
    max_tokens: usize,
    max_per_file: usize,
    snippet_lines: usize,
) -> Vec<SearchHit> {
    fit_budget_explained(hits, max_tokens, max_per_file, snippet_lines).0
}

/// [`fit_budget`], plus the fate of every input hit (same order).
pub(crate) fn fit_budget_explained(
    hits: Vec<SearchHit>,
    max_tokens: usize,
    max_per_file: usize,
    snippet_lines: usize,
) -> (Vec<SearchHit>, Vec<Fate>) {
    let mut per_file: HashMap<String, usize> = HashMap::new();
    let mut used = 0;
    let mut out = Vec::new();
    let mut fates = Vec::with_capacity(hits.len());
    let mut full = false;
    for mut h in hits {
        if full {
            fates.push(Fate::TokenCap);
            continue;
        }
        let n = per_file.entry(h.path.clone()).or_default();
        if *n >= max_per_file {
            fates.push(Fate::FileCap);
            continue;
        }
        let lines = h.text.lines().count();
        h.text = h
            .text
            .lines()
            .take(snippet_lines)
            .collect::<Vec<_>>()
            .join("\n");
        let cost = tokens(render_hit(&h).chars().count());
        if used + cost > max_tokens {
            full = true;
            fates.push(Fate::TokenCap);
            continue;
        }
        used += cost;
        *n += 1;
        out.push(h);
        fates.push(Fate::Kept {
            trimmed_from: (lines > snippet_lines).then_some(lines),
        });
    }
    (out, fates)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn big(id: u32) -> SearchHit {
        SearchHit {
            chunk_id: id,
            score: 0.5,
            path: format!("{}/f{id}.rs", "very_long_directory_name/".repeat(8)),
            start_line: 1,
            end_line: 400,
            text: (0..40)
                .map(|_| "y".repeat(90))
                .collect::<Vec<_>>()
                .join("\n"),
            source_changed: true,
        }
    }

    fn hit(id: u32, path: &str, lines: usize) -> SearchHit {
        SearchHit {
            chunk_id: id,
            score: 1.0,
            path: path.into(),
            start_line: 1,
            end_line: lines as u32,
            text: (1..=lines)
                .map(|i| format!("line {i}"))
                .collect::<Vec<_>>()
                .join("\n"),
            source_changed: false,
        }
    }

    #[test]
    fn fit_budget_caps_three_hits_per_file() {
        let hits = (0..5)
            .map(|i| hit(i, "a.rs", 3))
            .chain([hit(9, "b.rs", 3)])
            .collect();
        let ids: Vec<u32> = fit_budget(hits, 4000, 3, 15)
            .iter()
            .map(|h| h.chunk_id)
            .collect();
        assert_eq!(ids, [0, 1, 2, 9]);
    }

    #[test]
    fn fit_budget_trims_snippets_to_15_lines() {
        let out = fit_budget(vec![hit(0, "a.rs", 40)], 4000, 3, 15);
        assert_eq!(out[0].text.lines().count(), 15);
        assert!(out[0].text.ends_with("line 15"));
    }

    #[test]
    fn fit_budget_stops_before_exceeding_token_cap() {
        // One trimmed hit costs about 410 tokens, so exactly one fits in 600.
        let out = fit_budget((0..10).map(big).collect(), 600, 3, 15);
        assert_eq!(out.len(), 1);
        let used: usize = out
            .iter()
            .map(|h| tokens(render_hit(h).chars().count()))
            .sum();
        assert!(used <= 600, "{used}");
        // Order is kept: it stops, it does not skip ahead.
        assert_eq!(out.last().unwrap().chunk_id as usize, out.len() - 1);
    }

    #[test]
    fn fates_explain_every_dropped_or_trimmed_hit() {
        let hits = vec![
            hit(0, "a.rs", 40),
            hit(1, "a.rs", 3),
            hit(2, "a.rs", 3),
            hit(3, "a.rs", 3),
            hit(4, "b.rs", 3),
            big(5),
            hit(6, "c.rs", 3),
        ];
        let (kept, fates) = fit_budget_explained(hits.clone(), 200, 3, 15);
        assert_eq!(
            fates,
            [
                Fate::Kept {
                    trimmed_from: Some(40)
                },
                Fate::Kept { trimmed_from: None },
                Fate::Kept { trimmed_from: None },
                Fate::FileCap,
                Fate::Kept { trimmed_from: None },
                Fate::TokenCap,
                Fate::TokenCap,
            ]
        );
        assert_eq!(kept.len(), 4);
        // `context` adds the k cut and renders what Claude receives.
        let c = context(&hits, None, 0, 0, 2);
        assert_eq!(c.fates[2], Fate::BeyondK);
        assert_eq!(c.fates[4], Fate::BeyondK);
        assert_eq!(c.tokens, tokens(c.text.chars().count()));
        assert!(c.text.starts_with("indexed 0 seconds ago\n") && c.tokens <= MAX_TOKENS);
        assert_eq!(c.text.matches("score 1.00").count(), 2);
    }

    #[test]
    fn rendered_response_with_markers_stays_under_cap() {
        let header = staleness(Some(&"a".repeat(64)), 0, 10_000_000);
        let budget = 4000 - tokens(header.chars().count());
        let hits = fit_budget((0..20).map(big).collect(), budget, 3, 15);
        let out = render_response(&header, &hits);
        assert!(out.contains("[source changed") && out.contains("more lines: get_chunk"));
        assert!(tokens(out.chars().count()) <= 4000, "{}", out.len());
    }
}
