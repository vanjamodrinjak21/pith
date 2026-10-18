use crate::commands::clean;
use crate::SearchHit;
use std::collections::HashMap;

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
pub(crate) fn fit_budget(
    hits: Vec<SearchHit>,
    max_tokens: usize,
    max_per_file: usize,
    snippet_lines: usize,
) -> Vec<SearchHit> {
    let mut per_file: HashMap<String, usize> = HashMap::new();
    let mut used = 0;
    let mut out = Vec::new();
    for mut h in hits {
        let n = per_file.entry(h.path.clone()).or_default();
        if *n >= max_per_file {
            continue;
        }
        h.text = h
            .text
            .lines()
            .take(snippet_lines)
            .collect::<Vec<_>>()
            .join("\n");
        let cost = tokens(render_hit(&h).chars().count());
        if used + cost > max_tokens {
            break;
        }
        used += cost;
        *n += 1;
        out.push(h);
    }
    out
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
    fn rendered_response_with_markers_stays_under_cap() {
        let header = super::super::tools::staleness(Some(&"a".repeat(64)), 0, 10_000_000);
        let budget = 4000 - tokens(header.chars().count());
        let hits = fit_budget((0..20).map(big).collect(), budget, 3, 15);
        let out = super::super::tools::render_response(&header, &hits);
        assert!(out.contains("[source changed") && out.contains("more lines: get_chunk"));
        assert!(tokens(out.chars().count()) <= 4000, "{}", out.len());
    }
}
