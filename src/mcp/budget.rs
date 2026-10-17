use crate::SearchHit;
use std::collections::HashMap;

/// Fixed per-hit cost of the rendered header line (location, score, id), in chars.
const HEADER_CHARS: usize = 48;

/// Rough token estimate: chars / 4.
pub(crate) fn tokens(chars: usize) -> usize {
    chars.div_ceil(4)
}

/// Keeps hits in rank order, at most `max_per_file` per path, each text cut to its first
/// `snippet_lines` lines, and stops at the first hit that would push the total estimated
/// tokens (text + path + header, chars / 4) over `max_tokens`.
pub fn fit_budget(
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
        let cost = tokens(h.text.chars().count() + h.path.chars().count() + HEADER_CHARS);
        if used + cost > max_tokens {
            break;
        }
        used += cost;
        *n += 1;
        out.push(h);
    }
    out
}
