//! Window chunker. Guarantees every chunk is at most `CHUNK_MAX_CHARS` chars and
//! `MAX_LINES` lines.
//!
//! Consecutive windows share up to `OVERLAP` (10) lines. A literal 10-line overlap is
//! impossible when lines are long, so the overlap shrinks, down to 0, when the
//! character limit crowds the window (200-char lines overlap by 5; lines of 700+
//! chars do not overlap). Coverage and forward progress always hold. A line longer
//! than the character limit is split into pieces at char boundaries, never mid-codepoint.

use super::SourceChunk;
use crate::CHUNK_MAX_CHARS;

const MAX_LINES: usize = 60;
const OVERLAP: usize = 10;

/// A line, or a slice of an over-long line. `cont` pieces continue the previous piece's line.
struct Piece<'a> {
    text: &'a str,
    line: u32,
    cont: bool,
}

fn pieces(text: &str) -> Vec<Piece<'_>> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line_no = u32::try_from(i + 1).unwrap_or(u32::MAX);
        let mut rest = line;
        let mut cont = false;
        loop {
            // Split at the byte offset of the (MAX+1)th char, never inside a codepoint.
            let cut = rest
                .char_indices()
                .nth(CHUNK_MAX_CHARS)
                .map_or(rest.len(), |(b, _)| b);
            out.push(Piece {
                text: &rest[..cut],
                line: line_no,
                cont,
            });
            rest = &rest[cut..];
            if rest.is_empty() {
                break;
            }
            cont = true;
        }
    }
    out
}

/// Exclusive end of the window that starts at piece `s`; always > `s`.
fn window_end(p: &[Piece], s: usize) -> usize {
    let mut chars = 0;
    let mut e = s;
    while e < p.len() && e - s < MAX_LINES {
        let sep = usize::from(e > s && !p[e].cont);
        let add = sep + p[e].text.chars().count();
        if chars + add > CHUNK_MAX_CHARS {
            break;
        }
        chars += add;
        e += 1;
    }
    e
}

pub(super) fn chunk_text(path: &str, text: &str) -> Vec<SourceChunk> {
    let p = pieces(text);
    let mut out = Vec::new();
    let mut s = 0;
    while s < p.len() {
        let e = window_end(&p, s);
        let mut body = String::new();
        for (i, piece) in p[s..e].iter().enumerate() {
            if i > 0 && !piece.cont {
                body.push('\n');
            }
            body.push_str(piece.text);
        }
        out.push(SourceChunk {
            path: path.to_owned(),
            start_line: p[s].line,
            end_line: p[e - 1].line,
            text: body,
        });
        if e == p.len() {
            break;
        }
        // Back up to share up to OVERLAP pieces, but only as far as still makes progress.
        s = (0..=OVERLAP.min(e - s - 1))
            .rev()
            .map(|k| e - k)
            .find(|&ns| window_end(&p, ns) > e)
            .unwrap_or(e);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn line_strategy() -> impl Strategy<Value = String> {
        prop_oneof![
            4 => "[a-z é€𝄞\t]{0,40}",
            1 => "[a-zé€𝄞]{1300,4200}",
            1 => Just(String::new()),
        ]
    }

    #[test]
    fn overlap_shrinks_when_chars_crowd_the_window() {
        let text = |len: usize| (0..30).map(|_| "x".repeat(len) + "\n").collect::<String>();
        let ranges = |len| -> Vec<(u32, u32)> {
            chunk_text("f", &text(len))
                .iter()
                .map(|c| (c.start_line, c.end_line))
                .collect()
        };
        // 6 lines of 200 chars fit (1205 chars); 7 do not. Overlap 5.
        assert_eq!(&ranges(200)[..2], [(1, 6), (2, 7)]);
        // 699-char lines: 2 fit (1399). Overlap 1.
        assert_eq!(&ranges(699)[..2], [(1, 2), (2, 3)]);
        // 700-char lines: only 1 fits (2 need 1401). Overlap 0.
        assert_eq!(&ranges(700)[..2], [(1, 1), (2, 2)]);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]
        #[test]
        fn chunks_obey_limits_and_cover_every_line(
            lines in prop::collection::vec(line_strategy(), 0..200),
            trailing_newline in any::<bool>(),
        ) {
            let mut text = lines.join("\n");
            if trailing_newline { text.push('\n'); }
            let total = text.lines().count() as u32;
            let chunks = chunk_text("f", &text);

            let mut covered = vec![false; total as usize + 1];
            let mut prev: Option<&SourceChunk> = None;
            for c in &chunks {
                prop_assert!(c.text.chars().count() <= CHUNK_MAX_CHARS);
                prop_assert!(c.end_line - c.start_line < MAX_LINES as u32);
                prop_assert!(c.start_line >= 1 && c.start_line <= c.end_line && c.end_line <= total);
                prop_assert!(c.text.lines().count() <= MAX_LINES);
                match prev {
                    None => prop_assert_eq!(c.start_line, 1),
                    Some(p) => {
                        prop_assert!(c.start_line <= p.end_line + 1, "gap");
                        prop_assert!(c.start_line >= p.start_line && c.end_line >= p.end_line);
                        prop_assert!(c != p);
                    }
                }
                for l in c.start_line..=c.end_line { covered[l as usize] = true; }
                prev = Some(c);
            }
            if total > 0 {
                prop_assert_eq!(prev.unwrap().end_line, total);
            } else {
                prop_assert!(chunks.is_empty());
            }
            for (i, l) in text.lines().enumerate() {
                if !l.is_empty() { prop_assert!(covered[i + 1], "line {} uncovered", i + 1); }
            }
        }

        /// Without over-long lines the overlap is exactly 10 lines.
        #[test]
        fn short_lines_overlap_exactly_ten(n in 61usize..400) {
            let text: String = (0..n).map(|i| format!("l{i}\n")).collect();
            let chunks = chunk_text("f", &text);
            for w in chunks.windows(2) {
                if w[0].end_line - w[0].start_line == 59 {
                    prop_assert_eq!(w[1].start_line, w[0].end_line - 9);
                }
            }
        }
    }
}
