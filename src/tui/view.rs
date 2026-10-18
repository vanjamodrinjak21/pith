//! Drawing. Every string that comes from the database passes through `clean`.

use super::model::{bar, Tree, COLOURS};
use super::state::{Indexing, State, View};
use super::Data;
use crate::budget::{Fate, DEFAULT_K, MAX_PER_FILE, MAX_TOKENS, SNIPPET_LINES};
use crate::commands::{age, clean, unix_now, valid_head};
use crate::storage::StoredChunk;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::canvas::{Canvas, Points};
use ratatui::widgets::{Block, Clear, List, ListItem, ListState, Paragraph, Tabs};
use ratatui::Frame;

/// Preview limits: lines per chunk and chars per line.
const PREVIEW_LINES: usize = 30;
const PREVIEW_COLS: usize = 200;

/// Map colours for the largest top-level entries; the rest are grey.
const PALETTE: [Color; COLOURS] = [
    Color::Cyan,
    Color::Green,
    Color::Yellow,
    Color::Magenta,
    Color::Blue,
    Color::Red,
    Color::LightCyan,
    Color::LightGreen,
];
const OTHER: Color = Color::DarkGray;
const SELECTED: Style = Style::new().add_modifier(Modifier::REVERSED);

fn group_color(g: Option<usize>) -> Color {
    g.and_then(|g| PALETTE.get(g).copied()).unwrap_or(OTHER)
}

/// Cuts `s` to `max` chars, marking a cut with an ellipsis.
fn fit(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
    if max > 0 {
        out.push('…');
    }
    out
}

pub(crate) fn draw(f: &mut Frame, s: &State, d: &Data) {
    let [tabs, body, status] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(0),
        Constraint::Length(1),
    ])
    .areas(f.area());
    let selected = match s.view {
        View::Stack => 0,
        View::Map => 1,
        View::Rag => 2,
        View::Index => 3,
    };
    f.render_widget(
        Tabs::new(["1 Stack", "2 Map", "3 RAG", "4 Index"])
            .select(selected)
            .highlight_style(SELECTED),
        tabs,
    );
    match s.view {
        View::Stack => stack(f, body, s, d),
        View::Map => map(f, body, s, d),
        View::Rag => rag(f, body, s, d),
        View::Index => index(f, body, s),
    }
    f.render_widget(Paragraph::new(status_line(s, d)), status);
    if s.help {
        help(f, body);
    }
}

fn status_line(s: &State, d: &Data) -> Line<'static> {
    let Some(db) = &d.db else {
        return Line::from(vec![
            Span::styled(
                " no index yet · i to index · ? help",
                Style::new().fg(Color::DarkGray),
            ),
            Span::styled(
                format!("  {}", clean(&s.status)),
                Style::new().fg(Color::Yellow),
            ),
        ]);
    };
    let meta = db.meta();
    let mut text = format!(
        " {} chunks · {} · indexed {}",
        d.tree.total(),
        clean(db.model_id()),
        age(unix_now().saturating_sub(meta.indexed_at_unix))
    );
    if let Some(h) = meta.git_head.as_deref().filter(|h| valid_head(h)) {
        text.push_str(&format!(" · git {}", &h[..12]));
    }
    text.push_str(" · ? help");
    let mut spans = vec![Span::styled(text, Style::new().fg(Color::DarkGray))];
    if !s.status.is_empty() {
        spans.push(Span::styled(
            format!("  {}", clean(&s.status)),
            Style::new().fg(Color::Yellow),
        ));
    }
    Line::from(spans)
}

/// Header plus the first lines of a chunk, capped and scrubbed.
fn preview(c: &StoredChunk) -> Vec<Line<'static>> {
    let mut out = vec![Line::styled(
        format!("{}:{}-{}", clean(&c.path), c.start_line, c.end_line),
        Style::new().add_modifier(Modifier::BOLD),
    )];
    out.extend(
        c.text
            .lines()
            .take(PREVIEW_LINES)
            .map(|l| Line::raw(clean(&l.chars().take(PREVIEW_COLS).collect::<String>()))),
    );
    out
}

fn preview_pane(f: &mut Frame, area: Rect, chunk: Option<&StoredChunk>, empty: &str) {
    let lines = chunk.map_or_else(|| vec![Line::raw(empty.to_string())], preview);
    f.render_widget(
        Paragraph::new(lines).block(Block::bordered().title(" Preview ")),
        area,
    );
}

fn message(f: &mut Frame, area: Rect, text: &str) {
    f.render_widget(
        Paragraph::new(text.to_string()).block(Block::bordered()),
        area,
    );
}

fn stack(f: &mut Frame, area: Rect, s: &State, d: &Data) {
    let t = &d.tree;
    if t.total() == 0 {
        return message(
            f,
            area,
            "The index is empty: run `pith index` after adding files.",
        );
    }
    let [left, right] =
        Layout::horizontal([Constraint::Percentage(55), Constraint::Percentage(45)]).areas(area);
    let w = left.width.saturating_sub(2) as usize;
    let bar_w = w / 3;
    let label_w = w.saturating_sub(bar_w + 8);
    let items: Vec<ListItem> = t
        .rows(&s.expanded)
        .into_iter()
        .map(|(i, depth)| {
            let n = &t.nodes[i];
            let mark = match (t.is_dir(i), s.expanded[i]) {
                (false, _) => "  ",
                (true, false) => "▸ ",
                (true, true) => "▾ ",
            };
            let label = fit(
                &format!("{}{mark}{}", "  ".repeat(depth), clean(&n.name)),
                label_w,
            );
            ListItem::new(Line::from(vec![
                Span::raw(format!("{label:<label_w$}")),
                Span::styled(
                    format!(" {:>6} ", n.count),
                    Style::new().fg(Color::DarkGray),
                ),
                Span::styled(bar(n.count, t.total(), bar_w), Style::new().fg(Color::Cyan)),
            ]))
        })
        .collect();
    let tree_style = if s.in_chunks {
        Style::new().add_modifier(Modifier::DIM)
    } else {
        SELECTED
    };
    f.render_stateful_widget(
        List::new(items)
            .block(Block::bordered().title(" Stack "))
            .highlight_style(tree_style),
        left,
        &mut ListState::default().with_selected(Some(s.row)),
    );
    let chunks = s.file_chunks(t);
    let list_h = (chunks.len() + 2).min(right.height as usize / 3).max(3) as u16;
    let [list, prev] =
        Layout::vertical([Constraint::Length(list_h), Constraint::Min(0)]).areas(right);
    let items: Vec<ListItem> = chunks
        .iter()
        .filter_map(|&id| d.chunk(id))
        .map(|c| ListItem::new(format!("lines {}-{}", c.start_line, c.end_line)))
        .collect();
    let chunk_style = if s.in_chunks { SELECTED } else { Style::new() };
    f.render_stateful_widget(
        List::new(items)
            .block(Block::bordered().title(" Chunks "))
            .highlight_style(chunk_style),
        list,
        &mut ListState::default().with_selected((!chunks.is_empty()).then_some(s.chunk)),
    );
    let chunk = chunks.get(s.chunk).and_then(|&id| d.chunk(id));
    preview_pane(f, prev, chunk, "Select a file to see its chunks.");
}

fn map(f: &mut Frame, area: Rect, s: &State, d: &Data) {
    let m = &d.map;
    if d.pending.is_some() {
        return message(f, area, "projecting...");
    }
    if d.map_failed {
        return message(f, area, "map failed: the vectors could not be projected.");
    }
    if m.points.is_empty() {
        return message(f, area, "Nothing to map: the index has no vectors.");
    }
    let [left, right] =
        Layout::horizontal([Constraint::Percentage(60), Constraint::Percentage(40)]).areas(area);
    let [canvas, legend] =
        Layout::vertical([Constraint::Min(0), Constraint::Length(1)]).areas(left);
    let (xb, yb) = m.bounds();
    // Chunks retrieved in the RAG view: the ones sent to Claude, and the rest of the candidates.
    let (mut sent, mut dropped) = (Vec::new(), Vec::new());
    for (i, p) in s.hit_points.iter().enumerate() {
        let Some(p) = p else { continue };
        let kept = s
            .context
            .as_ref()
            .is_some_and(|c| matches!(c.fates.get(i), Some(Fate::Kept { .. })));
        if kept { &mut sent } else { &mut dropped }.push((p[0] as f64, p[1] as f64));
    }
    f.render_widget(
        Canvas::default()
            .block(Block::bordered().title(" Map (PCA) "))
            .marker(Marker::Braille)
            .x_bounds(xb)
            .y_bounds(yb)
            .paint(|ctx| {
                for (g, coords) in d.groups.iter().enumerate() {
                    let color = group_color(Some(g));
                    ctx.draw(&Points { coords, color });
                }
                ctx.layer();
                ctx.draw(&Points {
                    coords: &dropped,
                    color: Color::Gray,
                });
                for &(x, y) in &sent {
                    ctx.print(
                        x,
                        y,
                        Span::styled(
                            "•",
                            Style::new().fg(Color::White).add_modifier(Modifier::BOLD),
                        ),
                    );
                }
                if let Some(q) = s.query_point {
                    ctx.print(
                        q[0] as f64,
                        q[1] as f64,
                        Span::styled("◆", Style::new().fg(Color::LightRed)),
                    );
                }
                ctx.print(
                    s.cursor[0] as f64,
                    s.cursor[1] as f64,
                    Span::styled(
                        "+",
                        Style::new().fg(Color::White).add_modifier(Modifier::BOLD),
                    ),
                );
            }),
        canvas,
    );
    shade_density(f, canvas, &m.points, (xb, yb));
    f.render_widget(Paragraph::new(legend_line(&d.tree)), legend);
    let chunk = s.near.and_then(|i| d.chunk(m.id(i)));
    preview_pane(f, right, chunk, "");
}

/// Background shade per cell by how many points fall in it (log scale, 4 levels), so dense
/// regions read as clusters even where the dots saturate.
fn shade_density(
    f: &mut Frame,
    area: Rect,
    points: &[[f32; 2]],
    ([x0, x1], [y0, y1]): ([f64; 2], [f64; 2]),
) {
    let inner = Block::bordered().inner(area);
    let (w, h) = (inner.width as usize, inner.height as usize);
    if w == 0 || h == 0 {
        return;
    }
    let mut counts = vec![0u32; w * h];
    for p in points {
        let cx = ((p[0] as f64 - x0) / (x1 - x0) * w as f64) as usize;
        let cy = ((y1 - p[1] as f64) / (y1 - y0) * h as f64) as usize;
        counts[cy.min(h - 1) * w + cx.min(w - 1)] += 1;
    }
    let max = (*counts.iter().max().unwrap_or(&0) as f64).ln_1p();
    if max == 0.0 {
        return;
    }
    let buf = f.buffer_mut();
    for (i, &c) in counts.iter().enumerate() {
        if c > 0 {
            let level = ((c as f64).ln_1p() / max * 4.0).ceil() as u8;
            let pos = (inner.x + (i % w) as u16, inner.y + (i / w) as u16);
            buf[pos].set_bg(Color::Indexed(232 + 2 * level));
        }
    }
}

fn legend_line(t: &Tree) -> Line<'static> {
    let top = &t.nodes[0].children;
    let mut spans: Vec<Span> = top
        .iter()
        .take(PALETTE.len())
        .enumerate()
        .map(|(g, &i)| {
            Span::styled(
                format!("■ {} ", clean(&t.nodes[i].name)),
                Style::new().fg(group_color(Some(g))),
            )
        })
        .collect();
    if top.len() > PALETTE.len() {
        spans.push(Span::styled("■ other", Style::new().fg(OTHER)));
    }
    Line::from(spans)
}

fn fate_label(f: Fate) -> String {
    match f {
        Fate::Kept { trimmed_from: None } => "sent".into(),
        Fate::Kept {
            trimmed_from: Some(n),
        } => format!("sent, trimmed {n}->{SNIPPET_LINES} lines"),
        Fate::FileCap => format!("dropped: {MAX_PER_FILE} per file"),
        Fate::TokenCap => format!("dropped: {MAX_TOKENS}-token cap"),
        Fate::BeyondK => format!("dropped: over k={DEFAULT_K}"),
    }
}

fn rag(f: &mut Frame, area: Rect, s: &State, d: &Data) {
    let [query, rest] = Layout::vertical([Constraint::Length(3), Constraint::Min(0)]).areas(area);
    let caret = if s.editing { "▏" } else { "" };
    let title = if s.editing {
        " Query (Enter search, Esc done) "
    } else {
        " Query (/ to edit, Tab ranked/context) "
    };
    f.render_widget(
        Paragraph::new(format!("{}{caret}", s.query)).block(Block::bordered().title(title)),
        query,
    );
    let Some(c) = s.context.as_ref().filter(|_| !s.hits.is_empty()) else {
        return message(
            f,
            rest,
            "Press / and type a query; Enter shows what search_code would send Claude.",
        );
    };
    if s.show_context {
        let lines: Vec<Line> = c.text.lines().map(|l| Line::raw(clean(l))).collect();
        let title = format!(
            " Context as Claude sees it: {} / {MAX_TOKENS} tokens ",
            c.tokens
        );
        return f.render_widget(
            Paragraph::new(lines)
                .scroll((s.scroll, 0))
                .block(Block::bordered().title(title)),
            rest,
        );
    }
    let [list, side] =
        Layout::horizontal([Constraint::Percentage(55), Constraint::Percentage(45)]).areas(rest);
    let items: Vec<ListItem> = s
        .hits
        .iter()
        .zip(&c.fates)
        .map(|(h, &fate)| {
            let style = match fate {
                Fate::Kept { .. } => Style::new(),
                _ => Style::new().fg(Color::DarkGray),
            };
            ListItem::new(Line::from(vec![
                Span::styled(
                    format!("{:<6}", bar((h.score.max(0.0) * 1000.0) as usize, 1000, 6)),
                    Style::new().fg(Color::Cyan),
                ),
                Span::styled(
                    format!(
                        " {:.2}  {}:{}-{}  {}{}",
                        h.score,
                        clean(&h.path),
                        h.start_line,
                        h.end_line,
                        fate_label(fate),
                        if h.source_changed {
                            "  [source changed]"
                        } else {
                            ""
                        }
                    ),
                    style,
                ),
            ]))
        })
        .collect();
    f.render_stateful_widget(
        List::new(items)
            .block(Block::bordered().title(" Ranked candidates "))
            .highlight_style(SELECTED),
        list,
        &mut ListState::default().with_selected(Some(s.hit)),
    );
    // Token meter and per-file counts against the caps.
    let mut files: Vec<(&str, usize)> = Vec::new();
    for (h, fate) in s.hits.iter().zip(&c.fates) {
        let n = usize::from(matches!(fate, Fate::Kept { .. }));
        match files.iter_mut().find(|(p, _)| *p == h.path) {
            Some((_, k)) => *k += n,
            None => files.push((&h.path, n)),
        }
    }
    let mut lines = vec![
        Line::from(vec![
            Span::raw(format!("tokens {:>4} / {MAX_TOKENS} ", c.tokens)),
            Span::styled(
                bar(c.tokens, MAX_TOKENS, 16),
                Style::new().fg(Color::Yellow),
            ),
        ]),
        Line::raw(format!(
            "sent {} of {} candidates",
            c.fates
                .iter()
                .filter(|f| matches!(f, Fate::Kept { .. }))
                .count(),
            c.fates.len()
        )),
        Line::raw(""),
    ];
    lines.extend(
        files
            .iter()
            .map(|(p, n)| Line::raw(format!("{n}/{MAX_PER_FILE}  {}", clean(p)))),
    );
    let meter_h = (lines.len() + 2).min(side.height as usize / 2) as u16;
    let [meter, prev] =
        Layout::vertical([Constraint::Length(meter_h), Constraint::Min(0)]).areas(side);
    f.render_widget(
        Paragraph::new(lines).block(Block::bordered().title(" Budget ")),
        meter,
    );
    let chunk = s.hits.get(s.hit).and_then(|h| d.chunk(h.chunk_id));
    preview_pane(f, prev, chunk, "");
}

fn index(f: &mut Frame, area: Rect, s: &State) {
    let Some(ix) = &s.indexing else {
        return message(
            f,
            area,
            "Press i to index this project now. Esc or c cancels; the previous index stays until the new one is written.",
        );
    };
    let [left, right] =
        Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).areas(area);
    f.render_widget(
        Paragraph::new(index_lines(ix, left.width.saturating_sub(2) as usize))
            .block(Block::bordered().title(" Indexing ")),
        left,
    );
    let (xb, yb) = ix.bounds();
    f.render_widget(
        Canvas::default()
            .block(
                Block::bordered().title(format!(" Embedded chunks ({} sampled) ", ix.points.len())),
            )
            .marker(Marker::Braille)
            .x_bounds(xb)
            .y_bounds(yb)
            .paint(|ctx| {
                ctx.draw(&Points {
                    coords: &ix.points,
                    color: Color::Cyan,
                })
            }),
        right,
    );
}

fn index_lines(ix: &Indexing, width: usize) -> Vec<Line<'static>> {
    let secs = ix.secs();
    let rate = if secs > 0.0 {
        ix.chunks_done as f64 / secs
    } else {
        0.0
    };
    let eta = if rate > 0.0 && ix.running {
        format!(
            "{:.0}s",
            ix.chunks.saturating_sub(ix.chunks_done) as f64 / rate
        )
    } else {
        "-".into()
    };
    let state = match (ix.running, ix.chunks_done == ix.chunks) {
        (true, _) => "running",
        (false, true) => "finished",
        (false, false) => "stopped",
    };
    let mut out = vec![
        Line::raw(format!("{state}, {secs:.1}s")),
        Line::from(vec![
            Span::raw(format!("chunks {:>6} / {:<6} ", ix.chunks_done, ix.chunks)),
            Span::styled(
                bar(ix.chunks_done, ix.chunks, width.saturating_sub(23)),
                Style::new().fg(Color::Cyan),
            ),
        ]),
        Line::raw(format!("files  {:>6} / {}", ix.files_done, ix.files)),
        Line::raw(format!("rate   {rate:.1} chunks/s   eta {eta}")),
        Line::raw(format!(
            "file   {}",
            fit(&ix.current, width.saturating_sub(7))
        )),
        Line::raw(""),
        Line::raw(format!("skipped {} files", ix.skipped.len())),
    ];
    out.extend(ix.skipped.iter().map(|(p, why)| {
        Line::styled(
            fit(&format!("  {p}  {why}"), width),
            Style::new().fg(Color::DarkGray),
        )
    }));
    out
}

const HELP: &[&str] = &[
    "Tab / Shift-Tab, 1-4     switch view (Tab in RAG: ranked / context)",
    "j k / arrows             move",
    "Enter / Space            expand or collapse; open a file's chunks",
    "h l / left right         collapse, expand; leave chunks",
    "hjkl / arrows (Map)      move the cursor",
    "/                        type a query; Enter runs it like search_code",
    "i                        index now (Esc or c cancels)",
    "r                        reload the index",
    "q / Esc                  quit",
    "",
    "Any key closes this help.",
];

fn help(f: &mut Frame, area: Rect) {
    let w = (HELP.iter().map(|l| l.len()).max().unwrap_or(0) as u16 + 4).min(area.width);
    let h = (HELP.len() as u16 + 2).min(area.height);
    let r = Rect {
        x: area.x + (area.width - w) / 2,
        y: area.y + (area.height - h) / 2,
        width: w,
        height: h,
    };
    f.render_widget(Clear, r);
    f.render_widget(
        Paragraph::new(HELP.iter().map(|l| Line::raw(*l)).collect::<Vec<_>>())
            .block(Block::bordered().title(" Keys ")),
        r,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::FakeEmbedder;
    use crate::tui::state::tests::key;
    use crossterm::event::KeyCode;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn data(files: &[(&str, &str)]) -> (tempfile::TempDir, Data) {
        let dir = tempfile::tempdir().unwrap();
        for (p, text) in files {
            let p = dir.path().join(p);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, text).unwrap();
        }
        crate::index_project(dir.path(), &FakeEmbedder, false).unwrap();
        let mut d = Data::load(dir.path()).unwrap();
        d.poll_map(true);
        (dir, d)
    }

    fn render(s: &State, d: &Data, w: u16, h: u16) -> String {
        let mut t = Terminal::new(TestBackend::new(w, h)).unwrap();
        t.draw(|f| draw(f, s, d)).unwrap();
        let buf = t.backend().buffer();
        (0..h)
            .map(|y| (0..w).map(|x| buf[(x, y)].symbol()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn sample() -> (tempfile::TempDir, Data) {
        let big: String = (0..300).map(|i| format!("fn f{i}() {{ {i} }}\n")).collect();
        data(&[
            ("src/big.rs", &big),
            ("src/small.rs", "fn one() {}\n"),
            ("docs/a.md", "# heading\n\x1b[31mred\n"),
        ])
    }

    #[test]
    fn stack_bars_are_proportional() {
        let (_d, d) = sample();
        let s = State::new(&d.tree, &d.map);
        let out = render(&s, &d, 100, 20);
        let src = out.lines().find(|l| l.contains("▸ src")).expect(&out);
        let docs = out.lines().find(|l| l.contains("▸ docs")).expect(&out);
        let full = |l: &str| l.chars().filter(|&c| c == '█').count();
        assert!(full(src) > 10 * full(docs).max(1) / 2, "{out}");
        assert!(
            out.contains("chunks ·") && out.contains("fake-384"),
            "{out}"
        );
        // The src row comes first (sorted by count).
        assert!(out.find("▸ src").unwrap() < out.find("▸ docs").unwrap());
    }

    #[test]
    fn preview_is_scrubbed_and_capped() {
        let long = format!("{}\n", "x".repeat(500)).repeat(40);
        let (_d, d) = data(&[("a.txt", "\x1b[2Jboom\n"), ("b.txt", &long)]);
        let mut s = State::new(&d.tree, &d.map);
        let rows = d.tree.rows(&s.expanded);
        s.row = rows
            .iter()
            .position(|&(i, _)| d.tree.nodes[i].name == "a.txt")
            .unwrap();
        let out = render(&s, &d, 120, 30);
        assert!(out.contains("?[2Jboom") && !out.contains('\x1b'), "{out}");
        let c = d.chunk(d.tree.nodes[rows[1 - s.row].0].chunks[0]).unwrap();
        let p = preview(c);
        assert!(p.len() <= PREVIEW_LINES + 1);
        assert!(p.iter().all(|l| l.width() <= PREVIEW_COLS.max(40)));
    }

    #[test]
    fn map_draws_points_and_legend() {
        let (_d, d) = sample();
        let mut s = State::new(&d.tree, &d.map);
        s.update(key(KeyCode::Char('2')), &d.tree, &d.map);
        let out = render(&s, &d, 100, 30);
        let braille = out
            .chars()
            .filter(|c| ('\u{2801}'..='\u{28FF}').contains(c))
            .count();
        assert!(braille > 0, "{out}");
        assert!(
            out.contains("■ src") && out.contains("■ docs") && out.contains('+'),
            "{out}"
        );
        assert!(out.contains("Preview"), "{out}");
    }

    fn searched(dir: &std::path::Path, d: &Data, q: &str) -> State {
        let mut s = State::new(&d.tree, &d.map);
        s.view = View::Rag;
        let db = d.db.as_ref().unwrap();
        s.hits =
            crate::search_project(dir, db, &FakeEmbedder, q, DEFAULT_K * MAX_PER_FILE).unwrap();
        s.context = Some(crate::budget::context(&s.hits, None, 0, 0, DEFAULT_K));
        s.hit_points = vec![Some([0.0, 0.0]); s.hits.len()];
        s
    }

    #[test]
    fn rag_view_explains_the_budget() {
        let (dir, d) = sample();
        let mut s = State::new(&d.tree, &d.map);
        s.view = View::Rag;
        assert!(render(&s, &d, 80, 20).contains("Press /"));
        let mut s = searched(dir.path(), &d, "fn one() {}");
        let out = render(&s, &d, 140, 30);
        assert!(out.contains("1.00  src/small.rs:1-1  sent"), "{out}");
        assert!(out.contains("dropped: 3 per file"), "{out}");
        assert!(out.contains("sent, trimmed"), "{out}");
        assert!(out.contains("tokens") && out.contains("/ 4000"), "{out}");
        assert!(out.contains("3/3  src/big.rs"), "{out}");
        s.update(key(KeyCode::Tab), &d.tree, &d.map);
        let out = render(&s, &d, 140, 30);
        assert!(
            out.contains("Context as Claude sees it") && out.contains("chunk_id"),
            "{out}"
        );
        assert!(out.contains("indexed "), "header line: {out}");
        // Only the sent hits are marked as retrieved on the map.
        s.view = View::Map;
        assert!(render(&s, &d, 100, 30).contains('•'));
    }

    #[test]
    fn index_view_and_density() {
        let (_d, d) = sample();
        let mut s = State::new(&d.tree, &d.map);
        s.view = View::Index;
        assert!(render(&s, &d, 100, 20).contains("Press i"));
        let mut ix = crate::tui::state::Indexing::new(std::time::Instant::now());
        ix.apply(crate::tui::state::Delta {
            files: 3,
            chunks: 40,
            files_done: 1,
            chunks_done: 20,
            current: "src/big.rs".into(),
            points: vec![(0.0, 0.0), (1.0, 2.0)],
            skipped: Some(std::sync::Arc::new(vec![(
                "key.pem".into(),
                "Denied".into(),
            )])),
        });
        assert_eq!(ix.bounds(), ([-0.05, 1.05], [-0.1, 2.1]));
        s.indexing = Some(ix);
        let out = render(&s, &d, 100, 20);
        for want in [
            "chunks     20 / 40",
            "files       1 / 3",
            "src/big.rs",
            "key.pem  Denied",
            "2 sampled",
        ] {
            assert!(out.contains(want), "{want}: {out}");
        }
        // The map shades dense cells.
        s.view = View::Map;
        let mut t = Terminal::new(TestBackend::new(100, 30)).unwrap();
        t.draw(|f| draw(f, &s, &d)).unwrap();
        let shaded = t
            .backend()
            .buffer()
            .content
            .iter()
            .filter(|c| matches!(c.bg, Color::Indexed(_)))
            .count();
        assert!(shaded > 0);
    }

    #[test]
    fn map_failure_is_shown() {
        let (_d, mut d) = sample();
        d.map_failed = true;
        d.map = Default::default();
        let mut s = State::new(&d.tree, &d.map);
        s.view = View::Map;
        assert!(render(&s, &d, 80, 10).contains("map failed"));
    }

    #[test]
    fn no_index_yet() {
        let dir = tempfile::tempdir().unwrap();
        let d = Data::empty(dir.path().to_path_buf());
        for view in [View::Stack, View::Map, View::Rag, View::Index] {
            let mut s = State::new(&d.tree, &d.map);
            s.view = view;
            assert!(render(&s, &d, 80, 12).contains("no index yet"));
        }
    }

    #[test]
    fn empty_index_and_tiny_terminals() {
        let (_e, empty) = data(&[]);
        let mut s = State::new(&empty.tree, &empty.map);
        assert!(render(&s, &empty, 80, 10).contains("The index is empty"));
        s.view = View::Map;
        assert!(render(&s, &empty, 80, 10).contains("Nothing to map"));
        let (_d, d) = sample();
        for (w, h) in [(40, 10), (10, 3), (1, 1), (0, 0)] {
            for view in [View::Stack, View::Map, View::Rag, View::Index] {
                let mut s = State::new(&d.tree, &d.map);
                s.view = view;
                s.help = true;
                s.indexing = Some(crate::tui::state::Indexing::new(std::time::Instant::now()));
                render(&s, &d, w, h);
            }
        }
    }
}
