//! UI state and the key reducer. No terminal and no I/O: side effects are returned as [`Action`]s.

use super::model::{grow, padded, Map, Tree};
use crate::budget::Context;
use crate::SearchHit;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use std::sync::Arc;
use std::time::Instant;

/// Longest query accepted on the query line, in chars.
const MAX_QUERY: usize = 200;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum View {
    Stack,
    Map,
    Rag,
    Index,
}

/// What the runner must do after a key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Action {
    None,
    Quit,
    Search(String),
    Reload,
    StartIndex,
    CancelIndex,
}

/// The live indexing run, as last reported by the worker.
#[derive(Debug, Clone)]
pub(crate) struct Indexing {
    pub started: Instant,
    /// Set when the run ends; `running` is then false.
    pub ended: Option<Instant>,
    pub running: bool,
    pub files: usize,
    pub chunks: usize,
    pub files_done: usize,
    pub chunks_done: usize,
    pub current: String,
    /// (path, reason), already scrubbed; shared with the worker, sent once.
    pub skipped: Arc<Vec<(String, String)>>,
    /// Sampled embedded chunks projected onto the plane so far, and their raw extent.
    pub points: Vec<(f64, f64)>,
    pub extent: Option<[[f64; 2]; 2]>,
}

/// One progress report from the indexing worker: counters plus only the new points.
#[derive(Debug, Clone, Default)]
pub(crate) struct Delta {
    pub files: usize,
    pub chunks: usize,
    pub files_done: usize,
    pub chunks_done: usize,
    pub current: String,
    pub points: Vec<(f64, f64)>,
    /// Set when the skipped list changed (normally once, after the walk).
    pub skipped: Option<Arc<Vec<(String, String)>>>,
}

impl Indexing {
    pub fn apply(&mut self, d: Delta) {
        (self.files, self.chunks) = (d.files, d.chunks);
        (self.files_done, self.chunks_done) = (d.files_done, d.chunks_done);
        self.current = d.current;
        for &(x, y) in &d.points {
            grow(&mut self.extent, [x, y]);
        }
        self.points.extend(d.points);
        if let Some(s) = d.skipped {
            self.skipped = s;
        }
    }

    pub fn bounds(&self) -> ([f64; 2], [f64; 2]) {
        padded(self.extent)
    }

    pub fn finish(&mut self) {
        self.running = false;
        self.ended = Some(Instant::now());
    }

    pub fn secs(&self) -> f64 {
        let end = self.ended.unwrap_or_else(Instant::now);
        end.saturating_duration_since(self.started).as_secs_f64()
    }

    pub fn new(started: Instant) -> Indexing {
        Indexing {
            started,
            ended: None,
            running: true,
            files: 0,
            chunks: 0,
            files_done: 0,
            chunks_done: 0,
            current: String::new(),
            skipped: Arc::default(),
            points: Vec::new(),
            extent: None,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct State {
    pub view: View,
    pub help: bool,
    pub expanded: Vec<bool>,
    /// Selected row of the Stack tree.
    pub row: usize,
    /// Keys move the chunk list of the selected file instead of the tree.
    pub in_chunks: bool,
    pub chunk: usize,
    /// Map cursor in plane coordinates.
    pub cursor: [f32; 2],
    pub editing: bool,
    pub query: String,
    /// Ranked candidates (over-fetched like `search_code`) and what the budget made of them.
    pub hits: Vec<SearchHit>,
    pub context: Option<Context>,
    /// RAG view shows the response text instead of the ranked list; `scroll` is its first line.
    pub show_context: bool,
    pub scroll: u16,
    pub indexing: Option<Indexing>,
    /// Plane position of each hit and of the query (None if it could not be projected).
    pub hit_points: Vec<Option<[f32; 2]>>,
    pub query_point: Option<[f32; 2]>,
    pub hit: usize,
    pub status: String,
}

impl State {
    pub fn new(tree: &Tree, map: &Map) -> State {
        let ([x0, x1], [y0, y1]) = map.bounds();
        State {
            view: View::Stack,
            help: false,
            expanded: vec![false; tree.nodes.len()],
            row: 0,
            in_chunks: false,
            chunk: 0,
            cursor: [((x0 + x1) / 2.0) as f32, ((y0 + y1) / 2.0) as f32],
            editing: false,
            query: String::new(),
            hits: Vec::new(),
            context: None,
            show_context: false,
            scroll: 0,
            indexing: None,
            hit_points: Vec::new(),
            query_point: None,
            hit: 0,
            status: String::new(),
        }
    }

    /// The node on the selected Stack row.
    pub fn selected_node(&self, tree: &Tree) -> Option<usize> {
        tree.rows(&self.expanded).get(self.row).map(|&(i, _)| i)
    }

    /// The selected file's chunks (empty when a directory is selected).
    pub fn file_chunks<'a>(&self, tree: &'a Tree) -> &'a [u32] {
        self.selected_node(tree)
            .map_or(&[], |i| tree.nodes[i].chunks.as_slice())
    }

    pub fn update(&mut self, key: KeyEvent, tree: &Tree, map: &Map) -> Action {
        let ctrl_c =
            key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL);
        if ctrl_c {
            return Action::Quit;
        }
        if self.help {
            self.help = false;
            return Action::None;
        }
        if self.editing {
            return self.edit(key);
        }
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            return match key.code {
                KeyCode::Char('c') => Action::Quit,
                _ => Action::None,
            };
        }
        let running = self.indexing.as_ref().is_some_and(|i| i.running);
        match key.code {
            KeyCode::Esc | KeyCode::Char('c') if running => return Action::CancelIndex,
            KeyCode::Char('q') | KeyCode::Esc => return Action::Quit,
            KeyCode::Char('i') => {
                self.view = View::Index;
                if !running {
                    return Action::StartIndex;
                }
            }
            KeyCode::Char('?') => self.help = true,
            KeyCode::Char('r') => return Action::Reload,
            KeyCode::Char('/') => {
                self.view = View::Rag;
                self.editing = true;
            }
            KeyCode::Char('1') => self.view = View::Stack,
            KeyCode::Char('2') => self.view = View::Map,
            KeyCode::Char('3') => self.view = View::Rag,
            KeyCode::Char('4') => self.view = View::Index,
            KeyCode::Tab if self.view == View::Rag => {
                self.show_context = !self.show_context;
                self.scroll = 0;
            }
            KeyCode::Tab => self.view = self.cycle(1),
            KeyCode::BackTab => self.view = self.cycle(3),
            _ => match self.view {
                View::Stack => self.stack(key.code, tree),
                View::Map => self.map(key.code, map),
                View::Rag => self.search(key.code),
                View::Index => {}
            },
        }
        Action::None
    }

    fn cycle(&self, by: usize) -> View {
        const ALL: [View; 4] = [View::Stack, View::Map, View::Rag, View::Index];
        let i = ALL.iter().position(|&v| v == self.view).unwrap_or(0);
        ALL[(i + by) % ALL.len()]
    }

    fn edit(&mut self, key: KeyEvent) -> Action {
        match key.code {
            KeyCode::Enter => {
                self.editing = false;
                if !self.query.trim().is_empty() {
                    return Action::Search(self.query.clone());
                }
            }
            KeyCode::Esc => self.editing = false,
            KeyCode::Backspace => {
                self.query.pop();
            }
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                return Action::Quit
            }
            KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.query.clear()
            }
            KeyCode::Char(c) if !c.is_control() && self.query.chars().count() < MAX_QUERY => {
                self.query.push(c)
            }
            _ => {}
        }
        Action::None
    }

    fn stack(&mut self, code: KeyCode, tree: &Tree) {
        let rows = tree.rows(&self.expanded);
        let Some(&(node, _)) = rows.get(self.row) else {
            return;
        };
        let chunks = tree.nodes[node].chunks.len();
        match code {
            KeyCode::Char('j') | KeyCode::Down if self.in_chunks => {
                self.chunk = (self.chunk + 1).min(chunks.saturating_sub(1))
            }
            KeyCode::Char('k') | KeyCode::Up if self.in_chunks => {
                self.chunk = self.chunk.saturating_sub(1)
            }
            KeyCode::Char('h') | KeyCode::Left if self.in_chunks => self.in_chunks = false,
            KeyCode::Char('j') | KeyCode::Down => {
                self.row = (self.row + 1).min(rows.len() - 1);
                self.chunk = 0;
            }
            KeyCode::Char('k') | KeyCode::Up => {
                self.row = self.row.saturating_sub(1);
                self.chunk = 0;
            }
            KeyCode::Enter | KeyCode::Char(' ') | KeyCode::Char('l') | KeyCode::Right => {
                if tree.is_dir(node) {
                    // Right/l only opens; Enter/Space toggles.
                    let open = !matches!(code, KeyCode::Enter | KeyCode::Char(' '))
                        || !self.expanded[node];
                    self.expanded[node] = open;
                } else if chunks > 0 {
                    self.in_chunks = true;
                }
            }
            KeyCode::Char('h') | KeyCode::Left if tree.is_dir(node) => self.expanded[node] = false,
            _ => {}
        }
    }

    fn map(&mut self, code: KeyCode, map: &Map) {
        let ([x0, x1], [y0, y1]) = map.bounds();
        let (dx, dy) = (((x1 - x0) / 40.0) as f32, ((y1 - y0) / 20.0) as f32);
        let c = &mut self.cursor;
        match code {
            KeyCode::Char('h') | KeyCode::Left => c[0] -= dx,
            KeyCode::Char('l') | KeyCode::Right => c[0] += dx,
            KeyCode::Char('k') | KeyCode::Up => c[1] += dy,
            KeyCode::Char('j') | KeyCode::Down => c[1] -= dy,
            _ => {}
        }
        c[0] = c[0].clamp(x0 as f32, x1 as f32);
        c[1] = c[1].clamp(y0 as f32, y1 as f32);
    }

    fn search(&mut self, code: KeyCode) {
        match code {
            KeyCode::Char('j') | KeyCode::Down if self.show_context => {
                self.scroll = self.scroll.saturating_add(1)
            }
            KeyCode::Char('k') | KeyCode::Up if self.show_context => {
                self.scroll = self.scroll.saturating_sub(1)
            }
            KeyCode::Char('j') | KeyCode::Down => {
                self.hit = (self.hit + 1).min(self.hits.len().saturating_sub(1))
            }
            KeyCode::Char('k') | KeyCode::Up => self.hit = self.hit.saturating_sub(1),
            _ => {}
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn fixture() -> (Tree, Map) {
        let tree = Tree::build(["src/a.rs", "src/a.rs", "src/b.rs", "README.md"]);
        let q = [0i8, 0, 10, 0, 0, 10, 5, 5];
        (tree, Map::build(&q, 2, 4, 100))
    }

    fn press(s: &mut State, keys: &[KeyCode], t: &Tree, m: &Map) -> Vec<Action> {
        keys.iter().map(|&k| s.update(key(k), t, m)).collect()
    }

    #[test]
    fn global_keys() {
        let (t, m) = fixture();
        let mut s = State::new(&t, &m);
        assert_eq!(s.update(key(KeyCode::Char('q')), &t, &m), Action::Quit);
        assert_eq!(s.update(key(KeyCode::Esc), &t, &m), Action::Quit);
        let ctrl_c = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert_eq!(s.update(ctrl_c, &t, &m), Action::Quit);
        assert_eq!(s.update(key(KeyCode::Char('r')), &t, &m), Action::Reload);
        press(&mut s, &[KeyCode::Tab], &t, &m);
        assert_eq!(s.view, View::Map);
        press(&mut s, &[KeyCode::Tab], &t, &m);
        assert_eq!(s.view, View::Rag);
        // In the RAG view Tab toggles ranked hits / context; Shift-Tab still switches views.
        press(&mut s, &[KeyCode::Tab, KeyCode::Char('j')], &t, &m);
        assert!(s.show_context && s.view == View::Rag && s.scroll == 1);
        press(&mut s, &[KeyCode::Tab], &t, &m);
        assert!(!s.show_context && s.scroll == 0);
        press(
            &mut s,
            &[KeyCode::BackTab, KeyCode::BackTab, KeyCode::BackTab],
            &t,
            &m,
        );
        assert_eq!(s.view, View::Index);
        press(&mut s, &[KeyCode::Char('1'), KeyCode::BackTab], &t, &m);
        assert_eq!(s.view, View::Index);
        press(&mut s, &[KeyCode::Char('2')], &t, &m);
        assert_eq!(s.view, View::Map);
        // Help swallows the next key, even q.
        press(&mut s, &[KeyCode::Char('?')], &t, &m);
        assert!(s.help);
        assert_eq!(s.update(key(KeyCode::Char('q')), &t, &m), Action::None);
        assert!(!s.help);
        // ...but Ctrl-C always quits, even over help or the query line.
        press(&mut s, &[KeyCode::Char('?')], &t, &m);
        assert_eq!(s.update(ctrl_c, &t, &m), Action::Quit);
        press(&mut s, &[KeyCode::Esc, KeyCode::Char('/')], &t, &m);
        assert_eq!(s.update(ctrl_c, &t, &m), Action::Quit);
    }

    #[test]
    fn indexing_keys() {
        let (t, m) = fixture();
        let mut s = State::new(&t, &m);
        assert_eq!(
            s.update(key(KeyCode::Char('i')), &t, &m),
            Action::StartIndex
        );
        assert_eq!(s.view, View::Index);
        s.indexing = Some(Indexing::new(Instant::now()));
        // While a run is active: i does not start another; Esc and c cancel instead of quitting.
        assert_eq!(s.update(key(KeyCode::Char('i')), &t, &m), Action::None);
        assert_eq!(s.update(key(KeyCode::Esc), &t, &m), Action::CancelIndex);
        assert_eq!(
            s.update(key(KeyCode::Char('c')), &t, &m),
            Action::CancelIndex
        );
        assert_eq!(s.update(key(KeyCode::Char('q')), &t, &m), Action::Quit);
        s.indexing.as_mut().unwrap().finish();
        assert_eq!(s.update(key(KeyCode::Char('c')), &t, &m), Action::None);
        assert_eq!(s.update(key(KeyCode::Esc), &t, &m), Action::Quit);
    }

    #[test]
    fn stack_navigation() {
        let (t, m) = fixture();
        let mut s = State::new(&t, &m);
        assert_eq!(t.rows(&s.expanded).len(), 2);
        press(&mut s, &[KeyCode::Char('k'), KeyCode::Enter], &t, &m);
        assert_eq!(t.rows(&s.expanded).len(), 4, "src expanded");
        press(&mut s, &[KeyCode::Char('j')], &t, &m);
        assert_eq!(s.file_chunks(&t), [0, 1]);
        press(
            &mut s,
            &[KeyCode::Enter, KeyCode::Char('j'), KeyCode::Char('j')],
            &t,
            &m,
        );
        assert!(s.in_chunks);
        assert_eq!(s.chunk, 1, "clamped to the last chunk");
        press(&mut s, &[KeyCode::Char('h'), KeyCode::Char('j')], &t, &m);
        assert!(!s.in_chunks);
        assert_eq!((s.row, s.chunk), (2, 0));
        press(&mut s, &[KeyCode::Char('j'); 9], &t, &m);
        assert_eq!(s.row, 3, "clamped to the last row");
        press(&mut s, &[KeyCode::Char('k'); 9], &t, &m);
        press(&mut s, &[KeyCode::Char(' ')], &t, &m);
        assert_eq!(t.rows(&s.expanded).len(), 2, "space collapses");
        press(&mut s, &[KeyCode::Right, KeyCode::Right], &t, &m);
        assert_eq!(t.rows(&s.expanded).len(), 4, "right only opens");
        press(&mut s, &[KeyCode::Left], &t, &m);
        assert_eq!(t.rows(&s.expanded).len(), 2);
        // An empty tree ignores navigation.
        let (et, em) = (Tree::build([]), Map::default());
        let mut e = State::new(&et, &em);
        press(&mut e, &[KeyCode::Char('j'), KeyCode::Enter], &et, &em);
        assert_eq!(e.row, 0);
    }

    #[test]
    fn map_cursor_stays_in_bounds() {
        let (t, m) = fixture();
        let mut s = State::new(&t, &m);
        s.view = View::Map;
        let start = s.cursor;
        press(&mut s, &[KeyCode::Char('l')], &t, &m);
        assert!(s.cursor[0] > start[0]);
        press(&mut s, &[KeyCode::Up], &t, &m);
        assert!(s.cursor[1] > start[1]);
        press(&mut s, &[KeyCode::Char('h'); 100], &t, &m);
        press(&mut s, &[KeyCode::Char('j'); 100], &t, &m);
        let ([x0, _], [y0, _]) = m.bounds();
        assert_eq!(s.cursor, [x0 as f32, y0 as f32]);
    }

    #[test]
    fn query_line_and_hits() {
        let (t, m) = fixture();
        let mut s = State::new(&t, &m);
        press(&mut s, &[KeyCode::Char('/')], &t, &m);
        assert!(s.editing && s.view == View::Rag);
        // While editing, q and digits are text, not commands.
        press(
            &mut s,
            &[KeyCode::Char('q'), KeyCode::Char('1'), KeyCode::Char('x')],
            &t,
            &m,
        );
        press(&mut s, &[KeyCode::Backspace], &t, &m);
        assert_eq!(s.query, "q1");
        assert_eq!(
            s.update(key(KeyCode::Enter), &t, &m),
            Action::Search("q1".into())
        );
        assert!(!s.editing);
        // Empty query: Enter just closes the line.
        press(&mut s, &[KeyCode::Char('/')], &t, &m);
        s.update(
            KeyEvent::new(KeyCode::Char('u'), KeyModifiers::CONTROL),
            &t,
            &m,
        );
        assert_eq!(s.update(key(KeyCode::Enter), &t, &m), Action::None);
        press(&mut s, &[KeyCode::Char('/'), KeyCode::Esc], &t, &m);
        assert!(!s.editing, "Esc leaves the line without quitting");
        press(&mut s, &[KeyCode::Char('/')], &t, &m);
        press(&mut s, &[KeyCode::Char('a'); MAX_QUERY + 5], &t, &m);
        assert_eq!(s.query.chars().count(), MAX_QUERY);
        press(&mut s, &[KeyCode::Esc, KeyCode::Char('j')], &t, &m);
        assert_eq!(s.hit, 0, "no hits: stays at 0");
    }
}
