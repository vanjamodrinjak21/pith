//! Pure data for the UI: the directory tree, stacked-bar text, and a 2-D PCA map of the vectors.
//! Nothing here touches the terminal.

use std::collections::HashMap;

/// At most this many vectors are projected onto the map, evenly strided.
pub(crate) const MAP_CAP: usize = 20_000;

/// One directory or file. Children are sorted by chunk count, descending, then by name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Node {
    pub name: String,
    pub count: usize,
    pub children: Vec<usize>,
    /// Chunk ids, in id order; non-empty only for files.
    pub chunks: Vec<u32>,
}

/// Arena of nodes; node 0 is the root (empty name, count = all chunks).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Tree {
    pub nodes: Vec<Node>,
}

impl Tree {
    /// `paths[i]` is the stored path of chunk `i`, `/`-separated.
    pub fn build<'a>(paths: impl IntoIterator<Item = &'a str>) -> Tree {
        let node = |name: &str| Node {
            name: name.to_string(),
            count: 0,
            children: Vec::new(),
            chunks: Vec::new(),
        };
        let mut nodes = vec![node("")];
        let mut index: HashMap<(usize, &str), usize> = HashMap::new();
        for (id, path) in paths.into_iter().enumerate() {
            let mut cur = 0;
            nodes[0].count += 1;
            for part in path.split('/').filter(|p| !p.is_empty()) {
                cur = *index.entry((cur, part)).or_insert_with(|| {
                    nodes.push(node(part));
                    let new = nodes.len() - 1;
                    nodes[cur].children.push(new);
                    new
                });
                nodes[cur].count += 1;
            }
            nodes[cur].chunks.push(id as u32);
        }
        for i in 0..nodes.len() {
            let mut c = std::mem::take(&mut nodes[i].children);
            c.sort_by(|&a, &b| {
                nodes[b]
                    .count
                    .cmp(&nodes[a].count)
                    .then_with(|| nodes[a].name.cmp(&nodes[b].name))
            });
            nodes[i].children = c;
        }
        Tree { nodes }
    }

    pub fn total(&self) -> usize {
        self.nodes[0].count
    }

    pub fn is_dir(&self, i: usize) -> bool {
        !self.nodes[i].children.is_empty()
    }

    /// Visible rows as (node, depth), depth-first; a directory's children show when it is expanded.
    pub fn rows(&self, expanded: &[bool]) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let mut stack: Vec<(usize, usize)> = self.nodes[0]
            .children
            .iter()
            .rev()
            .map(|&c| (c, 0))
            .collect();
        while let Some((i, d)) = stack.pop() {
            out.push((i, d));
            if expanded.get(i).copied().unwrap_or(false) {
                stack.extend(self.nodes[i].children.iter().rev().map(|&c| (c, d + 1)));
            }
        }
        out
    }

    /// Index of `path`'s top-level entry among the root's children (its map colour group).
    pub fn group(&self, path: &str) -> Option<usize> {
        let top = path.split('/').find(|p| !p.is_empty())?;
        self.nodes[0]
            .children
            .iter()
            .position(|&c| self.nodes[c].name == top)
    }
}

const EIGHTHS: [char; 8] = [' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉'];

/// A bar `count / total` of `width` cells at 1/8-cell resolution, never wider than `width`.
/// Rounds down, except that a nonzero count always shows at least one eighth.
pub(crate) fn bar(count: usize, total: usize, width: usize) -> String {
    if total == 0 || width == 0 {
        return String::new();
    }
    let e = (count.min(total) as u128 * width as u128 * 8 / total as u128) as usize;
    let e = if count > 0 { e.max(1) } else { 0 };
    let mut s = "█".repeat(e / 8);
    if e % 8 > 0 {
        s.push(EIGHTHS[e % 8]);
    }
    s
}

/// `m = min(n, cap)` ids evenly strided over `0..n`, starting at 0.
pub(crate) fn sample_ids(n: usize, cap: usize) -> impl Iterator<Item = usize> {
    let m = n.min(cap);
    (0..m).map(move |i| (i as u128 * n as u128 / m as u128) as usize)
}

/// Two principal components of int8 rows, plus their mean.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Pca {
    pub mean: Vec<f32>,
    pub axes: [Vec<f32>; 2],
}

const ITERS: usize = 60;

fn normalize(v: &mut [f64]) {
    let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    // Degenerate (no variance left): the axis is zero and projects everything to 0.
    let n = if n > 1e-9 { n } else { f64::INFINITY };
    v.iter_mut().for_each(|x| *x /= n);
}

/// Sign convention: the coordinate with the largest magnitude (first on ties) is positive.
fn fix_sign(v: &mut [f64]) {
    let mut best = 0.0f64;
    for &x in v.iter() {
        if x.abs() > best.abs() {
            best = x;
        }
    }
    if best < 0.0 {
        v.iter_mut().for_each(|x| *x = -*x);
    }
}

impl Pca {
    /// Fits on rows `ids` of `q` (row-major, `dim` per row; int8 vectors, or f32 while indexing)
    /// by orthogonal power iteration from a fixed start, so the result is deterministic. None if
    /// `dim` is 0, `q` is not whole rows, an id is out of range, or there are no ids.
    pub fn fit<T: Copy + Into<f64>>(q: &[T], dim: usize, ids: &[usize]) -> Option<Pca> {
        if dim == 0 || !q.len().is_multiple_of(dim) || ids.is_empty() {
            return None;
        }
        let rows = q.len() / dim;
        if ids.iter().any(|&i| i >= rows) {
            return None;
        }
        let row = |i: usize| &q[i * dim..(i + 1) * dim];
        let mut mean = vec![0f64; dim];
        for &i in ids {
            mean.iter_mut()
                .zip(row(i))
                .for_each(|(m, &x)| *m += x.into());
        }
        mean.iter_mut().for_each(|m| *m /= ids.len() as f64);
        // Fixed, generic start vectors (not orthogonal to typical axes).
        let mut a: Vec<f64> = (0..dim).map(|j| (j % 7 + 1) as f64).collect();
        let mut b: Vec<f64> = (0..dim).map(|j| (j % 5) as f64 - 2.5).collect();
        let dot = |x: &[f64], y: &[f64]| x.iter().zip(y).map(|(p, q)| p * q).sum::<f64>();
        for _ in 0..ITERS {
            // w = C v without forming C: sum over rows of ((x - mean) . v) (x - mean).
            let (ma, mb) = (dot(&mean, &a), dot(&mean, &b));
            let (mut wa, mut wb) = (vec![0f64; dim], vec![0f64; dim]);
            let (mut sa, mut sb) = (0f64, 0f64);
            for &i in ids {
                let r = row(i);
                let (mut pa, mut pb) = (0f64, 0f64);
                for ((&x, &u), &v) in r.iter().zip(&a).zip(&b) {
                    let x: f64 = x.into();
                    pa += x * u;
                    pb += x * v;
                }
                let (pa, pb) = (pa - ma, pb - mb);
                for ((&x, ua), vb) in r.iter().zip(&mut wa).zip(&mut wb) {
                    let x: f64 = x.into();
                    *ua += pa * x;
                    *vb += pb * x;
                }
                sa += pa;
                sb += pb;
            }
            wa.iter_mut().zip(&mean).for_each(|(w, m)| *w -= sa * m);
            wb.iter_mut().zip(&mean).for_each(|(w, m)| *w -= sb * m);
            normalize(&mut wa);
            let p = dot(&wb, &wa);
            wb.iter_mut().zip(&wa).for_each(|(w, u)| *w -= p * u);
            normalize(&mut wb);
            (a, b) = (wa, wb);
        }
        fix_sign(&mut a);
        fix_sign(&mut b);
        let f = |v: Vec<f64>| v.into_iter().map(|x| x as f32).collect::<Vec<_>>();
        Some(Pca {
            mean: f(mean),
            axes: [f(a), f(b)],
        })
    }

    /// Plane coordinates of a row given as f32 values in int8 units. None on a dimension mismatch.
    pub fn project(&self, x: impl ExactSizeIterator<Item = f32> + Clone) -> Option<[f32; 2]> {
        if x.len() != self.mean.len() {
            return None;
        }
        let along = |axis: &[f32]| {
            x.clone()
                .zip(&self.mean)
                .zip(axis)
                .map(|((v, m), a)| (v - m) * a)
                .sum::<f32>()
        };
        Some([along(&self.axes[0]), along(&self.axes[1])])
    }

    pub fn project_row(&self, row: &[i8]) -> Option<[f32; 2]> {
        self.project(row.iter().map(|&v| v as f32))
    }

    /// A full-precision vector (e.g. a query) mapped into int8 units with the index `scale`.
    pub fn project_vector(&self, v: &[f32], scale: f32) -> Option<[f32; 2]> {
        self.project(v.iter().map(|&x| x / scale))
    }
}

/// The projected sample: `points[i]` is chunk `id(i)`. Keeps only two f32 per point.
#[derive(Debug, Clone, Default)]
pub(crate) struct Map {
    pub pca: Option<Pca>,
    pub points: Vec<[f32; 2]>,
    n: usize,
    /// Raw extent of `points`, computed once.
    extent: Option<[[f64; 2]; 2]>,
}

impl Map {
    /// `q` holds `n` rows of `dim` int8 values.
    pub fn build(q: &[i8], dim: usize, n: usize, cap: usize) -> Map {
        let ids: Vec<usize> = sample_ids(n, cap).collect();
        let Some(pca) = Pca::fit(q, dim, &ids) else {
            return Map::default();
        };
        let points: Vec<[f32; 2]> = ids
            .iter()
            .filter_map(|&i| pca.project_row(&q[i * dim..(i + 1) * dim]))
            .collect();
        let mut extent = None;
        for p in &points {
            grow(&mut extent, [p[0] as f64, p[1] as f64]);
        }
        Map {
            pca: Some(pca),
            points,
            n,
            extent,
        }
    }

    /// For a map built from an already-strided sample: point ids then refer to `n` chunks.
    pub fn with_total(mut self, n: usize) -> Map {
        self.n = n;
        self
    }

    /// Chunk id of point `i`.
    pub fn id(&self, i: usize) -> u32 {
        (i as u128 * self.n as u128 / self.points.len().max(1) as u128) as u32
    }

    /// (min, max) over x and y, padded so a single point or a line still has an area.
    pub fn bounds(&self) -> ([f64; 2], [f64; 2]) {
        padded(self.extent)
    }
}

/// Grows the raw `[[x0, x1], [y0, y1]]` extent of a point set by `p`.
pub(crate) fn grow(extent: &mut Option<[[f64; 2]; 2]>, p: [f64; 2]) {
    let e = extent.get_or_insert([[p[0], p[0]], [p[1], p[1]]]);
    for k in 0..2 {
        e[k] = [e[k][0].min(p[k]), e[k][1].max(p[k])];
    }
}

/// An extent padded by 5% (at least 1e-3); no points gives [-1, 1] on both axes.
pub(crate) fn padded(extent: Option<[[f64; 2]; 2]>) -> ([f64; 2], [f64; 2]) {
    let b = extent.unwrap_or([[-1.0, 1.0]; 2]).map(|[lo, hi]| {
        if extent.is_none() {
            return [lo, hi];
        }
        let pad = ((hi - lo) * 0.05).max(1e-3);
        [lo - pad, hi + pad]
    });
    (b[0], b[1])
}

/// Index of the point nearest to `p` (first on ties).
pub(crate) fn nearest(points: &[[f32; 2]], p: [f32; 2]) -> Option<usize> {
    let d = |q: &[f32; 2]| (q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2);
    let mut best: Option<(usize, f32)> = None;
    for (i, q) in points.iter().enumerate() {
        if best.is_none_or(|(_, bd)| d(q) < bd) {
            best = Some((i, d(q)));
        }
    }
    best.map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_aggregates_and_sorts_by_count() {
        let paths = [
            "src/a.rs",
            "src/a.rs",
            "src/b.rs",
            "README.md",
            "src/x/c.rs",
        ];
        let t = Tree::build(paths);
        assert_eq!(t.total(), 5);
        let top: Vec<_> = t.nodes[0]
            .children
            .iter()
            .map(|&c| (t.nodes[c].name.as_str(), t.nodes[c].count))
            .collect();
        assert_eq!(top, [("src", 4), ("README.md", 1)]);
        let src = t.nodes[0].children[0];
        let kids: Vec<_> = t.nodes[src]
            .children
            .iter()
            .map(|&c| t.nodes[c].name.as_str())
            .collect();
        assert_eq!(kids, ["a.rs", "b.rs", "x"], "ties sort by name");
        let a = t.nodes[src].children[0];
        assert_eq!(t.nodes[a].chunks, [0, 1]);
        assert!(t.is_dir(src) && !t.is_dir(a));
        assert_eq!(t.group("src/x/c.rs"), Some(0));
        assert_eq!(t.group("README.md"), Some(1));
        assert_eq!(t.group("nope"), None);
        assert_eq!(Tree::build([]).total(), 0);
    }

    #[test]
    fn rows_follow_expansion() {
        let t = Tree::build(["d/a", "d/b", "e"]);
        let mut ex = vec![false; t.nodes.len()];
        let names = |ex: &[bool]| {
            t.rows(ex)
                .into_iter()
                .map(|(i, d)| format!("{d}{}", t.nodes[i].name))
                .collect::<Vec<_>>()
        };
        assert_eq!(names(&ex), ["0d", "0e"]);
        ex[t.nodes[0].children[0]] = true;
        assert_eq!(names(&ex), ["0d", "1a", "1b", "0e"]);
    }

    fn cells(s: &str) -> usize {
        s.chars().count()
    }

    #[test]
    fn bar_widths_are_exact_and_bounded() {
        assert_eq!(bar(1, 1, 10), "█".repeat(10));
        assert_eq!(bar(1, 2, 10), "█".repeat(5));
        assert_eq!(bar(8, 16, 1), "▌", "half a cell");
        assert_eq!(bar(1, 8, 1), "▏", "one eighth");
        assert_eq!(bar(7, 8, 1), "▉");
        assert_eq!(bar(15, 16, 2), "█▉", "no rounding up into the next cell");
        assert_eq!(bar(0, 5, 10), "");
        assert_eq!(bar(1, 1_000_000, 10), "▏", "nonzero is visible");
        assert_eq!(bar(3, 0, 10), "");
        assert_eq!(bar(3, 3, 0), "");
        assert_eq!(bar(9, 3, 4), "████", "count above total is clamped");
        assert_eq!(cells(&bar(usize::MAX - 1, usize::MAX, 3)), 3, "no overflow");
        for total in 1..40 {
            for count in 0..=total {
                for width in 0..30 {
                    let b = bar(count, total, width);
                    assert!(cells(&b) <= width, "{count}/{total} w{width}: {b}");
                    if count == total && width > 0 {
                        assert_eq!(cells(&b), width);
                    }
                }
            }
        }
    }

    #[test]
    fn sample_is_strided_and_capped() {
        assert_eq!(sample_ids(5, 10).collect::<Vec<_>>(), [0, 1, 2, 3, 4]);
        assert_eq!(sample_ids(10, 5).collect::<Vec<_>>(), [0, 2, 4, 6, 8]);
        assert_eq!(sample_ids(0, 5).count(), 0);
        assert_eq!(sample_ids(5, 0).count(), 0);
        let v: Vec<_> = sample_ids(100_003, 20_000).collect();
        assert_eq!(v.len(), 20_000);
        assert!(v.windows(2).all(|w| w[0] < w[1]) && *v.last().unwrap() < 100_003);
    }

    /// Points with big spread along (1,1,0), less along (0,0,1), none along (1,-1,0).
    fn synthetic() -> Vec<i8> {
        let mut q = Vec::new();
        for i in -10i32..=10 {
            for j in [-2i32, 0, 2] {
                q.extend([(5 * i + 3) as i8, (5 * i + 3) as i8, (j * 3 - 7) as i8]);
            }
        }
        q
    }

    #[test]
    fn pca_finds_known_axes_with_stable_sign() {
        let q = synthetic();
        let ids: Vec<usize> = (0..q.len() / 3).collect();
        let p = Pca::fit(&q, 3, &ids).unwrap();
        let h = std::f32::consts::FRAC_1_SQRT_2;
        let near = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-3);
        assert!(near(&p.axes[0], &[h, h, 0.0]), "{:?}", p.axes[0]);
        assert!(near(&p.axes[1], &[0.0, 0.0, 1.0]), "{:?}", p.axes[1]);
        assert!(near(&p.mean, &[3.0, 3.0, -7.0]), "{:?}", p.mean);
        // Deterministic: same input, same output; row order does not flip signs.
        let rev: Vec<usize> = ids.iter().rev().copied().collect();
        assert!(near(&Pca::fit(&q, 3, &rev).unwrap().axes[0], &p.axes[0]));
        assert_eq!(Pca::fit(&q, 3, &ids).unwrap(), p);
        let c = p.project_row(&q[..3]).unwrap();
        assert!((c[0] - -50.0 * 2f32.sqrt()).abs() < 0.1, "{c:?}");
        assert_eq!(p.project_vector(&[3.0, 3.0, -7.0], 1.0), Some([0.0, 0.0]));
        assert_eq!(
            p.project_vector(&[0.03, 0.03, -0.07], 0.01)
                .map(|c| c[0].abs() < 1e-4),
            Some(true)
        );
        assert_eq!(p.project_row(&[1, 2]), None, "dim mismatch");
    }

    #[test]
    fn pca_degenerate_inputs_do_not_panic() {
        assert_eq!(Pca::fit::<i8>(&[], 3, &[]), None);
        let f = [1.0f32, 0.0, 3.0, 0.0];
        assert_eq!(
            Pca::fit(&f, 2, &[0, 1]).unwrap().axes[0],
            [1.0, 0.0],
            "f32 rows"
        );
        assert_eq!(Pca::fit(&[1, 2, 3], 0, &[0]), None);
        assert_eq!(Pca::fit(&[1, 2, 3, 4], 3, &[0]), None, "not whole rows");
        assert_eq!(Pca::fit(&[1, 2, 3], 3, &[1]), None, "id out of range");
        // One point, or identical points: every point maps to the origin.
        for q in [vec![4i8, 5, 6], vec![4, 5, 6, 4, 5, 6]] {
            let ids: Vec<usize> = (0..q.len() / 3).collect();
            let p = Pca::fit(&q, 3, &ids).unwrap();
            assert_eq!(p.project_row(&q[..3]), Some([0.0, 0.0]));
        }
        // Two points: the first axis is their difference.
        let p = Pca::fit(&[0, 0, 10, 0], 2, &[0, 1]).unwrap();
        assert_eq!(p.axes[0], [1.0, 0.0]);
        assert_eq!(p.project_row(&[10, 0]), Some([5.0, 0.0]));
        let m = Map::build(&[], 384, 0, MAP_CAP);
        assert!(m.points.is_empty() && m.pca.is_none());
        assert_eq!(m.bounds(), ([-1.0, 1.0], [-1.0, 1.0]));
        assert_eq!(
            Map::build(&[1, 2, 3], 2, 1, MAP_CAP).points.len(),
            0,
            "dim mismatch"
        );
    }

    #[test]
    fn map_ids_bounds_and_nearest() {
        let q = synthetic();
        let n = q.len() / 3;
        let m = Map::build(&q, 3, n, 7);
        assert_eq!(m.points.len(), 7);
        let ids: Vec<u32> = (0..7).map(|i| m.id(i)).collect();
        assert_eq!(ids, sample_ids(n, 7).map(|i| i as u32).collect::<Vec<_>>());
        // Built from the pre-sampled rows, the ids and points are the same.
        let rows: Vec<i8> = sample_ids(n, 7)
            .flat_map(|i| q[i * 3..i * 3 + 3].to_vec())
            .collect();
        let s = Map::build(&rows, 3, 7, 7).with_total(n);
        assert_eq!((0..7).map(|i| s.id(i)).collect::<Vec<_>>(), ids);
        assert_eq!(s.points, m.points);
        let ([x0, x1], _) = m.bounds();
        assert!(m.points.iter().all(|p| (x0..=x1).contains(&(p[0] as f64))));
        assert_eq!(nearest(&[], [0.0, 0.0]), None);
        let pts = [[0.0, 0.0], [5.0, 5.0], [1.0, 1.0]];
        assert_eq!(nearest(&pts, [0.9, 0.8]), Some(2));
        assert_eq!(nearest(&pts, [9.0, 9.0]), Some(1));
        assert_eq!(nearest(&[[1.0, 1.0], [1.0, 1.0]], [0.0, 0.0]), Some(0));
    }
}
