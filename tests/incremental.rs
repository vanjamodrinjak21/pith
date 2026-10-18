//! Incremental `pith index`: unchanged chunks reuse their vectors, and the result equals a full
//! reindex.

use pithdb::embed::{Embedder, FakeEmbedder};
use pithdb::storage::Db;
use pithdb::{index_project, index_project_with_progress, Error, IndexReport, Result};
use std::cell::RefCell;
use std::fs;
use std::path::Path;

/// FakeEmbedder that records every passage it embeds.
#[derive(Default)]
struct Counting(RefCell<Vec<String>>);
impl Embedder for Counting {
    fn model_id(&self) -> &str {
        FakeEmbedder.model_id()
    }
    fn dim(&self) -> usize {
        FakeEmbedder.dim()
    }
    fn embed_passages(&self, t: &[String]) -> Result<Vec<Vec<f32>>> {
        self.0.borrow_mut().extend_from_slice(t);
        FakeEmbedder.embed_passages(t)
    }
    fn embed_query(&self, t: &str) -> Result<Vec<f32>> {
        FakeEmbedder.embed_query(t)
    }
}

impl Counting {
    fn index(&self, root: &Path) -> IndexReport {
        self.0.borrow_mut().clear();
        index_project(root, self, false).unwrap()
    }
    fn embedded(&self) -> Vec<String> {
        self.0.borrow().clone()
    }
}

/// `n` numbered lines, so a file of 200 lines makes several chunks.
fn lines(tag: &str, n: usize) -> String {
    (0..n).map(|i| format!("{tag} line {i}\n")).collect()
}

fn project() -> tempfile::TempDir {
    let d = tempfile::tempdir().unwrap();
    fs::write(d.path().join("a.rs"), lines("alpha", 200)).unwrap();
    fs::write(d.path().join("b.rs"), lines("beta", 30)).unwrap();
    fs::create_dir(d.path().join("sub")).unwrap();
    fs::write(d.path().join("sub/c.md"), lines("gamma", 5)).unwrap();
    d
}

/// Chunks, f32 vectors and int8 vectors of the index under `root`.
type Content = (Vec<pithdb::storage::StoredChunk>, Vec<f32>, Vec<i8>);
fn content(root: &Path) -> Content {
    let db = Db::open(&root.join(".pith"), None).unwrap();
    let i = db.index();
    (
        db.chunks().to_vec(),
        i.full().to_vec(),
        i.quantized().to_vec(),
    )
}

/// The incremental index equals a from-scratch one (which then replaces it).
fn assert_equals_full_reindex(root: &Path) {
    let inc = content(root);
    index_project(root, &FakeEmbedder, true).unwrap();
    assert!(
        inc == content(root),
        "incremental index differs from a full reindex"
    );
}

#[test]
fn editing_one_chunk_embeds_only_that_chunk() {
    let d = project();
    let e = Counting::default();
    let first = e.index(d.path());
    assert_eq!((first.reused, first.embedded), (0, first.chunks));
    assert_eq!(e.embedded().len(), first.chunks);

    // b.rs fits one chunk: changing a line re-embeds exactly that chunk.
    fs::write(
        d.path().join("b.rs"),
        lines("beta", 30).replace("line 7", "LINE 7"),
    )
    .unwrap();
    let r = e.index(d.path());
    let want = Db::open(&d.path().join(".pith"), None)
        .unwrap()
        .chunks()
        .iter()
        .filter(|c| c.path == "b.rs")
        .map(|c| c.text.clone())
        .collect::<Vec<_>>();
    assert_eq!(e.embedded(), want);
    assert_eq!(
        (r.chunks, r.reused, r.embedded),
        (first.chunks, first.chunks - 1, 1)
    );
    assert!(!r.up_to_date);
    assert!(r
        .summary(0.5)
        .contains(&format!("({} reused, 1 embedded)", r.reused)));
    assert_equals_full_reindex(d.path());
}

#[test]
fn add_delete_rename_and_duplicate() {
    let d = project();
    let e = Counting::default();
    let first = e.index(d.path());

    // Rename and duplicate: same text, so nothing is embedded.
    fs::rename(d.path().join("b.rs"), d.path().join("renamed.rs")).unwrap();
    fs::copy(d.path().join("sub/c.md"), d.path().join("copy.md")).unwrap();
    let r = e.index(d.path());
    assert_eq!((r.embedded, r.files), (0, 4));
    assert!(
        !r.up_to_date,
        "paths changed, so the file must be rewritten"
    );
    assert_equals_full_reindex(d.path());

    // Delete: the chunks vanish, nothing embedded.
    fs::remove_file(d.path().join("a.rs")).unwrap();
    let r = e.index(d.path());
    assert_eq!(r.embedded, 0);
    assert!(r.chunks < first.chunks);
    assert!(content(d.path()).0.iter().all(|c| c.path != "a.rs"));
    assert_equals_full_reindex(d.path());

    // Add: only the new file is embedded.
    fs::write(d.path().join("new.rs"), "fn brand_new() {}\n").unwrap();
    let r = e.index(d.path());
    assert_eq!(e.embedded(), ["fn brand_new() {}"]);
    assert_eq!(r.embedded, 1);
    assert_equals_full_reindex(d.path());
}

#[test]
fn unchanged_project_is_up_to_date_and_not_rewritten() {
    let d = project();
    let e = Counting::default();
    e.index(d.path());
    let db = d.path().join(".pith");
    let (before, mtime) = (
        fs::read(&db).unwrap(),
        fs::metadata(&db).unwrap().modified().unwrap(),
    );
    std::thread::sleep(std::time::Duration::from_millis(1100));
    let r = e.index(d.path());
    assert!(r.up_to_date);
    assert_eq!((r.embedded, r.reused), (0, r.chunks));
    assert!(e.embedded().is_empty());
    assert!(
        r.summary(0.1).starts_with("index is up to date"),
        "{}",
        r.summary(0.1)
    );
    assert_eq!(fs::read(&db).unwrap(), before);
    assert_eq!(fs::metadata(&db).unwrap().modified().unwrap(), mtime);
}

#[test]
fn reindex_embeds_everything() {
    let d = project();
    let e = Counting::default();
    let first = e.index(d.path());
    e.0.borrow_mut().clear();
    let r = index_project(d.path(), &e, true).unwrap();
    assert_eq!(
        (r.embedded, r.reused, r.up_to_date),
        (first.chunks, 0, false)
    );
    assert_eq!(e.embedded().len(), first.chunks);
}

#[test]
fn progress_counts_reused_chunks_and_cancel_keeps_old_index() {
    let d = project();
    index_project(d.path(), &FakeEmbedder, false).unwrap();
    let db = d.path().join(".pith");
    let before = fs::read(&db).unwrap();
    fs::write(d.path().join("b.rs"), "changed\n").unwrap();

    // Cancel after the first group.
    let mut calls = 0;
    let r = index_project_with_progress(d.path(), &FakeEmbedder, false, &mut |_| {
        calls += 1;
        calls < 2
    });
    assert!(matches!(r, Err(Error::Cancelled)), "{r:?}");
    assert_eq!(fs::read(&db).unwrap(), before);

    // Uncancelled: every chunk, reused or not, is reported, and the last call is complete.
    let (mut vectors_seen, mut last) = (0, (0, 0));
    let r = index_project_with_progress(d.path(), &FakeEmbedder, false, &mut |p| {
        vectors_seen += p.vectors.len();
        last = (p.chunks_done, p.chunks);
        true
    })
    .unwrap();
    assert_eq!(vectors_seen, r.chunks);
    assert_eq!(last, (r.chunks, r.chunks));
    assert!(r.reused > 0 && r.embedded == 1);
}

#[test]
fn old_format_file_asks_for_reindex() {
    let d = project();
    index_project(d.path(), &FakeEmbedder, false).unwrap();
    let db = d.path().join(".pith");
    let mut b = fs::read(&db).unwrap();
    b[8..12].copy_from_slice(&1u32.to_le_bytes());
    fs::write(&db, &b).unwrap();
    let e = index_project(d.path(), &FakeEmbedder, false).unwrap_err();
    assert!(matches!(e, Error::UnsupportedFormat(1)), "{e:?}");
    assert_eq!(
        e.to_string(),
        "unsupported database format version 1; run `pith index --reindex`"
    );
    assert_eq!(fs::read(&db).unwrap(), b, "old file must be left alone");
    let r = index_project(d.path(), &FakeEmbedder, true).unwrap();
    assert_eq!(r.reused, 0);
    assert!(Db::open(&db, None).is_ok());
}

#[test]
fn randomized_edits_equal_full_reindex() {
    let d = tempfile::tempdir().unwrap();
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut rand = |n: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % n as u64) as usize
    };
    let e = Counting::default();
    for round in 0..12 {
        for _ in 0..3 {
            let name = d.path().join(format!("f{}.rs", rand(6)));
            match rand(4) {
                0 => {
                    let _ = fs::remove_file(&name);
                }
                1 => {
                    let to = d.path().join(format!("f{}.rs", rand(6)));
                    let _ = fs::rename(&name, to);
                }
                _ => {
                    let mut text = fs::read_to_string(&name).unwrap_or_default();
                    let at = rand(text.len() + 1);
                    let at = (0..=at).rev().find(|&i| text.is_char_boundary(i)).unwrap();
                    text.insert_str(at, &lines(&format!("r{round}"), 1 + rand(80)));
                    fs::write(&name, text).unwrap();
                }
            }
        }
        e.index(d.path());
        assert_equals_full_reindex(d.path());
    }
}
