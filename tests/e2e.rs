use pithdb::embed::{Embedder, FakeEmbedder};
use pithdb::index::VectorIndex;
use pithdb::storage::{Db, IndexMeta, StoredChunk};
use pithdb::{index_project, open_project, search_project, stats_project, Error, Result};
use std::fs;
use std::path::Path;
use std::process::Command;

fn project() -> tempfile::TempDir {
    let d = tempfile::tempdir().unwrap();
    fs::write(
        d.path().join("a.rs"),
        "fn alpha() {\n    println!(\"alpha\");\n}\n",
    )
    .unwrap();
    fs::write(
        d.path().join("b.rs"),
        "fn beta() {\n    println!(\"beta\");\n}\n",
    )
    .unwrap();
    fs::write(d.path().join("c.md"), "# gamma\nsome notes about gamma\n").unwrap();
    d
}

fn open(root: &Path) -> (std::path::PathBuf, Db) {
    open_project(root).unwrap()
}

struct Other(FakeEmbedder);
impl Embedder for Other {
    fn model_id(&self) -> &str {
        "other-384"
    }
    fn dim(&self) -> usize {
        384
    }
    fn embed_passages(&self, t: &[String]) -> Result<Vec<Vec<f32>>> {
        self.0.embed_passages(t)
    }
    fn embed_query(&self, t: &str) -> Result<Vec<f32>> {
        self.0.embed_query(t)
    }
}

#[test]
fn index_then_search_finds_the_matching_chunk() {
    let d = project();
    let r = index_project(d.path(), &FakeEmbedder, false).unwrap();
    assert_eq!((r.chunks, r.files), (3, 3));
    let (root, db) = open(d.path());
    let want = (0..3u32)
        .map(|i| db.chunk(i).unwrap())
        .find(|c| c.path == "b.rs")
        .unwrap();
    let hits = search_project(&root, &db, &FakeEmbedder, &want.text, 2).unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].path, "b.rs");
    assert!(hits[0].score > 0.99);
    assert_eq!(
        (hits[0].start_line, hits[0].text.as_str()),
        (want.start_line, want.text.as_str())
    );
    assert!(!hits[0].source_changed);
}

#[test]
fn search_before_index_errors_clearly() {
    let d = tempfile::tempdir().unwrap();
    let e = open_project(d.path()).err().unwrap();
    assert!(matches!(e, Error::NoIndex(_)));
    assert!(e.to_string().contains("run `pith index`"));
}

#[test]
fn reindex_required_when_model_id_differs() {
    let d = project();
    index_project(d.path(), &FakeEmbedder, false).unwrap();
    let e = index_project(d.path(), &Other(FakeEmbedder), false).unwrap_err();
    assert!(matches!(e, Error::ModelMismatch { .. }));
    assert!(e.to_string().contains("pith index --reindex"));
    index_project(d.path(), &Other(FakeEmbedder), true).unwrap();
    assert_eq!(open(d.path()).1.model_id(), "other-384");
    // Searching with the wrong embedder is refused too.
    let (root, db) = open(d.path());
    assert!(search_project(&root, &db, &FakeEmbedder, "x", 3).is_err());
}

#[test]
fn corrupt_index_requires_reindex_and_is_not_deleted() {
    let d = project();
    fs::write(d.path().join(".pith"), vec![7u8; 4096]).unwrap();
    let e = index_project(d.path(), &FakeEmbedder, false).unwrap_err();
    assert!(e.to_string().contains("--reindex"), "{e}");
    assert_eq!(fs::read(d.path().join(".pith")).unwrap(), vec![7u8; 4096]);
    index_project(d.path(), &FakeEmbedder, true).unwrap();
}

#[test]
fn empty_project_indexes_zero_chunks_and_search_is_empty() {
    let d = tempfile::tempdir().unwrap();
    let r = index_project(d.path(), &FakeEmbedder, false).unwrap();
    assert_eq!((r.chunks, r.files), (0, 0));
    let (root, db) = open(d.path());
    assert!(search_project(&root, &db, &FakeEmbedder, "anything", 5)
        .unwrap()
        .is_empty());
}

#[test]
fn source_changed_flag_true_after_file_edit() {
    let d = project();
    index_project(d.path(), &FakeEmbedder, false).unwrap();
    let (root, db) = open(d.path());
    let q = db.chunk(0).unwrap().text.clone();
    assert!(search_project(&root, &db, &FakeEmbedder, &q, 3)
        .unwrap()
        .iter()
        .all(|h| !h.source_changed));
    std::thread::sleep(std::time::Duration::from_millis(1100));
    fs::write(d.path().join("a.rs"), "fn alpha() { changed }\n").unwrap();
    fs::remove_file(d.path().join("c.md")).unwrap();
    let hits = search_project(&root, &db, &FakeEmbedder, &q, 3).unwrap();
    for h in &hits {
        assert_eq!(h.source_changed, h.path != "b.rs", "{}", h.path);
    }
}

#[cfg(unix)]
#[test]
fn symlinked_source_file_is_flagged_changed() {
    let d = project();
    index_project(d.path(), &FakeEmbedder, false).unwrap();
    let (root, db) = open(d.path());
    fs::remove_file(d.path().join("b.rs")).unwrap();
    std::os::unix::fs::symlink("/etc/hosts", d.path().join("b.rs")).unwrap();
    let q = db.chunk(1).unwrap().text.clone();
    let hits = search_project(&root, &db, &FakeEmbedder, &q, 3).unwrap();
    assert!(
        hits.iter()
            .find(|h| h.path == "b.rs")
            .unwrap()
            .source_changed
    );
}

fn git(dir: &Path, args: &[&str]) {
    let ok = Command::new("git")
        .args(args)
        .current_dir(dir)
        .output()
        .unwrap()
        .status
        .success();
    assert!(ok, "git {args:?}");
}

#[test]
fn index_sets_git_head_only_in_git_repos() {
    // The non-git half is always asserted; only the git half needs the git binary.
    let plain = project();
    index_project(plain.path(), &FakeEmbedder, false).unwrap();
    assert_eq!(open(plain.path()).1.meta().git_head, None);
    if Command::new("git").arg("--version").output().is_err() {
        eprintln!("git half of index_sets_git_head_only_in_git_repos skipped: git not found");
        return;
    }

    let repo = project();
    git(repo.path(), &["init", "-q"]);
    git(repo.path(), &["add", "."]);
    git(
        repo.path(),
        &[
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "-m",
            "x",
        ],
    );
    index_project(repo.path(), &FakeEmbedder, false).unwrap();
    let head = open(repo.path()).1.meta().git_head.clone().unwrap();
    assert!(matches!(head.len(), 40 | 64) && head.bytes().all(|b| b.is_ascii_hexdigit()));

    // A repo without commits has no HEAD: None, not an error.
    let empty = project();
    git(empty.path(), &["init", "-q"]);
    index_project(empty.path(), &FakeEmbedder, false).unwrap();
    assert_eq!(open(empty.path()).1.meta().git_head, None);
}

#[test]
fn stats_reports_chunks_files_age() {
    let d = project();
    index_project(d.path(), &FakeEmbedder, false).unwrap();
    let s = {
        let (r, db) = open(d.path());
        stats_project(&r, &db).unwrap()
    };
    assert_eq!((s.chunks, s.files), (3, 3));
    assert_eq!(s.model_id, "fake-384");
    assert!(s.age_secs < 60 && s.db_bytes > 0);
}

#[test]
fn gitignore_is_appended_only_when_present_and_never_rewritten() {
    let d = project();
    index_project(d.path(), &FakeEmbedder, false).unwrap();
    assert!(!d.path().join(".gitignore").exists());

    fs::write(d.path().join(".gitignore"), "target/\n.pith").unwrap(); // no trailing newline
    index_project(d.path(), &FakeEmbedder, false).unwrap();
    assert_eq!(
        fs::read_to_string(d.path().join(".gitignore")).unwrap(),
        "target/\n.pith\n.pith.lock\n"
    );
    index_project(d.path(), &FakeEmbedder, false).unwrap();
    assert_eq!(
        fs::read_to_string(d.path().join(".gitignore")).unwrap(),
        "target/\n.pith\n.pith.lock\n"
    );
}

#[test]
fn secret_file_never_appears_in_results() {
    let d = project();
    fs::write(d.path().join(".env"), "API_KEY=supersecretvalue123\n").unwrap();
    let r = index_project(d.path(), &FakeEmbedder, false).unwrap();
    assert!(r.skipped.iter().any(|(p, _)| p == ".env"));
    let (root, db) = open(d.path());
    let hits = search_project(
        &root,
        &db,
        &FakeEmbedder,
        "API_KEY=supersecretvalue123\n",
        100,
    )
    .unwrap();
    assert!(hits
        .iter()
        .all(|h| h.path != ".env" && !h.text.contains("supersecret")));
}

#[test]
fn k_is_capped_at_100_and_zero_is_an_error() {
    let d = tempfile::tempdir().unwrap();
    for i in 0..150 {
        fs::write(
            d.path().join(format!("f{i}.txt")),
            format!("file number {i}\n"),
        )
        .unwrap();
    }
    assert_eq!(
        index_project(d.path(), &FakeEmbedder, false)
            .unwrap()
            .chunks,
        150
    );
    let (root, db) = open(d.path());
    assert_eq!(
        search_project(&root, &db, &FakeEmbedder, "x", usize::MAX)
            .unwrap()
            .len(),
        100
    );
    assert_eq!(
        search_project(&root, &db, &FakeEmbedder, "x", 7)
            .unwrap()
            .len(),
        7
    );
    let e = search_project(&root, &db, &FakeEmbedder, "x", 0).unwrap_err();
    assert!(matches!(e, Error::Invalid(ref m) if m.contains("k must be at least 1")));
}

#[test]
fn empty_query_is_rejected() {
    let d = project();
    index_project(d.path(), &FakeEmbedder, false).unwrap();
    let (root, db) = open(d.path());
    for q in ["", "   \n\t"] {
        let e = search_project(&root, &db, &FakeEmbedder, q, 3).unwrap_err();
        assert!(matches!(e, Error::Invalid(ref m) if m.contains("query must not be empty")));
    }
}

#[cfg(unix)]
#[test]
fn symlinked_gitignore_is_left_untouched_and_index_succeeds() {
    let d = project();
    let outside = tempfile::tempdir().unwrap();
    let target = outside.path().join("shared-ignore");
    fs::write(&target, "target/\n").unwrap();
    std::os::unix::fs::symlink(&target, d.path().join(".gitignore")).unwrap();
    index_project(d.path(), &FakeEmbedder, false).unwrap();
    assert_eq!(fs::read_to_string(&target).unwrap(), "target/\n");
    assert!(d
        .path()
        .join(".gitignore")
        .symlink_metadata()
        .unwrap()
        .file_type()
        .is_symlink());
}

#[test]
fn second_writer_fails_fast_with_locked() {
    let d = project();
    let _held = pithdb::storage::WriterLock::acquire(&d.path().join(".pith")).unwrap();
    let e = index_project(d.path(), &FakeEmbedder, false).unwrap_err();
    assert!(matches!(e, Error::Locked));
}

#[cfg(unix)]
#[test]
fn unreadable_existing_index_reports_the_io_error_not_corruption() {
    use std::os::unix::fs::PermissionsExt;
    if rustix::process::geteuid().is_root() {
        eprintln!("skipping unreadable_existing_index test: running as root ignores mode bits");
        return;
    }
    let d = project();
    index_project(d.path(), &FakeEmbedder, false).unwrap();
    let p = d.path().join(".pith");
    fs::set_permissions(&p, fs::Permissions::from_mode(0o000)).unwrap();
    let r = index_project(d.path(), &FakeEmbedder, false);
    fs::set_permissions(&p, fs::Permissions::from_mode(0o600)).unwrap();
    let e = r.unwrap_err();
    assert!(matches!(e, Error::Io(_)), "{e}");
    assert!(!e.to_string().contains("--reindex"), "{e}");
}

fn tampered_db(root: &Path, model: &str, head: &str) {
    let idx = VectorIndex::build(vec![vec![1.0; 384]]).unwrap();
    let c = StoredChunk {
        path: "a\x1b[2Jb.rs".into(),
        start_line: 1,
        end_line: 1,
        text: "t".into(),
    };
    let meta = IndexMeta {
        indexed_at_unix: 1,
        git_head: Some(head.into()),
    };
    Db::write(&root.join(".pith"), model, vec![c], &idx, meta).unwrap();
}

#[test]
fn escape_sequences_from_a_tampered_db_never_reach_output() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path().canonicalize().unwrap();
    tampered_db(&root, "m\x1b[31m-384", "\x1b]0;pwn\x07");
    let out = Command::new(env!("CARGO_BIN_EXE_pith"))
        .arg("stats")
        .current_dir(&root)
        .output()
        .unwrap();
    assert!(out.status.success());
    assert!(!out.stdout.contains(&0x1b) && !out.stderr.contains(&0x1b));
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        text.contains("(invalid)") && text.contains("m?[31m-384"),
        "{text}"
    );

    let e = index_project(&root, &FakeEmbedder, false).unwrap_err();
    assert!(matches!(e, Error::ModelMismatch { .. }));
    assert!(!e.to_string().contains('\x1b'), "{e}");

    let (_, db) = open(&root);
    let hits = vec![pithdb::SearchHit {
        chunk_id: 0,
        score: 1.0,
        path: db.chunk(0).unwrap().path.clone(),
        start_line: 1,
        end_line: 1,
        text: "\x1b[0m".into(),
        source_changed: true,
    }];
    assert!(!pithdb::render_search(&hits, 1, 2).contains('\x1b'));
}

#[cfg(feature = "onnx")]
#[test]
#[ignore = "needs a pulled model: set PITH_HOME and run `pith model pull`"]
fn real_model_smoke() {
    let embedder = pithdb::load_default_embedder().unwrap();
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let d = tempfile::tempdir().unwrap();
    let dst = d.path().join("src");
    fn copy(a: &Path, b: &Path) {
        fs::create_dir_all(b).unwrap();
        for e in fs::read_dir(a).unwrap() {
            let e = e.unwrap();
            let t = b.join(e.file_name());
            if e.path().is_dir() {
                copy(&e.path(), &t)
            } else {
                fs::copy(e.path(), t).unwrap();
            }
        }
    }
    copy(&src, &dst);
    let t = std::time::Instant::now();
    let r = index_project(d.path(), embedder.as_ref(), false).unwrap();
    let secs = t.elapsed().as_secs_f64();
    eprintln!(
        "indexed {} chunks / {} files in {secs:.1}s = {:.1} chunks/s",
        r.chunks,
        r.files,
        r.chunks as f64 / secs
    );
    let (root, db) = open(d.path());
    for h in search_project(&root, &db, embedder.as_ref(), "mmap vector index", 5).unwrap() {
        eprintln!("{:.2} {}:{}-{}", h.score, h.path, h.start_line, h.end_line);
    }
}

#[test]
fn file_as_root_keeps_the_not_a_directory_error() {
    let d = project();
    for reindex in [false, true] {
        let e = index_project(&d.path().join("a.rs"), &FakeEmbedder, reindex).unwrap_err();
        assert!(
            matches!(e, Error::Invalid(ref m) if m.contains("is not a directory")),
            "{e}"
        );
    }
}

#[test]
fn lock_file_is_gitignored_even_when_indexing_fails() {
    let d = project();
    tampered_db(d.path(), "other-model", &"a".repeat(40));
    fs::write(d.path().join(".gitignore"), "target/\n").unwrap();
    let e = index_project(d.path(), &FakeEmbedder, false).unwrap_err();
    assert!(matches!(e, Error::ModelMismatch { .. }));
    let g = fs::read_to_string(d.path().join(".gitignore")).unwrap();
    assert!(g.lines().any(|l| l == ".pith.lock"), "{g}");
}

#[cfg(unix)]
#[test]
fn error_messages_with_escape_bytes_in_paths_are_scrubbed() {
    let d = tempfile::tempdir().unwrap();
    let evil = d.path().canonicalize().unwrap().join("x\x1b[2Jy");
    fs::create_dir(&evil).unwrap();
    for args in [
        vec!["stats"],
        vec!["search", "q"],
        vec!["index", "nonexistent\x1b[2J"],
    ] {
        let out = Command::new(env!("CARGO_BIN_EXE_pith"))
            .args(&args)
            .current_dir(&evil)
            .output()
            .unwrap();
        assert!(!out.status.success(), "{args:?}");
        assert!(
            !out.stdout.contains(&0x1b) && !out.stderr.contains(&0x1b),
            "{args:?}"
        );
    }
    let out = Command::new(env!("CARGO_BIN_EXE_pith"))
        .arg("stats")
        .current_dir(&evil)
        .output()
        .unwrap();
    assert!(String::from_utf8_lossy(&out.stderr).contains("x?[2Jy"));
}
