use std::process::Command;

#[test]
fn version_prints_package_version() {
    let out = Command::new(env!("CARGO_BIN_EXE_pith"))
        .arg("--version")
        .output()
        .expect("run pith");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(stdout, format!("pith {}\n", env!("CARGO_PKG_VERSION")));
}

fn pith(dir: &std::path::Path) -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_pith"));
    c.current_dir(dir).env("PITH_HOME", dir.join("home"));
    c
}

#[test]
fn stats_before_and_after_index() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path().canonicalize().unwrap();
    std::fs::write(root.join("a.txt"), "hello world\n").unwrap();
    let out = pith(&root).arg("stats").output().unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("run `pith index`"));

    pithdb::index_project(&root, &pithdb::embed::FakeEmbedder, false).unwrap();
    let out = pith(&root).arg("stats").output().unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        text.contains("chunks:  1") && text.contains("files:   1") && text.contains("fake-384"),
        "{text}"
    );
}

#[test]
fn init_without_a_tty_does_not_download() {
    let d = tempfile::tempdir().unwrap();
    let out = pith(d.path())
        .arg("init")
        .stdin(std::process::Stdio::null())
        .output()
        .unwrap();
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("pith model pull"));
    assert!(
        !d.path().join("home").exists(),
        "nothing may be created or downloaded"
    );
}

#[test]
fn model_list_reports_missing_files_offline() {
    let d = tempfile::tempdir().unwrap();
    let out = pith(d.path()).args(["model", "list"]).output().unwrap();
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("missing"));
}

#[test]
fn search_modes_without_a_model() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path().canonicalize().unwrap();
    std::fs::write(root.join("a.rs"), "fn open_writer_lock() {}\n").unwrap();
    std::fs::write(root.join("b.rs"), "fn unrelated() {}\n").unwrap();
    pithdb::index_project(&root, &pithdb::embed::FakeEmbedder, false).unwrap();
    for mode in [None, Some("hybrid"), Some("keyword")] {
        let mut c = pith(&root);
        c.args(["search", "WriterLock"]);
        if let Some(m) = mode {
            c.args(["--mode", m]);
        }
        let out = c.output().unwrap();
        let (stdout, stderr) = (
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
        assert!(out.status.success(), "{mode:?}: {stderr}");
        assert!(
            stdout.starts_with("1. 1.00  a.rs:1-1"),
            "{mode:?}: {stdout}"
        );
        assert!(!stdout.contains("b.rs"), "{stdout}");
        assert_eq!(
            stderr.contains(pithdb::KEYWORD_ONLY),
            mode != Some("keyword"),
            "{mode:?}: {stderr}"
        );
    }
    let out = pith(&root)
        .args(["search", "x", "--mode", "vector"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("pith model pull") || stderr.contains("onnx"),
        "{stderr}"
    );
    let out = pith(&root)
        .args(["search", "x", "--mode", "fuzzy"])
        .output()
        .unwrap();
    assert!(!out.status.success());
}

#[test]
fn eval_reports_every_mode_and_names_bad_lines() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path().canonicalize().unwrap();
    std::fs::write(root.join("a.rs"), "fn open_writer_lock() {}\n").unwrap();
    std::fs::write(root.join("b.rs"), "fn unrelated() {}\n").unwrap();
    pithdb::index_project(&root, &pithdb::embed::FakeEmbedder, false).unwrap();
    std::fs::write(
        root.join("cases.jsonl"),
        "{\"query\": \"WriterLock\", \"expect\": [\"a.rs:1-1\"]}\n\
         {\"query\": \"nothing here\", \"expect\": [\"b.rs\"]}\n",
    )
    .unwrap();
    let out = pith(&root)
        .args(["eval", "cases.jsonl", "--mode", "keyword", "--verbose"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.starts_with("2 cases, k = 10\n"), "{stdout}");
    assert!(
        stdout.contains("keyword    0.50   0.50    0.50   0.50"),
        "{stdout}"
    );
    assert!(
        stdout.contains("miss [keyword] \"nothing here\": \n"),
        "{stdout}"
    );
    // Without a model, vector mode is an error.
    assert!(!pith(&root)
        .args(["eval", "cases.jsonl"])
        .output()
        .unwrap()
        .status
        .success());

    std::fs::write(root.join("bad.jsonl"), "\n{\"query\": 1}\n").unwrap();
    let out = pith(&root).args(["eval", "bad.jsonl"]).output().unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("line 2:"));
}
