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
