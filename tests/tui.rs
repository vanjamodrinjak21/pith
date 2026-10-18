#![cfg(feature = "tui")]

use std::process::{Command, Stdio};

#[test]
fn ui_without_a_terminal_fails_cleanly() {
    let d = tempfile::tempdir().unwrap();
    std::fs::write(d.path().join("a.txt"), "hello\n").unwrap();
    pithdb::index_project(d.path(), &pithdb::embed::FakeEmbedder, false).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_pith"))
        .args(["ui", "--project"])
        .arg(d.path())
        .env("PITH_HOME", d.path().join("home"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(out.stdout.is_empty(), "no escape codes on a pipe");
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        err,
        "pith: invalid input: pith ui needs an interactive terminal\n"
    );
}
