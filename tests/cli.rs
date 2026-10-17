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

// Stub contract; delete when Task 6 implements the subcommands.
#[test]
fn stub_stats_reports_not_implemented() {
    let out = Command::new(env!("CARGO_BIN_EXE_pith"))
        .arg("stats")
        .output()
        .expect("run pith");
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("not implemented"));
}
