use std::process::Command;

#[test]
fn version_prints_package_version() {
    let out = Command::new(env!("CARGO_BIN_EXE_pith"))
        .arg("--version")
        .output()
        .expect("run pith");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains(env!("CARGO_PKG_VERSION")), "got {stdout}");
}
