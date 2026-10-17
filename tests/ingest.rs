use pithdb::ingest::{collect, is_denied, CollectReport, SkipReason, MAX_FILE_BYTES};
use std::fs;
use std::path::Path;

fn write(root: &Path, rel: &str, body: impl AsRef<[u8]>) {
    let p = root.join(rel);
    fs::create_dir_all(p.parent().unwrap()).unwrap();
    fs::write(p, body).unwrap();
}

fn paths(r: &CollectReport) -> Vec<&str> {
    let mut v: Vec<&str> = r.chunks.iter().map(|c| c.path.as_str()).collect();
    v.dedup();
    v
}

const DENIED: &[&str] = &[
    ".env",
    ".env.local",
    ".env.production",
    "a/.ENV",
    "server.pem",
    "x/tls.KEY",
    "c.p12",
    "c.pfx",
    "r.jks",
    "release.keystore",
    "db.kdbx",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "putty.ppk",
    ".npmrc",
    ".pypirc",
    ".netrc",
    "credentials",
    "credentials.json",
    "secrets.json",
    "secrets.yaml",
    "secrets.yml",
    "service-account.json",
    "service-account-prod.json",
    ".aws/credentials",
    ".ssh/config",
    ".ssh/deep/known_hosts",
    ".gnupg/pubring.kbx",
    "Sub/.SSH/x",
];

#[test]
fn denied_files_never_returned_even_if_not_gitignored() {
    let d = tempfile::tempdir().unwrap();
    for p in DENIED {
        write(d.path(), p, "SECRET=1\n");
        assert!(is_denied(Path::new(p)), "{p} should be denied");
    }
    write(d.path(), "ok.rs", "fn main() {}\n");
    let r = collect(d.path()).unwrap();
    assert_eq!(paths(&r), ["ok.rs"]);
}

#[test]
fn allowed_lookalikes_are_indexed() {
    let d = tempfile::tempdir().unwrap();
    let ok = [
        ".env.example",
        ".env.sample",
        ".env.template",
        "id_rsa.pub",
        "src/credentials/mod.rs",
        "keys.md",
    ];
    for p in ok {
        write(d.path(), p, "KEY=\n");
        assert!(!is_denied(Path::new(p)), "{p} should be allowed");
    }
    let r = collect(d.path()).unwrap();
    let mut want = ok.to_vec();
    want.sort();
    assert_eq!(paths(&r), want);
}

#[cfg(unix)]
#[test]
fn deny_check_runs_before_any_read() {
    use std::os::unix::fs::PermissionsExt;
    let d = tempfile::tempdir().unwrap();
    write(d.path(), ".env", "S=1\n");
    write(d.path(), ".ssh/id", "k\n");
    write(d.path(), "ok.txt", "hi\n");
    fs::set_permissions(d.path().join(".env"), fs::Permissions::from_mode(0o000)).unwrap();
    let r = collect(d.path()).unwrap();
    assert_eq!(paths(&r), ["ok.txt"]);
    assert!(
        r.skipped.contains(&(".env".into(), SkipReason::Denied)),
        "{:?}",
        r.skipped
    );
}

#[cfg(unix)]
#[test]
fn unreadable_file_is_skipped_not_fatal() {
    use std::os::unix::fs::PermissionsExt;
    let d = tempfile::tempdir().unwrap();
    write(d.path(), "locked.txt", "x\n");
    write(d.path(), "ok.txt", "hi\n");
    fs::set_permissions(
        d.path().join("locked.txt"),
        fs::Permissions::from_mode(0o000),
    )
    .unwrap();
    let r = collect(d.path()).unwrap();
    assert_eq!(paths(&r), ["ok.txt"]);
    assert!(matches!(r.skipped.as_slice(), [(p, SkipReason::Unreadable(_))] if p == "locked.txt"));
}

#[test]
fn gitignore_and_pithignore_respected() {
    let d = tempfile::tempdir().unwrap();
    write(d.path(), ".gitignore", "ignored.txt\nbuild/\n");
    write(d.path(), ".pithignore", "private/\n*.gen.rs\n");
    write(d.path(), "ignored.txt", "x\n");
    write(d.path(), "build/o.txt", "x\n");
    write(d.path(), "private/p.txt", "x\n");
    write(d.path(), "a.gen.rs", "x\n");
    write(d.path(), "keep.txt", "x\n");
    let r = collect(d.path()).unwrap();
    assert_eq!(paths(&r), [".gitignore", ".pithignore", "keep.txt"]);
}

#[test]
fn works_without_git_repo_and_skips_dot_git() {
    let d = tempfile::tempdir().unwrap();
    write(d.path(), ".gitignore", "*.log\n");
    write(d.path(), "a.log", "x\n");
    write(d.path(), "b.txt", "x\n");
    write(d.path(), ".git/config", "x\n");
    write(d.path(), ".git/HEAD", "x\n");
    let r = collect(d.path()).unwrap();
    assert_eq!(paths(&r), [".gitignore", "b.txt"]);
}

#[test]
fn global_gitignore_is_not_used() {
    let d = tempfile::tempdir().unwrap();
    let home = tempfile::tempdir().unwrap();
    write(home.path(), ".gitignore_global", "global.txt\n");
    write(
        home.path(),
        ".gitconfig",
        format!(
            "[core]\n\texcludesfile = {}/.gitignore_global\n",
            home.path().display()
        ),
    );
    write(d.path(), "global.txt", "x\n");
    // Hermetic: point git at the fake home for this process.
    std::env::set_var("HOME", home.path());
    std::env::set_var("XDG_CONFIG_HOME", home.path());
    let r = collect(d.path()).unwrap();
    assert_eq!(paths(&r), ["global.txt"]);
}

#[test]
fn database_files_are_never_indexed() {
    let d = tempfile::tempdir().unwrap();
    for p in [".pith", ".pith.lock", ".pith.tmpAbC123", ".pith.tmp"] {
        write(d.path(), p, "data\n");
    }
    write(d.path(), "real.txt", "x\n");
    let r = collect(d.path()).unwrap();
    assert_eq!(paths(&r), ["real.txt"]);
}

#[test]
fn binary_and_invalid_utf8_skipped() {
    let d = tempfile::tempdir().unwrap();
    write(d.path(), "bin.dat", b"abc\0def");
    write(
        d.path(),
        "late_nul.txt",
        [vec![b'a'; 9000], vec![0u8]].concat(),
    ); // NUL after 8 KiB: not sniffed as binary
    write(d.path(), "bad.txt", [0x66u8, 0x6f, 0xff, 0xfe, 0x0a]);
    write(d.path(), "empty.txt", "");
    write(d.path(), "good.txt", "héllo\n");
    let r = collect(d.path()).unwrap();
    assert!(r.skipped.contains(&("bin.dat".into(), SkipReason::Binary)));
    assert!(r.skipped.contains(&("bad.txt".into(), SkipReason::NotUtf8)));
    assert!(r.skipped.contains(&("empty.txt".into(), SkipReason::Empty)));
    assert!(paths(&r).contains(&"good.txt"));
    assert!(!paths(&r).contains(&"bin.dat") && !paths(&r).contains(&"bad.txt"));
}

#[test]
fn oversized_files_skipped() {
    let d = tempfile::tempdir().unwrap();
    write(
        d.path(),
        "at_limit.txt",
        vec![b'a'; MAX_FILE_BYTES as usize],
    );
    write(
        d.path(),
        "over.txt",
        vec![b'a'; MAX_FILE_BYTES as usize + 1],
    );
    let r = collect(d.path()).unwrap();
    assert_eq!(paths(&r), ["at_limit.txt"]);
    assert!(r
        .skipped
        .contains(&("over.txt".into(), SkipReason::TooLarge)));
}

#[cfg(unix)]
#[test]
fn symlink_loop_does_not_hang_and_links_are_not_followed() {
    use std::os::unix::fs::symlink;
    let d = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    write(outside.path(), "secret.txt", "outside\n");
    write(d.path(), "real.txt", "x\n");
    write(d.path(), "sub/inner.txt", "y\n");
    symlink(d.path(), d.path().join("sub/loop")).unwrap();
    symlink(outside.path(), d.path().join("out_dir")).unwrap();
    symlink(
        outside.path().join("secret.txt"),
        d.path().join("out_file.txt"),
    )
    .unwrap();
    let r = collect(d.path()).unwrap();
    assert_eq!(paths(&r), ["real.txt", "sub/inner.txt"]);
    let links: Vec<_> = r
        .skipped
        .iter()
        .filter(|(_, s)| *s == SkipReason::Symlink)
        .collect();
    assert_eq!(links.len(), 3, "{:?}", r.skipped);
}

#[test]
fn window_line_ranges_are_contiguous_with_overlap() {
    let d = tempfile::tempdir().unwrap();
    let body: String = (1..=150).map(|i| format!("line {i}\n")).collect();
    write(d.path(), "f.txt", &body);
    let r = collect(d.path()).unwrap();
    let ranges: Vec<(u32, u32)> = r
        .chunks
        .iter()
        .map(|c| (c.start_line, c.end_line))
        .collect();
    assert_eq!(ranges, [(1, 60), (51, 110), (101, 150)]);
    assert!(r.chunks[1].text.starts_with("line 51\n"));
}

#[test]
fn single_huge_line_is_split_to_char_limit() {
    let d = tempfile::tempdir().unwrap();
    let line = "é".repeat(5_000);
    write(d.path(), "min.js", &line);
    let r = collect(d.path()).unwrap();
    assert!(r.chunks.len() >= 4);
    for c in &r.chunks {
        assert!(c.text.chars().count() <= 1_400);
        assert_eq!((c.start_line, c.end_line), (1, 1));
        assert!(!c.text.contains('\n'));
    }
}

#[test]
fn empty_dir_yields_no_chunks() {
    let d = tempfile::tempdir().unwrap();
    let r = collect(d.path()).unwrap();
    assert!(r.chunks.is_empty() && r.skipped.is_empty());
}

#[test]
fn missing_root_is_an_error() {
    assert!(collect(Path::new("/definitely/not/here")).is_err());
}

#[test]
fn output_is_sorted_and_paths_are_relative_slashed() {
    let d = tempfile::tempdir().unwrap();
    for p in ["z.txt", "b/c.txt", "a.txt", "b/a.txt"] {
        write(d.path(), p, "x\n");
    }
    let r = collect(d.path()).unwrap();
    assert_eq!(paths(&r), ["a.txt", "b/a.txt", "b/c.txt", "z.txt"]);
}
