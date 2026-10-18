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

#[cfg(unix)]
fn is_root() -> bool {
    rustix::process::geteuid().is_root()
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
    "x/.aws/credentials",
    "server.PEM",
    "A/B.Key",
    "trail.pem.",
    "id_ed25519 ",
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
    if is_root() {
        // libtest captures this for passing tests; run with --nocapture to see it.
        eprintln!("SKIPPED (running as root): unreadable_file_is_skipped_not_fatal");
        return;
    }
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

/// Child half of `global_gitignore_is_not_used`; only runs when the parent spawns it.
#[test]
#[ignore = "spawned by global_gitignore_is_not_used"]
fn global_gitignore_child() {
    let root = std::env::var("PITH_TEST_ROOT").expect("run via the parent test");
    let r = collect(Path::new(&root)).unwrap();
    assert_eq!(paths(&r), ["global.txt"]);
}

#[test]
fn global_gitignore_is_not_used() {
    let d = tempfile::tempdir().unwrap();
    let home = tempfile::tempdir().unwrap();
    write(home.path(), ".gitignore_global", "global.txt\n");
    let cfg = format!(
        "[core]\n\texcludesfile = {}/.gitignore_global\n",
        home.path().display()
    );
    write(home.path(), ".gitconfig", cfg);
    write(d.path(), "global.txt", "x\n");
    // A child process gets the fake HOME; this process's environment is never mutated.
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--ignored", "--exact", "global_gitignore_child"])
        .env("HOME", home.path())
        .env("XDG_CONFIG_HOME", home.path())
        .env("PITH_TEST_ROOT", d.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success() && stdout.contains("1 passed"),
        "{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
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
    // A NUL after the first 8 KiB is not sniffed: the file is valid UTF-8 and IS indexed.
    assert!(paths(&r).contains(&"late_nul.txt"));
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

#[test]
fn deny_table() {
    let denied = [
        "id_rsa.bak",
        "id_ed25519_old",
        "id_ecdsa-work",
        "server.pem.bak",
        "key.pfx.old",
        "prod.env",
        ".envrc",
        ".git-credentials",
        ".docker/config.json",
        ".kube/config",
        "terraform.tfstate",
        "terraform.tfstate.backup",
        "prod.tfvars",
        "a.tfvars.json",
        ".htpasswd",
        "service_account.json",
        "service_account_prod.json",
        "secrets.json.bak",
        "credentials.bak",
        "credentials.json.old",
        ".npmrc.bak",
        "deep/dir/.NETRC",
        ".env.production.local",
    ];
    let allowed = [
        ".env.example",
        ".env.sample",
        ".env.template",
        "id_rsa.pub",
        "id_ed25519.pub",
        "credentials.rs",
        "credentials_test.go",
        "src/credentials/mod.rs",
        "env.rs",
        "environment.ts",
        "keys.md",
        "README.md",
        ".gitignore",
        "src/ssh/mod.rs",
        "monkey.rs",
        "turkey.txt",
        "secrets.rs",
        "secrets_test.py",
        "service_accounts.rs",
        ".docker/Dockerfile",
        "config",
        "docs/aws/credentials.md",
        "src/main.rs",
    ];
    for p in denied {
        assert!(is_denied(Path::new(p)), "{p} should be denied");
    }
    for p in allowed {
        assert!(!is_denied(Path::new(p)), "{p} should be allowed");
    }
    // And end to end, for the ones that need a real tree.
    let d = tempfile::tempdir().unwrap();
    for p in denied.iter().chain(allowed.iter()) {
        write(d.path(), p, "x\n");
    }
    let r = collect(d.path()).unwrap();
    let got: Vec<&str> = paths(&r);
    let mut want = allowed.to_vec();
    want.sort();
    assert_eq!(got, want);
}

#[test]
fn private_key_content_is_skipped_whatever_the_name() {
    let d = tempfile::tempdir().unwrap();
    for (i, kind) in ["RSA", "EC", "OPENSSH", "ENCRYPTED", ""].iter().enumerate() {
        let hdr = format!("-----BEGIN {kind} PRIVATE KEY-----").replace("  ", " ");
        write(
            d.path(),
            &format!("notes{i}.txt"),
            format!("see below\n{hdr}\nMIIE\n"),
        );
    }
    write(
        d.path(),
        "public.txt",
        format!("-----BEGIN {}-----\nMIIB\n", "PUBLIC KEY"),
    );
    write(
        d.path(),
        "prose.txt",
        "the words PRIVATE KEY----- alone are fine\n",
    );
    let r = collect(d.path()).unwrap();
    assert_eq!(paths(&r), ["prose.txt", "public.txt"]);
    let secret = r
        .skipped
        .iter()
        .filter(|(_, s)| *s == SkipReason::Secret)
        .count();
    assert_eq!(secret, 5, "{:?}", r.skipped);
}

#[test]
fn root_inside_secrets_directory_is_refused() {
    use pithdb::Error;
    let d = tempfile::tempdir().unwrap();
    for dir in [".ssh", ".gnupg", ".aws", "home/.SSH/sub"] {
        let root = d.path().join(dir);
        fs::create_dir_all(&root).unwrap();
        write(&root, "f.txt", "x\n");
        assert!(matches!(collect(&root), Err(Error::Invalid(_))), "{dir}");
    }
}

#[test]
fn crlf_no_trailing_newline_and_blank_files() {
    let d = tempfile::tempdir().unwrap();
    write(d.path(), "crlf.txt", "a\r\nb\r\n");
    write(d.path(), "nonl.txt", "a\nb");
    write(d.path(), "blank.txt", "\n\n  \n\t\n");
    let r = collect(d.path()).unwrap();
    let get = |p: &str| r.chunks.iter().filter(|c| c.path == p).collect::<Vec<_>>();
    assert_eq!(get("crlf.txt").len(), 1);
    assert_eq!(get("crlf.txt")[0].text, "a\nb");
    assert_eq!(
        (get("crlf.txt")[0].start_line, get("crlf.txt")[0].end_line),
        (1, 2)
    );
    assert_eq!(get("nonl.txt")[0].text, "a\nb");
    assert!(get("blank.txt").is_empty());
    assert!(r.skipped.contains(&("blank.txt".into(), SkipReason::Empty)));
}

#[cfg(unix)]
#[test]
fn fifo_is_not_opened_and_is_recorded() {
    let d = tempfile::tempdir().unwrap();
    write(d.path(), "ok.txt", "hi\n");
    let fifo = d.path().join("pipe");
    assert!(std::process::Command::new("mkfifo")
        .arg(&fifo)
        .status()
        .unwrap()
        .success());
    let r = collect(d.path()).unwrap(); // would hang if the FIFO were opened for reading
    assert_eq!(paths(&r), ["ok.txt"]);
    assert!(
        r.skipped
            .contains(&("pipe".into(), SkipReason::SpecialFile)),
        "{:?}",
        r.skipped
    );
}

#[cfg(unix)]
#[test]
fn unreadable_directory_is_reported_with_its_path() {
    use std::os::unix::fs::PermissionsExt;
    if is_root() {
        // libtest captures this for passing tests; run with --nocapture to see it.
        eprintln!("SKIPPED (running as root): unreadable_directory_is_reported_with_its_path");
        return;
    }
    let d = tempfile::tempdir().unwrap();
    write(d.path(), "ok.txt", "hi\n");
    write(d.path(), "locked/inner.txt", "x\n");
    let locked = d.path().join("locked");
    fs::set_permissions(&locked, fs::Permissions::from_mode(0o000)).unwrap();
    let r = collect(d.path());
    fs::set_permissions(&locked, fs::Permissions::from_mode(0o755)).unwrap();
    let r = r.unwrap();
    assert_eq!(paths(&r), ["ok.txt"]);
    assert!(
        matches!(r.skipped.as_slice(), [(p, SkipReason::Unreadable(_))] if p == "locked"),
        "{:?}",
        r.skipped
    );
}

#[test]
fn hardlinked_pem_under_another_name_is_caught_by_content() {
    let d = tempfile::tempdir().unwrap();
    let hdr = format!("-----BEGIN {} PRIVATE KEY-----", "RSA");
    write(d.path(), "server.pem", format!("{hdr}\nMIIE\n"));
    fs::hard_link(d.path().join("server.pem"), d.path().join("notes.txt")).unwrap();
    let r = collect(d.path()).unwrap();
    assert!(r.chunks.is_empty(), "{:?}", r.chunks);
    assert!(
        r.skipped
            .contains(&("notes.txt".into(), SkipReason::Secret)),
        "{:?}",
        r.skipped
    );
    assert!(r
        .skipped
        .contains(&("server.pem".into(), SkipReason::Denied)));
}
