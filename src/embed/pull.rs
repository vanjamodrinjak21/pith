//! Verified model download. The only code in Pith that touches the network.

use super::manifest::{ModelFile, DEFAULT_MODEL};
use crate::{Error, Result};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{Read, Write};
use std::net::IpAddr;
use std::path::Path;
use std::time::{Duration, Instant};

const TEMP_PREFIX: &str = ".pull-";
const LOCK_NAME: &str = ".pull.lock";
const MAX_REDIRECTS: usize = 5;
const LOCK_TIMEOUT: Duration = Duration::from_secs(5 * 60);

/// Downloads the default model into `models_dir/<model id>` (see [`pull_files`]).
pub fn pull(models_dir: &Path, base_url: &str) -> Result<()> {
    pull_files(
        &models_dir.join(DEFAULT_MODEL.id),
        base_url,
        &DEFAULT_MODEL.files,
    )
}

/// Downloads `files` from `base_url/<remote>` into `dir`. Each file is streamed into a fresh
/// temp file in `dir`, capped at its manifest size, sha256-verified and fsynced; only after every
/// file verified are they renamed into place, one rename per file (each rename is atomic, the set
/// is not; every file on disk is always a complete verified one). On any failure before the
/// renames the temp files are removed and the existing files are untouched. Files already present
/// with the right hash are not downloaded. Concurrent pulls into one dir serialize on a lock file.
/// URL policy (see [`check_url`]): a loopback base URL (tests) may use http and its redirects must
/// stay on loopback. Any other base URL and every redirect from it must be https to a DNS name
/// whose last label is letters only, which rules out every IP-literal form (dotted, decimal, hex,
/// short, IPv6, IPv4-mapped) and `localhost`/`.localhost`/`.local`/`.internal`/`.localdomain`.
/// Not covered: a public DNS name that resolves to a private address (no DNS lookups here, to
/// avoid rebinding races); TLS certificate validation is the backstop.
pub fn pull_files(dir: &Path, base_url: &str, files: &[ModelFile<'_>]) -> Result<()> {
    for f in files {
        check_local_name(f.local)?;
    }
    let origin_local = check_url(base_url, false, true)?;
    create_private_dir(dir)?;
    let _lock = acquire_lock(&dir.join(LOCK_NAME), LOCK_TIMEOUT)?;
    // Under the lock no other pull is live, so every temp file left here is from a crash.
    for e in fs::read_dir(dir)?.flatten() {
        if e.file_name().to_string_lossy().starts_with(TEMP_PREFIX) {
            let _ = fs::remove_file(e.path());
        }
    }
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .max_redirects(0)
        .max_redirects_will_error(false)
        .http_status_as_error(false)
        .timeout_connect(Some(Duration::from_secs(30)))
        .timeout_recv_response(Some(Duration::from_secs(60)))
        // A stalled body cannot hang forever: 35 MB in 15 min needs only ~40 KB/s.
        .timeout_global(Some(Duration::from_secs(15 * 60)))
        .user_agent(concat!("pith/", env!("CARGO_PKG_VERSION")))
        .build()
        .into();
    let mut staged = Vec::new();
    for f in files {
        let dest = dir.join(f.local);
        if is_verified(&dest, f)? {
            continue;
        }
        let url = format!("{}/{}", base_url.trim_end_matches('/'), f.remote);
        staged.push((download(&agent, &url, origin_local, dir, f)?, dest));
    }
    for (tmp, dest) in staged {
        tmp.persist(&dest).map_err(|e| e.error)?;
    }
    fs::File::open(dir)?.sync_all()?;
    Ok(())
}

fn download(
    agent: &ureq::Agent,
    url: &str,
    origin_local: bool,
    dir: &Path,
    f: &ModelFile<'_>,
) -> Result<tempfile::NamedTempFile> {
    let mut url = url.to_string();
    let mut hops = 0;
    let resp = loop {
        let resp = agent
            .get(&url)
            .call()
            .map_err(|e| dl(format!("{}: {e}", redact(&url))))?;
        let status = resp.status();
        if status.is_redirection() {
            hops += 1;
            if hops > MAX_REDIRECTS {
                return Err(dl(format!("{}: too many redirects", f.remote)));
            }
            let loc = resp
                .headers()
                .get("location")
                .and_then(|v| v.to_str().ok())
                .ok_or_else(|| {
                    dl(format!(
                        "{}: redirect without a valid Location",
                        redact(&url)
                    ))
                })?;
            url = join(&url, loc)?;
            check_url(&url, true, origin_local)?;
            continue;
        }
        if !status.is_success() {
            return Err(dl(format!("{}: HTTP {status}", redact(&url))));
        }
        break resp;
    };
    if let Some(len) = resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
    {
        if len.trim().parse::<u64>().ok() != Some(f.size) {
            return Err(dl(format!(
                "{}: server reports {len} bytes, manifest says {}",
                f.remote, f.size
            )));
        }
    }
    let mut tmp = tempfile::Builder::new()
        .prefix(TEMP_PREFIX)
        .tempfile_in(dir)?;
    let mut body = resp.into_body().into_reader().take(f.size + 1);
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024];
    let mut total = 0u64;
    loop {
        let n = body
            .read(&mut buf)
            .map_err(|e| dl(format!("{}: {e}", f.remote)))?;
        if n == 0 {
            break;
        }
        total += n as u64;
        if total > f.size {
            return Err(dl(format!(
                "{}: larger than the manifest size {}",
                f.remote, f.size
            )));
        }
        hasher.update(&buf[..n]);
        tmp.write_all(&buf[..n])?;
    }
    if total != f.size {
        return Err(dl(format!(
            "{}: got {total} bytes, expected {}",
            f.remote, f.size
        )));
    }
    let got = hex(&hasher.finalize());
    if got != f.sha256 {
        return Err(Error::Integrity(format!(
            "{}: sha256 {got}, expected {}",
            f.remote, f.sha256
        )));
    }
    tmp.as_file().sync_all()?;
    Ok(tmp)
}

/// True if `path` exists with the manifest size and hash.
fn is_verified(path: &Path, f: &ModelFile<'_>) -> Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(m) if m.file_type().is_file() && m.len() == f.size => {
            Ok(sha256_hex(&fs::read(path)?) == f.sha256)
        }
        Ok(_) => Ok(false),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e.into()),
    }
}

/// Creates `dir` (and missing parents) with mode 0700. An existing dir keeps its mode; a
/// symlink at `dir` is refused.
fn create_private_dir(dir: &Path) -> Result<()> {
    match fs::symlink_metadata(dir) {
        Ok(m) if m.file_type().is_dir() => return Ok(()),
        Ok(_) => {
            return Err(Error::Invalid(format!(
                "{} exists and is not a directory (symlinks are refused)",
                dir.display()
            )))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.into()),
    }
    let mut b = fs::DirBuilder::new();
    b.recursive(true);
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut b, 0o700); // applies only to dirs created here
    b.create(dir)?;
    if !fs::symlink_metadata(dir)?.file_type().is_dir() {
        return Err(Error::Invalid(format!(
            "{} is not a directory",
            dir.display()
        )));
    }
    Ok(())
}

/// A manifest file name must be a single plain path component.
fn check_local_name(name: &str) -> Result<()> {
    let ok = !name.is_empty()
        && name != "."
        && name != ".."
        && !name.contains(['/', '\\', ':', '\0'])
        && !name.starts_with(TEMP_PREFIX)
        && name != LOCK_NAME;
    if ok {
        Ok(())
    } else {
        Err(Error::Invalid(format!("bad model file name {name:?}")))
    }
}

pub(super) fn sha256_hex(data: &[u8]) -> String {
    hex(&Sha256::digest(data))
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn dl(msg: String) -> Error {
    Error::Download(msg)
}

/// URL without query or fragment (CDN redirects carry signatures there).
fn redact(url: &str) -> &str {
    url.split(['?', '#']).next().unwrap_or(url)
}

fn split(url: &str) -> Option<(&str, &str)> {
    let (scheme, rest) = url.split_once("://")?;
    Some((scheme, rest.split(['/', '?', '#']).next().unwrap_or("")))
}

/// Checks a URL; returns whether its host is loopback. With `local_ok` (base URL check, or a
/// redirect from a loopback base) a loopback host (`localhost`, 127.0.0.0/8 dotted quad, `::1`)
/// is accepted over http or https. Otherwise the URL must be https to a plain DNS name (see
/// [`is_public_dns_name`]); a redirect from a loopback base may not leave loopback.
fn check_url(url: &str, redirect: bool, local_ok: bool) -> Result<bool> {
    let what = if redirect {
        "refusing redirect to"
    } else {
        "refusing URL"
    };
    let bad = |why: &str| dl(format!("{what} {}: {why}", redact(url)));
    let (scheme, authority) = split(url).ok_or_else(|| bad("not an absolute URL"))?;
    if authority.contains('@') {
        return Err(bad("credentials in URLs are not supported"));
    }
    let scheme = scheme.to_ascii_lowercase();
    if scheme != "https" && scheme != "http" {
        return Err(bad("only https is allowed"));
    }
    let host = match authority.strip_prefix('[') {
        Some(v6) => format!("[{}]", v6.split(']').next().unwrap_or("")),
        None => authority.split(':').next().unwrap_or("").to_string(),
    }
    .to_ascii_lowercase();
    let host = host.strip_suffix('.').unwrap_or(&host);
    if local_ok && is_loopback(host) {
        return Ok(true);
    }
    if redirect && local_ok {
        return Err(bad(
            "redirects from a loopback base URL must stay on loopback",
        ));
    }
    if scheme != "https" {
        return Err(bad(
            "plain http is only allowed to loopback, from a loopback base URL",
        ));
    }
    if !is_public_dns_name(host) {
        return Err(bad(
            "host must be a public DNS name (no IP literals or local names)",
        ));
    }
    Ok(false)
}

fn is_loopback(host: &str) -> bool {
    let inner = host.strip_prefix('[').and_then(|h| h.strip_suffix(']'));
    host == "localhost"
        || inner
            .unwrap_or(host)
            .parse::<IpAddr>()
            .is_ok_and(|ip| ip.is_loopback())
}

/// Lowercased host (one trailing dot stripped): LDH labels, last label letters only, and not a
/// local-only name. Shape-based on purpose: no IP literal of any spelling can pass.
fn is_public_dns_name(host: &str) -> bool {
    let labels: Vec<&str> = host.split('.').collect();
    let ldh = labels.iter().all(|l| {
        !l.is_empty()
            && l.len() <= 63
            && l.bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
    });
    let last = labels.last().copied().unwrap_or("");
    let local = ["localhost", "local", "internal", "localdomain"].contains(&last);
    ldh && labels.len() >= 2 && last.bytes().all(|b| b.is_ascii_lowercase()) && !local
}

/// Opens and exclusively locks the pull lock file, waiting up to `timeout`. A new lock file gets
/// mode 0600 (the mode applies only on creation; an existing file keeps its mode). A symlink at
/// the lock path is refused (O_NOFOLLOW).
fn acquire_lock(path: &Path, timeout: Duration) -> Result<fs::File> {
    let mut o = fs::OpenOptions::new();
    o.create(true).truncate(false).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.mode(0o600)
            .custom_flags(rustix::fs::OFlags::NOFOLLOW.bits() as i32);
    }
    let file = o
        .open(path)
        .map_err(|e| dl(format!("cannot open lock file {} ({e})", path.display())))?;
    let start = Instant::now();
    loop {
        match file.try_lock() {
            Ok(()) => return Ok(file), // released on drop
            Err(fs::TryLockError::WouldBlock) if start.elapsed() < timeout => {
                std::thread::sleep(Duration::from_millis(100))
            }
            Err(fs::TryLockError::WouldBlock) => {
                return Err(dl(format!(
                    "another `pith model pull` holds {} (waited {:.1} s); retry when it finishes",
                    path.display(),
                    timeout.as_secs_f32()
                )))
            }
            Err(fs::TryLockError::Error(e)) => {
                return Err(dl(format!(
                    "cannot lock {} ({e}); the filesystem may not support file locks, set \
                     PITH_HOME to a local directory",
                    path.display()
                )))
            }
        }
    }
}

/// Resolves a Location header against the current URL (absolute, scheme-relative, or
/// path-absolute).
fn join(base: &str, loc: &str) -> Result<String> {
    let (scheme, authority) = split(base).ok_or_else(|| dl("bad base URL".into()))?;
    if loc.contains("://") {
        Ok(loc.to_string())
    } else if let Some(rest) = loc.strip_prefix("//") {
        Ok(format!("{scheme}://{rest}"))
    } else if loc.starts_with('/') {
        Ok(format!("{scheme}://{authority}{loc}"))
    } else {
        Err(dl(format!(
            "unsupported relative redirect from {}",
            redact(base)
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redirect_policy() {
        // production: https origin; check_url(.., true, origin_local) is exactly what download()
        // runs on every hop
        let local = check_url("https://huggingface.co/x", false, true).unwrap();
        assert!(!local);
        assert!(!check_url("HTTPS://HUGGINGFACE.CO/x", false, true).unwrap());
        for to in [
            "http://cdn.example.com/m",
            "http://127.0.0.1/m",
            "https://127.0.0.1/m",
            "https://2130706433/m",
            "https://0x7f000001/m",
            "https://127.1/m",
            "https://0/m",
            "https://0.0.0.0/m",
            "https://100.64.0.1/m",
            "https://localhost/m",
            "https://localhost./m",
            "https://LOCALHOST/m",
            "https://foo.localhost/m",
            "https://printer.local/m",
            "https://metadata.google.internal/m",
            "https://box.localdomain/m",
            "https://[::1]/m",
            "https://[::ffff:127.0.0.1]/m",
            "https://[::ffff:10.0.0.1]/m",
            "https://10.0.0.1/m",
            "https://192.168.1.1/m",
            "https://172.16.0.1/m",
            "https://169.254.169.254/latest",
            "https://[fd00::1]/m",
            "https://8.8.8.8/m",
            "https://single/m",
        ] {
            let err = check_url(to, true, local).unwrap_err();
            assert!(err.to_string().contains("refusing redirect"), "{to}: {err}");
        }
        for ok in [
            "https://cas-bridge.xethub.hf.co/xet-bridge-us/abc?sig=1",
            "https://us.aws.cdn.hf.co/x",
            "https://cdn-lfs.huggingface.co./x",
            "HTTPS://CDN.EXAMPLE.COM/m",
        ] {
            assert!(check_url(ok, true, local).is_ok(), "{ok}");
        }
        // loopback base (tests): may use http, redirects must stay on loopback
        assert!(check_url("http://127.0.0.1:9/m", false, true).unwrap());
        assert!(check_url("http://127.0.0.1:9/m", true, true).is_ok());
        assert!(check_url("http://[::1]:9/m", true, true).is_ok());
        assert!(check_url("http://example.com/m", true, true).is_err());
        assert!(check_url("https://example.com/m", true, true).is_err());
        assert!(check_url("http://192.168.1.1/m", true, true).is_err());
        // plain http base off loopback is refused outright
        assert!(check_url("http://example.com/m", false, true).is_err());
    }

    #[test]
    fn lock_times_out_with_a_clear_error() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join(LOCK_NAME);
        let _held = acquire_lock(&p, Duration::ZERO).unwrap();
        let t = Instant::now();
        let err = acquire_lock(&p, Duration::from_millis(300)).unwrap_err();
        assert!(t.elapsed() >= Duration::from_millis(300));
        let msg = err.to_string();
        assert!(
            msg.contains(".pull.lock") && msg.contains("another"),
            "{msg}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn lock_keeps_existing_mode_and_refuses_symlink() {
        use std::os::unix::fs::PermissionsExt;
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join(LOCK_NAME);
        fs::write(&p, b"").unwrap();
        fs::set_permissions(&p, fs::Permissions::from_mode(0o644)).unwrap();
        drop(acquire_lock(&p, Duration::ZERO).unwrap());
        assert_eq!(
            fs::metadata(&p).unwrap().permissions().mode() & 0o777,
            0o644
        );
        let link = d.path().join("link.lock");
        std::os::unix::fs::symlink(&p, &link).unwrap();
        let err = acquire_lock(&link, Duration::ZERO).unwrap_err();
        assert!(err.to_string().contains("cannot open lock file"), "{err}");
    }

    #[test]
    fn redirect_joins() {
        let b = "https://huggingface.co/a/resolve/x/f?q=1";
        assert_eq!(
            join(b, "https://cdn.hf.co/p?s=2").unwrap(),
            "https://cdn.hf.co/p?s=2"
        );
        assert_eq!(join(b, "//cdn.hf.co/p").unwrap(), "https://cdn.hf.co/p");
        assert_eq!(join(b, "/api/p").unwrap(), "https://huggingface.co/api/p");
        assert_eq!(
            join("http://127.0.0.1:5/x", "/y").unwrap(),
            "http://127.0.0.1:5/y"
        );
        assert!(join(b, "relative/p").is_err());
        // a scheme-relative join keeps https, so it can never become a downgrade
        let j = join(b, "//cdn.hf.co/p").unwrap();
        assert!(check_url(&j, true, false).is_ok());
        assert!(check_url(&join(b, "//127.0.0.1/p").unwrap(), true, false).is_err());
        assert!(check_url(&join(b, "/p").unwrap(), true, false).is_ok());
    }

    #[test]
    fn local_names_must_be_plain() {
        for bad in [
            "",
            ".",
            "..",
            "a/b",
            "../x",
            "/etc/passwd",
            "a\\b",
            "c:model",
            ".pull-x",
            ".pull.lock",
        ] {
            assert!(check_local_name(bad).is_err(), "{bad:?}");
        }
        assert!(check_local_name("tokenizer.json").is_ok());
    }
}
