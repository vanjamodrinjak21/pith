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
/// Plain `http` is accepted only to loopback and only when `base_url` itself is loopback (tests);
/// redirects may never downgrade from https to http, and from a non-loopback origin they may not
/// point at loopback, private, link-local or unspecified IP literals. (Hostnames that resolve to
/// private addresses are not caught; the pinned origin is Hugging Face over TLS.)
pub fn pull_files(dir: &Path, base_url: &str, files: &[ModelFile<'_>]) -> Result<()> {
    for f in files {
        check_local_name(f.local)?;
    }
    let origin_local = check_url(base_url, false, true)?;
    create_private_dir(dir)?;
    let _lock = acquire_lock(&dir.join(LOCK_NAME))?;
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

/// Checks a URL and returns whether its host is loopback. https is always allowed except that,
/// unless `local_ok` (the caller's own base URL was loopback), the host may not be a loopback,
/// private, link-local or unspecified IP literal. http is allowed only to loopback and only when
/// `local_ok`. Userinfo is never allowed.
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
    let host = match authority.strip_prefix('[') {
        Some(v6) => v6.split(']').next().unwrap_or(""),
        None => authority.split(':').next().unwrap_or(""),
    };
    if host.is_empty() {
        return Err(bad("no host"));
    }
    let loopback = is_loopback(host);
    if !local_ok && (loopback || is_private(host)) {
        return Err(bad("local or private addresses are not allowed"));
    }
    match scheme {
        "https" => Ok(loopback),
        "http" if local_ok && loopback => Ok(true),
        "http" => Err(bad(
            "plain http is only allowed to loopback, from a loopback base URL",
        )),
        _ => Err(bad("only https is allowed")),
    }
}

fn is_loopback(host: &str) -> bool {
    host.eq_ignore_ascii_case("localhost")
        || host.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback())
}

fn is_private(host: &str) -> bool {
    match host.parse::<IpAddr>() {
        Ok(IpAddr::V4(ip)) => ip.is_private() || ip.is_link_local() || ip.is_unspecified(),
        Ok(IpAddr::V6(ip)) => {
            let s0 = ip.segments()[0];
            ip.is_unspecified() || s0 & 0xfe00 == 0xfc00 || s0 & 0xffc0 == 0xfe80
        }
        Err(_) => false,
    }
}

/// Opens (mode 0600) and exclusively locks the pull lock file, waiting up to `LOCK_TIMEOUT`.
fn acquire_lock(path: &Path) -> Result<fs::File> {
    let mut o = fs::OpenOptions::new();
    o.create(true).truncate(false).write(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut o, 0o600);
    let file = o.open(path)?;
    let start = Instant::now();
    loop {
        match file.try_lock() {
            Ok(()) => return Ok(file), // released on drop
            Err(fs::TryLockError::WouldBlock) if start.elapsed() < LOCK_TIMEOUT => {
                std::thread::sleep(Duration::from_millis(100))
            }
            Err(fs::TryLockError::WouldBlock) => {
                return Err(dl(format!(
                    "another `pith model pull` holds {} (waited {} s); retry when it finishes",
                    path.display(),
                    LOCK_TIMEOUT.as_secs()
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
        // production: https origin (not local)
        let local = check_url("https://huggingface.co/x", false, true).unwrap();
        assert!(!local);
        for to in [
            "http://127.0.0.1/m",
            "http://localhost:8080/m",
            "http://[::1]/m",
            "http://cdn.example/m",
            "https://127.0.0.1/m",
            "https://localhost/m",
            "https://[::1]/m",
            "https://10.0.0.5/m",
            "https://192.168.1.1/m",
            "https://172.16.0.1/m",
            "https://169.254.169.254/latest",
            "https://0.0.0.0/m",
            "https://[fd00::1]/m",
            "https://[fe80::1]/m",
        ] {
            let err = check_url(to, true, local).unwrap_err();
            assert!(err.to_string().contains("refusing redirect"), "{to}: {err}");
        }
        assert!(check_url("https://cdn.example/m", true, local).is_ok());
        assert!(check_url("https://8.8.8.8/m", true, local).is_ok());
        // loopback-http base (tests) may redirect within loopback, never to non-loopback http
        assert!(check_url("http://127.0.0.1:9/m", false, true).unwrap());
        assert!(check_url("http://127.0.0.1:9/m", true, true).is_ok());
        assert!(check_url("http://example.com/m", true, true).is_err());
        assert!(check_url("http://192.168.1.1/m", true, true).is_err());
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
