//! Verified model download. The only code in Pith that touches the network.

use super::manifest::{ModelFile, DEFAULT_MODEL};
use crate::{Error, Result};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{Read, Write};
use std::net::Ipv4Addr;
use std::path::Path;
use std::time::Duration;

const TEMP_PREFIX: &str = ".pull-";
const MAX_REDIRECTS: usize = 5;

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
/// file verified are they renamed into place. On any failure the temp files are removed and the
/// existing files are untouched. Files already present with the right hash are not downloaded.
/// Plain `http` is accepted only for loopback hosts (tests); redirects must stay on https.
pub fn pull_files(dir: &Path, base_url: &str, files: &[ModelFile<'_>]) -> Result<()> {
    check_url(base_url, false)?;
    create_private_dir(dir)?;
    // shortcut: a concurrent pull into the same dir loses its temp files and fails cleanly; no lock.
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
        staged.push((download(&agent, &url, dir, f)?, dest));
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
            check_url(&url, true)?;
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
    match fs::metadata(path) {
        Ok(m) if m.len() == f.size => Ok(sha256_hex(&fs::read(path)?) == f.sha256),
        Ok(_) => Ok(false),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e.into()),
    }
}

fn create_private_dir(dir: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
        fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(dir)?;
        fs::set_permissions(dir, fs::Permissions::from_mode(0o700))?;
    }
    #[cfg(not(unix))]
    fs::create_dir_all(dir)?;
    Ok(())
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

/// https anywhere; http only to loopback; no userinfo.
fn check_url(url: &str, redirect: bool) -> Result<()> {
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
    match scheme {
        "https" if !authority.is_empty() => Ok(()),
        "http" => {
            let host = match authority.strip_prefix('[') {
                Some(v6) => v6.split(']').next().unwrap_or(""),
                None => authority.split(':').next().unwrap_or(""),
            };
            let loopback = host == "localhost"
                || host == "::1"
                || host.parse::<Ipv4Addr>().is_ok_and(|ip| ip.is_loopback());
            if loopback {
                Ok(())
            } else {
                Err(bad("plain http is only allowed to loopback"))
            }
        }
        _ => Err(bad("only https is allowed")),
    }
}

/// Resolves a Location header against the current URL (absolute, scheme-relative, or path-absolute).
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
