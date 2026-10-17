//! Secrets deny list. Pure path matching, so it runs before any file is opened.
//!
//! Names are matched case-insensitively with trailing dots and spaces trimmed
//! (Windows ignores them, so `server.pem.` is `server.pem`).

use std::path::{Component, Path};

const SAFE_ENV: [&str; 3] = [".env.example", ".env.sample", ".env.template"];
/// Final extensions that are secret key material (`server.pem`, `tls.key`).
const DENIED_EXT: [&str; 8] = ["pem", "key", "p12", "pfx", "jks", "keystore", "kdbx", "ppk"];
const KEY_PREFIXES: [&str; 4] = ["id_rsa", "id_dsa", "id_ecdsa", "id_ed25519"];
const DENIED_NAMES: [&str; 12] = [
    ".npmrc",
    ".pypirc",
    ".netrc",
    ".git-credentials",
    ".htpasswd",
    ".envrc",
    "credentials",
    "credentials.json",
    "secrets.json",
    "secrets.yaml",
    "secrets.yml",
    ".env",
];
/// A denied name plus one of these is a backup of a secret (`credentials.bak`);
/// `credentials.rs` is source and stays indexed.
const BACKUP_SUFFIXES: [&str; 10] = [
    ".bak", ".old", ".orig", ".save", ".backup", ".tmp", ".swp", ".copy", ".dist", "~",
];
/// Final extensions of backups / temp copies; with a denied extension before them
/// (`server.pem.bak`, `key.pfx.old`) the file is denied.
const BACKUP_EXT: [&str; 9] = [
    "bak", "old", "orig", "save", "backup", "tmp", "swp", "copy", "dist",
];
/// Source and document extensions: a secret-looking name part alone (`api.key.ts`,
/// `id_rsa_helper.ts`) does not deny these. Exact secret names still do.
const SOURCE_EXT: [&str; 18] = [
    "ts", "tsx", "js", "mjs", "rs", "py", "go", "swift", "kt", "java", "c", "h", "cpp", "md",
    "txt", "toml", "yaml", "yml",
];
/// `(directory, file)` pairs, matched on consecutive components.
const DENIED_PAIRS: [(&str, &str); 3] = [
    (".aws", "credentials"),
    (".docker", "config.json"),
    (".kube", "config"),
];
/// A project root inside one of these directories is refused outright.
const SENSITIVE_ROOTS: [&str; 5] = [".ssh", ".gnupg", ".aws", ".docker", ".kube"];

fn norm(s: &std::ffi::OsStr) -> String {
    s.to_string_lossy()
        .to_lowercase()
        .trim_end_matches(['.', ' '])
        .to_owned()
}

fn lower_components(p: &Path) -> Vec<String> {
    p.components()
        .filter_map(|c| match c {
            Component::Normal(s) => Some(norm(s)),
            _ => None,
        })
        .collect()
}

fn dir_denied(parts: &[String]) -> bool {
    parts.iter().any(|c| c == ".ssh" || c == ".gnupg")
}

fn name_denied(n: &str) -> bool {
    if SAFE_ENV.contains(&n) {
        return false;
    }
    let core = BACKUP_SUFFIXES
        .iter()
        .find_map(|s| n.strip_suffix(s))
        .unwrap_or(n);
    if DENIED_NAMES.contains(&core)
        || ((core.starts_with("service-account") || core.starts_with("service_account"))
            && core.ends_with(".json"))
    {
        return true;
    }
    let trimmed = n.trim_end_matches('~');
    let segs: Vec<&str> = trimmed.split('.').collect();
    let last = if segs.len() > 1 {
        segs[segs.len() - 1]
    } else {
        ""
    };
    if SOURCE_EXT.contains(&last) {
        return false;
    }
    if KEY_PREFIXES.iter().any(|p| n.starts_with(p)) && !n.ends_with(".pub") {
        return true;
    }
    // Earlier segments only count when the name ends in a backup suffix.
    let earlier_denied = segs.len() > 2
        && segs[1..segs.len() - 1]
            .iter()
            .any(|s| DENIED_EXT.contains(s));
    n.starts_with(".env.")
        || n.ends_with(".env")
        || DENIED_EXT.contains(&last)
        || (BACKUP_EXT.contains(&last) && earlier_denied)
        || segs
            .iter()
            .skip(1)
            .any(|seg| *seg == "tfvars" || seg.starts_with("tfstate"))
}

/// True when `rel_path` (relative to the project root) must never be read or indexed.
/// `.env.example`-style templates and `.pub` keys stay allowed.
pub fn is_denied(rel_path: &Path) -> bool {
    let parts = lower_components(rel_path);
    if dir_denied(&parts) {
        return true;
    }
    if parts
        .windows(2)
        .any(|w| DENIED_PAIRS.iter().any(|(d, f)| w[0] == *d && w[1] == *f))
    {
        return true;
    }
    parts.last().is_some_and(|n| name_denied(n))
}

/// Directory-only check (file-name rules must not hide a directory called `credentials`).
pub(super) fn is_denied_dir(rel_path: &Path) -> bool {
    dir_denied(&lower_components(rel_path))
}

/// True when the (canonical) project root lies inside a secrets directory, where the
/// relative-path rules above cannot see the sensitive component.
pub(super) fn is_denied_root(canonical_root: &Path) -> bool {
    lower_components(canonical_root)
        .iter()
        .any(|c| SENSITIVE_ROOTS.contains(&c.as_str()))
}

/// The database and its lock / temp sidecars.
pub(super) fn is_db_file(name: &str) -> bool {
    let n = name.to_lowercase();
    n == ".pith" || n == ".pith.lock" || n.starts_with(".pith.tmp")
}
