//! Secrets deny list. Pure path matching, so it runs before any file is opened.

use std::path::{Component, Path};

const SAFE_ENV: [&str; 3] = [".env.example", ".env.sample", ".env.template"];
const DENIED_EXT: [&str; 8] = ["pem", "key", "p12", "pfx", "jks", "keystore", "kdbx", "ppk"];
const DENIED_NAMES: [&str; 13] = [
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    ".npmrc",
    ".pypirc",
    ".netrc",
    "credentials",
    "credentials.json",
    "secrets.json",
    "secrets.yaml",
    "secrets.yml",
    ".env",
];

fn lower_components(p: &Path) -> Vec<String> {
    p.components()
        .filter_map(|c| match c {
            Component::Normal(s) => Some(s.to_string_lossy().to_lowercase()),
            _ => None,
        })
        .collect()
}

fn dir_denied(parts: &[String]) -> bool {
    parts.iter().any(|c| c == ".ssh" || c == ".gnupg")
}

/// True when `rel_path` (relative to the project root) must never be read or indexed.
/// Case-insensitive; `.env.example`-style templates and `.pub` keys stay allowed.
pub fn is_denied(rel_path: &Path) -> bool {
    let parts = lower_components(rel_path);
    if dir_denied(&parts) {
        return true;
    }
    if parts
        .windows(2)
        .any(|w| w[0] == ".aws" && w[1] == "credentials")
    {
        return true;
    }
    let Some(name) = parts.last() else {
        return false;
    };
    if SAFE_ENV.contains(&name.as_str()) {
        return false;
    }
    DENIED_NAMES.contains(&name.as_str())
        || name.starts_with(".env.")
        || (name.starts_with("service-account") && name.ends_with(".json"))
        || name
            .rsplit_once('.')
            .is_some_and(|(_, ext)| DENIED_EXT.contains(&ext))
}

/// Directory-only check (file-name rules must not hide a directory called `credentials`).
pub(super) fn is_denied_dir(rel_path: &Path) -> bool {
    dir_denied(&lower_components(rel_path))
}

/// The database and its lock / temp sidecars.
pub(super) fn is_db_file(name: &str) -> bool {
    let n = name.to_lowercase();
    n == ".pith" || n == ".pith.lock" || n.starts_with(".pith.tmp")
}
