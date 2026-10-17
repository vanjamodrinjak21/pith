//! Secrets deny list. Pure path matching, so it runs before any file is opened.
//!
//! Content is not inspected here (see `walk.rs` for the PEM header check). That check is
//! header-only: raw base64 or DER keys without a header are not detected by content (DER
//! is binary and skipped as such). Unicode-confusable file names are not normalized: they
//! are different file names from the real secret names, so they are not a bypass.
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
/// Backup / temp / swap suffixes, stripped repeatedly by `normalize_name`
/// (`credentials.bak.old` is `credentials`); `credentials.rs` is source and stays indexed.
const BACKUP_SUFFIXES: [&str; 11] = [
    ".bak", ".old", ".orig", ".save", ".backup", ".tmp", ".swp", ".swo", ".swn", ".copy", ".dist",
];
const SWAP_SUFFIXES: [&str; 3] = [".swp", ".swo", ".swn"];
/// Source and document extensions: a secret-looking name part alone (`api.key.ts`,
/// `id_rsa_helper.ts`) does not deny these. Exact secret names still do.
const SOURCE_EXT: [&str; 18] = [
    "ts", "tsx", "js", "mjs", "rs", "py", "go", "swift", "kt", "java", "c", "h", "cpp", "md",
    "txt", "toml", "yaml", "yml",
];
/// `(directory, file)` pairs, matched on consecutive components.
const DENIED_PAIRS: [(&str, &str); 1] = [(".docker", "config.json")];
/// Directories denied wholesale: `.aws/config` and `.aws/sso/cache/*` hold tokens, and
/// `.kube/config.yaml` is a kubeconfig under a common name.
const DENIED_DIRS: [&str; 4] = [".ssh", ".gnupg", ".aws", ".kube"];
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
    parts.iter().any(|c| DENIED_DIRS.contains(&c.as_str()))
}

/// Strips editor and backup decorations until none are left: `#x#`, trailing `~`,
/// the backup suffixes, and all-digit suffixes of 1 to 14 digits (`credentials.1`,
/// `secrets.yaml.1697500000`, `credentials.20261017`). Numeric suffixes only matter when
/// what remains is an exact secret name, so `config.v2.json` and `config.v2` are untouched.
fn normalize_name(n: &str) -> String {
    let mut s = n;
    loop {
        if s.len() >= 2 && s.starts_with('#') && s.ends_with('#') {
            s = &s[1..s.len() - 1];
        } else if let Some(t) = s.strip_suffix('~') {
            s = t;
        } else if let Some(t) = BACKUP_SUFFIXES.iter().find_map(|x| s.strip_suffix(x)) {
            s = t;
        } else if let Some((t, d)) = s.rsplit_once('.') {
            if t.is_empty()
                || d.is_empty()
                || d.len() > 14
                || !d.bytes().all(|b| b.is_ascii_digit())
            {
                return s.to_owned();
            }
            s = t;
        } else {
            return s.to_owned();
        }
    }
}

/// Normalized forms of a name. A vim swap file `.x.swp` is the swap of `x`, but `.env.swp`
/// is also the swap of `.env`, so both readings are tried.
fn variants(n: &str) -> Vec<String> {
    let mut v = vec![normalize_name(n)];
    if n.starts_with('.') && SWAP_SUFFIXES.iter().any(|x| n.ends_with(x)) {
        v.push(normalize_name(&n[1..]));
    }
    v
}

/// An `env` segment after the first (`prod.env.yaml`) denies the file, except the
/// `.example` / `.sample` / `.template` forms (`app.env.example`).
fn env_segment_denied(segs: &[&str]) -> bool {
    let Some(i) = segs.iter().skip(1).position(|s| *s == "env").map(|i| i + 1) else {
        return false;
    };
    !(i + 2 == segs.len() && ["example", "sample", "template"].contains(&segs[i + 1]))
}

/// Rules on one normalized name.
fn name_denied(n: &str) -> bool {
    if SAFE_ENV.contains(&n) {
        return false;
    }
    if DENIED_NAMES.contains(&n)
        || ((n.starts_with("service-account") || n.starts_with("service_account"))
            && n.ends_with(".json"))
    {
        return true;
    }
    let segs: Vec<&str> = n.split('.').collect();
    let last = if segs.len() > 1 {
        segs[segs.len() - 1]
    } else {
        ""
    };
    // Only real key-pair names (`id_rsa.pub`) get the public-key allowance.
    let pubkey = n.ends_with(".pub") && KEY_PREFIXES.iter().any(|p| n.starts_with(p));
    // Never softened by a source extension: `.env.yaml`, `id_rsa.txt`, `x.tfvars.yaml`.
    if n.starts_with(".env.")
        || n.ends_with(".env")
        || DENIED_EXT.contains(&last)
        || env_segment_denied(&segs)
        || segs
            .iter()
            .skip(1)
            .any(|seg| *seg == "tfvars" || seg.starts_with("tfstate"))
        || (!pubkey
            && KEY_PREFIXES
                .iter()
                .any(|p| n == *p || n.starts_with(&format!("{p}."))))
    {
        return true;
    }
    // Heuristics a source or document extension softens: a key prefix inside a longer
    // identifier (`id_rsa_helper.ts`) and a key extension in a middle segment (`api.key.ts`).
    if SOURCE_EXT.contains(&last) {
        return false;
    }
    (!pubkey && KEY_PREFIXES.iter().any(|p| n.starts_with(p)))
        || (segs.len() > 2
            && segs[1..segs.len() - 1]
                .iter()
                .any(|s| DENIED_EXT.contains(s)))
}

/// True when `rel_path` (relative to the project root) must never be read or indexed.
/// `.env.example`-style templates and `.pub` keys stay allowed.
pub fn is_denied(rel_path: &Path) -> bool {
    let parts = lower_components(rel_path);
    if dir_denied(&parts) {
        return true;
    }
    if parts.windows(2).any(|w| {
        variants(&w[1])
            .iter()
            .any(|f| DENIED_PAIRS.iter().any(|(d, df)| w[0] == *d && f == df))
    }) {
        return true;
    }
    parts
        .last()
        .is_some_and(|n| variants(n).iter().any(|v| name_denied(v)))
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

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    const DECOR: [&str; 14] = [
        ".bak", ".old", ".orig", ".save", ".backup", ".tmp", ".swp", ".swo", ".swn", ".copy",
        ".dist", ".1", ".12", "~",
    ];
    const SECRET_NAMES: [&str; 14] = [
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
        "service-account.json",
        "service_account_x.json",
    ];
    const SOURCE_NAMES: [&str; 8] = [
        "credentials.rs",
        "env.ts",
        "api.key.ts",
        "id_rsa_helper.ts",
        "config.v2.json",
        "config.rs",
        "config.json.schema.ts",
        "secrets.rs",
    ];

    /// 0..=3 stacked decorations, then optionally one editor wrapper.
    fn decorate(name: &str, picks: &[usize], wrap: u8) -> String {
        let mut s = name.to_owned();
        for &i in picks {
            s.push_str(DECOR[i % DECOR.len()]);
        }
        match wrap % 4 {
            1 => format!(".{s}.swp"),
            2 => format!("#{s}#"),
            3 => format!("{s}~"),
            _ => s,
        }
    }

    proptest! {
        #[test]
        fn strong_rules_ignore_source_extensions(
            ni in 0usize..6,
            ei in 0usize..SOURCE_EXT.len(),
        ) {
            let base = [".env", ".env.local", "x.tfvars", "x.tfstate", "id_rsa", "id_ed25519"][ni];
            let f = format!("{base}.{}", SOURCE_EXT[ei]);
            prop_assert!(is_denied(Path::new(&f)), "{f}");
        }

        #[test]
        fn decorated_secret_names_are_denied(
            ni in 0usize..SECRET_NAMES.len(),
            picks in prop::collection::vec(0usize..DECOR.len(), 0..=3),
            wrap in 0u8..4,
        ) {
            let f = decorate(SECRET_NAMES[ni], &picks, wrap);
            prop_assert!(is_denied(Path::new(&f)), "{f}");
            prop_assert!(is_denied(&Path::new("sub").join(&f)), "sub/{f}");
        }

        #[test]
        fn decorated_pairs_are_denied(
            pi in 0usize..DENIED_PAIRS.len(),
            picks in prop::collection::vec(0usize..DECOR.len(), 0..=3),
            wrap in 0u8..4,
        ) {
            let (d, f) = DENIED_PAIRS[pi];
            let p = format!("{d}/{}", decorate(f, &picks, wrap));
            prop_assert!(is_denied(Path::new(&p)), "{p}");
            prop_assert!(is_denied(Path::new(&format!("x/{p}"))), "x/{p}");
        }
    }

    #[test]
    fn whole_aws_and_kube_dirs_are_denied() {
        for p in [
            ".aws/config",
            ".aws/sso/cache/x.json",
            ".kube/config.yaml",
            ".kube/config.rs",
        ] {
            assert!(is_denied(Path::new(p)), "{p}");
        }
        assert!(is_denied_dir(Path::new("a/.kube")));
    }

    #[test]
    fn known_source_names_stay_allowed() {
        for n in SOURCE_NAMES {
            assert!(!is_denied(Path::new(n)), "{n}");
            assert!(!is_denied(&Path::new(".docker").join(n)), ".docker/{n}");
        }
    }
}
