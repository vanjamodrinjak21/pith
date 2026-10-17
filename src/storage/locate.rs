use std::path::{Path, PathBuf};

/// Nearest trusted `.pith` file for `start`, bounded by `$HOME`.
/// See [`locate_project_within`].
pub fn locate_project(start: &Path) -> Option<PathBuf> {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    locate_project_within(start, home.as_deref())
}

/// Walks up from `start` looking for a `.pith` file, so a database planted in a
/// shared parent (such as `/tmp`) is never picked up:
/// - the walk stops at the first directory containing `.git` (the repo root) and
///   never goes above `home`; if `start` is not under `home` (or there is no
///   home), only `start` itself is checked;
/// - a `.pith` that is a symlink, not a regular file, or (on unix) owned by
///   another user is skipped.
pub fn locate_project_within(start: &Path, home: Option<&Path>) -> Option<PathBuf> {
    let home = home.filter(|h| start.starts_with(h));
    for dir in start.ancestors() {
        let candidate = dir.join(".pith");
        if trusted(&candidate) {
            return Some(candidate);
        }
        if home.is_none_or(|h| dir == h) || dir.join(".git").exists() {
            return None;
        }
    }
    None
}

fn trusted(p: &Path) -> bool {
    // symlink_metadata does not follow links, so a symlink is not `is_file`.
    let Ok(m) = p.symlink_metadata() else {
        return false;
    };
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if m.uid() != rustix::process::getuid().as_raw() {
            return false;
        }
    }
    m.is_file()
}
