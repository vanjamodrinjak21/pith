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
    #[cfg(unix)]
    let uid = Some(rustix::process::getuid().as_raw());
    #[cfg(not(unix))]
    let uid = None;
    locate(start, home, uid)
}

/// `owner`: required file owner uid (unix); `None` skips the owner check.
fn locate(start: &Path, home: Option<&Path>, owner: Option<u32>) -> Option<PathBuf> {
    let home = home.filter(|h| start.starts_with(h));
    for dir in start.ancestors() {
        let candidate = dir.join(".pith");
        if trusted(&candidate, owner) {
            return Some(candidate);
        }
        if home.is_none_or(|h| dir == h) || dir.join(".git").exists() {
            return None;
        }
    }
    None
}

fn trusted(p: &Path, owner: Option<u32>) -> bool {
    // symlink_metadata does not follow links, so a symlink is not `is_file`.
    let Ok(m) = p.symlink_metadata() else {
        return false;
    };
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if owner.is_some_and(|uid| m.uid() != uid) {
            return false;
        }
    }
    #[cfg(not(unix))]
    let _ = owner;
    m.is_file()
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::MetadataExt;

    #[test]
    fn file_owned_by_other_uid_is_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join(".pith");
        std::fs::write(&p, b"x").unwrap();
        let mine = p.metadata().unwrap().uid();
        assert_eq!(locate(dir.path(), None, Some(mine)), Some(p.clone()));
        assert_eq!(locate(dir.path(), None, Some(mine.wrapping_add(1))), None);
        // The public entry point passes the real uid, which owns this file.
        assert_eq!(locate_project_within(dir.path(), None), Some(p));
    }

    #[test]
    fn symlink_is_skipped_even_with_matching_owner() {
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("real");
        std::fs::write(&real, b"x").unwrap();
        let link = dir.path().join(".pith");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let mine = link.symlink_metadata().unwrap().uid();
        assert_eq!(locate(dir.path(), None, Some(mine)), None);
        assert_eq!(locate(dir.path(), None, None), None);
    }
}
