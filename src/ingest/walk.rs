use super::{chunk::chunk_text, deny, CollectReport, SkipReason};
use crate::{Error, Result};
use ignore::WalkBuilder;
use std::fs::{File, Metadata, OpenOptions};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Files larger than this (1 MiB) are skipped: generated or minified output is
/// noise for retrieval and slow to embed.
pub const MAX_FILE_BYTES: u64 = 1024 * 1024;

/// How much of a file is sniffed for a NUL byte before the rest is read. A NUL after
/// this prefix does NOT make a file binary; such a file is indexed.
const SNIFF_BYTES: u64 = 8 * 1024;

/// Walk `root`, honoring `.gitignore` and `.pithignore` (no global gitignore, no git
/// repo required), and window every readable text file into chunks.
///
/// Symlinks are never followed, `.git` is never entered, the secrets deny list is
/// applied to the path before any file is opened, and the database's own files are
/// skipped. A single unreadable file is recorded in the report, not an error. A root
/// that is inside a secrets directory (`~/.ssh`, `.gnupg`, `.aws`, ...) is refused.
///
/// Files are opened with `O_NOFOLLOW | O_NONBLOCK` and the open handle must be the same
/// regular file (device and inode) the walk saw. Residual risk: an intermediate
/// directory swapped for a symlink mid-walk is not detected by that check; the
/// mitigation is that only paths under the root are produced by the walk, and an
/// attacker able to race the project tree can already edit the files directly.
///
/// Memory: all chunks of the project are held in the returned report, which is
/// acceptable for this slice (files are capped at [`MAX_FILE_BYTES`]).
pub fn collect(root: &Path) -> Result<CollectReport> {
    if !root.metadata()?.is_dir() {
        return Err(Error::Invalid(format!(
            "{} is not a directory",
            root.display()
        )));
    }
    let canonical = root.canonicalize()?;
    if deny::is_denied_root(&canonical) {
        return Err(Error::Invalid(format!(
            "refusing to index {}: it is inside a secrets directory",
            canonical.display()
        )));
    }
    let mut report = CollectReport::default();
    let denied_dirs = Arc::new(Mutex::new(Vec::<String>::new()));
    let walker = WalkBuilder::new(root)
        .hidden(false) // dotfiles must be seen so the deny list, not the walker, decides
        .parents(false)
        .git_global(false)
        .require_git(false)
        .follow_links(false)
        .add_custom_ignore_filename(".pithignore")
        .filter_entry({
            let root = root.to_owned();
            let denied_dirs = Arc::clone(&denied_dirs);
            move |e| {
                if !e.file_type().is_some_and(|t| t.is_dir()) {
                    return true;
                }
                let Ok(rel) = e.path().strip_prefix(&root) else {
                    return true;
                };
                if e.file_name() == ".git" {
                    return false;
                }
                if deny::is_denied_dir(rel) {
                    if let Ok(mut v) = denied_dirs.lock() {
                        v.push(rel_string(rel));
                    }
                    return false;
                }
                true
            }
        })
        .build();

    for entry in walker {
        let entry = match entry {
            Ok(e) => e,
            Err(err) => {
                let path = err_path(&err)
                    .map(|p| rel_string(p.strip_prefix(root).unwrap_or(&p)))
                    .unwrap_or_else(|| err.to_string());
                report
                    .skipped
                    .push((path, SkipReason::Unreadable(err.to_string())));
                continue;
            }
        };
        let Some(ft) = entry.file_type() else {
            continue;
        };
        if ft.is_dir() {
            continue;
        }
        let Ok(rel) = entry.path().strip_prefix(root) else {
            let p = entry.path().display().to_string();
            report
                .skipped
                .push((p, SkipReason::Unreadable("path outside root".into())));
            continue;
        };
        if rel.as_os_str().is_empty() {
            continue;
        }
        let Some(rel_str) = rel
            .to_str()
            .map(|s| s.replace(std::path::MAIN_SEPARATOR, "/"))
        else {
            report
                .skipped
                .push((rel.to_string_lossy().into_owned(), SkipReason::NotUtf8));
            continue;
        };
        if deny::is_db_file(&entry.file_name().to_string_lossy()) {
            continue;
        }
        if deny::is_denied(rel) {
            report.skipped.push((rel_str, SkipReason::Denied));
            continue;
        }
        if ft.is_symlink() {
            report.skipped.push((rel_str, SkipReason::Symlink));
            continue;
        }
        if !ft.is_file() {
            report.skipped.push((rel_str, SkipReason::SpecialFile));
            continue;
        }
        let walked = match std::fs::symlink_metadata(entry.path()) {
            Ok(m) => m,
            Err(e) => {
                report
                    .skipped
                    .push((rel_str, SkipReason::Unreadable(e.to_string())));
                continue;
            }
        };
        match read_text(entry.path(), &walked) {
            Ok(text) => {
                let chunks: Vec<_> = chunk_text(&rel_str, &text)
                    .into_iter()
                    .filter(|c| !c.text.trim().is_empty())
                    .collect();
                if chunks.is_empty() {
                    report.skipped.push((rel_str, SkipReason::Empty)); // only blank lines
                } else {
                    report.chunks.extend(chunks);
                }
            }
            Err(reason) => report.skipped.push((rel_str, reason)),
        }
    }
    if let Ok(v) = denied_dirs.lock() {
        report
            .skipped
            .extend(v.iter().map(|p| (p.clone(), SkipReason::Denied)));
    }
    report.chunks.sort_by(|a, b| {
        (&a.path, a.start_line, a.end_line).cmp(&(&b.path, b.start_line, b.end_line))
    });
    report.skipped.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(report)
}

fn rel_string(p: &Path) -> String {
    p.to_string_lossy().replace(std::path::MAIN_SEPARATOR, "/")
}

/// The path an `ignore::Error` is about, looking through nested variants.
fn err_path(e: &ignore::Error) -> Option<PathBuf> {
    match e {
        ignore::Error::WithPath { path, .. } => Some(path.clone()),
        ignore::Error::WithLineNumber { err, .. } | ignore::Error::WithDepth { err, .. } => {
            err_path(err)
        }
        ignore::Error::Partial(v) => v.iter().find_map(err_path),
        ignore::Error::Loop { child, .. } => Some(child.clone()),
        _ => None,
    }
}

/// Open without following a final symlink and without blocking on a FIFO.
fn open_nofollow(path: &Path) -> std::io::Result<File> {
    let mut o = OpenOptions::new();
    o.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        let flags = rustix::fs::OFlags::NOFOLLOW | rustix::fs::OFlags::NONBLOCK;
        o.custom_flags(flags.bits() as i32);
    }
    o.open(path)
}

/// `walked` is the `symlink_metadata` taken during the walk; the open handle must be
/// the same regular file.
fn read_text(path: &Path, walked: &Metadata) -> std::result::Result<String, SkipReason> {
    let unreadable = |e: std::io::Error| SkipReason::Unreadable(e.to_string());
    let file = open_nofollow(path).map_err(unreadable)?;
    let meta = file.metadata().map_err(unreadable)?;
    if !meta.is_file() {
        return Err(SkipReason::SpecialFile);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if meta.dev() != walked.dev() || meta.ino() != walked.ino() {
            return Err(SkipReason::Unreadable("file changed during walk".into()));
        }
    }
    #[cfg(not(unix))]
    let _ = walked;
    if meta.len() > MAX_FILE_BYTES {
        return Err(SkipReason::TooLarge);
    }
    // Sniff the prefix for NUL before reading the rest; `take` bounds the read even if
    // the file grows after the metadata check.
    let mut bytes = Vec::new();
    (&file)
        .take(SNIFF_BYTES)
        .read_to_end(&mut bytes)
        .map_err(unreadable)?;
    if bytes.is_empty() {
        return Err(SkipReason::Empty);
    }
    if bytes.contains(&0) {
        return Err(SkipReason::Binary);
    }
    (&file)
        .take(MAX_FILE_BYTES + 1 - bytes.len() as u64)
        .read_to_end(&mut bytes)
        .map_err(unreadable)?;
    if bytes.len() as u64 > MAX_FILE_BYTES {
        return Err(SkipReason::TooLarge);
    }
    let text = String::from_utf8(bytes).map_err(|_| SkipReason::NotUtf8)?;
    if has_private_key_header(&text) {
        return Err(SkipReason::Secret);
    }
    Ok(text)
}

/// `-----BEGIN ... PRIVATE KEY-----` on one line (RSA, EC, OPENSSH, ENCRYPTED, plain).
fn has_private_key_header(text: &str) -> bool {
    text.lines().any(|l| {
        l.find("-----BEGIN")
            .is_some_and(|i| l[i..].contains("PRIVATE KEY-----"))
    })
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;

    #[test]
    fn symlink_swapped_in_after_walk_is_not_followed() {
        let d = tempfile::tempdir().unwrap();
        let (f, target) = (d.path().join("f.txt"), d.path().join("target.txt"));
        std::fs::write(&f, "walked\n").unwrap();
        std::fs::write(&target, "secret target\n").unwrap();
        let walked = std::fs::symlink_metadata(&f).unwrap();
        std::fs::remove_file(&f).unwrap();
        symlink(&target, &f).unwrap();
        let r = read_text(&f, &walked);
        assert!(matches!(r, Err(SkipReason::Unreadable(_))), "{r:?}");
    }

    #[test]
    fn different_file_swapped_in_after_walk_is_rejected() {
        let d = tempfile::tempdir().unwrap();
        let (f, other) = (d.path().join("f.txt"), d.path().join("other.txt"));
        std::fs::write(&f, "walked\n").unwrap();
        std::fs::write(&other, "other\n").unwrap();
        let walked = std::fs::symlink_metadata(&f).unwrap();
        std::fs::rename(&other, &f).unwrap(); // other's inode is distinct from the walked one
        assert_eq!(
            read_text(&f, &walked),
            Err(SkipReason::Unreadable("file changed during walk".into()))
        );
    }

    #[test]
    fn unchanged_file_is_read() {
        let d = tempfile::tempdir().unwrap();
        let f = d.path().join("f.txt");
        std::fs::write(&f, "hi\n").unwrap();
        assert_eq!(
            read_text(&f, &std::fs::symlink_metadata(&f).unwrap()).unwrap(),
            "hi\n"
        );
    }
}
