use super::{chunk::chunk_text, deny, CollectReport, SkipReason};
use crate::{Error, Result};
use ignore::WalkBuilder;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Files larger than this (1 MiB) are skipped: generated or minified output is
/// noise for retrieval and slow to embed.
pub const MAX_FILE_BYTES: u64 = 1024 * 1024;

/// How much of a file is sniffed for a NUL byte.
const SNIFF_BYTES: usize = 8 * 1024;

/// Walk `root`, honoring `.gitignore` and `.pithignore` (no global gitignore, no git
/// repo required), and window every readable text file into chunks.
///
/// Symlinks are never followed, `.git` is never entered, the secrets deny list is
/// applied to the path before any file is opened, and the database's own files are
/// skipped. A single unreadable file is recorded in the report, not an error.
pub fn collect(root: &Path) -> Result<CollectReport> {
    if !root.metadata()?.is_dir() {
        return Err(Error::Invalid(format!(
            "{} is not a directory",
            root.display()
        )));
    }
    let mut report = CollectReport::default();
    let walker = WalkBuilder::new(root)
        .hidden(false) // dotfiles must be seen so the deny list, not the walker, decides
        .parents(false)
        .git_global(false)
        .require_git(false)
        .follow_links(false)
        .add_custom_ignore_filename(".pithignore")
        .filter_entry({
            let root = root.to_owned();
            move |e| {
                let rel = e.path().strip_prefix(&root).unwrap_or(e.path());
                let is_dir = e.file_type().is_some_and(|t| t.is_dir());
                !(is_dir && (e.file_name() == ".git" || deny::is_denied_dir(rel)))
            }
        })
        .build();

    for entry in walker {
        let entry = match entry {
            Ok(e) => e,
            Err(err) => {
                report
                    .skipped
                    .push((err.to_string(), SkipReason::Unreadable(err.to_string())));
                continue;
            }
        };
        let Some(ft) = entry.file_type() else {
            continue;
        };
        let rel = entry.path().strip_prefix(root).unwrap_or(entry.path());
        if rel.as_os_str().is_empty() || ft.is_dir() {
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
            continue; // sockets, fifos: opening could block
        }
        match read_text(entry.path()) {
            Ok(text) => {
                let chunks = chunk_text(&rel_str, &text);
                report
                    .chunks
                    .extend(chunks.into_iter().filter(|c| !c.text.trim().is_empty()));
            }
            Err(reason) => report.skipped.push((rel_str, reason)),
        }
    }
    report
        .chunks
        .sort_by(|a, b| (&a.path, a.start_line).cmp(&(&b.path, b.start_line)));
    report.skipped.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(report)
}

fn read_text(path: &Path) -> std::result::Result<String, SkipReason> {
    let unreadable = |e: std::io::Error| SkipReason::Unreadable(e.to_string());
    let file = File::open(path).map_err(unreadable)?;
    let meta = file.metadata().map_err(unreadable)?;
    if !meta.is_file() {
        return Err(SkipReason::Unreadable("not a regular file".into()));
    }
    if meta.len() > MAX_FILE_BYTES {
        return Err(SkipReason::TooLarge);
    }
    // `take` bounds the read even if the file grows after the metadata check.
    let mut bytes = Vec::new();
    file.take(MAX_FILE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(unreadable)?;
    if bytes.len() as u64 > MAX_FILE_BYTES {
        return Err(SkipReason::TooLarge);
    }
    if bytes.is_empty() {
        return Err(SkipReason::Empty);
    }
    if bytes[..bytes.len().min(SNIFF_BYTES)].contains(&0) {
        return Err(SkipReason::Binary);
    }
    String::from_utf8(bytes).map_err(|_| SkipReason::NotUtf8)
}
