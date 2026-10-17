use std::path::{Path, PathBuf};

/// Nearest `.pith` file in `start` or any ancestor directory.
pub fn locate_project(start: &Path) -> Option<PathBuf> {
    start
        .ancestors()
        .map(|d| d.join(".pith"))
        .find(|p| p.is_file())
}
