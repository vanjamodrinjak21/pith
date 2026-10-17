//! Writer lock: an advisory `flock`-style lock (std `File::try_lock`) on `<db>.lock`.
//! The OS drops it when the process exits or is killed, so a crash never leaves it stuck.

use crate::{Error, Result};
use std::fs::{File, OpenOptions, TryLockError};
use std::path::{Path, PathBuf};

pub struct WriterLock {
    _file: File,
}

pub(crate) fn lock_path(db: &Path) -> PathBuf {
    let mut name = db.file_name().unwrap_or_default().to_os_string();
    name.push(".lock");
    db.with_file_name(name)
}

impl WriterLock {
    /// Fails immediately with `Error::Locked` if another writer holds it; never blocks.
    pub fn acquire(db: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(lock_path(db))?;
        match file.try_lock() {
            Ok(()) => Ok(Self { _file: file }),
            Err(TryLockError::WouldBlock) => Err(Error::Locked),
            Err(TryLockError::Error(e)) => Err(e.into()),
        }
    }
}
