//! Writer lock: an advisory `flock`-style lock (std `File::try_lock`) on `<db>.lock`.
//! The OS drops it when the process exits or is killed, so a crash never leaves it stuck.

use crate::{Error, Result};
use std::fs::{File, OpenOptions, TryLockError};
use std::path::{Path, PathBuf};

const RETRY_MS: u64 = 5;
const GRACE_MS: u64 = 250;

pub struct WriterLock {
    _file: File,
}

pub(crate) fn lock_path(db: &Path) -> PathBuf {
    let mut name = db.file_name().unwrap_or_default().to_os_string();
    name.push(".lock");
    db.with_file_name(name)
}

impl WriterLock {
    /// Fails with `Error::Locked` if another writer still holds the lock after a short grace
    /// period (about 250 ms); never blocks longer than that.
    pub fn acquire(db: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(lock_path(db))?;
        // A thread that forks (e.g. `Command::spawn`) while another thread holds this lock gives
        // the child a copy of the fd, and flock belongs to the open file description, so the lock
        // stays held until the child execs or exits (CLOEXEC only acts at exec). That window is
        // milliseconds; a real second writer holds for far longer. Retry briefly, then give up.
        let mut waited = 0;
        loop {
            match file.try_lock() {
                Ok(()) => return Ok(Self { _file: file }),
                Err(TryLockError::WouldBlock) if waited < GRACE_MS => {
                    std::thread::sleep(std::time::Duration::from_millis(RETRY_MS));
                    waited += RETRY_MS;
                }
                Err(TryLockError::WouldBlock) => return Err(Error::Locked),
                Err(TryLockError::Error(e)) => return Err(e.into()),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn waits_for_a_briefly_held_lock() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join(".pith");
        let held = WriterLock::acquire(&p).unwrap();
        let t = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            drop(held);
        });
        assert!(WriterLock::acquire(&p).is_ok());
        t.join().unwrap();
    }

    #[test]
    fn a_long_held_lock_is_locked_well_under_a_second() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join(".pith");
        let _held = WriterLock::acquire(&p).unwrap();
        let t = Instant::now();
        assert!(matches!(WriterLock::acquire(&p), Err(Error::Locked)));
        assert!(t.elapsed() < Duration::from_secs(1));
    }
}
