//! Regression: a fork on another thread briefly shares the lock fd; acquire must ride it out.
use pithdb::storage::WriterLock;
use std::sync::atomic::{AtomicBool, Ordering::*};
#[test]
fn lock_survives_a_concurrent_fork_window() {
    let stop = std::sync::Arc::new(AtomicBool::new(false));
    let s2 = stop.clone();
    let forker = std::thread::spawn(move || {
        while !s2.load(Relaxed) {
            let _ = std::process::Command::new("true").status();
        }
    });
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join(".pith");
    let mut fails = 0;
    for _ in 0..1000 {
        let a = WriterLock::acquire(&p);
        if a.is_err() {
            fails += 1;
        }
        drop(a);
    }
    stop.store(true, Relaxed);
    forker.join().unwrap();

    assert_eq!(fails, 0);
}
