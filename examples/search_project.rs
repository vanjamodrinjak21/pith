//! Index a directory and search it, using Pith as a library.
//!
//! Needs the embedding model (`pith init` once). It writes `<directory>/.pith` (and a `.pith.lock`)
//! and refuses to touch an existing index built with another model. Run with:
//! `cargo run --example search_project -- <directory> "<query>"`

use std::path::Path;

fn main() -> pithdb::Result<()> {
    let mut args = std::env::args().skip(1);
    let (Some(dir), Some(query)) = (args.next(), args.next()) else {
        eprintln!("usage: search_project <directory> \"<query>\"");
        std::process::exit(2);
    };
    let root = Path::new(&dir);

    let embedder = pithdb::load_default_embedder()?;
    let report = pithdb::index_project(root, embedder.as_ref(), false)?;
    println!(
        "indexed {} chunks from {} files",
        report.chunks, report.files
    );

    let (found, db) = pithdb::open_project(root)?;
    // open_project walks up the tree: make sure it found the index we just wrote.
    assert_eq!(
        found.canonicalize()?,
        root.canonicalize()?,
        "found another project's index"
    );
    let root = found;
    for hit in pithdb::search_project(&root, &db, embedder.as_ref(), &query, 5)? {
        let stale = if hit.source_changed {
            "  [source changed since indexing]"
        } else {
            ""
        };
        println!(
            "{:.2}  {}:{}-{}{stale}",
            hit.score, hit.path, hit.start_line, hit.end_line
        );
    }
    Ok(())
}
