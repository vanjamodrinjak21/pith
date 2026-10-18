//! Index a directory and search it, using Pith as a library.
//!
//! Needs the embedding model (`pith init` once). Run with:
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
    let report = pithdb::index_project(root, embedder.as_ref(), true)?;
    println!("indexed {} chunks from {} files", report.chunks, report.files);

    let (root, db) = pithdb::open_project(root)?;
    for hit in pithdb::search_project(&root, &db, embedder.as_ref(), &query, 5)? {
        println!("{:.2}  {}:{}-{}", hit.score, hit.path, hit.start_line, hit.end_line);
    }
    Ok(())
}
