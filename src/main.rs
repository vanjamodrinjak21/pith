use clap::{Parser, Subcommand};
use pithdb::embed::{models_dir, pull, DEFAULT_MODEL};
use pithdb::{
    index_project, is_yes, load_default_embedder, open_project, render_error, render_search,
    render_stats, search_project, skip_summary, stats_project, Error, Result,
};
use std::io::{BufRead, IsTerminal};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "pith", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Download the embedding model (asks first)
    Init {
        /// Download without asking
        #[arg(short, long)]
        yes: bool,
    },
    /// Index a project into <path>/.pith
    Index {
        #[arg(default_value = ".")]
        path: PathBuf,
        /// Overwrite an index built with another model, or an unreadable one
        #[arg(long)]
        reindex: bool,
    },
    /// Search the indexed project
    Search {
        query: String,
        #[arg(short, default_value_t = 10)]
        k: usize,
    },
    /// Show index statistics
    Stats,
    Mcp,
    /// Manage the embedding model
    Model {
        #[command(subcommand)]
        cmd: ModelCmd,
    },
}

#[derive(Subcommand)]
enum ModelCmd {
    /// Download the default model (network)
    Pull,
    /// Show the default model and whether its files are present
    List,
}

fn pull_default() -> Result<()> {
    let dir = models_dir()?;
    eprintln!("downloading {} to {}", DEFAULT_MODEL.id, dir.display());
    pull(&dir, DEFAULT_MODEL.base_url)?;
    println!("model {} installed", DEFAULT_MODEL.id);
    Ok(())
}

/// Presence by size only; the embedder verifies hashes on load.
fn model_files(dir: &Path) -> Vec<(&'static str, &'static str)> {
    DEFAULT_MODEL
        .files
        .iter()
        .map(|f| {
            let state = match std::fs::symlink_metadata(dir.join(DEFAULT_MODEL.id).join(f.local)) {
                Err(_) => "missing",
                Ok(m) if m.is_file() && m.len() == f.size => "present",
                Ok(_) => "wrong size",
            };
            (f.local, state)
        })
        .collect()
}

fn init(yes: bool) -> Result<()> {
    let dir = models_dir()?;
    if model_files(&dir).iter().all(|(_, s)| *s == "present") {
        println!("model {} is already installed", DEFAULT_MODEL.id);
        return Ok(());
    }
    let go = yes || {
        if !std::io::stdin().is_terminal() {
            false
        } else {
            eprint!(
                "Download {} (~34 MB) to {}? [Y/n] (Enter or y downloads; anything else cancels) ",
                DEFAULT_MODEL.id,
                dir.display()
            );
            let mut line = String::new();
            std::io::stdin().lock().read_line(&mut line)? > 0 && is_yes(&line)
        }
    };
    if !go {
        println!("model not downloaded; run `pith model pull` to download it");
        return Ok(());
    }
    pull_default()
}

fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Init { yes } => init(yes),
        Command::Model {
            cmd: ModelCmd::Pull,
        } => pull_default(),
        Command::Model {
            cmd: ModelCmd::List,
        } => {
            let dir = models_dir()?;
            println!(
                "default model: {} ({}-d)",
                DEFAULT_MODEL.id, DEFAULT_MODEL.dim
            );
            for (f, s) in model_files(&dir) {
                println!("  {f}: {s}");
            }
            println!("location: {}", dir.join(DEFAULT_MODEL.id).display());
            Ok(())
        }
        Command::Index { path, reindex } => {
            // Fail before loading the model; index_project re-checks.
            if !path.is_dir() {
                return Err(Error::Invalid(format!(
                    "{} is not a directory",
                    path.display()
                )));
            }
            let embedder = load_default_embedder()?;
            let t = Instant::now();
            let r = index_project(&path, embedder.as_ref(), reindex)?;
            if !r.skipped.is_empty() {
                eprintln!("{}", skip_summary(&r.skipped));
            }
            let secs = t.elapsed().as_secs_f64();
            println!(
                "indexed {} chunks from {} files in {secs:.1}s",
                r.chunks, r.files
            );
            Ok(())
        }
        Command::Search { query, k } => {
            let (root, db) = open_project(&std::env::current_dir()?)?;
            let embedder = load_default_embedder()?;
            let hits = search_project(&root, &db, embedder.as_ref(), &query, k)?;
            print!(
                "{}",
                render_search(
                    &hits,
                    db.meta().indexed_at_unix,
                    pithdb::commands::unix_now()
                )
            );
            Ok(())
        }
        Command::Stats => {
            let (root, db) = open_project(&std::env::current_dir()?)?;
            print!("{}", render_stats(&stats_project(&root, &db)?));
            Ok(())
        }
        Command::Mcp => Err(Error::NotImplemented("mcp")),
    }
}

fn main() {
    if let Err(e) = run(Cli::parse()) {
        eprintln!("pith: {}", render_error(&e));
        std::process::exit(1);
    }
}
