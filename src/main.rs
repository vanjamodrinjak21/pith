use clap::{Parser, Subcommand};
use pithdb::embed::{models_dir, pull, DEFAULT_MODEL};
use pithdb::{
    index_project, load_default_embedder, open_project, search_project, skip_summary,
    stats_project, Error, Result,
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
fn model_present(dir: &Path) -> Vec<(&'static str, &'static str)> {
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
    if model_present(&dir).iter().all(|(_, s)| *s == "present") {
        println!("model {} is already installed", DEFAULT_MODEL.id);
        return Ok(());
    }
    let go = yes || {
        if !std::io::stdin().is_terminal() {
            false
        } else {
            eprint!(
                "Download {} (~34 MB) to {}? [Y/n] ",
                DEFAULT_MODEL.id,
                dir.display()
            );
            let mut line = String::new();
            let n = std::io::stdin().lock().read_line(&mut line)?;
            n > 0 && matches!(line.trim().to_ascii_lowercase().as_str(), "" | "y" | "yes")
        }
    };
    if !go {
        println!("model not downloaded; run `pith model pull` to download it");
        return Ok(());
    }
    pull_default()
}

fn age(secs: u64) -> String {
    match secs {
        0..=59 => format!("{secs} seconds ago"),
        60..=3599 => format!("{} minutes ago", secs / 60),
        3600..=86399 => format!("{} hours ago", secs / 3600),
        _ => format!("{} days ago", secs / 86400),
    }
}

fn snippet(text: &str) -> String {
    text.lines()
        .take(3)
        .map(|l| {
            let l: String = l.chars().take(100).collect();
            format!(
                "    {}",
                l.chars()
                    .map(|c| if c.is_control() { ' ' } else { c })
                    .collect::<String>()
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
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
            for (f, s) in model_present(&dir) {
                println!("  {f}: {s}");
            }
            println!("location: {}", dir.join(DEFAULT_MODEL.id).display());
            Ok(())
        }
        Command::Index { path, reindex } => {
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
            for (i, h) in hits.iter().enumerate() {
                let mark = if h.source_changed {
                    "  [source changed]"
                } else {
                    ""
                };
                println!(
                    "{}. {:.2}  {}:{}-{}{mark}\n{}",
                    i + 1,
                    h.score,
                    pithdb::commands::clean(&h.path),
                    h.start_line,
                    h.end_line,
                    snippet(&h.text)
                );
            }
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs());
            let changed = hits.iter().filter(|h| h.source_changed).count();
            println!(
                "indexed {}; {changed} of {} results have changed sources",
                age(now.saturating_sub(db.meta().indexed_at_unix)),
                hits.len()
            );
            Ok(())
        }
        Command::Stats => {
            let cwd = std::env::current_dir()?;
            let path = pithdb::storage::locate_project(&cwd)
                .ok_or_else(|| Error::NoIndex(cwd.display().to_string()))?;
            let s = stats_project(&path)?;
            println!("chunks:  {}", s.chunks);
            println!("files:   {}", s.files);
            println!("model:   {}", s.model_id);
            println!("db size: {} bytes", s.db_bytes);
            println!("indexed: {}", age(s.age_secs));
            if let Some(h) = s.git_head {
                println!("git head: {h}");
            }
            Ok(())
        }
        Command::Mcp => Err(Error::NotImplemented("mcp")),
    }
}

fn main() {
    if let Err(e) = run(Cli::parse()) {
        match e {
            Error::Locked => eprintln!(
                "pith: another pith process is writing this index; wait for it to finish and retry"
            ),
            e => eprintln!("pith: {e}"),
        }
        std::process::exit(1);
    }
}
