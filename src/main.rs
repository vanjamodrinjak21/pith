use clap::{Parser, Subcommand};
use pithdb::{Error, Result};

#[derive(Parser)]
#[command(name = "pith", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Init,
    Index,
    Search,
    Stats,
    Mcp,
    Model,
}

fn run(cli: Cli) -> Result<()> {
    let name = match cli.command {
        Command::Init => "init",
        Command::Index => "index",
        Command::Search => "search",
        Command::Stats => "stats",
        Command::Mcp => "mcp",
        Command::Model => "model",
    };
    Err(Error::NotImplemented(name))
}

fn main() {
    if let Err(e) = run(Cli::parse()) {
        eprintln!("pith: {e}");
        std::process::exit(1);
    }
}
