use super::{budget::fit_budget, State};
use crate::commands::{age, clean, unix_now, valid_head};
use crate::{render_error, render_stats, search_project, stats_project, Error};
use rmcp::handler::server::{router::tool::ToolRouter, wrapper::Parameters};
use rmcp::model::{Implementation, ServerCapabilities, ServerConfig};
use rmcp::schemars::JsonSchema;
use rmcp::{tool, tool_handler, tool_router, ServerHandler, ServiceExt};
use serde::Deserialize;
use std::sync::Arc;

const MAX_TOKENS: usize = 4000;
const MAX_PER_FILE: usize = 3;
const SNIPPET_LINES: usize = 15;
const DEFAULT_K: u32 = 8;
const MAX_K: u32 = 20;
const MAX_QUERY_CHARS: usize = 2000;

#[derive(Deserialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
pub(crate) struct SearchArgs {
    /// What to look for, in natural language or code.
    query: String,
    /// Maximum results (1-20, default 8).
    k: Option<u32>,
}

#[derive(Deserialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
pub(crate) struct ChunkArgs {
    /// A `chunk_id` from a search_code result.
    chunk_id: u32,
}

#[derive(Clone)]
pub(crate) struct Server {
    state: Arc<State>,
    tool_router: ToolRouter<Self>,
}

/// Tool errors are shown to the model, so they say what to run and are sanitised.
fn message(e: &Error) -> String {
    match e {
        Error::ModelMissing(_) => "embedding model is not installed; run `pith init`".into(),
        e => render_error(e),
    }
}

fn reply(r: crate::Result<String>) -> Result<String, String> {
    r.map_err(|e| message(&e))
}

#[tool_router]
impl Server {
    /// Semantic search over the indexed project. Returns a few capped snippets (about 15 lines
    /// each, at most 3 per file) with path, line range, score and chunk_id, plus index staleness.
    #[tool]
    async fn search_code(&self, Parameters(a): Parameters<SearchArgs>) -> Result<String, String> {
        reply(self.search(&a.query, a.k))
    }

    /// Full text of one chunk by its chunk_id from search_code.
    #[tool]
    async fn get_chunk(&self, Parameters(a): Parameters<ChunkArgs>) -> Result<String, String> {
        reply(self.state.with_db(|_, db| {
            let c = db
                .chunk(a.chunk_id)
                .ok_or_else(|| Error::Invalid(format!("no chunk with id {}", a.chunk_id)))?;
            Ok(format!(
                "{}:{}-{}\n{}\n",
                clean(&c.path),
                c.start_line,
                c.end_line,
                clean_block(&c.text)
            ))
        }))
    }

    /// Index statistics: chunk and file counts, model, size, age and git head.
    #[tool]
    async fn stats(&self) -> Result<String, String> {
        reply(
            self.state
                .with_db(|root, db| Ok(render_stats(&stats_project(root, db)?))),
        )
    }
}

/// `clean` per line, keeping the newlines.
fn clean_block(text: &str) -> String {
    text.lines().map(clean).collect::<Vec<_>>().join("\n")
}

impl Server {
    fn search(&self, query: &str, k: Option<u32>) -> crate::Result<String> {
        if query.trim().is_empty() {
            return Err(Error::Invalid("query must not be empty".into()));
        }
        if query.chars().count() > MAX_QUERY_CHARS {
            return Err(Error::Invalid(format!(
                "query is longer than {MAX_QUERY_CHARS} characters"
            )));
        }
        let k = k.unwrap_or(DEFAULT_K).clamp(1, MAX_K) as usize;
        self.state.with_db(|root, db| {
            // Over-fetch so the per-file cap still leaves up to k results.
            let hits = self
                .state
                .with_embedder(|e| search_project(root, db, e, query, k * MAX_PER_FILE))?;
            let mut hits = fit_budget(hits, MAX_TOKENS, MAX_PER_FILE, SNIPPET_LINES);
            hits.truncate(k);
            let mut out = format!(
                "indexed {}",
                age(unix_now().saturating_sub(db.meta().indexed_at_unix))
            );
            if let Some(h) = db.meta().git_head.as_deref().filter(|h| valid_head(h)) {
                out.push_str(&format!(" at git {h}"));
            }
            out.push_str(&format!("; {} results\n", hits.len()));
            for h in &hits {
                let shown = h.text.lines().count() as u32;
                let full = h.end_line.saturating_sub(h.start_line) + 1;
                out.push_str(&format!(
                    "\n{}:{}-{}  score {:.2}  chunk_id {}{}\n{}\n",
                    clean(&h.path),
                    h.start_line,
                    h.end_line,
                    h.score,
                    h.chunk_id,
                    if h.source_changed {
                        "  [source changed since indexing]"
                    } else {
                        ""
                    },
                    clean_block(&h.text)
                ));
                if shown < full {
                    out.push_str(&format!(
                        "[{} more lines: get_chunk {}]\n",
                        full - shown,
                        h.chunk_id
                    ));
                }
            }
            Ok(out)
        })
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for Server {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("pith", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "Read-only code search over this project's Pith index. Use search_code before \
                 reading whole files; use get_chunk for a full chunk.",
            )
    }
}

pub(crate) async fn serve(state: Arc<State>) -> crate::Result<()> {
    let server = Server {
        state,
        tool_router: Server::tool_router(),
    };
    let running = server
        .serve(rmcp::transport::stdio())
        .await
        .map_err(|e| Error::Io(std::io::Error::other(e.to_string())))?;
    running
        .waiting()
        .await
        .map_err(|e| Error::Io(std::io::Error::other(e.to_string())))?;
    Ok(())
}
