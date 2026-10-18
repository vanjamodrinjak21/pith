# pithdb [![crates.io](https://img.shields.io/crates/v/pithdb.svg)](https://crates.io/crates/pithdb) [![CI](https://github.com/vanjamodrinjak21/pith/actions/workflows/ci.yml/badge.svg)](https://github.com/vanjamodrinjak21/pith/actions) [![docs.rs](https://docs.rs/pithdb/badge.svg)](https://docs.rs/pithdb)

Pith is a lightweight, local vector database with built-in ONNX embeddings, plus an
[MCP](https://modelcontextprotocol.io) server that lets Claude Code search your codebase by
meaning. The crate is `pithdb`; the command is `pith`.

Large projects waste context every session: the model re-reads files it has already seen. Pith
indexes your code once into a single `.pith` file, like a SQLite file but for vectors. Claude then
asks for the few relevant chunks (capped at about 4,000 tokens per call) instead of re-reading
whole files. One binary, no daemon, no API key, no cloud: the only network access is a one-time
model download.

> **The on-disk format is unstable until 1.0.** Any release before 1.0 may change it. Pith
> refuses to open an index it does not understand; rebuild with `pith index --reindex`. Treat
> `.pith` as a cache of your source tree, not as durable data.

Features:
 * Single-file database: checksummed sections, zstd-compressed chunk table, memory-mapped
   vectors, atomic writes, one writer at a time
 * Exact int8 SIMD scan (NEON, AVX2) with an f32 rescore of the top candidates; cosine similarity
 * Hybrid search by default: embeddings fused with an in-memory BM25 keyword index that knows
   identifiers (`search_with` also matches `search` and `with`), so exact names and error
   strings are found; keyword search works without the model
 * Built-in embeddings: `bge-small-en-v1.5` (int8, 384 dimensions) through a statically linked
   ONNX Runtime, downloaded once, pinned to a commit and sha256-verified
 * Gitignore-aware indexing (`.gitignore`, `.ignore`, `.pithignore`) that skips secrets by name
   and by content
 * Read-only MCP server: `search_code`, `get_chunk`, `stats`; results show how stale the index is
 * Also usable as a Rust library (storage, index and ingest are separate modules)

## Status

Pith is `0.1` and needs Rust 1.92 or newer to build. macOS on Apple Silicon is the supported platform: that is where it is built and
tested (an M4 Pro; the release binary is about 38 MB). Linux and Windows are untested. There is
no HNSW (search is a brute-force scan, see Performance), no incremental indexing and no
non-code ingestion.

## Install

```sh
cargo install pithdb      # installs the `pith` binary
pith init                 # asks, then downloads the embedding model (~34 MB), once
```

`cargo install` fetches a prebuilt, hash-verified ONNX Runtime archive (about 80 MB, cached in
`~/Library/Caches/ort.pyke.io`) from pyke's CDN at build time. That is the only network access
during the build. Offline builds: point `ORT_LIB_PATH` at a self-built static ONNX Runtime.
`cargo install pithdb --no-default-features` builds without ONNX Runtime and without embeddings.

## Use

```sh
cd your-project
pith index                # writes ./.pith; --reindex overwrites a stale or unreadable one
pith search "where are retries handled" -k 5
pith stats
```

```
1. 0.69  src/storage/lock.rs:1-35
    //! Writer lock: an advisory `flock`-style lock (std `File::try_lock`) on `<db>.lock`.
    //! The OS drops it when the process exits or is killed, so a crash never leaves it stuck.
    ...
indexed 9 seconds ago; 0 of 3 results have changed sources
```

`--mode` picks how results are ranked:

| Mode | Ranking | Needs the model |
| --- | --- | --- |
| `hybrid` (default) | reciprocal rank fusion (k = 60) of the top 50 by vector and the top 50 by BM25 | no: without it, keyword only, with a one-line notice |
| `vector` | cosine similarity of embeddings | yes (`run pith init`) |
| `keyword` | BM25 (k1 = 1.2, b = 0.75) over identifier-aware tokens | no |

The score shown is the cosine similarity whenever the model is loaded (hybrid results are
ordered by fused rank, so scores need not decrease); keyword-only scores are BM25 divided by the
top hit's, so the first is 1.00. The keyword index is built in memory from the chunk texts on the
first keyword or hybrid query and never stored. A chunk contributes at most 1,024 tokens, and once
the index holds 16 Mi postings (about 128 MiB) later chunks are not keyword-indexed.

Run `pith index` again after edits: chunks whose text is unchanged keep their vectors, so only
changed chunks are embedded, and an unchanged project is reported up to date without rewriting
`.pith`. `--reindex` embeds everything again.

Files over 1 MiB are skipped. `pith index` adds `.pith` and `.pith.lock` to an existing
`.gitignore`; it never creates one.

### Claude Code (MCP)

```sh
claude mcp add pith -- pith mcp                        # serves the project in the current directory
claude mcp add pith -- pith mcp --project /path/to/project
```

| Tool | Arguments | Returns |
| --- | --- | --- |
| `search_code` | `query`, `k` (default 8, max 20) | ranked chunks (hybrid search) with path, line range, score and chunk id |
| `get_chunk` | `chunk_id` | the full text of one chunk |
| `stats` | none | chunk count, model, index age |

Each response is capped at about 4,000 tokens, with at most 3 chunks per file and about 15 lines
per snippet. The server never indexes or writes: run `pith index` first, and again after big
changes (a reindex is picked up without restarting). The embedding model loads on the first
search and is dropped after 5 idle minutes. Without the model, `search_code` returns keyword
results and says so in its second line.

## Terminal UI

```sh
pith ui                   # explore the index of the project in the current directory
pith ui --index           # index first (works before the first `pith index`), then explore
```

Four views:

- **Stack**: directory -> file -> chunk, sorted by chunk count, with a bar for each entry's share
  of the index and a preview of the selected chunk.
- **Map**: every chunk vector projected to 2-D (PCA, up to 20,000 points), coloured by top-level
  directory and shaded by density. Move the cursor to preview the nearest chunk. After a query
  it marks the query and the chunks sent to Claude.
- **RAG**: runs a query exactly like the MCP `search_code` tool and shows what Claude would
  receive: the ranked candidates with scores, which were sent, trimmed or dropped (3-per-file
  cap, 4,000-token cap, k = 8), a token meter, and the final text.
- **Index**: reindexes the project live: files and chunks done, chunks/s, ETA, the current
  file, skipped files and the reasons, and the embedded chunks filling a map. Cancelling keeps
  the previous index.

| Key | Action |
| --- | --- |
| `1`-`4`, `Tab`, `Shift-Tab` | switch view (`Tab` in RAG toggles ranked hits / context) |
| `j` `k`, arrows | move; scroll the context |
| `Enter`, `Space`, `h` `l` | expand or collapse a directory; open or leave a file's chunks |
| `/` | type a query; `Enter` runs it |
| `i` | index now; `Esc` or `c` cancels |
| `r` | reload the index after a `pith index` elsewhere |
| `?` | help |
| `q`, `Esc` | quit |

`pith ui` needs an interactive terminal. Built with the default `tui` feature (ratatui).

## Retrieval quality

`pith eval <cases.jsonl> [-k 10] [--mode vector|keyword|hybrid] [--verbose]` measures search
quality on hand-written cases, one JSON object per line:

```json
{"query": "where is the git head recorded", "expect": ["src/commands.rs:131-150"]}
```

An expectation is a path (any chunk of that file) or `path:start-end` (a hit must overlap those
lines). It prints hit@1, hit@3, hit@k and MRR per mode (all three without `--mode`), plus token
cost: the average size of what `search_code` would send against the average size of the expected
whole files. `--verbose` lists the misses with what was returned.

[`eval/pith.jsonl`](eval/pith.jsonl) holds 42 cases for this repository (26 identifier and
error-string lookups, 16 questions), checked by hand against the code. On this repository
(526 chunks, `bge-small-en-v1.5`, k = 10):

| Mode | hit@1 | hit@3 | hit@10 | MRR | tokens sent | whole files |
| --- | --- | --- | --- | --- | --- | --- |
| vector | 0.50 | 0.67 | 0.90 | 0.62 | 1,325 | 4,961 |
| keyword | 0.48 | 0.60 | 0.93 | 0.60 | 1,404 | 4,961 |
| hybrid | 0.52 | 0.76 | 0.93 | 0.65 | 1,348 | 4,961 |

Hybrid is the default because it beats vector search on the identifier cases (MRR 0.63 vs 0.60,
hit@3 0.73 vs 0.65) without losing on the questions (MRR 0.69 vs 0.66, hit@3 0.81 vs 0.69).
Keyword search alone is best at hit@1 on identifiers (0.54) but weakest on questions. A
`search_code` call costs about 3.7x fewer tokens than reading the expected files whole.

## Performance

Measured on an M4 Pro, release build.

| What | Result |
| --- | --- |
| Search, 100k vectors x 384, k = 10 (`cargo bench --bench search`) | mean 1.20 ms; p50 1.19, p95 1.22, p99 1.28 ms over 2,000 queries, embedding excluded |
| Embedding, 64 chunks, 1 thread (`cargo bench --bench index`) | about 2.45 s, roughly 26 chunks/s |
| Embedding, 64 chunks, 8 threads | about 0.75 s, roughly 85 chunks/s |
| `pith index` on this repository, first run (485 chunks, 47 files) | 11.9 s |
| `pith index` again after editing one file (484 reused, 1 embedded) | 0.13 s wall clock, model load included |
| Keyword index build, this repository (526 chunks) | under 10 ms (`pith search --mode keyword` takes 0.01 s in all) |
| Keyword index build, 100k synthetic chunks of about 600 chars | about 1.4 s, once per process (`cargo test --release --lib build_time -- --ignored`) |

Embedding dominates indexing time. Batch size (8, 16, 32 or 64) makes no measurable difference,
only the thread count does. Pith embeds with batch size 1 by default: the int8 model quantizes
activations dynamically with scales shared across a batch, so a text's vector depends on its
batch neighbours (cosine to the batch-1 vector 0.996 to 0.9995). Queries and the index therefore
always use the same batch size.

Search is an int8 scan with an f32 rescore of the top candidates, so it is not strictly exact:
in principle a true top-k hit can be lost in the int8 pass.

## Known limitations

- The format is unstable until 1.0 (above).
- `source_changed` in results compares the file's mtime, in whole seconds, with the index time,
  so an edit in the same second as indexing is missed. (A file that was only touched is handled:
  the next `pith index` rewrites the index time.)
- Secret detection is path-based plus a PEM-header content check. An in-tree hardlink to a secret
  under another name is not caught by name, and headerless raw base64 keys are not caught by
  content. Directories named `.ssh`, `.gnupg`, `.aws` and `.kube` are never indexed.
- The chunk table is bounded on decompression: an index whose table compresses better than
  16,384:1 (plus a small allowance) is refused when opened.
- Model-download redirects are validated by host-name shape (https to a DNS name; no IP literals,
  `localhost`, `.local`, `.internal` and similar). There is no DNS lookup, so a public-looking
  name that resolves to a private address is not blocked.
- A `.pith` file is trusted as-is: its text is shown next to real file paths, in search output
  and to Claude through the MCP server. Do not commit `.pith` files, and do not use one from an
  untrusted source.
- Two `pith index` runs at once both succeed and the last write wins; the file itself is never
  corrupted.
- A mapped `.pith` file that is truncated by another program while Pith runs crashes the process
  (SIGBUS). Pith itself only replaces the file by atomic rename.

## Using Pith as a library

See [`examples/search_project.rs`](examples/search_project.rs): index a directory and search it in
about 20 lines (`cargo run --example search_project -- <directory> "<query>"`).

## Development

```sh
cargo test                # unit and integration tests
cargo bench               # search always; embedding only if the model is installed
cargo clippy --all-targets -- -D warnings
```

The ignored embedding tests need the real model (`pith init` first). The non-default
`test-hooks` feature exists only for this crate's own tests (it lets `pith mcp` use a fake
embedder) and is compiled only in debug builds, so release and bench builds never contain it.

## Security

See [SECURITY.md](SECURITY.md) for what Pith defends against and how to report a vulnerability.

## License

Pith is licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your
option. The embedding model is MIT-licensed (derived from
[BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), mirrored by
[Xenova](https://huggingface.co/Xenova/bge-small-en-v1.5)).

## Contribution

Any kind of contribution is welcome as a pull request. Unless you explicitly state otherwise,
any contribution intentionally submitted for inclusion in Pith by you shall be dual licensed as
above, without any additional terms or conditions.
