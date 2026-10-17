# Pith

A local, single-file vector database with built-in ONNX embeddings, plus an MCP server so Claude
Code can search your codebase by meaning. One binary, no daemon, no API key, no cloud: the only
network access is a one-time model download.

> **The on-disk format is unstable until 1.0.** Any release before 1.0 may change it. Pith
> refuses to open an index it does not understand; rebuild with `pith index --reindex`. Do not
> treat `.pith/` as durable data: it is a cache of your source tree.

Pith is `0.1`: indexing, search, stats and the MCP server work; there is no HNSW (search is an
exact-ish brute-force scan, see Performance), no incremental indexing, and no non-code ingestion.

## Install

```sh
cargo install pithdb      # installs the `pith` binary
pith init                 # asks, then downloads the embedding model (~34 MB), once
```

## Use

```sh
cd your-project
pith index                # writes ./.pith (a single file); --reindex overwrites a stale one
pith search "where are retries handled" -k 5
pith stats
```

Respects `.gitignore`, `.ignore` and a project-local `.pithignore`. Secrets (`.env`, keys,
credentials) are skipped by name and by content. Files over 1 MiB are skipped.

### Claude Code (MCP)

```sh
claude mcp add pith -- pith mcp          # serves the project in the current directory
claude mcp add pith -- pith mcp --project /path/to/project
```

Read-only tools: `search_code {query, k?}`, `get_chunk {chunk_id}`, `stats`. Results are capped
at about 4,000 tokens (3 chunks per file, k up to 20). The server never indexes: run
`pith index` first and re-run it after big changes (a reindex is picked up without restarting).
The model is loaded on the first search and dropped after 5 idle minutes.

## Model and linking

- Model: `bge-small-en-v1.5` int8, 384 dimensions, from the `Xenova/bge-small-en-v1.5`
  Hugging Face mirror, pinned to a commit and sha256-verified. Stored in `~/.pith/models`
  (`$PITH_HOME/models` if set). License: MIT (derived from BAAI/bge-small-en-v1.5).
- **ONNX Runtime is statically linked**: no separate install. The `ort` crate (2.0.0-rc.14,
  features `download-binaries`, `tls-rustls`) fetches a prebuilt, hash-verified ONNX Runtime
  archive (~80 MB, cached in `~/Library/Caches/ort.pyke.io` or `~/.cache/ort.pyke.io`) from pyke's
  CDN **at build time**. That is the only network access during the build. Offline or air-gapped
  builds: point `ORT_LIB_PATH` at a self-built static ONNX Runtime.
- Verified on macOS arm64 only (M4 Pro; release binary about 38 MB with the MCP server). Linux
  x86_64 uses the same mechanism and is covered by CI, but I have not run it by hand.
- `cargo install pithdb --no-default-features` builds without ONNX Runtime (no embeddings).
- Embedding uses batch size 1 by default. The int8 model quantizes activations dynamically with
  scales shared across a batch, so a text's vector depends on its batch neighbours (cosine to the
  batch-1 vector 0.996 to 0.9995 when measured), and batching gave no speedup. An index must be
  queried with the same batch size it was built with; the defaults guarantee that.

## Performance

Measured on an M4 Pro, release build.

| What | Result |
| --- | --- |
| Search, 100k x 384 random vectors, k = 10 (`cargo bench --bench search`) | mean 1.20 ms; p50 1.19, p95 1.22, p99 1.28 ms over 2,000 queries (embedding excluded) |
| Embedding, batch 1, 8 threads, 830-char chunks | 43.5 chunks/s |
| Embedding, batch 32, 8 threads | 45.4 chunks/s |
| Embedding, batch 1, 4 threads / 1 thread | 33.3 / 15.5 chunks/s |
| Indexing this repo's `src/` (134 chunks) | 3.2 s |

Embedding dominates: a 3,000-chunk repository takes about a minute to index. The batch 8/16/64
and per-thread grid in `cargo bench --bench index` is not measured yet (it needs the model
installed; the bench skips itself otherwise). Search is a brute-force int8 scan with an f32
rescore of the top candidates, not strictly exact: the true top-k can in principle be lost in the
int8 pass.

## Known limitations

- Format unstable until 1.0 (above).
- `source_changed` in results compares the file's mtime in whole seconds with the index time, so
  an edit within the same second as indexing is missed.
- Secret detection is path-based plus a PEM-header content check: an in-tree hardlink to a secret
  under another name is not caught by name, and headerless raw base64 keys are not caught by
  content. Directories named `.ssh`, `.gnupg`, `.aws` and `.kube` are never indexed.
- The index file bounds table decompression, so an index whose chunk table compresses better than
  16,384:1 (plus a small allowance) is refused at write time.
- Model download redirects are validated by host-name shape (https to a DNS name; no IP literals,
  `localhost`, `.local`, `.internal` and similar). No DNS lookup is made, so a public-looking
  host name that resolves to a private address is not blocked.
- Unix only in practice: on platforms with neither `F_GETPATH` nor `/proc` (BSDs) every file is
  skipped as unreadable.

## Development

```sh
cargo test                # includes the AVX2 path where the CPU has it
cargo bench               # search always; embedding only if the model is installed
```

The non-default `test-hooks` feature exists for this crate's own tests (it lets `pith mcp` use a
fake embedder). It is test-only: never enable it in anything you ship.

## License

MIT OR Apache-2.0.
