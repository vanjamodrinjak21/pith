# Pith

A local, single-file vector database with built-in ONNX embeddings, plus an MCP server so Claude
Code can search your codebase by meaning. One binary, no daemon, no API key, no cloud: the only
network access is a one-time model download.

> **The on-disk format is unstable until 1.0.** Any release before 1.0 may change it. Pith
> refuses to open an index it does not understand; rebuild with `pith index --reindex`. Do not
> treat `.pith` as durable data: it is a cache of your source tree.

Pith is `0.1`: indexing, search, stats and the MCP server work; there is no HNSW (search is an
exact-ish brute-force scan, see Performance), no incremental indexing, and no non-code ingestion.

## Install

```sh
cargo install pithdb      # installs the `pith` binary (once published; until then `cargo install --path .`)
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
- **macOS (Apple Silicon) is the supported platform for 0.1** (verified on an M4 Pro; release
  binary about 38 MB with the MCP server). Linux and Windows are not tested or supported yet; the
  code has an AVX2 path for x86_64 that has never run on real hardware.
- `cargo install pithdb --no-default-features` builds without ONNX Runtime (no embeddings).
- Embedding uses batch size 1 by default. The int8 model quantizes activations dynamically with
  scales shared across a batch, so a text's vector depends on its batch neighbours (cosine to the
  batch-1 vector 0.996 to 0.9995 when measured), and batching gave no speedup. An index must be
  queried with the same batch size it was built with; the defaults guarantee that.

## Performance

Measured on an M4 Pro, release build. The search row is from this repo's bench. The embedding
rows come from the ignored real-model embedding test (`cargo test --release --test embed -- --ignored
--nocapture --test-threads=1`, 830-char chunks), not from `cargo bench`.

| What | Result |
| --- | --- |
| Search, 100k x 384 random vectors, k = 10 (`cargo bench --bench search`) | mean 1.20 ms; p50 1.19, p95 1.22, p99 1.28 ms over 2,000 queries (embedding excluded) |
| Embedding, batch 1, 8 threads, 830-char chunks | 43.5 chunks/s |
| Embedding, batch 32, 8 threads | 45.4 chunks/s |
| Embedding, batch 1, 4 threads / 1 thread | 33.3 / 15.5 chunks/s |
| `pith index` on an early copy of this repo's `src/` (134 chunks) | 3.2 s |

Embedding dominates: a 3,000-chunk repository takes about a minute to index. `cargo bench --bench
index` (64 chunks, M4 Pro, model installed) agrees: batch size 8/16/32/64 makes no difference
(about 2.45 s at 1 thread, 0.75 s at 8 threads, roughly 26 and 85 chunks/s); only the thread count
matters. The bench skips itself when the model is not installed. Indexing this repo (342 chunks)
took 8.4 s. Search is a brute-force int8 scan with an f32
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
- A `.pith` file is trusted as-is: its chunk text is shown next to real file paths, in search
  output and to Claude through the MCP server. A `.pith` committed to a repository you clone is
  used like your own. Do not commit `.pith` files, and do not use one from an untrusted source.
- Unix only in practice: on platforms with neither `F_GETPATH` nor `/proc` (BSDs) every file is
  skipped as unreadable.

## Development

```sh
cargo test                # includes the AVX2 path where the CPU has it
cargo bench               # search always; embedding only if the model is installed
```

The non-default `test-hooks` feature exists for this crate's own tests (it lets `pith mcp` use a
fake embedder). It is compiled only in debug builds, so release and bench profiles never
contain it even if the feature is switched on.

## License

MIT OR Apache-2.0.
