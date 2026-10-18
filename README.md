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
 * Built-in embeddings: `bge-small-en-v1.5` (int8, 384 dimensions) through a statically linked
   ONNX Runtime, downloaded once, pinned to a commit and sha256-verified
 * Gitignore-aware indexing (`.gitignore`, `.ignore`, `.pithignore`) that skips secrets by name
   and by content
 * Read-only MCP server: `search_code`, `get_chunk`, `stats`; results show how stale the index is
 * Also usable as a Rust library (storage, index and ingest are separate modules)

## Status

Pith is `0.1`. macOS on Apple Silicon is the supported platform: that is where it is built and
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

Files over 1 MiB are skipped. `pith index` adds `.pith` and `.pith.lock` to an existing
`.gitignore`; it never creates one.

### Claude Code (MCP)

```sh
claude mcp add pith -- pith mcp                        # serves the project in the current directory
claude mcp add pith -- pith mcp --project /path/to/project
```

| Tool | Arguments | Returns |
| --- | --- | --- |
| `search_code` | `query`, `k` (default 8, max 20) | ranked chunks with path, line range, score and chunk id |
| `get_chunk` | `chunk_id` | the full text of one chunk |
| `stats` | none | chunk count, model, index age |

Each response is capped at about 4,000 tokens, with at most 3 chunks per file and about 15 lines
per snippet. The server never indexes or writes: run `pith index` first, and again after big
changes (a reindex is picked up without restarting). The embedding model loads on the first
search and is dropped after 5 idle minutes.

## Performance

Measured on an M4 Pro, release build.

| What | Result |
| --- | --- |
| Search, 100k vectors x 384, k = 10 (`cargo bench --bench search`) | mean 1.20 ms; p50 1.19, p95 1.22, p99 1.28 ms over 2,000 queries, embedding excluded |
| Embedding, 64 chunks, 1 thread (`cargo bench --bench index`) | about 2.45 s, roughly 26 chunks/s |
| Embedding, 64 chunks, 8 threads | about 0.75 s, roughly 85 chunks/s |
| `pith index` on this repository (342 chunks, 38 files) | 8.4 s |

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
  so an edit in the same second as indexing is missed.
- Secret detection is path-based plus a PEM-header content check. An in-tree hardlink to a secret
  under another name is not caught by name, and headerless raw base64 keys are not caught by
  content. Directories named `.ssh`, `.gnupg`, `.aws` and `.kube` are never indexed.
- The chunk table is bounded on decompression: an index whose table compresses better than
  16,384:1 (plus a small allowance) is refused at write time.
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

## Development

```sh
cargo test                # unit and integration tests
cargo bench               # search always; embedding only if the model is installed
cargo clippy --all-targets -- -D warnings
```

The ignored embedding tests need the real model (`pith init` first). The non-default
`test-hooks` feature exists only for this crate's own tests (it lets `pith mcp` use a fake
embedder) and is compiled only in debug builds, so release and bench builds never contain it.

## License

Pith is licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your
option. The embedding model is MIT-licensed (derived from
[BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), mirrored by
[Xenova](https://huggingface.co/Xenova/bge-small-en-v1.5)).

## Contribution

Any kind of contribution is welcome as a pull request. Unless you explicitly state otherwise,
any contribution intentionally submitted for inclusion in Pith by you shall be dual licensed as
above, without any additional terms or conditions.
