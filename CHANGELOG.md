# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html),
specifically the [variant used by Rust](http://doc.crates.io/manifest.html#the-version-field).
The on-disk format is unstable until 1.0.

## [Unreleased] - TBD

## [0.4.0] - 2026-10-18
### Added
- Hybrid search, the new default: reciprocal rank fusion (k = 60) of the top 50 vector hits and
  the top 50 BM25 hits. The BM25 index (k1 = 1.2, b = 0.75) is built in memory from the chunk
  texts on the first keyword or hybrid query in each process, with an identifier-aware
  tokenizer (`parseHttpURL` also yields `parse`, `http`, `url`); the file format is unchanged.
  Memory is bounded at 1,024 tokens per chunk and a 32 Mi budget of terms plus postings (a
  100,000-chunk index used about a third of it; peak RSS +260 MiB, build 0.8 to 1.1 s).
- `pith search --mode vector|keyword|hybrid`.
- Without the embedding model, hybrid search (CLI, MCP `search_code`, `pith ui`) falls back to
  keyword search with a one-line notice instead of failing. `--mode vector` still asks for
  `pith init`.
- `pith eval <cases.jsonl>`: hit@1, hit@3, hit@k, MRR and token cost per mode (vector is skipped
  without a model), and `eval/pith.jsonl`, 42 hand-checked cases for this repository. Results
  in the README: hybrid MRR 0.57 and hit@10 0.95, vector 0.56 and 0.88, keyword 0.54 and 0.93.
- Library: `search_with`, `SearchMode`, `KEYWORD_ONLY`, `is_no_model` and
  `load_optional_embedder`. The `eval` module is public for the binary but hidden and unstable.

### Changed
- `search_project` uses hybrid search. Hit scores stay cosine similarities when the model is
  loaded; keyword-only scores are BM25 normalised to the top hit.
- The missing-model error now says to run `pith init` (was `pith model pull`).
- `.pithignore` excludes `eval/` from this repository's own index.

## [0.3.0] - 2026-10-18
**The on-disk format changed (version 2): rebuild existing indexes with `pith index --reindex`.**
### Added
- Incremental `pith index`: a chunk whose text is unchanged reuses its vector from the existing
  index (looked up by a blake3 hash of the text), so only new or edited chunks are embedded.
  Renamed, moved and duplicated content is not embedded again, and identical new chunks are
  embedded once. When no chunk, file time or git head changed, `.pith` is left as it was and the
  run reports "index is up to date". `--reindex` still embeds
  everything. The summary line now reads `indexed N chunks from M files (X reused, Y embedded)`.
- `IndexReport::{reused, embedded, up_to_date, summary}`, `Db::chunks` and `Db::text_hash`.

### Changed
- On-disk format version 2: each chunk stores the blake3 hash of its text, verified on open.
  Version-1 files are refused with "run `pith index --reindex`".

## [0.2.0] - 2026-10-18
### Added
- `pith ui`: a terminal UI (default `tui` feature) with four views: Stack (directory -> file ->
  chunk with share-of-index bars), Map (2-D PCA of the vectors with density shading), RAG (what
  `search_code` sends Claude, with dropped/trimmed hits and a token meter) and Index (live,
  cancellable reindex with progress). `pith ui --index` indexes first.
- `index_project_with_progress`, `Progress` and `Error::Cancelled` in the library.
- `VectorIndex::scale`.

## [0.1.1] - 2026-10-18
### Added
- `SECURITY.md`, `CHANGELOG.md` and a library example (`examples/search_project.rs`).
- A CI job that builds and tests on the MSRV, now documented in the README.

### Fixed
- `rust-version` is 1.92 (0.1.0 declared 1.89, but the ONNX Runtime bindings require 1.92).

## [0.1.0] - 2026-10-18
### Added
- Single-file `.pith` database: checksummed sections, zstd chunk table, memory-mapped vectors,
  atomic writes and a single-writer lock.
- Exact int8 SIMD vector scan (NEON, AVX2) with f32 rescoring; cosine similarity.
- Gitignore-aware indexing with a secrets deny list (by name and PEM content).
- Built-in ONNX embeddings (`bge-small-en-v1.5`, int8, 384 dimensions) with a verified,
  commit-pinned model download.
- CLI: `pith init`, `index`, `search`, `stats`, `model`, `mcp`.
- Read-only stdio MCP server with `search_code`, `get_chunk` and `stats`, capped responses and
  index staleness in every result.
