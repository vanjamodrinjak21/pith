# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html),
specifically the [variant used by Rust](http://doc.crates.io/manifest.html#the-version-field).
The on-disk format is unstable until 1.0.

## [Unreleased] - TBD
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
