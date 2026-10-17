# Pith

Lightweight, embeddable vector database with built-in ONNX embeddings. Work in progress.

## Installing

```sh
cargo install pithdb   # installs the `pith` binary
pith init              # downloads the embedding model (~35 MB), once
```

- **ONNX Runtime is statically linked**: no separate install. The `ort` crate (2.0.0-rc.14, features
  `download-binaries`, `tls-rustls`) fetches a prebuilt, hash-verified ONNX Runtime 1.31 archive
  (~80 MB, cached in `~/Library/Caches/ort.pyke.io` or `~/.cache/ort.pyke.io`) from pyke's CDN
  **at build time**. That is the only network access during the build. Offline or air-gapped
  builds: point `ORT_LIB_PATH` at a self-built static ONNX Runtime.
- Verified on macOS arm64 (clean cache, ~16 s release build on an M4 Pro, binary ~32 MB, ~23 MB
  stripped; ORT accounts for ~30 MB). Linux x86_64 uses the same mechanism (a prebuilt archive
  exists for `x86_64-unknown-linux-gnu`) but has not been verified yet.
- `cargo install pithdb --no-default-features` builds without ONNX Runtime (no embeddings).
- At runtime Pith uses the network only for the model download (`pith init` / `pith model pull`):
  `bge-small-en-v1.5` int8 from the `Xenova/bge-small-en-v1.5` Hugging Face mirror, pinned to a
  commit and sha256-verified, stored in `~/.pith/models` (`$PITH_HOME/models` if set).
  License: MIT (derived from BAAI/bge-small-en-v1.5).
