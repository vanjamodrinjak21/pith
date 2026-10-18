# Security Policy

## Supported versions

Pith is pre-1.0. Only the latest release on [crates.io](https://crates.io/crates/pithdb) receives
security fixes.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Report it privately through GitHub:
[Report a vulnerability](https://github.com/vanjamodrinjak21/pith/security/advisories/new).

Include the Pith version (`pith --version`), your platform, and the smallest steps or input file
that reproduce the problem. You can expect an acknowledgement within a week. Fixes ship as a new
patch release and the advisory credits you unless you prefer otherwise.

## What Pith defends against

Pith reads your source tree, writes one `.pith` file, makes one network request (the model
download) and answers MCP requests from a local client. These are the boundaries it hardens:

- **Secrets stay out of the index.** Indexing skips secret files by name (`.env`, keys,
  credentials, `.ssh`, `.gnupg`, `.aws`, `.kube`) and by content (PEM private-key headers), never
  follows symlinks out of the project, and skips files over 1 MiB.
- **A `.pith` file is checked before it is used.** Every section is checksummed, sizes are bounded
  before anything is decompressed (zstd window and ratio limits), and a malformed file produces an
  error, never a panic or an unbounded allocation.
- **Output is sanitised.** Text that comes from the index or the file system has terminal control
  and bidirectional characters removed before it is printed or sent to the MCP client.
- **The MCP server is read-only.** It exposes `search_code`, `get_chunk` and `stats`, never takes
  the writer lock, caps every response at about 4,000 tokens and every input line at 1 MiB.
- **The model download is verified.** The manifest is compiled in and pinned to a commit. Each
  file is size-capped, sha256-checked, written to a private temporary file and renamed into
  place. Only HTTPS is accepted, and redirects must stay on public DNS names.
- **Writes are atomic.** A single writer holds an advisory lock; a crash cannot leave a
  half-written index.

## Known limits (not treated as vulnerabilities)

These are documented in the README under "Known limitations":

- A `.pith` file is trusted as written. Its text appears next to real file paths, so a `.pith`
  from an untrusted source can show a model misleading content. Do not commit or download them.
- Secret detection is name- and header-based, not a scanner: an in-tree hardlink to a secret under
  another name, or a key without a PEM header, can be indexed.
- Redirect checks look at host-name shape and do not resolve DNS, so a public-looking name that
  resolves to a private address is not blocked.
- Truncating a mapped `.pith` file from another program crashes the process (SIGBUS).
- Supply chain: `cargo install` downloads a hash-verified ONNX Runtime archive at build time.
