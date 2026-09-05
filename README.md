# cadalloc

A minimal, configurable, embeddable allocator with segregated free lists.

`cadalloc` is a general-purpose memory allocator built around segregated free
lists. It targets multithreaded environments but deliberately does *not* use
thread-local free lists, trading a little contended-allocation throughput for a
small, predictable core that is easy to embed and reason about.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
<!-- For a published crate, add:
[![Crates.io](https://img.shields.io/crates/v/cadalloc)](https://crates.io/crates/cadalloc)
[![Docs.rs](https://img.shields.io/docsrs/cadalloc)](https://docs.rs/cadalloc)
-->

## Features

- **Segregated free lists** — allocations are served from size-class free
  lists, keeping the common-case allocate/free path short and predictable.
- **Multithread-ready, no thread-local caches** — safe under concurrent use
  without the per-thread state (and cross-thread free complexity) that
  thread-local free lists introduce.
- **Configurable** — size classes and policy are meant to be tuned to the
  target workload rather than fixed by the crate.
- **Minimal and embeddable** — a small core with no required dependencies,
  intended as a building block.

## Installation

```toml
[dependencies]
cadalloc = "0.1"
```

## Usage

```rust
fn main() {
    println!("{}", cadalloc::version());
}
```

The allocator API is under active development; see [`PLANNED.md`](PLANNED.md)
for the design surface.

## Development

```sh
cargo test                                   # tests, doctests
cargo clippy --all-targets --all-features    # what CI gates on
cargo fmt --check
cargo doc --no-deps --open
```

CI runs on `master` and `main`: [`ci.yml`](.github/workflows/ci.yml) tests the
stable/beta/nightly × Linux/macOS/Windows matrix, and
[`check.yml`](.github/workflows/check.yml) gates clippy, formatting, and docs.

## License

MIT — see [LICENSE](LICENSE).
