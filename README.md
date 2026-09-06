# cadalloc

A minimal, configurable, embeddable allocator with segregated free lists —
built for managing a **fixed region of memory** from many threads.

`cadalloc` is `#![no_std]`, assumes a **64-bit** target, and hands out
[`Slice`]s (`{ ptr: u64, len: u64 }`) rather than raw pointers. It coordinates
its size-class free lists with atomics *inside the managed heap*, so it needs no
global allocator, no thread-local state, and no dependencies.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Crates.io](https://img.shields.io/crates/v/cadalloc)](https://crates.io/crates/cadalloc)
[![Docs.rs](https://img.shields.io/docsrs/cadalloc)](https://docs.rs/cadalloc)

> **Status: experimental (`0.1`).** The API is not yet stable. Read the
> tradeoffs below before depending on it.

## Is this the allocator you want?

`cadalloc` is deliberately narrow. It is a good fit when all of these hold:

- You own a **single, fixed region** of memory to sub-allocate — a static
  buffer, an `mmap`, a device/shared-memory window, a VM/interpreter heap — and
  its size is known up front.
- You want to address that memory by **`(offset, len)`** rather than by native
  pointer. This is the unusual, differentiating property: the region need not be
  in the allocating CPU's own address space. Managing a **GPU/device buffer**, a
  **shared-memory arena between processes**, or a **WASM-style linear memory** is
  exactly what the `u64`-slice API is for.
- The workload's allocation sizes are **reasonably homogeneous / bounded**, or
  churn is high enough that freed blocks get reused at the same sizes.
- You need it to be **`no_std`, small, or dependency-free**.

It is **not** a general-purpose `malloc` replacement:

- **It never coalesces free blocks.** Blocks are class-exact and are returned to
  their size-class free list on `free`; adjacent free blocks are never merged.
  This keeps the hot path short and predictable, but means fragmentation is
  **permanent**: memory freed at one size class cannot later satisfy a larger
  class. Under a long-lived, heterogeneous workload it will strand memory. Under
  a fixed-heap workload with bounded sizes — the target — it is fast and stable.
- Growing a block via `realloc` always **relocates** (allocate + copy + free),
  precisely because there is no coalescing.
- **The optional `#[global_allocator]` support is limited.** There is a
  `GlobalAlloc` impl behind the `alloc` feature (see
  [As the global allocator](#as-the-global-allocator)), but it inherits the two
  points above and can only satisfy alignments up to `MIN_ALIGN` — so it fits a
  fixed-heap program, not arbitrary `malloc` traffic. The unstable
  `core::alloc::Allocator` trait is not implemented.

If you want a general heap allocator with coalescing, reach for [`talc`],
[`linked_list_allocator`], [`buddy_system_allocator`], or [`rlsf`] instead.
`cadalloc` trades generality for a tiny, offset-addressed core.

## Design at a glance

- **Segregated free lists.** Size classes are computed at compile time from four
  power-of-two constants: a linear region (step `MIN_ALIGN`), an exponential
  region (each octave split in two), and one oversized bin above `EXP_CEIL`.
- **Lock-free bins + one refill lock.** Each fixed size class is a lock-free
  Treiber stack (with a low-bit ABA tag); the bump pointer, the oversized list,
  and block splitting run under a single spinlock. There are **no thread-local
  caches** — one shared set of bins, kept simple.
- **One-word headers, no footer.** Every block starts with a single `u64`
  packing its size and a free bit (the low alignment bits are always zero, so no
  shifting). No footers, no coalescing metadata.
- **Configuration fingerprint.** An initialized heap is stamped with a
  `"CADALLOC"` marker and a 64-bit fingerprint of the layout constants;
  [`verify`] rejects a heap laid out by a different configuration before it can
  corrupt anything — useful for persistent or shared-memory regions.

## Installation

```toml
[dependencies]
cadalloc = "0.1"

# Or, to also use it as the program's `#[global_allocator]`:
# cadalloc = { version = "0.1", features = ["alloc"] }
```

## Configuring an allocator

The whole configuration surface is one trait, [`Config`]: name an atomics
backend and the four size-class constants (heap region optional at compile time).

```rust
use cadalloc::{Config, CoreAtomics};

struct MyConfig;

impl Config for MyConfig {
    type Atomics = CoreAtomics;

    const MIN_ALIGN: u64 = 16;    // min alignment & linear class step
    const LNR_FLOOR: u64 = 32;    // smallest class (>= 2 * MIN_ALIGN)
    const EXP_FLOOR: u64 = 256;   // linear -> exponential boundary
    const EXP_CEIL: u64 = 65536;  // largest class; above spills oversized

    const HEAP_BASE: u64 = 0x2000_0000; // a real, mapped region
    const HEAP_SIZE: u64 = 1 << 20;     // 1 MiB
}

// Optional but recommended: validate the constants at compile time.
const _: () = cadalloc::assert_config_valid::<MyConfig>();
```

When the region is only known at runtime, leave `HEAP_BASE`/`HEAP_SIZE` at their
defaults and override the `heap_base()` / `heap_size()` methods instead.

## Usage

```rust,no_run
use cadalloc::{CadAlloc, Config, CoreAtomics};

# struct MyConfig;
# impl Config for MyConfig {
#     type Atomics = CoreAtomics;
#     const MIN_ALIGN: u64 = 16;
#     const LNR_FLOOR: u64 = 32;
#     const EXP_FLOOR: u64 = 256;
#     const EXP_CEIL: u64 = 65536;
#     const HEAP_BASE: u64 = 0x2000_0000;
#     const HEAP_SIZE: u64 = 1 << 20;
# }
let a = CadAlloc::<MyConfig>::new();
a.init().expect("prepare the heap");   // once, before any allocation
assert!(a.verify().is_ok());           // marker + config fingerprint

let s = a.alloc(100);                  // Slice { ptr, len }; len may exceed 100
if !s.is_null() {
    // ... use the region [s.ptr, s.ptr + s.len) ...
    let s = a.realloc(s, 4000);        // grow (relocates) or shrink in place
    a.free(s);
}
```

The `CadAlloc` handle is zero-sized — all state lives in the heap — so it can be
reconstructed anywhere (e.g. a `static`) and shared across threads once `init`
has run.

### Exporting to C

`export_c_api!` emits unmangled `extern "C"` shims bound to one configuration:

```rust
# use cadalloc::{Config, CoreAtomics};
# struct MyConfig;
# impl Config for MyConfig {
#     type Atomics = CoreAtomics;
#     const MIN_ALIGN: u64 = 16;
#     const LNR_FLOOR: u64 = 32;
#     const EXP_FLOOR: u64 = 256;
#     const EXP_CEIL: u64 = 65536;
#     const HEAP_BASE: u64 = 0x2000_0000;
#     const HEAP_SIZE: u64 = 1 << 20;
# }
cadalloc::export_c_api! {
    config: MyConfig,
    init: cad_init,
    alloc: cad_alloc,
    realloc: cad_realloc,
    free: cad_free,
    verify: cad_verify,
}
```

`Slice` is `#[repr(C)]` (`struct { uint64_t ptr, len; }`), passed and returned by
value. See the `ffi` module docs for the ABI and status codes.

### As the global allocator

With the `alloc` feature, a configuration can back the `alloc` crate:

```rust,ignore
#[global_allocator]
static GLOBAL: cadalloc::CadAlloc<MyConfig> = cadalloc::CadAlloc::new();

fn main() {
    GLOBAL.init().expect("prepare the heap"); // before the first allocation
    // `Box`, `Vec`, ... now allocate from the fixed heap.
}
```

Two caveats (see the `global` module docs): `init()` must run before anything
allocates — there is no lazy initialization — and only alignments up to
`MIN_ALIGN` can be satisfied; a stronger request returns null, which the `alloc`
crate turns into an allocation-error abort.

## Two region types

Following C++'s `string` / `string_view`, memory is described by two types:

- **`Slice`** — an *owning handle* to an allocation, returned by `alloc` and
  consumed by `free`/`realloc`. It is **move-only** (not `Copy`/`Clone`), so the
  compiler rules out double frees and use-after-free through the handle: once you
  pass a `Slice` to `free`, you cannot name it again.
- **`SliceView<'a>`** — a cheap, `Copy` *non-owning view*, borrowed for `'a`. It
  carries the inspection and sub-region operations (`subslice`, `split_at`,
  `contains`, `as_bytes`, …). Borrow an owner as one with `slice.view()`; the
  view — and any `&'a [u8]` taken from it — is tied to that borrow, so the owner
  cannot be freed while a view of it is alive.

## Safety

- **Constructing either type from a raw address is `unsafe`.** `Slice::new` /
  `SliceView::new` assert that the address really names owned memory (and, for a
  view, that it lives for the chosen `'a`) — you only reach for them to
  reconstruct an address you already own, e.g. across FFI. Handles from `alloc`
  and views from `slice.view()` need no `unsafe`.
- **Double-free detection is best-effort and debug-only.** The move-only `Slice`
  makes an honest double free a *compile* error; the runtime guard exists for the
  FFI path, where a caller can fabricate a second handle. `free` refuses a block
  already marked free (and asserts in debug builds), but release builds do not
  defend against every misuse.

## Concurrency testing

Because the bins are lock-free, serial tests can't establish correctness. Two
layers guard it:

- **A multithreaded stress test** (`cargo test`) hammers one heap from many
  threads doing alloc/free/realloc churn, fingerprinting every live block so any
  overlap or lost/duplicated free is caught as corruption.
- **A [`loom`] model check** exhaustively explores thread interleavings and
  memory orderings for the isolated Treiber-stack push/pop protocol (including
  its ABA tag). Run it with:

  ```sh
  RUSTFLAGS="--cfg loom" cargo test --lib treiber
  ```

  `loom` is a dev-dependency only under `--cfg loom`; a normal `cargo test`
  never compiles or downloads it. Loom cannot model the whole allocator (a real
  heap is far too many atomic locations, and it addresses memory by raw `u64`),
  so it targets the one genuinely subtle lock-free component while the stress
  test exercises the whole thing.

## Development

```sh
cargo test                                            # unit + stress + doctests
cargo test --features alloc                           # + the GlobalAlloc adapter
cargo clippy --all-targets --all-features -- -D warnings   # what CI gates on
cargo fmt --check
cargo doc --no-deps --all-features --open
cargo build --target thumbv7em-none-eabihf            # bare-metal smoke build
RUSTFLAGS="--cfg loom" cargo test --lib treiber       # loom model check
```

CI runs on `master` and `main`: [`ci.yml`](.github/workflows/ci.yml) tests the
stable/beta/nightly × Linux/macOS/Windows matrix, and
[`check.yml`](.github/workflows/check.yml) gates clippy, formatting, and docs.

See [`PLANNED.md`](PLANNED.md) for the design surface and [`CHANGELOG.md`](CHANGELOG.md)
for changes.

## License

MIT — see [LICENSE](LICENSE).

[`Slice`]: https://docs.rs/cadalloc/latest/cadalloc/struct.Slice.html
[`Config`]: https://docs.rs/cadalloc/latest/cadalloc/trait.Config.html
[`verify`]: https://docs.rs/cadalloc/latest/cadalloc/struct.CadAlloc.html#method.verify
[`loom`]: https://docs.rs/loom
[`talc`]: https://crates.io/crates/talc
[`linked_list_allocator`]: https://crates.io/crates/linked_list_allocator
[`buddy_system_allocator`]: https://crates.io/crates/buddy_system_allocator
[`rlsf`]: https://crates.io/crates/rlsf
