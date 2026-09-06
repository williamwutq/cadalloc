# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `#![no_std]` crate targeting 64-bit platforms, handing out `Slice`
  (`{ ptr, len }`) values rather than raw pointers.
- `Config` trait: the single configuration surface, with the size-class
  constants `MIN_ALIGN`, `LNR_FLOOR`, `EXP_FLOOR`, `EXP_CEIL`, the
  `HEAP_BASE`/`HEAP_SIZE` compile-time heap region and `heap_base`/`heap_size`
  runtime accessors, and an `Atomics` backend as an associated type.
- Compile-time validation of the size constants (powers of two and ordering)
  via `assert_config_valid` and the `Config::VALIDATE` associated constant.
- `const_heap_base` / `const_heap_size` free functions for reading the
  compile-time heap region in `const` contexts.
- `Atomics` trait defining the eight required atomic operations (`atomic_load`,
  `atomic_store`, `atomic_cas`, `atomic_swap`, `atomic_inc`, `atomic_dec`,
  `atomic_add`, `atomic_sub`), with default compare-and-swap-loop
  implementations for the derived operations.
- `CoreAtomics`, an `AtomicU64`-based backend gated on
  `target_has_atomic = "64"`.
- Statically-computed bin counts on `Config` (`NUM_LINEAR`, `NUM_EXP`,
  `NUM_BINS`) derived from the size constants.
- Allocator core: `Allocator<C>` (a zero-sized handle whose state lives in the
  managed heap) with `init`, `alloc`, and `free`. Fixed size-class bins are
  lock-free Treiber stacks with an ABA tag; the bump pointer and oversized free
  list run under one refill spinlock. Block headers are a single `u64`
  (`size << 8 | free-byte`) in the `MIN_ALIGN` header region; no footer, no
  coalescing. Oversized blocks split subject to `SPLIT_MIN`/`CARVE_MAX`.
  `InitError` reports a misaligned or too-small heap.
- Heap metadata and `Allocator::verify`: an initialized heap begins with the
  `"CADALLOC"` marker (little-endian) and a 64-bit configuration fingerprint
  (packed base-2 logarithms of the power-of-two constants, avalanched with
  SplitMix64); `verify` guards against running over a heap laid out by a
  different configuration. `InitError` and `VerifyError` are `#[repr(i32)]`,
  so they are FFI-safe and each variant's discriminant is the status code the C
  shims return (success is `0`).
- `export_c_api!` macro (module `ffi`): emits unmangled `extern` entry points
  (`init`/`alloc`/`free`/`verify`) bound to one concrete configuration, so the
  allocator can be linked from C and other non-Rust callers. The calling
  convention defaults to `"C"` and can be overridden with an `abi:` line (e.g.
  `"system"`, `"sysv64"`, `"aapcs"`).

### Changed

### Deprecated

### Removed

### Fixed

### Security
