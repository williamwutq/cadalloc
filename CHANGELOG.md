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

### Changed

### Deprecated

### Removed

### Fixed

### Security
