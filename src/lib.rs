//! `cadalloc` — a minimal, configurable, embeddable allocator.
//!
//! `cadalloc` is a general-purpose allocator built around segregated free
//! lists. It targets multithreaded environments but deliberately does *not*
//! use thread-local free lists: the design goal is a small, predictable core
//! that is easy to embed and reason about, rather than the last few percent of
//! contended-allocation throughput.
//!
//! The crate is `#![no_std]` and assumes a **64-bit** target: addresses and
//! lengths are `u64` throughout, and the allocator hands out [`Slice`]s
//! (`{ ptr, len }`) rather than raw pointers.
//!
//! # Configuring an allocator
//!
//! All tuning lives in one trait, [`Config`]. Implementing it is the entire
//! configuration step — pick an [`Atomics`] backend and the four size-class
//! constants:
//!
//! ```
//! use cadalloc::{Config, CoreAtomics};
//!
//! struct MyConfig;
//!
//! impl Config for MyConfig {
//!     type Atomics = CoreAtomics;
//!
//!     const MIN_ALIGN: u64 = 16;    // min alignment & linear class step
//!     const LNR_FLOOR: u64 = 32;    // smallest class
//!     const EXP_FLOOR: u64 = 256;   // linear -> exponential boundary
//!     const EXP_CEIL: u64 = 65536;  // largest class; above spills oversized
//! }
//!
//! // Optional but recommended: validate the constants at compile time.
//! const _: () = cadalloc::assert_config_valid::<MyConfig>();
//! ```
//!
//! See the [`config`] module for the size-class scheme and the heap-region
//! accessors, and the [`atomic`] module for the atomic contract. The allocator
//! core that consumes a `Config` is still being built out.

#![no_std]

pub mod allocator;
pub mod atomic;
pub mod config;
pub mod ffi;
pub mod slice;

pub use allocator::{CadAlloc, InitError, VerifyError};
pub use atomic::Atomics;
#[cfg(target_has_atomic = "64")]
pub use atomic::CoreAtomics;
pub use config::{Config, assert_config_valid, const_heap_base, const_heap_size};
pub use slice::Slice;

/// Computes the power of two `2^n` as a `u64`.
///
/// A thin wrapper over `1u64 << n`: it just spells the intent out at the call
/// site, which is handy for the power-of-two size-class constants a
/// [`Config`](crate::Config) is built from. It is `const`-evaluable, so it works
/// anywhere a constant is expected. `n` must be less than 64, or the shift
/// overflows (a compile error in `const` context).
///
/// # Examples
///
/// ```
/// use cadalloc::pow2;
///
/// const EXP_FLOOR: u64 = pow2!(8); // 256
/// const EXP_CEIL: u64 = pow2!(16); // 65536
/// assert_eq!(pow2!(0), 1);
/// assert_eq!(EXP_FLOOR, 256);
/// ```
#[macro_export]
macro_rules! pow2 {
    ($n:expr) => {
        1u64 << ($n)
    };
}

#[cfg(test)]
mod test;
