//! The configuration surface: the single trait a user implements to stand up an
//! allocator, plus compile-time validation of its constants.
//!
//! Everything the allocator needs to be tuned is gathered into one
//! [`Config`] trait. Implementing it is the whole configuration step:
//!
//! ```
//! use cadalloc::{Config, CoreAtomics};
//!
//! struct MyConfig;
//!
//! impl Config for MyConfig {
//!     type Atomics = CoreAtomics;
//!
//!     const MIN_ALIGN: u64 = 16;
//!     const LNR_FLOOR: u64 = 16;
//!     const EXP_FLOOR: u64 = 256;
//!     const EXP_CEIL: u64 = 65536;
//! }
//!
//! // The constants are checked at compile time; this fails to build if they
//! // violate the power-of-two or ordering rules.
//! const _: () = cadalloc::assert_config_valid::<MyConfig>();
//! ```
//!
//! ## Size classes
//!
//! A request size maps to one of three regions, delimited by the four
//! constants (all of which must be powers of two):
//!
//! * **Linear** — from [`LNR_FLOOR`](Config::LNR_FLOOR) up to
//!   [`EXP_FLOOR`](Config::EXP_FLOOR), classes step by
//!   [`MIN_ALIGN`](Config::MIN_ALIGN). Small allocations, tightly spaced.
//! * **Exponential** — from [`EXP_FLOOR`](Config::EXP_FLOOR) up to
//!   [`EXP_CEIL`](Config::EXP_CEIL), each power-of-two octave `[2^n, 2^(n+1))`
//!   is split into two classes: `2^n` and `3 * 2^(n-1)` (i.e. `1.5 * 2^n`).
//! * **Oversized** — anything larger than [`EXP_CEIL`](Config::EXP_CEIL) spills
//!   into a single oversized bucket.
//!
//! [`MIN_ALIGN`](Config::MIN_ALIGN) is also the minimum alignment of *every*
//! allocation, so it must be greater than `2^3` (i.e. at least 16).
//!
//! ## Heap region
//!
//! [`HEAP_BASE`](Config::HEAP_BASE) / [`HEAP_SIZE`](Config::HEAP_SIZE) describe
//! the region the allocator manages *when it is known at compile time* — the
//! common embedded case of a statically reserved buffer. When the region is
//! only known at runtime, leave those at `0` ("not applicable") and override
//! [`heap_base`](Config::heap_base) / [`heap_size`](Config::heap_size). The
//! free functions [`const_heap_base`] / [`const_heap_size`] read the
//! compile-time values in `const` contexts.

use crate::atomic::Atomics;

/// The complete configuration of a `cadalloc` allocator.
///
/// A `Config` is a zero-sized policy type carrying only associated constants,
/// an [`Atomics`] backend, and (optionally) heap accessors. See the
/// [module documentation](self) for the size-class scheme and an example.
///
/// The four size constants plus `SPLIT_MIN` must all be **powers of two**, and
/// the size constants must satisfy
/// `MIN_ALIGN <= LNR_FLOOR <= EXP_FLOOR <= EXP_CEIL`, with `MIN_ALIGN > 8`.
/// (`CARVE_MAX` is unconstrained.) These rules are checked at compile time by
/// [`assert_config_valid`] and by the [`VALIDATE`](Config::VALIDATE) associated
/// constant.
pub trait Config {
    /// The atomic backend used to coordinate the free lists.
    ///
    /// Use [`CoreAtomics`](crate::CoreAtomics) on hosted 64-bit targets, or a
    /// custom [`Atomics`] implementation elsewhere.
    type Atomics: Atomics;

    /// Minimum alignment of every allocation, and the step of the linear size
    /// classes. Must be a power of two greater than `2^3` (so, at least 16).
    const MIN_ALIGN: u64;

    /// Smallest allocation class. Must be a power of two and `>= MIN_ALIGN`.
    const LNR_FLOOR: u64;

    /// Boundary between the linear and exponential regions. Must be a power of
    /// two and `>= LNR_FLOOR`.
    const EXP_FLOOR: u64;

    /// Largest managed class; requests above this spill into the oversized
    /// bucket. Must be a power of two and `>= EXP_FLOOR`.
    const EXP_CEIL: u64;

    /// Minimum excess, in bytes, worth carving off a block into a separate free
    /// block instead of retaining as internal slack.
    ///
    /// When a chosen block is larger than the request, the leftover is only
    /// split out as its own free block if it is at least `SPLIT_MIN` bytes;
    /// otherwise the whole block is handed out and the excess stays as internal
    /// fragmentation. Must be a power of two.
    ///
    /// Defaults to [`EXP_CEIL`](Config::EXP_CEIL), which is typically the
    /// optimal threshold.
    const SPLIT_MIN: u64 = Self::EXP_CEIL;

    /// Maximum number of blocks to carve out of an oversized block (one larger
    /// than [`EXP_CEIL`](Config::EXP_CEIL)) during an `alloc` or `realloc` when
    /// the block is too large for the request.
    ///
    /// Bounds how finely a single oversized block is subdivided in one
    /// operation. Not required to be a power of two. Defaults to `2`.
    const CARVE_MAX: u64 = 2;

    /// Compile-time base address of the managed heap, or `0` if the base is
    /// only known at runtime. Read by [`const_heap_base`] and, by default, by
    /// [`heap_base`](Config::heap_base).
    const HEAP_BASE: u64 = 0;

    /// Compile-time size of the managed heap, or `0` if only known at runtime.
    /// Read by [`const_heap_size`] and, by default, by
    /// [`heap_size`](Config::heap_size).
    const HEAP_SIZE: u64 = 0;

    /// Runtime base address of the managed heap.
    ///
    /// Defaults to the compile-time [`HEAP_BASE`](Config::HEAP_BASE). Override
    /// when the region is discovered at runtime.
    #[must_use]
    fn heap_base() -> u64 {
        Self::HEAP_BASE
    }

    /// Runtime size of the managed heap.
    ///
    /// Defaults to the compile-time [`HEAP_SIZE`](Config::HEAP_SIZE). Override
    /// when the region is discovered at runtime.
    #[must_use]
    fn heap_size() -> u64 {
        Self::HEAP_SIZE
    }

    /// Compile-time validation of the size constants.
    ///
    /// Evaluating this constant (which [`assert_config_valid`] forces) triggers
    /// a compile error if any constant is not a power of two or the ordering
    /// `MIN_ALIGN > 8`, `MIN_ALIGN <= LNR_FLOOR <= EXP_FLOOR <= EXP_CEIL` does
    /// not hold. There is no reason to reference this directly; call
    /// [`assert_config_valid`] instead.
    const VALIDATE: () = {
        assert!(
            Self::MIN_ALIGN.is_power_of_two(),
            "[cadalloc config] MIN_ALIGN must be a power of two"
        );
        assert!(
            Self::MIN_ALIGN > 8,
            "[cadalloc config] MIN_ALIGN must be greater than 2^3 (i.e. >= 16)"
        );
        assert!(
            Self::LNR_FLOOR.is_power_of_two(),
            "[cadalloc config] LNR_FLOOR must be a power of two"
        );
        assert!(
            Self::LNR_FLOOR >= Self::MIN_ALIGN,
            "[cadalloc config] LNR_FLOOR must be >= MIN_ALIGN"
        );
        assert!(
            Self::EXP_FLOOR.is_power_of_two(),
            "[cadalloc config] EXP_FLOOR must be a power of two"
        );
        assert!(
            Self::EXP_FLOOR >= Self::LNR_FLOOR,
            "[cadalloc config] EXP_FLOOR must be >= LNR_FLOOR"
        );
        assert!(
            Self::EXP_CEIL.is_power_of_two(),
            "[cadalloc config] EXP_CEIL must be a power of two"
        );
        assert!(
            Self::EXP_CEIL >= Self::EXP_FLOOR,
            "[cadalloc config] EXP_CEIL must be >= EXP_FLOOR"
        );
        assert!(
            Self::SPLIT_MIN.is_power_of_two(),
            "[cadalloc config] SPLIT_MIN must be a power of two"
        );
    };
}

/// Forces compile-time validation of a [`Config`]'s constants.
///
/// Call this in a `const` context to turn an invalid configuration into a build
/// error at the point of use:
///
/// ```
/// # use cadalloc::{Config, CoreAtomics};
/// # struct MyConfig;
/// # impl Config for MyConfig {
/// #     type Atomics = CoreAtomics;
/// #     const MIN_ALIGN: u64 = 16;
/// #     const LNR_FLOOR: u64 = 16;
/// #     const EXP_FLOOR: u64 = 256;
/// #     const EXP_CEIL: u64 = 65536;
/// # }
/// const _: () = cadalloc::assert_config_valid::<MyConfig>();
/// ```
pub const fn assert_config_valid<C: Config>() {
    C::VALIDATE
}

/// Returns the compile-time heap base of `C`, usable in `const` contexts.
///
/// This is [`Config::HEAP_BASE`]; it is `0` when the base is only known at
/// runtime, in which case use [`Config::heap_base`] instead.
#[must_use]
pub const fn const_heap_base<C: Config>() -> u64 {
    C::HEAP_BASE
}

/// Returns the compile-time heap size of `C`, usable in `const` contexts.
///
/// This is [`Config::HEAP_SIZE`]; it is `0` when the size is only known at
/// runtime, in which case use [`Config::heap_size`] instead.
#[must_use]
pub const fn const_heap_size<C: Config>() -> u64 {
    C::HEAP_SIZE
}
