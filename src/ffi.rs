//! Exporting a `cadalloc` allocator to C and other non-Rust callers.
//!
//! The public methods on [`Allocator`](crate::Allocator) are generic over the
//! [`Config`](crate::Config), so they monomorphize to a distinct function per
//! configuration and cannot themselves carry a stable, unmangled symbol name —
//! `#[no_mangle]` does not apply to a generic function. To link from C you pick
//! one concrete configuration and emit thin `extern "C"` shims for it, which is
//! exactly what [`export_c_api!`](crate::export_c_api) does.
//!
//! # ABI
//!
//! - [`Slice`](crate::Slice) is `#[repr(C)]` — `struct { uint64_t ptr, len; }` —
//!   and is passed and returned by value.
//! - `alloc(size, align) -> Slice` returns [`Slice::NULL`](crate::Slice::NULL)
//!   (a zero `ptr`) on failure.
//! - `free(Slice)` returns nothing; freeing the null slice is a no-op.
//! - `init() -> int32_t` and `verify() -> int32_t` return `0` on success, or a
//!   positive code otherwise:
//!   - `init`: `1` = misaligned heap, `2` = heap too small.
//!   - `verify`: `1` = bad/absent marker, `2` = configuration mismatch.
//!
//! The allocator is stateless (its state lives in the heap), so the shims
//! construct the handle on each call for free.
//!
//! # Example
//!
//! ```
//! use cadalloc::{Config, CoreAtomics};
//!
//! struct MyConfig;
//! impl Config for MyConfig {
//!     type Atomics = CoreAtomics;
//!     const MIN_ALIGN: u64 = 16;
//!     const LNR_FLOOR: u64 = 16;
//!     const EXP_FLOOR: u64 = 256;
//!     const EXP_CEIL: u64 = 65536;
//!     const HEAP_BASE: u64 = 0x2000_0000;
//!     const HEAP_SIZE: u64 = 1 << 20;
//! }
//!
//! cadalloc::export_c_api! {
//!     config: MyConfig,
//!     init: cad_init,
//!     alloc: cad_alloc,
//!     free: cad_free,
//!     verify: cad_verify,
//! }
//! ```
//!
//! This emits `cad_init`, `cad_alloc`, `cad_free`, and `cad_verify` as
//! unmangled `extern "C"` symbols bound to `MyConfig`.

/// Emits unmangled `extern "C"` entry points for one concrete
/// [`Config`](crate::Config).
///
/// See the [module documentation](crate::ffi) for the ABI and status codes. All
/// four names are required and given in the order shown:
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
/// #     const HEAP_BASE: u64 = 0x2000_0000;
/// #     const HEAP_SIZE: u64 = 1 << 20;
/// # }
/// cadalloc::export_c_api! {
///     config: MyConfig,
///     init: cad_init,
///     alloc: cad_alloc,
///     free: cad_free,
///     verify: cad_verify,
/// }
/// ```
#[macro_export]
macro_rules! export_c_api {
    (
        config: $config:ty,
        init: $init:ident,
        alloc: $alloc:ident,
        free: $free:ident,
        verify: $verify:ident $(,)?
    ) => {
        /// Prepares the heap. Returns `0` on success, `1` if the heap base is
        /// misaligned, or `2` if the heap is too small.
        #[unsafe(no_mangle)]
        pub extern "C" fn $init() -> i32 {
            match $crate::Allocator::<$config>::new().init() {
                Ok(()) => 0,
                Err($crate::InitError::MisalignedHeap) => 1,
                Err($crate::InitError::HeapTooSmall) => 2,
            }
        }

        /// Allocates `size` bytes with `align` alignment. Returns the payload
        /// slice, or the null slice (zero `ptr`) on failure.
        #[unsafe(no_mangle)]
        pub extern "C" fn $alloc(size: u64, align: u64) -> $crate::Slice {
            $crate::Allocator::<$config>::new().alloc(size, align)
        }

        /// Returns a block previously handed out by the matching `alloc`.
        /// Freeing the null slice is a no-op.
        #[unsafe(no_mangle)]
        pub extern "C" fn $free(block: $crate::Slice) {
            $crate::Allocator::<$config>::new().free(block)
        }

        /// Checks the heap marker and configuration fingerprint. Returns `0` on
        /// success, `1` if the marker is bad/absent, or `2` on a configuration
        /// mismatch.
        #[unsafe(no_mangle)]
        pub extern "C" fn $verify() -> i32 {
            match $crate::Allocator::<$config>::new().verify() {
                Ok(()) => 0,
                Err($crate::VerifyError::BadMagic) => 1,
                Err($crate::VerifyError::ConfigMismatch) => 2,
            }
        }
    };
}
