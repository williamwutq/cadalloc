//! [`GlobalAlloc`] adapter, behind the `alloc` feature.
//!
//! With the `alloc` feature enabled, [`CadAlloc<C>`] implements
//! [`core::alloc::GlobalAlloc`], so a concrete configuration can back the
//! `alloc` crate as the program's `#[global_allocator]`:
//!
//! ```ignore
//! use cadalloc::{CadAlloc, Config, CoreAtomics};
//!
//! struct MyConfig;
//! impl Config for MyConfig {
//!     type Atomics = CoreAtomics;
//!     const MIN_ALIGN: u64 = 16;
//!     const LNR_FLOOR: u64 = 32;
//!     const EXP_FLOOR: u64 = 256;
//!     const EXP_CEIL: u64 = 65536;
//!     const HEAP_BASE: u64 = 0x2000_0000; // a real, mapped region
//!     const HEAP_SIZE: u64 = 1 << 20;
//! }
//!
//! #[global_allocator]
//! static GLOBAL: CadAlloc<MyConfig> = CadAlloc::new();
//!
//! fn main() {
//!     // MUST run before the first allocation (see below).
//!     GLOBAL.init().expect("prepare the heap");
//!     let v = vec![1u8, 2, 3];
//!     assert_eq!(v.len(), 3);
//! }
//! ```
//!
//! # You must call [`init`](CadAlloc::init) first
//!
//! The `#[global_allocator]` static is used the moment anything allocates, and
//! [`init`](CadAlloc::init) — which stamps the heap and positions the bump
//! pointer — is **not** run automatically. Call it once, from a single thread,
//! before the first allocation (in practice, the first line of `main` or an
//! equivalent startup hook). Allocating before `init` fails (returns null),
//! which the `alloc` crate turns into an allocation-error abort. There is no
//! lazy initialization: it could not be done race-free within this crate's
//! single-`init` contract.
//!
//! # Alignment
//!
//! `cadalloc` guarantees payloads are aligned to `MIN_ALIGN` (always `>= 16`),
//! and nothing stronger. A request whose [`Layout`] demands
//! `align > MIN_ALIGN` therefore **cannot be satisfied and returns null** (an
//! allocation-error abort at the `alloc`-crate level). Every primitive type and
//! most aggregates need `align <= 16`, so this only bites over-aligned types
//! (e.g. some SIMD vectors, `#[repr(align(N))]` with large `N`). Raise
//! `MIN_ALIGN` in the configuration to widen the ceiling if such types matter.

use crate::allocator::CadAlloc;
use crate::config::Config;
use crate::slice::Slice;
use core::alloc::{GlobalAlloc, Layout};

// SAFETY: the `GlobalAlloc` contract is upheld by delegating to the allocator's
// own methods over the heap fixed by `C`:
//
// - `alloc` returns either null or a `MIN_ALIGN`-aligned block of at least
//   `layout.size()` bytes; it refuses (returns null) any `layout` whose
//   alignment exceeds `MIN_ALIGN`, so a non-null result always meets the
//   requested alignment.
// - `dealloc`/`realloc` receive a pointer previously returned by `alloc` on the
//   same allocator; `free`/`realloc` recover the true block size from its header
//   (the slice length is ignored), so the `Layout` need only supply the pointer.
unsafe impl<C: Config> GlobalAlloc for CadAlloc<C> {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // We only guarantee `MIN_ALIGN`; stronger alignment is unsatisfiable.
        if layout.align() as u64 > C::MIN_ALIGN {
            return core::ptr::null_mut();
        }
        CadAlloc::alloc(self, layout.size() as u64).ptr as *mut u8
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        if ptr.is_null() {
            return;
        }
        // SAFETY: `ptr` was handed out by `alloc` on this allocator; `free` reads
        // the block's true size from its header and ignores the slice length.
        let block = unsafe { Slice::new(ptr as u64, 0) };
        CadAlloc::free(self, block);
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // The alignment is unchanged from the original (already `<= MIN_ALIGN`),
        // and a relocating grow lands on `MIN_ALIGN` too, so no re-check is
        // needed. On failure `realloc` returns null and leaves `ptr` valid — the
        // `GlobalAlloc` contract exactly.
        // SAFETY: `ptr`/`layout` describe a live block from this allocator; the
        // slice length is ignored, so any value (here `layout.size()`) is fine.
        let block = unsafe { Slice::new(ptr as u64, layout.size() as u64) };
        CadAlloc::realloc(self, block, new_size as u64).ptr as *mut u8
    }
}
