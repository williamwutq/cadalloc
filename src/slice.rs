//! The fundamental unit handed out and taken back by the allocator.
//!
//! `cadalloc` traffics in *slices*, not raw pointers. A [`Slice`] is a
//! `(ptr, len)` pair of `u64`s: the crate assumes a 64-bit target, so an
//! address is always representable as a `u64` and there is no dependence on the
//! host pointer width. Working in slices keeps the length beside every address,
//! which is what a segregated-free-list allocator needs to place a block back
//! into the correct size class on free.

/// A contiguous region of memory, described as a base address and a length.
///
/// Both fields are raw `u64`s. A `Slice` carries no provenance and enforces no
/// invariants on its own; it is the caller's responsibility to ensure any
/// address it dereferences is valid, aligned, and part of the managed heap.
///
/// The all-zero slice ([`Slice::NULL`]) is reserved to mean "no region", and
/// [`is_null`](Slice::is_null) tests for it. A real allocation therefore never
/// has base address `0`, which matches every heap this allocator can manage
/// (address `0` is never a valid allocation).
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct Slice {
    /// Base address of the region.
    pub ptr: u64,
    /// Length of the region, in bytes.
    pub len: u64,
}

impl Slice {
    /// The null slice: base `0`, length `0`. Returned to signal failure.
    pub const NULL: Slice = Slice { ptr: 0, len: 0 };

    /// Constructs a slice from a base address and a length.
    #[must_use]
    #[inline(always)]
    pub const fn new(ptr: u64, len: u64) -> Self {
        Self { ptr, len }
    }

    /// Returns `true` if this is the [null slice](Slice::NULL) (base `0`).
    #[must_use]
    #[inline(always)]
    pub const fn is_null(&self) -> bool {
        self.ptr == 0
    }

    /// Returns the address one past the end of the region (`ptr + len`).
    ///
    /// The addition is wrapping; a well-formed slice never wraps the address
    /// space, so a wrapped result indicates a malformed slice.
    #[must_use]
    #[inline(always)]
    pub const fn end(&self) -> u64 {
        self.ptr.wrapping_add(self.len)
    }

    /// Returns `true` if `addr` lies within `[ptr, ptr + len)`.
    #[must_use]
    #[inline(always)]
    pub const fn contains(&self, addr: u64) -> bool {
        addr >= self.ptr && addr < self.end()
    }
}
