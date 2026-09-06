//! The two memory-region types the allocator traffics in.
//!
//! `cadalloc` works in `(ptr, len)` pairs of `u64`s rather than raw pointers
//! (the crate assumes a 64-bit target, so an address is always a `u64`). It
//! splits that pair into two types, along the same line as C++'s `std::string`
//! and `std::string_view`:
//!
//! - [`Slice`] is an **owning handle** to an allocation. It is what
//!   [`alloc`](crate::CadAlloc::alloc) hands out and what
//!   [`free`](crate::CadAlloc::free) / [`realloc`](crate::CadAlloc::realloc)
//!   take back. It is deliberately **not `Copy` or `Clone`**: an allocation has
//!   a single owner, so the handle moves, and the type system then rules out
//!   double frees and use-after-free at the API surface — once you pass a
//!   `Slice` to `free`, you cannot name it again.
//! - [`SliceView<'a>`] is a cheap, `Copy` **non-owning view** of a region,
//!   borrowed for `'a`. It carries all the inspection and sub-region operations.
//!   Borrow a `Slice` as one with [`view`](Slice::view); the view (and any byte
//!   slice taken from it) is then tied to that borrow, so the owner cannot be
//!   freed while a view of it is alive.
//!
//! The `'a` on a view is what makes the byte accessors sound: [`as_bytes`] and
//! friends return `&'a [u8]`, bounded by the region's borrow rather than an
//! invented lifetime. Constructing a view from a raw address
//! ([`SliceView::new`]) is `unsafe` precisely because there the caller *chooses*
//! `'a` and must ensure the region really lives that long.
//!
//! [`as_bytes`]: SliceView::as_bytes

use core::marker::PhantomData;
use core::ops::Range;

/// A non-owning view of a contiguous memory region, borrowed for `'a`: a base
/// address and a length (both raw `u64`s).
///
/// `SliceView` is `Copy` and freely duplicated — it borrows, it does not own.
/// The lifetime `'a` bounds the region's validity: [`as_bytes`](SliceView::as_bytes)
/// / [`as_bytes_mut`](SliceView::as_bytes_mut) hand back `&'a [u8]` /
/// `&'a mut [u8]`, so a byte slice can never outlive the region. Get one by
/// borrowing an owning [`Slice`] with [`view`](Slice::view) (then `'a` is that
/// borrow), or, for raw memory, with the `unsafe`
/// [`new`](SliceView::new) (then `'a` is the caller's promise). Deriving a
/// sub-view ([`subslice`](SliceView::subslice), [`split_at`](SliceView::split_at))
/// is safe and preserves `'a`.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct SliceView<'a> {
    pub(crate) ptr: u64,
    pub(crate) len: u64,
    _region: PhantomData<&'a [u8]>,
}

/// An owning handle to an allocation: a base address and a length.
///
/// Returned by [`alloc`](crate::CadAlloc::alloc) and consumed by
/// [`free`](crate::CadAlloc::free) / [`realloc`](crate::CadAlloc::realloc).
/// **Not `Copy` or `Clone`** — it represents sole ownership of a block, so it
/// moves, and passing it to `free` consumes it (no double free, no
/// use-after-free through the handle). Borrow it as a [`SliceView`] with
/// [`view`](Slice::view) to reach the view's operations; while any such view is
/// alive the handle is borrowed and cannot be freed.
///
/// The handle carries no region lifetime of its own: the allocator is stateless
/// (its state lives in the heap), so there is no heap value to borrow one from.
/// The move-only-ness guards the handle; the borrow on [`view`](Slice::view)
/// guards reads and writes of the memory.
#[repr(C)]
#[derive(PartialEq, Eq, Hash)]
pub struct Slice {
    pub(crate) ptr: u64,
    pub(crate) len: u64,
}

impl<'a> SliceView<'a> {
    /// The null view: base `0`, length `0`.
    pub const NULL: SliceView<'a> = SliceView {
        ptr: 0,
        len: 0,
        _region: PhantomData,
    };

    /// Constructs a view from a base address and a length, borrowed for `'a`.
    ///
    /// # Safety
    ///
    /// The region `[ptr, ptr + len)` must be valid memory the caller may access
    /// for the whole of `'a` — which the caller *chooses* here, so it is a
    /// promise the caller must keep. Reading it back through
    /// [`as_bytes`](SliceView::as_bytes) / [`as_bytes_mut`](SliceView::as_bytes_mut)
    /// dereferences it. Prefer [`Slice::view`], which fixes `'a` to the owner's
    /// borrow instead. The [null view](SliceView::NULL) needs no `unsafe`.
    #[must_use]
    #[inline(always)]
    pub const unsafe fn new(ptr: u64, len: u64) -> Self {
        Self {
            ptr,
            len,
            _region: PhantomData,
        }
    }

    /// The base address.
    #[must_use]
    #[inline(always)]
    pub const fn ptr(&self) -> u64 {
        self.ptr
    }

    /// The length, in bytes.
    #[must_use]
    #[inline(always)]
    pub const fn len(&self) -> u64 {
        self.len
    }

    /// Returns `true` for the [null view](SliceView::NULL) (base `0`).
    #[must_use]
    #[inline(always)]
    pub const fn is_null(&self) -> bool {
        self.ptr == 0
    }

    /// Returns `true` when the region is empty (length `0`).
    #[must_use]
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The address one past the end of the region (`ptr + len`, wrapping).
    #[must_use]
    #[inline(always)]
    pub const fn end(&self) -> u64 {
        self.ptr.wrapping_add(self.len)
    }

    /// The region as a half-open address range `ptr .. ptr + len`.
    #[must_use]
    #[inline(always)]
    pub const fn range(&self) -> Range<u64> {
        self.ptr..self.end()
    }

    /// Returns `true` if `addr` lies within `[ptr, ptr + len)`.
    #[must_use]
    #[inline(always)]
    pub const fn contains(&self, addr: u64) -> bool {
        addr >= self.ptr && addr < self.end()
    }

    /// Returns `true` if `other`'s range lies entirely within this one.
    #[must_use]
    #[inline(always)]
    pub const fn covers(&self, other: &SliceView<'_>) -> bool {
        other.ptr >= self.ptr && other.end() <= self.end()
    }

    /// Returns `true` if the two regions share any address.
    #[must_use]
    #[inline(always)]
    pub const fn overlaps(&self, other: &SliceView<'_>) -> bool {
        self.ptr < other.end() && other.ptr < self.end()
    }

    /// Returns `true` if the base address is a multiple of `align`
    /// (`false` if `align` is `0`).
    #[must_use]
    #[inline(always)]
    pub const fn is_aligned_to(&self, align: u64) -> bool {
        align != 0 && self.ptr % align == 0
    }

    /// The base address as a `*const u8`. Producing the pointer is safe;
    /// dereferencing it is not.
    #[must_use]
    #[inline(always)]
    pub const fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }

    /// The base address as a `*mut u8`. Producing the pointer is safe;
    /// dereferencing it is not.
    #[must_use]
    #[inline(always)]
    pub const fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr as *mut u8
    }

    /// A sub-view `[offset, offset + len)` of this region, or `None` if that
    /// range does not fit within `[0, self.len)`. The sub-view shares this
    /// region's lifetime `'a`.
    ///
    /// Safe: the parent view already bounds its region to `'a`, and the result
    /// is a `SliceView` (not an owner), so it can never be mistaken for a
    /// freeable allocation.
    #[must_use]
    #[inline]
    pub const fn subslice(&self, offset: u64, len: u64) -> Option<SliceView<'a>> {
        if let Some(hi) = offset.checked_add(len) {
            if hi <= self.len {
                return Some(SliceView {
                    ptr: self.ptr + offset,
                    len,
                    _region: PhantomData,
                });
            }
        }
        None
    }

    /// A sub-view `[offset, offset + len)`, without the bounds check.
    ///
    /// # Safety
    ///
    /// `offset + len` must not exceed `self.len` (and must not overflow), so the
    /// result stays within this region.
    #[must_use]
    #[inline(always)]
    pub const unsafe fn subslice_unchecked(&self, offset: u64, len: u64) -> SliceView<'a> {
        SliceView {
            ptr: self.ptr + offset,
            len,
            _region: PhantomData,
        }
    }

    /// Splits the region at `mid` bytes into `(head, tail)`, or `None` if
    /// `mid > self.len`.
    #[must_use]
    #[inline]
    pub const fn split_at(&self, mid: u64) -> Option<(SliceView<'a>, SliceView<'a>)> {
        if mid <= self.len {
            Some((
                SliceView {
                    ptr: self.ptr,
                    len: mid,
                    _region: PhantomData,
                },
                SliceView {
                    ptr: self.ptr + mid,
                    len: self.len - mid,
                    _region: PhantomData,
                },
            ))
        } else {
            None
        }
    }

    /// Views the region as a shared byte slice, bounded by the region's
    /// lifetime `'a`.
    ///
    /// # Safety
    ///
    /// The region must be valid and initialized for `len` bytes and not be
    /// mutated (through any alias) for `'a`.
    #[must_use]
    #[inline(always)]
    pub const unsafe fn as_bytes(&self) -> &'a [u8] {
        // SAFETY: the caller upholds validity/initialization/aliasing, and `'a`
        // (chosen at construction or fixed by `Slice::view`) bounds the region.
        unsafe { core::slice::from_raw_parts(self.ptr as *const u8, self.len as usize) }
    }

    /// Views the region as an exclusive byte slice, bounded by the region's
    /// lifetime `'a`.
    ///
    /// # Safety
    ///
    /// The region must be valid and initialized for `len` bytes and not be
    /// aliased for `'a`. (`SliceView` is `Copy`, so exclusivity is not enforced
    /// by the borrow checker here — it is the caller's obligation.)
    #[must_use]
    #[inline(always)]
    #[allow(clippy::mut_from_ref)] // a raw view: the exclusive borrow is the caller's obligation
    pub const unsafe fn as_bytes_mut(&self) -> &'a mut [u8] {
        // SAFETY: the caller upholds validity/initialization/aliasing per above.
        unsafe { core::slice::from_raw_parts_mut(self.ptr as *mut u8, self.len as usize) }
    }

    /// Promotes this view to an owning [`Slice`].
    ///
    /// # Safety
    ///
    /// The view must name a whole allocation that this allocator produced and
    /// still owns (so it may be handed to [`free`](crate::CadAlloc::free) /
    /// [`realloc`](crate::CadAlloc::realloc)). A view into the *middle* of a
    /// block, or one already owned by another `Slice`, must not be promoted —
    /// doing so and then freeing it corrupts the heap.
    #[must_use]
    #[inline(always)]
    pub const unsafe fn to_slice(&self) -> Slice {
        Slice {
            ptr: self.ptr,
            len: self.len,
        }
    }
}

impl Slice {
    /// The null handle: base `0`, length `0`. Returned to signal failure;
    /// freeing it is a no-op.
    pub const NULL: Slice = Slice { ptr: 0, len: 0 };

    /// Constructs an owning handle from a base address and a length.
    ///
    /// # Safety
    ///
    /// The handle is understood as owning a whole allocation: handing it to
    /// [`free`](crate::CadAlloc::free) / [`realloc`](crate::CadAlloc::realloc)
    /// reads a block header just below `ptr`. Constructing one asserts that
    /// `ptr`/`len` name such an allocation from this allocator (in practice, one
    /// previously returned by [`alloc`](crate::CadAlloc::alloc)). Fabricating a
    /// handle and freeing it is undefined behaviour. The
    /// [null handle](Slice::NULL) needs no `unsafe`.
    #[must_use]
    #[inline(always)]
    pub const unsafe fn new(ptr: u64, len: u64) -> Self {
        Self { ptr, len }
    }

    /// Borrows this handle as a [`SliceView`] for the borrow's lifetime. While
    /// the view (or any byte slice taken from it) is alive, the handle is
    /// borrowed and cannot be moved or freed.
    #[must_use]
    #[inline(always)]
    pub const fn view(&self) -> SliceView<'_> {
        SliceView {
            ptr: self.ptr,
            len: self.len,
            _region: PhantomData,
        }
    }

    /// Consumes this handle, returning an *unbounded* (`'static`) [`SliceView`]
    /// over the same region. The allocation is **not** freed — ownership is
    /// simply dropped — so this both leaks (unless the view is later promoted
    /// back and freed) and gives up the borrow check that [`view`](Slice::view)
    /// provides. Prefer [`view`](Slice::view) unless you must detach the region
    /// from the handle's scope.
    #[must_use]
    #[inline(always)]
    pub const fn into_view(self) -> SliceView<'static> {
        SliceView {
            ptr: self.ptr,
            len: self.len,
            _region: PhantomData,
        }
    }

    /// The base address.
    #[must_use]
    #[inline(always)]
    pub const fn ptr(&self) -> u64 {
        self.ptr
    }

    /// The length, in bytes.
    #[must_use]
    #[inline(always)]
    pub const fn len(&self) -> u64 {
        self.len
    }

    /// Returns `true` for the [null handle](Slice::NULL) (base `0`).
    #[must_use]
    #[inline(always)]
    pub const fn is_null(&self) -> bool {
        self.ptr == 0
    }

    /// Returns `true` when the region is empty (length `0`).
    #[must_use]
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The address one past the end of the region (`ptr + len`, wrapping).
    #[must_use]
    #[inline(always)]
    pub const fn end(&self) -> u64 {
        self.ptr.wrapping_add(self.len)
    }

    /// The region as a half-open address range `ptr .. ptr + len`.
    #[must_use]
    #[inline(always)]
    pub const fn range(&self) -> Range<u64> {
        self.ptr..self.end()
    }

    /// Returns `true` if `addr` lies within `[ptr, ptr + len)`.
    #[must_use]
    #[inline(always)]
    pub const fn contains(&self, addr: u64) -> bool {
        addr >= self.ptr && addr < self.end()
    }

    /// The base address as a `*const u8`.
    #[must_use]
    #[inline(always)]
    pub const fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }

    /// The base address as a `*mut u8`.
    #[must_use]
    #[inline(always)]
    pub const fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr as *mut u8
    }

    /// Views the payload as a shared byte slice, borrowed from the handle.
    ///
    /// # Safety
    ///
    /// The payload must be initialized for `len` bytes (freshly allocated memory
    /// is not). The shared borrow of `self` rules out concurrent mutation
    /// through this handle for the borrow.
    #[must_use]
    #[inline(always)]
    pub const unsafe fn as_bytes(&self) -> &[u8] {
        // SAFETY: an owning handle names a valid `len`-byte allocation; the
        // caller guarantees initialization. The borrow ties the slice to `self`.
        unsafe { core::slice::from_raw_parts(self.ptr as *const u8, self.len as usize) }
    }

    /// Views the payload as an exclusive byte slice, borrowed from the handle.
    ///
    /// # Safety
    ///
    /// The payload must be initialized for `len` bytes (freshly allocated memory
    /// is not). The exclusive borrow of `self` rules out aliasing for the borrow.
    #[must_use]
    #[inline(always)]
    pub const unsafe fn as_bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: as `as_bytes`; the `&mut self` borrow enforces exclusivity.
        unsafe { core::slice::from_raw_parts_mut(self.ptr as *mut u8, self.len as usize) }
    }
}

impl From<Slice> for SliceView<'static> {
    /// Consumes the owner, yielding an unbounded view (see
    /// [`Slice::into_view`]).
    #[inline(always)]
    fn from(s: Slice) -> SliceView<'static> {
        s.into_view()
    }
}

impl core::fmt::Debug for SliceView<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SliceView")
            .field("ptr", &format_args!("{:#x}", self.ptr))
            .field("len", &self.len)
            .finish()
    }
}

impl core::fmt::Debug for Slice {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Slice")
            .field("ptr", &format_args!("{:#x}", self.ptr))
            .field("len", &self.len)
            .finish()
    }
}
