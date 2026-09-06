//! The allocator core: size-class mapping, block headers, and the `alloc` /
//! `free` fast paths.
//!
//! # Layout
//!
//! A [`CadAlloc`] is stateless — a zero-sized marker for a [`Config`]. All
//! mutable state lives in the managed heap, in a *control region* at its base,
//! followed by the payload area blocks are carved from:
//!
//! ```text
//! heap_base ->  [ "CADALLOC" ][ config fingerprint ]
//!               [ bin head 0 ][ bin head 1 ] … [ bin head NUM_BINS-1 ]
//!               [ bump ][ lock ]
//!   (aligned) -> [ ------------------ payload area ------------------ ]
//! heap_base + heap_size ->
//! ```
//!
//! The heap opens with two metadata words: the `"CADALLOC"` marker (8 ASCII
//! bytes, little-endian) and a 64-bit fingerprint of the configuration, both
//! checked by [`verify`](CadAlloc::verify). Each bin head is a `u64`: the
//! address of the first free block in that bin (`0` when empty), with the low
//! alignment bits used as an ABA tag for the lock-free bins. The last bin
//! (`NUM_BINS - 1`) is the oversized bin.
//!
//! # Blocks
//!
//! Every block begins with a `MIN_ALIGN`-byte header region. Its first `u64`
//! packs the block's total size (header included) — a multiple of `MIN_ALIGN`,
//! so its low 4 bits are free — with the flags in those low bits (bit 0: `1` =
//! free). Its second `u64` (offset 8, always inside the header region because
//! `MIN_ALIGN >= 16`) holds the free-list `next` pointer while the block is
//! free. The payload starts at `block + MIN_ALIGN`. There is no footer and
//! blocks are never coalesced.
//!
//! # Concurrency
//!
//! Fixed size-class bins are lock-free Treiber stacks driven by the [`Atomics`]
//! backend; an ABA tag in the low (always-zero) alignment bits of each bin head
//! guards against the common ABA races. The bump pointer, the oversized free
//! list, and block splitting run under a single spinlock (the "refill lock").
//! Every access to a heap word goes through [`Atomics`], so there are no
//! non-atomic data races on heap memory.
//!
//! # Safety invariant
//!
//! The private `load` / `store` / `cas` helpers all assume their address names a
//! live, `MIN_ALIGN`-aligned `u64` inside the managed heap. Every internal call
//! site upholds this by construction (control words and block headers computed
//! from `heap_base`). [`free`](CadAlloc::free) trusts that the slice it is
//! given was produced by [`alloc`](CadAlloc::alloc) on this allocator.

use crate::atomic::Atomics;
use crate::config::Config;
use crate::slice::Slice;
use core::hint::spin_loop;
use core::marker::PhantomData;

/// Block sizes are always multiples of `MIN_ALIGN` (`> 8`, so `>= 16`), hence the
/// low 4 bits of a header word are always zero and carry the block's flags — no
/// shifting, and `size` keeps its full magnitude.
const FLAG_MASK: u64 = 0xF;
/// Free flag (bit 0 of a block header). Bits 1..4 are reserved.
const FREE: u64 = 1;
/// "In use" flag (bit 0 clear).
const USED: u64 = 0;

/// Packs a total block `size` (a multiple of `MIN_ALIGN`) and a `flag` into a
/// header word.
#[inline(always)]
const fn pack(size: u64, flag: u64) -> u64 {
    size | (flag & FLAG_MASK)
}

/// Extracts the total block size from a header word.
#[inline(always)]
const fn header_size(header: u64) -> u64 {
    header & !FLAG_MASK
}

/// Returns `true` if the header's block is marked free.
#[inline(always)]
const fn header_is_free(header: u64) -> bool {
    (header & FLAG_MASK) == FREE
}

/// Rounds `x` up to a multiple of the power-of-two `align`.
#[inline(always)]
const fn align_up(x: u64, align: u64) -> u64 {
    (x + (align - 1)) & !(align - 1)
}

/// The `"CADALLOC"` marker word (8 ASCII bytes, little-endian) written at the
/// very front of an initialized heap.
pub(crate) const MAGIC: u64 = u64::from_le_bytes(*b"CADALLOC");

/// Bytes of metadata (`MAGIC` + config fingerprint) preceding the bin heads.
const META_BYTES: u64 = 16;

/// The SplitMix64 finalizer — a bijective 64-bit avalanche mixer.
#[inline(always)]
const fn splitmix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A 64-bit fingerprint of the layout-defining configuration constants.
///
/// The power-of-two constants carry only their exponent, so their base-2
/// logarithms (each `< 64`, i.e. 6 bits) are packed losslessly into 30 bits;
/// `CARVE_MAX` is folded in by an odd multiplier, and the result is run through
/// the [`splitmix64`] finalizer to avalanche the bits. Every distinct
/// layout-affecting configuration therefore gets a distinct, well-dispersed
/// fingerprint. Purely a compile-time computation.
#[inline]
pub(crate) const fn config_fingerprint<C: Config>() -> u64 {
    let packed = (C::MIN_ALIGN.ilog2() as u64)
        | ((C::LNR_FLOOR.ilog2() as u64) << 6)
        | ((C::EXP_FLOOR.ilog2() as u64) << 12)
        | ((C::EXP_CEIL.ilog2() as u64) << 18)
        | ((C::SPLIT_MIN.ilog2() as u64) << 24);
    splitmix64(packed ^ C::CARVE_MAX.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

/// Returns the total block size of the size class held by `bin`.
///
/// For the oversized bin (`bin == NUM_BINS - 1`) there is no fixed class size,
/// so this returns [`u64::MAX`]. The returned value is always a multiple of
/// `MIN_ALIGN`.
#[inline]
pub(crate) const fn bin_class<C: Config>(bin: u64) -> u64 {
    if bin < C::NUM_LINEAR {
        C::LNR_FLOOR + bin * C::MIN_ALIGN
    } else if bin < C::NUM_LINEAR + C::NUM_EXP {
        let j = bin - C::NUM_LINEAR;
        let octave = j / 2;
        let f = C::EXP_FLOOR.ilog2() as u64;
        let base = 1u64 << (f + octave);
        let raw = if j % 2 == 0 { base } else { base + base / 2 };
        align_up(raw, C::MIN_ALIGN)
    } else {
        u64::MAX
    }
}

/// Maps a total block `size` to the bin whose class is the smallest that is
/// `>= size` (a ceiling map), or the oversized bin when `size > EXP_CEIL`.
///
/// Because every class value maps to itself, this doubles as the placement map
/// when freeing a block of a known class size.
#[inline]
pub(crate) const fn bin_index<C: Config>(size: u64) -> u64 {
    let min = C::MIN_ALIGN;
    if size <= C::LNR_FLOOR {
        0
    } else if size < C::EXP_FLOOR {
        // Smallest linear class >= size: ceil((size - LNR_FLOOR) / MIN_ALIGN).
        // For size just below EXP_FLOOR this yields NUM_LINEAR, i.e. the first
        // exponential bin (class EXP_FLOOR), which is exactly what fits.
        (size - C::LNR_FLOOR).div_ceil(min)
    } else if size <= C::EXP_CEIL {
        let m = size.ilog2() as u64; // floor(log2(size))
        let base = 1u64 << m;
        let f = C::EXP_FLOOR.ilog2() as u64;
        let (octave, within) = if size == base {
            (m - f, 0)
        } else if size <= base + base / 2 {
            (m - f, 1)
        } else {
            (m + 1 - f, 0)
        };
        C::NUM_LINEAR + 2 * octave + within
    } else {
        C::NUM_BINS - 1
    }
}

/// A `cadalloc` allocator, parameterized by a [`Config`].
///
/// The type is zero-sized: it carries no state of its own, only the `Config`
/// that fixes the size classes, heap region, and atomics backend. All state
/// lives in the managed heap. Because coordination is entirely through the
/// [`Atomics`] backend, a `CadAlloc` may be shared across threads (for
/// example as a `static`) once [`init`](CadAlloc::init) has run.
///
/// # Usage
///
/// ```no_run
/// use cadalloc::{CadAlloc, Config, CoreAtomics};
///
/// struct C;
/// impl Config for C {
///     type Atomics = CoreAtomics;
///     const MIN_ALIGN: u64 = 16;
///     const LNR_FLOOR: u64 = 32;
///     const EXP_FLOOR: u64 = 256;
///     const EXP_CEIL: u64 = 65536;
///     const HEAP_BASE: u64 = 0x2000_0000; // a real, mapped region
///     const HEAP_SIZE: u64 = 1 << 20;
/// }
///
/// let a = CadAlloc::<C>::new();
/// a.init().unwrap();
/// let s = a.alloc(100);
/// if !s.is_null() {
///     // ... use [s.ptr, s.ptr + s.len) ...
///     a.free(s);
/// }
/// ```
pub struct CadAlloc<C: Config> {
    _config: PhantomData<fn() -> C>,
}

impl<C: Config> Default for CadAlloc<C> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Why [`CadAlloc::init`] could not prepare the heap.
///
/// `#[repr(i32)]`, so it is FFI-safe and each variant's discriminant is the
/// status code the [`export_c_api!`](crate::export_c_api) `init` shim returns
/// (success is `0`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(i32)]
pub enum InitError {
    /// The heap base address is not aligned to `MIN_ALIGN`.
    MisalignedHeap = 1,
    /// The heap is too small to hold the control region.
    HeapTooSmall = 2,
}

/// Why [`CadAlloc::verify`] rejected a heap.
///
/// `#[repr(i32)]`, so it is FFI-safe and each variant's discriminant is the
/// status code the [`export_c_api!`](crate::export_c_api) `verify` shim returns
/// (success is `0`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(i32)]
pub enum VerifyError {
    /// The `"CADALLOC"` marker is absent — the heap was never initialized (or
    /// not by `cadalloc`).
    BadMagic = 1,
    /// The marker is present but the configuration fingerprint differs — the
    /// heap was laid out by a different configuration. Using it would corrupt
    /// memory.
    ConfigMismatch = 2,
}

impl<C: Config> CadAlloc<C> {
    /// Creates an allocator handle. Call [`init`](CadAlloc::init) once, before
    /// any allocation, to prepare the heap.
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        Self {
            _config: PhantomData,
        }
    }

    // --- control-region geometry -------------------------------------------

    /// Address of the `"CADALLOC"` marker word.
    #[inline(always)]
    fn magic_addr(&self) -> u64 {
        C::heap_base()
    }

    /// Address of the config-fingerprint word.
    #[inline(always)]
    fn fingerprint_addr(&self) -> u64 {
        C::heap_base() + 8
    }

    /// Address of bin `k`'s head word (after the metadata words).
    #[inline(always)]
    fn bin_head(&self, k: u64) -> u64 {
        C::heap_base() + META_BYTES + k * 8
    }

    /// Address of the bump-pointer word.
    #[inline(always)]
    fn bump_addr(&self) -> u64 {
        C::heap_base() + META_BYTES + C::NUM_BINS * 8
    }

    /// Address of the refill-lock word.
    #[inline(always)]
    fn lock_addr(&self) -> u64 {
        C::heap_base() + META_BYTES + (C::NUM_BINS + 1) * 8
    }

    /// Number of bytes reserved for the control region, aligned to `MIN_ALIGN`.
    #[inline(always)]
    const fn control_bytes() -> u64 {
        align_up(META_BYTES + (C::NUM_BINS + 2) * 8, C::MIN_ALIGN)
    }

    /// One-past-the-end address of the heap.
    #[inline(always)]
    fn heap_end(&self) -> u64 {
        C::heap_base() + C::heap_size()
    }

    /// The oversized bin index.
    #[inline(always)]
    const fn oversize_bin() -> u64 {
        C::NUM_BINS - 1
    }

    /// The fixed bin whose class is the *largest* `<= size` (a floor map), for
    /// `LNR_FLOOR <= size <= EXP_CEIL`.
    #[inline]
    const fn floor_bin(size: u64) -> u64 {
        let ceil = bin_index::<C>(size);
        if bin_class::<C>(ceil) <= size {
            ceil
        } else {
            ceil - 1
        }
    }

    /// Whether `size` bytes form a valid stand-alone block: an oversized region
    /// (`> EXP_CEIL`, whose exact size lives in its header) or a size that is
    /// *exactly* one of the fixed classes. A fixed-range size that is not a class
    /// is invalid — [`free`](CadAlloc::free) ceil-bins the header, so such a
    /// block would be reused as a larger class and overrun its neighbour.
    #[inline]
    const fn is_block_exact(size: u64) -> bool {
        size > C::EXP_CEIL || bin_class::<C>(bin_index::<C>(size)) == size
    }

    /// The address alignment required of a block in `bin`: `MIN_ALIGN` for linear
    /// bins, half the octave base (`2^(n-1)` for octave `2^n`) for exponential
    /// bins, and `EXP_CEIL` for the oversized bin (its octave, `2^(c+1)`, aligns
    /// to `2^c`); never below `MIN_ALIGN`.
    #[inline]
    const fn align_of_bin(bin: u64) -> u64 {
        if bin < C::NUM_LINEAR {
            C::MIN_ALIGN
        } else if bin < C::NUM_LINEAR + C::NUM_EXP {
            let octave = (bin - C::NUM_LINEAR) / 2;
            let base_log2 = C::EXP_FLOOR.ilog2() as u64 + octave;
            let align = 1u64 << (base_log2 - 1);
            if align < C::MIN_ALIGN {
                C::MIN_ALIGN
            } else {
                align
            }
        } else {
            C::EXP_CEIL
        }
    }

    // --- raw heap-word access (see module safety invariant) ----------------

    #[inline(always)]
    fn load(&self, addr: u64) -> u64 {
        // SAFETY: `addr` names a live, aligned heap word by construction.
        unsafe { C::Atomics::atomic_load(addr) }
    }

    #[inline(always)]
    fn store(&self, addr: u64, val: u64) {
        // SAFETY: see `load`.
        unsafe { C::Atomics::atomic_store(addr, val) }
    }

    /// Weak compare-and-swap, returning whether the store succeeded. Only valid
    /// inside a retry loop (may fail spuriously). Every CAS in the allocator
    /// sits in such a loop, so the weak form is always the right one.
    #[inline(always)]
    fn cas_weak(&self, addr: u64, current: u64, new: u64) -> bool {
        // SAFETY: see `load`.
        unsafe { C::Atomics::atomic_cas_weak(addr, current, new) }.1
    }

    // --- refill lock -------------------------------------------------------

    #[inline]
    fn lock(&self) {
        let addr = self.lock_addr();
        while !self.cas_weak(addr, 0, 1) {
            spin_loop();
        }
    }

    #[inline]
    fn unlock(&self) {
        self.store(self.lock_addr(), 0);
    }

    // --- lock-free Treiber stack (fixed-class bins) ------------------------

    /// Pushes a block onto bin `k`'s lock-free stack. The block is private to
    /// the caller until published by the final CAS.
    #[inline]
    pub(crate) fn push_fixed(&self, k: u64, block: u64) {
        let head = self.bin_head(k);
        let mask = C::MIN_ALIGN - 1;
        loop {
            let old = self.load(head);
            let old_addr = old & !mask;
            let new_tag = (old + 1) & mask;
            self.store(block + 8, old_addr); // block.next = current head
            if self.cas_weak(head, old, block | new_tag) {
                return;
            }
        }
    }

    /// Pops a block from bin `k`'s lock-free stack, or returns `0` if empty.
    #[inline]
    pub(crate) fn pop_fixed(&self, k: u64) -> u64 {
        let head = self.bin_head(k);
        let mask = C::MIN_ALIGN - 1;
        loop {
            let old = self.load(head);
            let old_addr = old & !mask;
            if old_addr == 0 {
                return 0;
            }
            // Speculative read; a failed CAS discards it. Offset 8 is header
            // padding, never user payload, so the read is always in-bounds.
            let next = self.load(old_addr + 8);
            let new_tag = (old + 1) & mask;
            if self.cas_weak(head, old, next | new_tag) {
                return old_addr;
            }
        }
    }

    // --- public API --------------------------------------------------------

    /// The `"CADALLOC"` marker written at the front of an initialized heap.
    #[must_use]
    #[inline]
    pub const fn magic() -> u64 {
        MAGIC
    }

    /// The compile-time fingerprint of this configuration's layout-defining
    /// constants (see [`verify`](CadAlloc::verify)).
    #[must_use]
    #[inline]
    pub const fn config_hash() -> u64 {
        config_fingerprint::<C>()
    }

    /// Prepares the heap: stamps the `"CADALLOC"` marker and the config
    /// fingerprint, zeroes the bin heads, positions the bump pointer past the
    /// control region, and clears the lock.
    ///
    /// Must be called exactly once, from a single thread, before any
    /// allocation. Returns an [`InitError`] if the heap base is misaligned or
    /// the heap is too small to hold the control region.
    #[inline]
    pub fn init(&self) -> Result<(), InitError> {
        let base = C::heap_base();
        if base & (C::MIN_ALIGN - 1) != 0 {
            return Err(InitError::MisalignedHeap);
        }
        if C::heap_size() < Self::control_bytes() {
            return Err(InitError::HeapTooSmall);
        }
        self.store(self.magic_addr(), MAGIC);
        self.store(self.fingerprint_addr(), config_fingerprint::<C>());
        let mut k = 0;
        while k < C::NUM_BINS {
            self.store(self.bin_head(k), 0);
            k += 1;
        }
        self.store(self.bump_addr(), base + Self::control_bytes());
        self.store(self.lock_addr(), 0);
        Ok(())
    }

    /// Checks that the heap was initialized by a `cadalloc` allocator with a
    /// matching configuration.
    ///
    /// Reads the marker and fingerprint words at the front of the heap and
    /// compares them against [`magic`](CadAlloc::magic) and this
    /// configuration's [`config_hash`](CadAlloc::config_hash). **Call this
    /// before touching a heap that some *other* build may have prepared** (for
    /// example a persistent or shared-memory region): running an allocator over
    /// a heap laid out by a different configuration corrupts memory silently,
    /// and this is the cheap guard against it. It reads only two words and does
    /// not need the lock.
    #[inline]
    pub fn verify(&self) -> Result<(), VerifyError> {
        if self.load(self.magic_addr()) != MAGIC {
            return Err(VerifyError::BadMagic);
        }
        if self.load(self.fingerprint_addr()) != config_fingerprint::<C>() {
            return Err(VerifyError::ConfigMismatch);
        }
        Ok(())
    }

    /// The exact total block size [`alloc`](CadAlloc::alloc) would carve for a
    /// `size`-byte request: the containing fixed class, or the header-rounded
    /// `need` for an oversized request.
    #[inline]
    const fn request_block_size(size: u64) -> u64 {
        let need = align_up(size, C::MIN_ALIGN) + C::MIN_ALIGN; // + header
        let bin = bin_index::<C>(need);
        if bin == Self::oversize_bin() {
            need
        } else {
            bin_class::<C>(bin)
        }
    }

    /// Allocates a block with at least `size` payload bytes, `MIN_ALIGN`-aligned,
    /// returning a [`Slice`] over its payload, or [`Slice::NULL`] on failure.
    ///
    /// The returned slice's `len` is the block's full usable capacity, which may
    /// exceed `size`. The payload is always `MIN_ALIGN`-aligned; stronger
    /// alignments are not currently supported.
    #[must_use]
    #[inline]
    pub fn alloc(&self, size: u64) -> Slice {
        let need = align_up(size, C::MIN_ALIGN) + C::MIN_ALIGN; // + header
        let bin = bin_index::<C>(need);
        let block = if bin == Self::oversize_bin() {
            self.alloc_oversize(need)
        } else {
            self.alloc_fixed(bin)
        };
        if block == 0 {
            return Slice::NULL;
        }
        let bsize = header_size(self.load(block));
        // SAFETY: `block` is a live block we just carved/popped; its payload
        // `[block + MIN_ALIGN, block + bsize)` is owned and in-heap.
        unsafe { Slice::new(block + C::MIN_ALIGN, bsize - C::MIN_ALIGN) }
    }

    /// Returns a block previously handed out by [`alloc`](CadAlloc::alloc) to
    /// its bin. Freeing [`Slice::NULL`] is a no-op.
    ///
    /// The slice's `ptr` must be one returned by this allocator; its `len` is
    /// ignored (the true size is read from the block header). A block that is
    /// already free — a double free, or a bogus slice — is refused (returning it
    /// to a bin twice would corrupt the free list); debug builds also assert.
    #[inline]
    pub fn free(&self, block: Slice) {
        if block.is_null() {
            return;
        }
        let hdr = block.ptr - C::MIN_ALIGN;
        let header = self.load(hdr);
        if header_is_free(header) {
            debug_assert!(false, "cadalloc: double free or invalid slice");
            return;
        }
        let size = header_size(header);
        // Every block is class-exact (fixed) or `> EXP_CEIL` (oversized), so the
        // ceiling bin index is also its exact home.
        let bin = bin_index::<C>(size);
        self.store(hdr, pack(size, FREE));
        if bin == Self::oversize_bin() {
            self.lock();
            self.push_oversize_locked(hdr);
            self.unlock();
        } else {
            self.push_fixed(bin, hdr);
        }
    }

    /// Resizes `block` to hold at least `new_size` payload bytes, `MIN_ALIGN`-
    /// aligned, returning a [`Slice`] over the result (possibly the same address,
    /// possibly moved), or [`Slice::NULL`] on failure.
    ///
    /// Follows C `realloc` conventions: reallocating [`Slice::NULL`] is
    /// equivalent to [`alloc`](CadAlloc::alloc), and on failure the original
    /// block is left untouched (not freed). Because blocks are never coalesced,
    /// growth beyond the current block relocates: a fresh block is allocated,
    /// the payload is copied, and the old block is freed. Shrinking stays in
    /// place, splitting off the tail only when the leftover is worth it (the
    /// `SPLIT_MIN` / `CARVE_MAX` rule, so with the default `SPLIT_MIN` a shrink
    /// within the fixed classes just keeps the block and its slack).
    #[inline]
    pub fn realloc(&self, block: Slice, new_size: u64) -> Slice {
        if block.is_null() {
            return self.alloc(new_size);
        }

        let hdr = block.ptr - C::MIN_ALIGN;
        let cur_size = header_size(self.load(hdr));
        let target = Self::request_block_size(new_size);

        if cur_size >= target {
            // Fits in place. Carve the excess only when it clears SPLIT_MIN;
            // otherwise avoid the lock and keep the block whole.
            if cur_size - target < C::SPLIT_MIN {
                // SAFETY: same live payload as the incoming `block`, kept whole.
                return unsafe { Slice::new(block.ptr, cur_size - C::MIN_ALIGN) };
            }
            self.lock();
            let alloc_size = self.split_and_free_tail(hdr, cur_size, target);
            self.unlock();
            // SAFETY: `block` still names a live payload, now of `alloc_size`.
            return unsafe { Slice::new(block.ptr, alloc_size - C::MIN_ALIGN) };
        }

        // Grow: relocate. Allocate first so a failure leaves the original valid.
        self.relocate(block, cur_size - C::MIN_ALIGN, self.alloc(new_size))
    }

    /// Copies `src`'s first `src_usable` payload bytes into `dst`, frees `src`,
    /// and returns `dst` — the shared tail of every relocating path. A null
    /// `dst` (allocation failed) leaves `src` untouched and returns the null
    /// slice, per C `realloc`.
    #[inline]
    fn relocate(&self, src: Slice, src_usable: u64, dst: Slice) -> Slice {
        if dst.is_null() {
            return Slice::NULL;
        }
        let copy = if src_usable < dst.len {
            src_usable
        } else {
            dst.len
        };
        // SAFETY: `src` and `dst` are distinct, caller-owned payloads, each valid
        // for `copy` bytes; neither is accessed concurrently.
        unsafe {
            core::ptr::copy_nonoverlapping(src.ptr as *const u8, dst.ptr as *mut u8, copy as usize);
        }
        self.free(src);
        dst
    }

    // --- fixed-class allocation --------------------------------------------

    /// Allocates a fixed-class block from `bin`, returning its address (header),
    /// or `0` if the heap is exhausted. Blocks in a fixed bin are always exactly
    /// the class size, so the returned block is stamped used at the class size.
    #[inline]
    fn alloc_fixed(&self, bin: u64) -> u64 {
        let class = bin_class::<C>(bin);

        // Fast path: a free block already in the bin.
        let block = self.pop_fixed(bin);
        if block != 0 {
            self.store(block, pack(class, USED));
            return block;
        }

        // Slow path: re-check under the lock (another thread may have freed
        // into this bin), else carve a fresh class-sized block from the bump
        // region. Exponential blocks must land on their class alignment, so the
        // bump pointer is rounded up and the gap freed; linear/oversized bins
        // have `MIN_ALIGN` alignment, for which the round-up is a no-op.
        let align = Self::align_of_bin(bin);
        self.lock();
        let mut block = self.pop_fixed(bin);
        if block == 0 {
            let bump = self.load(self.bump_addr());
            let aligned = align_up(bump, align);
            if aligned + class <= self.heap_end() {
                self.carve_gap(bump, aligned);
                self.store(self.bump_addr(), aligned + class);
                block = aligned;
            }
        }
        self.unlock();

        // The block is private to us now (not in any list), so it is safe to
        // stamp the header after releasing the lock.
        if block != 0 {
            self.store(block, pack(class, USED));
        }
        block
    }

    // --- oversized allocation (under the refill lock) ----------------------

    /// Links an already-headered free block onto the front of the oversized
    /// list. The caller must hold the refill lock.
    #[inline]
    fn push_oversize_locked(&self, addr: u64) {
        let head = self.bin_head(Self::oversize_bin());
        self.store(addr + 8, self.load(head));
        self.store(head, addr);
    }

    /// The largest block that can be carved off ending at address `end` from a
    /// region of `len` bytes: `(size, bin)`, or `(0, 0)` if nothing fits. The
    /// placement `end - size` meets the class alignment (`align | end`, since
    /// the class size is a multiple of its alignment).
    ///
    /// A region `> EXP_CEIL` yields one oversized block, aligned to `EXP_CEIL`:
    /// its start is `end - size` rounded up to that alignment, and the
    /// misaligned prefix is left for the next carve. If that rounding would
    /// shrink the block into the fixed range, the region is carved as fixed
    /// classes instead.
    ///
    /// Otherwise the class is bounded by both `len` and alignment, computed in
    /// O(1): alignment constrains only exponential classes (linear ones need
    /// just `MIN_ALIGN`, which an already-`MIN_ALIGN`-aligned `end` has). An
    /// exponential class `V` is aligned to `2^(floor(log2 V) - 1)`, which
    /// divides `end` iff it is `<= 1 << end.trailing_zeros()`; the largest such
    /// class is therefore `3 * (1 << end.trailing_zeros())`.
    #[inline]
    pub(crate) const fn largest_carvable(end: u64, len: u64) -> (u64, u64) {
        if len < C::LNR_FLOOR {
            return (0, 0);
        }
        // Oversized: align the block start to `EXP_CEIL`. `end - len` is the
        // region base and never underflows (the region lies in the heap).
        let mut cap = len;
        if len > C::EXP_CEIL {
            let start = align_up(end - len, C::EXP_CEIL);
            let size = end - start;
            if size > C::EXP_CEIL {
                return (size, Self::oversize_bin());
            }
            // Rounding dropped it into the fixed range; carve fixed classes.
            cap = C::EXP_CEIL;
        }
        let limit = if cap < C::EXP_FLOOR {
            // Linear target: alignment does not bind.
            cap
        } else {
            let exp_cap = 3 * (1u64 << end.trailing_zeros());
            if cap < exp_cap {
                cap
            } else if exp_cap >= C::EXP_FLOOR {
                exp_cap
            } else {
                // No exponential class is aligned at `end`; fall back to the
                // largest linear class, which is always aligned.
                C::EXP_FLOOR - C::MIN_ALIGN
            }
        };
        if limit < C::LNR_FLOOR {
            // Degenerate config with no linear classes and alignment excluding
            // every exponential class.
            return (0, 0);
        }
        let bin = Self::floor_bin(limit);
        (bin_class::<C>(bin), bin)
    }

    /// Frees the alignment gap `[lo, hi)` left when the bump pointer is rounded
    /// up so an exponential block lands on its class alignment. Carves the gap
    /// into class-exact, aligned free blocks, largest first, *without* a
    /// `CARVE_MAX` bound (the gap is bounded — smaller than the block's class).
    /// A sub-class sliver at the low end is only possible when
    /// `LNR_FLOOR > MIN_ALIGN`, and is then unavoidable dead space. Caller holds
    /// the refill lock.
    #[inline]
    fn carve_gap(&self, lo: u64, hi: u64) {
        let mut end = hi;
        while end - lo >= C::MIN_ALIGN {
            let (size, bin) = Self::largest_carvable(end, end - lo);
            if size == 0 {
                break;
            }
            end -= size;
            self.store(end, pack(size, FREE));
            if bin == Self::oversize_bin() {
                self.push_oversize_locked(end);
            } else {
                self.push_fixed(bin, end);
            }
        }
    }

    /// Carves the excess of an in-use block `[block, block + bsize)` beyond
    /// `need` into free blocks, marks the block used at its final size, and
    /// returns that size. The caller must hold the refill lock.
    ///
    /// If the excess is below `SPLIT_MIN` the whole block is kept (the excess
    /// stays as internal slack). Otherwise blocks are carved from the *top* of
    /// the excess, largest first (see [`largest_carvable`](Self::largest_carvable)),
    /// for roughly `CARVE_MAX` iterations; every carved block is class-exact (or
    /// one oversized block) and aligned, so it lands in the correct free list.
    ///
    /// The retained block must itself stay a valid block — an oversized region,
    /// or an *exact* fixed class — because [`free`](CadAlloc::free) later maps its
    /// header size to a bin by ceiling: a fixed-range, non-class-exact size would
    /// be reused as a larger class and overrun its neighbour. So carving may run
    /// a few iterations past `CARVE_MAX` to reach a valid retained size, and any
    /// final uncarvable sub-`LNR_FLOOR` sliver becomes dead space rather than
    /// inflating the block. The block's start does not move, and it stays aligned
    /// to its (no larger) class.
    #[inline]
    fn split_and_free_tail(&self, block: u64, bsize: u64, need: u64) -> u64 {
        if bsize - need < C::SPLIT_MIN {
            self.store(block, pack(bsize, USED));
            return bsize;
        }
        let ptr = block + need;
        let mut end = block + bsize;
        let mut iter = 0;
        // Carve class-exact (or oversized) blocks off the top. `CARVE_MAX` bounds
        // the work, but the retained block `[block, end)` must itself stay a
        // valid block: an oversized region or an exact class may keep its slack
        // and stop, whereas a fixed-range, non-class-exact size must be carved
        // down further — a later `free` ceil-bins the header, so a non-exact
        // block would be reused as a larger class and overrun its neighbour.
        // Reaching a valid size costs only a few extra carves past the budget.
        while end - ptr >= C::MIN_ALIGN
            && (iter < C::CARVE_MAX || !Self::is_block_exact(end - block))
        {
            let (size, bin) = Self::largest_carvable(end, end - ptr);
            if size == 0 {
                break;
            }
            end -= size;
            self.store(end, pack(size, FREE));
            if bin == Self::oversize_bin() {
                self.push_oversize_locked(end);
            } else {
                self.push_fixed(bin, end);
            }
            iter += 1;
        }
        // If only an unavoidable sub-`LNR_FLOOR` sliver was left uncarvable, the
        // retained size may still be non-exact; drop to the largest class that
        // fits and leave the sliver as dead space, keeping the block class-exact.
        let mut alloc_size = end - block;
        if !Self::is_block_exact(alloc_size) {
            alloc_size = bin_class::<C>(Self::floor_bin(alloc_size));
        }
        // Alignment holds without moving `block`: the retained class is no larger
        // than the original (we only carve off the top), and `align_of_bin` is
        // non-decreasing in class size, so the retained class's alignment divides
        // the original's — which `block` already met.
        debug_assert!(
            alloc_size > C::EXP_CEIL
                || block & (Self::align_of_bin(bin_index::<C>(alloc_size)) - 1) == 0,
            "cadalloc: retained block not aligned to its class"
        );
        self.store(block, pack(alloc_size, USED));
        alloc_size
    }

    /// Allocates an oversized block of total size `need` (`> EXP_CEIL`): a
    /// first-fit search of the oversized free list, else a bump carve. A block
    /// larger than needed has its excess carved (see `split_and_free_tail`).
    /// Returns the block address (header, marked used), or `0` on exhaustion.
    #[inline]
    fn alloc_oversize(&self, need: u64) -> u64 {
        self.lock();
        let head = self.bin_head(Self::oversize_bin());

        // First-fit over the singly linked oversized list. `prev_link` is the
        // address of the word that points at `cur`.
        let mut prev_link = head;
        let mut cur = self.load(head);
        let mut chosen = 0;
        let mut chosen_prev_link = 0;
        while cur != 0 {
            if header_size(self.load(cur)) >= need {
                chosen = cur;
                chosen_prev_link = prev_link;
                break;
            }
            prev_link = cur + 8;
            cur = self.load(cur + 8);
        }

        let block = if chosen != 0 {
            self.store(chosen_prev_link, self.load(chosen + 8)); // unlink
            let bsize = header_size(self.load(chosen));
            self.split_and_free_tail(chosen, bsize, need);
            chosen
        } else {
            // Bump a fresh block, aligned to EXP_CEIL; free the resulting gap.
            let bump = self.load(self.bump_addr());
            let aligned = align_up(bump, C::EXP_CEIL);
            if aligned + need > self.heap_end() {
                0
            } else {
                self.carve_gap(bump, aligned);
                self.store(self.bump_addr(), aligned + need);
                self.store(aligned, pack(need, USED));
                aligned
            }
        };

        self.unlock();
        block
    }
}
