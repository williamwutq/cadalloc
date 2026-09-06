//! The allocator core: size-class mapping, block headers, and the `alloc` /
//! `free` fast paths.
//!
//! # Layout
//!
//! An [`Allocator`] is stateless — a zero-sized marker for a [`Config`]. All
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
//! checked by [`verify`](Allocator::verify). Each bin head is a `u64`: the
//! address of the first free block in that bin (`0` when empty), with the low
//! alignment bits used as an ABA tag for the lock-free bins. The last bin
//! (`NUM_BINS - 1`) is the oversized bin.
//!
//! # Blocks
//!
//! Every block begins with a `MIN_ALIGN`-byte header region. Its first `u64`
//! packs the block's total size (header included) in the high 56 bits and the
//! free flag in the low byte (`1` = free). Its second `u64` (offset 8, always
//! inside the header region because `MIN_ALIGN >= 16`) holds the free-list
//! `next` pointer while the block is free. The payload starts at
//! `block + MIN_ALIGN`. There is no footer and blocks are never coalesced.
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
//! from `heap_base`). [`free`](Allocator::free) trusts that the slice it is
//! given was produced by [`alloc`](Allocator::alloc) on this allocator.

use crate::atomic::Atomics;
use crate::config::Config;
use crate::slice::Slice;
use core::hint::spin_loop;
use core::marker::PhantomData;

/// Free flag stored in the low byte of a block header.
const FREE: u64 = 1;
/// "In use" flag stored in the low byte of a block header.
const USED: u64 = 0;

/// Packs a total block `size` and a free/used `flag` into a header word.
#[inline(always)]
const fn pack(size: u64, flag: u64) -> u64 {
    (size << 8) | (flag & 0xFF)
}

/// Extracts the total block size from a header word.
#[inline(always)]
const fn header_size(header: u64) -> u64 {
    header >> 8
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
/// [`Atomics`] backend, an `Allocator` may be shared across threads (for
/// example as a `static`) once [`init`](Allocator::init) has run.
///
/// # Usage
///
/// ```no_run
/// use cadalloc::{Allocator, Config, CoreAtomics};
///
/// struct C;
/// impl Config for C {
///     type Atomics = CoreAtomics;
///     const MIN_ALIGN: u64 = 16;
///     const LNR_FLOOR: u64 = 16;
///     const EXP_FLOOR: u64 = 256;
///     const EXP_CEIL: u64 = 65536;
///     const HEAP_BASE: u64 = 0x2000_0000; // a real, mapped region
///     const HEAP_SIZE: u64 = 1 << 20;
/// }
///
/// let a = Allocator::<C>::new();
/// a.init().unwrap();
/// let s = a.alloc(100, 16);
/// if !s.is_null() {
///     // ... use [s.ptr, s.ptr + s.len) ...
///     a.free(s);
/// }
/// ```
pub struct Allocator<C: Config> {
    _config: PhantomData<fn() -> C>,
}

impl<C: Config> Default for Allocator<C> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Why [`Allocator::init`] could not prepare the heap.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InitError {
    /// The heap base address is not aligned to `MIN_ALIGN`.
    MisalignedHeap,
    /// The heap is too small to hold the control region.
    HeapTooSmall,
}

/// Why [`Allocator::verify`] rejected a heap.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VerifyError {
    /// The `"CADALLOC"` marker is absent — the heap was never initialized (or
    /// not by `cadalloc`).
    BadMagic,
    /// The marker is present but the configuration fingerprint differs — the
    /// heap was laid out by a different configuration. Using it would corrupt
    /// memory.
    ConfigMismatch,
}

impl<C: Config> Allocator<C> {
    /// Creates an allocator handle. Call [`init`](Allocator::init) once, before
    /// any allocation, to prepare the heap.
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        Self {
            _config: PhantomData,
        }
    }

    /// The total number of bins for this configuration (statically known).
    #[must_use]
    #[inline]
    pub const fn bin_count() -> u64 {
        C::NUM_BINS
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
    fn control_bytes() -> u64 {
        align_up(META_BYTES + (C::NUM_BINS + 2) * 8, C::MIN_ALIGN)
    }

    /// One-past-the-end address of the heap.
    #[inline(always)]
    fn heap_end(&self) -> u64 {
        C::heap_base() + C::heap_size()
    }

    /// The oversized bin index.
    #[inline(always)]
    fn oversize_bin() -> u64 {
        C::NUM_BINS - 1
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

    #[inline(always)]
    fn cas(&self, addr: u64, current: u64, new: u64) -> u64 {
        // SAFETY: see `load`.
        unsafe { C::Atomics::atomic_cas(addr, current, new) }
    }

    // --- refill lock -------------------------------------------------------

    #[inline]
    fn lock(&self) {
        let addr = self.lock_addr();
        while self.cas(addr, 0, 1) != 0 {
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
    fn push_fixed(&self, k: u64, block: u64) {
        let head = self.bin_head(k);
        let mask = C::MIN_ALIGN - 1;
        loop {
            let old = self.load(head);
            let old_addr = old & !mask;
            let new_tag = (old + 1) & mask;
            self.store(block + 8, old_addr); // block.next = current head
            if self.cas(head, old, block | new_tag) == old {
                return;
            }
        }
    }

    /// Pops a block from bin `k`'s lock-free stack, or returns `0` if empty.
    #[inline]
    fn pop_fixed(&self, k: u64) -> u64 {
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
            if self.cas(head, old, next | new_tag) == old {
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
    /// constants (see [`verify`](Allocator::verify)).
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
    #[inline(never)]
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
    /// compares them against [`magic`](Allocator::magic) and this
    /// configuration's [`config_hash`](Allocator::config_hash). **Call this
    /// before touching a heap that some *other* build may have prepared** (for
    /// example a persistent or shared-memory region): running an allocator over
    /// a heap laid out by a different configuration corrupts memory silently,
    /// and this is the cheap guard against it. It reads only two words and does
    /// not need the lock.
    #[inline(never)]
    pub fn verify(&self) -> Result<(), VerifyError> {
        if self.load(self.magic_addr()) != MAGIC {
            return Err(VerifyError::BadMagic);
        }
        if self.load(self.fingerprint_addr()) != config_fingerprint::<C>() {
            return Err(VerifyError::ConfigMismatch);
        }
        Ok(())
    }

    /// Allocates a block with at least `size` payload bytes and the given
    /// `align`, returning a [`Slice`] over its payload, or [`Slice::NULL`] on
    /// failure.
    ///
    /// `align` must be a power of two no greater than `MIN_ALIGN` (the payload
    /// is always `MIN_ALIGN`-aligned); larger alignments are not yet supported
    /// and return the null slice. The returned slice's `len` is the block's full
    /// usable capacity, which may exceed `size`.
    #[must_use]
    #[inline(never)]
    pub fn alloc(&self, size: u64, align: u64) -> Slice {
        if align > C::MIN_ALIGN || (align > 1 && !align.is_power_of_two()) {
            return Slice::NULL;
        }
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
        Slice::new(block + C::MIN_ALIGN, bsize - C::MIN_ALIGN)
    }

    /// Returns a block previously handed out by [`alloc`](Allocator::alloc) to
    /// its bin. Freeing [`Slice::NULL`] is a no-op.
    ///
    /// The slice's `ptr` must be one returned by `alloc` on this allocator; its
    /// `len` is ignored (the true size is read from the block header).
    #[inline(never)]
    pub fn free(&self, block: Slice) {
        if block.is_null() {
            return;
        }
        let hdr = block.ptr - C::MIN_ALIGN;
        let size = header_size(self.load(hdr));
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

    // --- fixed-class allocation --------------------------------------------

    /// Allocates a fixed-class block from `bin`, returning its address (header),
    /// or `0` if the heap is exhausted. The returned block's header is marked
    /// used with its class size.
    #[inline]
    fn alloc_fixed(&self, bin: u64) -> u64 {
        // Blocks in a fixed bin are always exactly the class size.
        let class = bin_class::<C>(bin);

        // Fast path: a free block already in the bin.
        let block = self.pop_fixed(bin);
        if block != 0 {
            self.store(block, pack(class, USED));
            return block;
        }

        // Slow path: re-check under the lock (another thread may have freed
        // into this bin), else carve a fresh class-sized block from the bump
        // region.
        self.lock();
        let mut block = self.pop_fixed(bin);
        if block == 0 {
            let bump = self.load(self.bump_addr());
            if bump + class <= self.heap_end() {
                self.store(self.bump_addr(), bump + class);
                block = bump;
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

    /// Inserts a free block of `size` bytes into its correct bin while holding
    /// the refill lock: the oversized list directly, or a fixed bin via its
    /// lock-free push.
    #[inline]
    fn insert_free_locked(&self, addr: u64, size: u64) {
        self.store(addr, pack(size, FREE));
        let bin = bin_index::<C>(size);
        if bin == Self::oversize_bin() {
            self.push_oversize_locked(addr);
        } else {
            self.push_fixed(bin, addr);
        }
    }

    /// Allocates an oversized block of total size `need` (`> EXP_CEIL`): a
    /// first-fit search of the oversized free list, else a bump carve. A block
    /// larger than needed is split when the leftover is worth it. Returns the
    /// block address (header, marked used), or `0` on exhaustion.
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
            // Unlink and (maybe) split.
            self.store(chosen_prev_link, self.load(chosen + 8));
            let bsize = header_size(self.load(chosen));
            let leftover = bsize - need;
            if leftover >= C::SPLIT_MIN && leftover >= C::MIN_ALIGN && C::CARVE_MAX >= 2 {
                self.store(chosen, pack(need, USED));
                self.insert_free_locked(chosen + need, leftover);
            } else {
                self.store(chosen, pack(bsize, USED));
            }
            chosen
        } else {
            // Bump a fresh block of exactly `need`.
            let bump = self.load(self.bump_addr());
            if bump + need > self.heap_end() {
                0
            } else {
                self.store(self.bump_addr(), bump + need);
                self.store(bump, pack(need, USED));
                bump
            }
        };

        self.unlock();
        block
    }
}
