//! The atomic primitives the allocator is built on.
//!
//! `cadalloc` assumes a multithreaded environment and coordinates its
//! segregated free lists with atomic operations on 64-bit words *inside the
//! managed heap*. It does not reach for `core::sync::atomic` directly, because
//! the intended targets range from hosted 64-bit systems to embedded parts
//! where 64-bit atomics must be emulated (e.g. behind a critical section).
//! Instead the concrete operations live behind the [`Atomics`] trait, and a
//! [`Config`](crate::Config) names the backend it wants.
//!
//! Every operation acts on a naturally-aligned `u64` at a raw address. They are
//! all `unsafe`: the caller must guarantee `addr` points at a valid, aligned,
//! 64-bit word that is part of the managed region and is only ever accessed
//! through these operations for the duration of concurrent use.
//!
//! ## Memory ordering
//!
//! The allocator relies on these operations being *sequentially consistent*
//! (or at minimum: `Acquire` on loads, `Release` on stores, and `AcqRel` on
//! read-modify-write). An implementation that provides weaker ordering is
//! unsound. The provided [`CoreAtomics`] backend uses
//! [`SeqCst`](core::sync::atomic::Ordering::SeqCst).
//!
//! ## What you must implement
//!
//! Only three operations are fundamental: [`atomic_load`](Atomics::atomic_load),
//! [`atomic_store`](Atomics::atomic_store), and
//! [`atomic_cas`](Atomics::atomic_cas). The remaining five have default
//! implementations built from a compare-and-swap loop. Override them when the
//! target has a cheaper native instruction (`fetch_add`, `swap`, …).

/// Atomic operations on 64-bit words at raw addresses.
///
/// Implementors are zero-sized policy types: every method is an associated
/// function taking the target address, so no instance is ever constructed. See
/// the [module documentation](self) for the safety and ordering contract.
pub trait Atomics {
    /// Atomically loads the `u64` at `addr`.
    ///
    /// # Safety
    ///
    /// `addr` must point to a valid, aligned 64-bit word governed by this
    /// backend.
    unsafe fn atomic_load(addr: u64) -> u64;

    /// Atomically stores `val` to the `u64` at `addr`.
    ///
    /// # Safety
    ///
    /// See [`atomic_load`](Atomics::atomic_load).
    unsafe fn atomic_store(addr: u64, val: u64);

    /// Atomic compare-and-swap.
    ///
    /// If the word at `addr` equals `current`, replaces it with `new`. Returns
    /// the value that was read *before* the attempt: the swap succeeded if and
    /// only if the returned value equals `current`.
    ///
    /// # Safety
    ///
    /// See [`atomic_load`](Atomics::atomic_load).
    unsafe fn atomic_cas(addr: u64, current: u64, new: u64) -> u64;

    /// Atomically stores `val` and returns the previous value.
    ///
    /// # Safety
    ///
    /// See [`atomic_load`](Atomics::atomic_load).
    #[inline]
    unsafe fn atomic_swap(addr: u64, val: u64) -> u64 {
        loop {
            let cur = unsafe { Self::atomic_load(addr) };
            if unsafe { Self::atomic_cas(addr, cur, val) } == cur {
                return cur;
            }
        }
    }

    /// Atomically adds `val` (wrapping) and returns the previous value.
    ///
    /// # Safety
    ///
    /// See [`atomic_load`](Atomics::atomic_load).
    #[inline]
    unsafe fn atomic_add(addr: u64, val: u64) -> u64 {
        loop {
            let cur = unsafe { Self::atomic_load(addr) };
            let new = cur.wrapping_add(val);
            if unsafe { Self::atomic_cas(addr, cur, new) } == cur {
                return cur;
            }
        }
    }

    /// Atomically subtracts `val` (wrapping) and returns the previous value.
    ///
    /// # Safety
    ///
    /// See [`atomic_load`](Atomics::atomic_load).
    #[inline]
    unsafe fn atomic_sub(addr: u64, val: u64) -> u64 {
        loop {
            let cur = unsafe { Self::atomic_load(addr) };
            let new = cur.wrapping_sub(val);
            if unsafe { Self::atomic_cas(addr, cur, new) } == cur {
                return cur;
            }
        }
    }

    /// Atomically increments the word at `addr` by one, returning the previous
    /// value.
    ///
    /// # Safety
    ///
    /// See [`atomic_load`](Atomics::atomic_load).
    #[inline]
    unsafe fn atomic_inc(addr: u64) -> u64 {
        unsafe { Self::atomic_add(addr, 1) }
    }

    /// Atomically decrements the word at `addr` by one, returning the previous
    /// value.
    ///
    /// # Safety
    ///
    /// See [`atomic_load`](Atomics::atomic_load).
    #[inline]
    unsafe fn atomic_dec(addr: u64) -> u64 {
        unsafe { Self::atomic_sub(addr, 1) }
    }
}

/// An [`Atomics`] backend built on [`core::sync::atomic::AtomicU64`].
///
/// Available only where the target has native 64-bit atomics
/// (`target_has_atomic = "64"`), which covers essentially every hosted 64-bit
/// platform. Every operation maps to the corresponding `AtomicU64` method with
/// [`SeqCst`](core::sync::atomic::Ordering::SeqCst), so the read-modify-write
/// operations are single native
/// instructions rather than compare-and-swap loops. Embedded targets without
/// 64-bit atomics should supply their own backend instead.
#[cfg(target_has_atomic = "64")]
#[derive(Clone, Copy, Debug, Default)]
pub struct CoreAtomics;

#[cfg(target_has_atomic = "64")]
impl CoreAtomics {
    /// Views `addr` as a shared reference to an `AtomicU64`.
    ///
    /// # Safety
    ///
    /// `addr` must be a valid, `AtomicU64`-aligned address that stays valid for
    /// the duration of the borrow.
    #[inline(always)]
    unsafe fn at<'a>(addr: u64) -> &'a core::sync::atomic::AtomicU64 {
        // 64-bit target: `u64` -> `usize` is lossless.
        unsafe { &*(addr as usize as *const core::sync::atomic::AtomicU64) }
    }
}

#[cfg(target_has_atomic = "64")]
impl Atomics for CoreAtomics {
    #[inline(always)]
    unsafe fn atomic_load(addr: u64) -> u64 {
        unsafe { Self::at(addr) }.load(core::sync::atomic::Ordering::SeqCst)
    }

    #[inline(always)]
    unsafe fn atomic_store(addr: u64, val: u64) {
        unsafe { Self::at(addr) }.store(val, core::sync::atomic::Ordering::SeqCst);
    }

    #[inline(always)]
    unsafe fn atomic_cas(addr: u64, current: u64, new: u64) -> u64 {
        use core::sync::atomic::Ordering::SeqCst;
        match unsafe { Self::at(addr) }.compare_exchange(current, new, SeqCst, SeqCst) {
            // On success the read value equals `current`; on failure it is the
            // value that was actually present. Either way, return what was read.
            Ok(prev) | Err(prev) => prev,
        }
    }

    #[inline(always)]
    unsafe fn atomic_swap(addr: u64, val: u64) -> u64 {
        unsafe { Self::at(addr) }.swap(val, core::sync::atomic::Ordering::SeqCst)
    }

    #[inline(always)]
    unsafe fn atomic_add(addr: u64, val: u64) -> u64 {
        unsafe { Self::at(addr) }.fetch_add(val, core::sync::atomic::Ordering::SeqCst)
    }

    #[inline(always)]
    unsafe fn atomic_sub(addr: u64, val: u64) -> u64 {
        unsafe { Self::at(addr) }.fetch_sub(val, core::sync::atomic::Ordering::SeqCst)
    }
}
