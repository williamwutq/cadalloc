//! Unit tests for the configuration surface and allocator core.

use crate::allocator::{bin_class, bin_index};
use crate::*;

/// A representative configuration used to exercise the config surface.
struct TestConfig;

impl Config for TestConfig {
    type Atomics = CoreAtomics;

    const MIN_ALIGN: u64 = 16;
    const LNR_FLOOR: u64 = 32;
    const EXP_FLOOR: u64 = 256;
    const EXP_CEIL: u64 = 65536;

    const HEAP_BASE: u64 = 0x1_0000;
    const HEAP_SIZE: u64 = 0x10_0000;
}

// Forces compile-time validation of `TestConfig`'s constants.
const _: () = assert_config_valid::<TestConfig>();

#[test]
fn const_heap_accessors_read_compile_time_values() {
    assert_eq!(const_heap_base::<TestConfig>(), 0x1_0000);
    assert_eq!(const_heap_size::<TestConfig>(), 0x10_0000);
    assert_eq!(TestConfig::heap_base(), 0x1_0000);
    assert_eq!(TestConfig::heap_size(), 0x10_0000);
}

#[test]
fn slice_null_and_bounds() {
    assert!(Slice::NULL.is_null());
    // SAFETY: a fabricated slice used only for pure address arithmetic below;
    // its bytes are never read, written, or handed to the allocator.
    let s = unsafe { Slice::new(0x1000, 0x100) };
    assert!(!s.is_null());
    assert_eq!(s.end(), 0x1100);
    assert!(s.contains(0x1000));
    assert!(s.contains(0x10FF));
    assert!(!s.contains(0x1100));
}

#[test]
fn core_atomics_round_trip() {
    // A single 64-bit word on the stack, addressed by value.
    let cell = core::sync::atomic::AtomicU64::new(0);
    let addr = &cell as *const _ as u64;
    // SAFETY: `addr` names a live, aligned `AtomicU64` for this scope.
    unsafe {
        CoreAtomics::atomic_store(addr, 10);
        assert_eq!(CoreAtomics::atomic_load(addr), 10);
        assert_eq!(CoreAtomics::atomic_add(addr, 5), 10);
        assert_eq!(CoreAtomics::atomic_load(addr), 15);
        assert_eq!(CoreAtomics::atomic_cas(addr, 15, 20), 15);
        assert_eq!(CoreAtomics::atomic_load(addr), 20);
        assert_eq!(CoreAtomics::atomic_cas(addr, 99, 0), 20); // no-op: 20 != 99
        assert_eq!(CoreAtomics::atomic_swap(addr, 7), 20);
        assert_eq!(CoreAtomics::atomic_dec(addr), 7);
        assert_eq!(CoreAtomics::atomic_load(addr), 6);

        // Bitwise fetch operations return the previous value.
        CoreAtomics::atomic_store(addr, 0b1010);
        assert_eq!(CoreAtomics::atomic_or(addr, 0b0100), 0b1010);
        assert_eq!(CoreAtomics::atomic_load(addr), 0b1110);
        assert_eq!(CoreAtomics::atomic_and(addr, 0b1100), 0b1110);
        assert_eq!(CoreAtomics::atomic_load(addr), 0b1100);

        // Weak CAS: loop to absorb spurious failure (single-threaded, so the
        // read is always the current value and only spurious failure retries).
        loop {
            let (read, ok) = CoreAtomics::atomic_cas_weak(addr, 0b1100, 42);
            assert_eq!(read, 0b1100);
            if ok {
                break;
            }
        }
        assert_eq!(CoreAtomics::atomic_load(addr), 42);
        // A mismatched expected value fails (never spuriously succeeds).
        let (read, ok) = CoreAtomics::atomic_cas_weak(addr, 999, 0);
        assert!(!ok);
        assert_eq!(read, 42);
    }
}

// --- bin-math tests (pure, parallel-safe) ----------------------------------

#[test]
fn bin_counts_are_statically_computed() {
    // MIN_ALIGN=16, LNR_FLOOR=32, EXP_FLOOR=256, EXP_CEIL=65536:
    // linear = (256-32)/16 = 14; exp = 2*(16-8)+1 = 17; +1 oversized = 32.
    assert_eq!(TestConfig::NUM_LINEAR, 14);
    assert_eq!(TestConfig::NUM_EXP, 17);
    assert_eq!(TestConfig::NUM_BINS, 32);
}

#[test]
fn bin_classes_are_ordered_and_aligned() {
    let fixed = TestConfig::NUM_BINS - 1;
    let mut prev = 0;
    for k in 0..fixed {
        let c = bin_class::<TestConfig>(k);
        assert!(c > prev, "bin {k} class {c} not increasing from {prev}");
        assert_eq!(c % TestConfig::MIN_ALIGN, 0, "bin {k} class {c} misaligned");
        prev = c;
    }
    // First/last known classes (14 linear: bins 0..13 = 32..240; exp from bin 14).
    assert_eq!(bin_class::<TestConfig>(0), 32);
    assert_eq!(bin_class::<TestConfig>(13), 240);
    assert_eq!(bin_class::<TestConfig>(14), 256);
    assert_eq!(bin_class::<TestConfig>(15), 384);
    assert_eq!(bin_class::<TestConfig>(fixed - 1), 65536);
    assert_eq!(bin_class::<TestConfig>(fixed), u64::MAX); // oversized sentinel
}

#[test]
fn bin_index_round_trips_every_class() {
    let fixed = TestConfig::NUM_BINS - 1;
    for k in 0..fixed {
        let c = bin_class::<TestConfig>(k);
        assert_eq!(
            bin_index::<TestConfig>(c),
            k,
            "class {c} did not map back to bin {k}"
        );
    }
}

#[test]
fn bin_index_always_fits() {
    for s in [
        1u64, 15, 16, 17, 100, 240, 241, 255, 256, 257, 384, 385, 512, 4096, 65535, 65536,
    ] {
        let k = bin_index::<TestConfig>(s);
        assert!(
            k < TestConfig::NUM_BINS - 1,
            "size {s} unexpectedly oversized"
        );
        assert!(
            bin_class::<TestConfig>(k) >= s,
            "bin {k} class too small for {s}"
        );
    }
    // Above EXP_CEIL is the oversized bin.
    assert_eq!(bin_index::<TestConfig>(65537), TestConfig::NUM_BINS - 1);
    assert_eq!(bin_index::<TestConfig>(1 << 30), TestConfig::NUM_BINS - 1);
}

#[test]
fn largest_carvable_matches_brute_force() {
    type C = TestConfig; // MIN=16, LNR=16, EXP_FLOOR=256, EXP_CEIL=65536

    fn align_up(x: u64, a: u64) -> u64 {
        (x + (a - 1)) & !(a - 1)
    }
    // The alignment a class of size `v` demands (mirrors `align_of_bin`).
    fn class_align(v: u64) -> u64 {
        if v < C::EXP_FLOOR {
            C::MIN_ALIGN
        } else {
            let a = 1u64 << (v.ilog2() - 1);
            if a < C::MIN_ALIGN { C::MIN_ALIGN } else { a }
        }
    }
    // Ground truth: a region past EXP_CEIL is one oversized block aligned to
    // EXP_CEIL (falling back to fixed if that rounding shrinks it into the fixed
    // range); otherwise the largest fixed class fitting `len` and aligned at
    // `end`.
    fn reference(end: u64, len: u64) -> u64 {
        let mut cap = len;
        if len > C::EXP_CEIL {
            let size = end - align_up(end - len, C::EXP_CEIL);
            if size > C::EXP_CEIL {
                return size;
            }
            cap = C::EXP_CEIL;
        }
        let mut best = 0;
        let fixed = C::NUM_BINS - 1;
        let mut bin = 0;
        while bin < fixed {
            let c = bin_class::<C>(bin);
            if c <= cap && end % class_align(c) == 0 && c > best {
                best = c;
            }
            bin += 1;
        }
        best
    }

    for ez in 4..21u32 {
        // Three addresses whose largest power-of-two divisor is exactly 2^ez,
        // kept well above the largest tested `len` so `end - len` is valid.
        for m in 0..3u64 {
            let end = ((2 * m + 1) + (1u64 << 20)) << ez;
            for &len in &[
                1u64, 16, 100, 240, 255, 256, 300, 500, 768, 1000, 2048, 3000, 4096, 6144, 60000,
                65536, 70000, 200000,
            ] {
                let (size, bin) = CadAlloc::<C>::largest_carvable(end, len);
                let expected = reference(end, len);
                assert_eq!(size, expected, "end={end:#x} len={len}: size mismatch");
                if size == 0 {
                    continue;
                }
                if bin == C::NUM_BINS - 1 {
                    assert!(size > C::EXP_CEIL, "oversized bin holds a small block");
                    assert_eq!((end - size) % C::EXP_CEIL, 0, "oversized misaligned");
                } else {
                    assert_eq!(bin_class::<C>(bin), size, "bin/size disagree");
                    assert_eq!(end % class_align(size), 0, "carved block misaligned");
                }
            }
        }
    }
}

// --- allocator integration test (over a real static heap) ------------------

/// `"CADALLOC"` as a little-endian u64, computed independently of the crate.
const MAGIC_LE: u64 = u64::from_le_bytes(*b"CADALLOC");

const HEAP_N: usize = 1 << 16;

#[repr(C, align(64))]
struct Heap([u8; HEAP_N]);

static mut HEAP: Heap = Heap([0; HEAP_N]);

fn heap_base_addr() -> u64 {
    &raw const HEAP as u64
}

/// Config backed by the static `HEAP` above, with a small `EXP_CEIL` so the
/// oversized path is reachable within the buffer.
struct HeapConfig;

impl Config for HeapConfig {
    type Atomics = CoreAtomics;

    const MIN_ALIGN: u64 = 16;
    const LNR_FLOOR: u64 = 32;
    const EXP_FLOOR: u64 = 256;
    const EXP_CEIL: u64 = 4096;

    fn heap_base() -> u64 {
        heap_base_addr()
    }
    fn heap_size() -> u64 {
        HEAP_N as u64
    }
}

/// Same heap as `HeapConfig` but a different `EXP_CEIL`, so its fingerprint
/// differs. Only reads the fixed-offset metadata words, so it is safe to use
/// against a `HeapConfig`-initialized heap.
struct MismatchConfig;

impl Config for MismatchConfig {
    type Atomics = CoreAtomics;
    const MIN_ALIGN: u64 = 16;
    const LNR_FLOOR: u64 = 32;
    const EXP_FLOOR: u64 = 256;
    const EXP_CEIL: u64 = 8192; // differs from HeapConfig's 4096
    fn heap_base() -> u64 {
        heap_base_addr()
    }
    fn heap_size() -> u64 {
        HEAP_N as u64
    }
}

#[test]
fn allocator_alloc_free_reuse() {
    let a = CadAlloc::<HeapConfig>::new();

    // A pristine (zeroed) heap has no marker.
    assert_eq!(a.verify(), Err(VerifyError::BadMagic));

    a.init().expect("init");

    // After init the marker and fingerprint check out...
    assert_eq!(a.verify(), Ok(()));
    assert_eq!(CadAlloc::<HeapConfig>::magic(), MAGIC_LE);
    // ...but a different configuration is rejected.
    assert_eq!(
        CadAlloc::<MismatchConfig>::new().verify(),
        Err(VerifyError::ConfigMismatch)
    );

    // Basic fixed-class allocation.
    let s1 = a.alloc(100);
    assert!(!s1.is_null());
    assert_eq!(s1.ptr % 16, 0, "payload must be MIN_ALIGN-aligned");
    assert!(s1.len >= 100, "capacity {} < requested 100", s1.len);

    // A second allocation must not overlap the first.
    let s2 = a.alloc(100);
    assert!(!s2.is_null());
    assert_ne!(s1.ptr, s2.ptr);
    assert!(
        s2.ptr >= s1.ptr + s1.len || s1.ptr >= s2.ptr + s2.len,
        "blocks overlap"
    );

    // Freeing then re-allocating the same class reuses the block (LIFO).
    let p1 = s1.ptr;
    a.free(s1);
    let s3 = a.alloc(100);
    assert_eq!(s3.ptr, p1, "freed block should be reused");

    a.free(s2);
    a.free(s3);

    // Oversized path: > EXP_CEIL (4096).
    let big = a.alloc(5000);
    assert!(!big.is_null());
    assert!(big.len >= 5000);
    let bp = big.ptr;
    a.free(big);
    // First-fit on the oversized list should recover the same block.
    let big2 = a.alloc(5000);
    assert_eq!(big2.ptr, bp, "oversized block should be reused");
    a.free(big2);
}

#[test]
fn init_rejects_bad_heap() {
    struct Misaligned;
    impl Config for Misaligned {
        type Atomics = CoreAtomics;
        const MIN_ALIGN: u64 = 16;
        const LNR_FLOOR: u64 = 32;
        const EXP_FLOOR: u64 = 256;
        const EXP_CEIL: u64 = 4096;
        fn heap_base() -> u64 {
            heap_base_addr() + 8 // deliberately off-alignment
        }
        fn heap_size() -> u64 {
            HEAP_N as u64
        }
    }
    assert_eq!(
        CadAlloc::<Misaligned>::new().init(),
        Err(InitError::MisalignedHeap)
    );

    struct Tiny;
    impl Config for Tiny {
        type Atomics = CoreAtomics;
        const MIN_ALIGN: u64 = 16;
        const LNR_FLOOR: u64 = 32;
        const EXP_FLOOR: u64 = 256;
        const EXP_CEIL: u64 = 4096;
        fn heap_base() -> u64 {
            heap_base_addr()
        }
        fn heap_size() -> u64 {
            16 // far too small for the control region
        }
    }
    assert_eq!(CadAlloc::<Tiny>::new().init(), Err(InitError::HeapTooSmall));
}

// --- FFI export shims (over their own static heap) -------------------------

#[repr(C, align(64))]
struct FfiHeap([u8; HEAP_N]);

static mut FFI_HEAP: FfiHeap = FfiHeap([0; HEAP_N]);

fn ffi_heap_base() -> u64 {
    &raw const FFI_HEAP as u64
}

struct FfiConfig;

impl Config for FfiConfig {
    type Atomics = CoreAtomics;
    const MIN_ALIGN: u64 = 16;
    const LNR_FLOOR: u64 = 32;
    const EXP_FLOOR: u64 = 256;
    const EXP_CEIL: u64 = 4096;
    fn heap_base() -> u64 {
        ffi_heap_base()
    }
    fn heap_size() -> u64 {
        HEAP_N as u64
    }
}

crate::export_c_api! {
    config: FfiConfig,
    init: cadtest_init,
    alloc: cadtest_alloc,
    realloc: cadtest_realloc,
    free: cadtest_free,
    verify: cadtest_verify,
}

#[test]
fn ffi_shims_round_trip() {
    // BadMagic before init, then the full round trip through the C entry points.
    assert_eq!(cadtest_verify(), 1);
    assert_eq!(cadtest_init(), 0);
    assert_eq!(cadtest_verify(), 0);

    let s = cadtest_alloc(100);
    assert!(!s.is_null());
    assert!(s.len >= 100);
    cadtest_free(s);

    // Reuse after free (LIFO), exercising both alloc and free shims.
    let s2 = cadtest_alloc(100);
    assert_eq!(s2.ptr, s.ptr);

    // realloc shim: grow, then free through the shim.
    let s3 = cadtest_realloc(s2, 2000);
    assert!(!s3.is_null());
    assert!(s3.len >= 2000);
    cadtest_free(s3);

    // realloc(NULL, ...) behaves like alloc.
    let s4 = cadtest_realloc(Slice::NULL, 64);
    assert!(!s4.is_null());
    cadtest_free(s4);
}

// --- double-free detection (own static heap) -------------------------------

#[repr(C, align(64))]
struct DfHeap([u8; HEAP_N]);

static mut DF_HEAP: DfHeap = DfHeap([0; HEAP_N]);

struct DfConfig;

impl Config for DfConfig {
    type Atomics = CoreAtomics;
    const MIN_ALIGN: u64 = 16;
    const LNR_FLOOR: u64 = 32;
    const EXP_FLOOR: u64 = 256;
    const EXP_CEIL: u64 = 4096;
    fn heap_base() -> u64 {
        &raw const DF_HEAP as u64
    }
    fn heap_size() -> u64 {
        HEAP_N as u64
    }
}

// Freeing an already-free block asserts in debug builds (and is a silent no-op
// in release, which this test can't observe — hence the debug gate).
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "double free")]
fn double_free_is_detected() {
    let a = CadAlloc::<DfConfig>::new();
    a.init().expect("init");
    let s = a.alloc(100);
    assert!(!s.is_null());
    a.free(s);
    a.free(s); // second free -> debug assertion
}

// --- realloc (over its own static heap) ------------------------------------

#[repr(C, align(64))]
struct ReallocHeap([u8; HEAP_N]);

static mut REALLOC_HEAP: ReallocHeap = ReallocHeap([0; HEAP_N]);

fn realloc_heap_base() -> u64 {
    &raw const REALLOC_HEAP as u64
}

struct ReallocConfig;

impl Config for ReallocConfig {
    type Atomics = CoreAtomics;
    const MIN_ALIGN: u64 = 16;
    const LNR_FLOOR: u64 = 32;
    const EXP_FLOOR: u64 = 256;
    const EXP_CEIL: u64 = 4096;
    fn heap_base() -> u64 {
        realloc_heap_base()
    }
    fn heap_size() -> u64 {
        HEAP_N as u64
    }
}

#[test]
fn realloc_grows_shrinks_relocates() {
    let a = CadAlloc::<ReallocConfig>::new();
    a.init().expect("init");

    // realloc(NULL, ...) behaves like alloc.
    let n = a.realloc(Slice::NULL, 50);
    assert!(!n.is_null());
    a.free(n);

    // Grow relocates and preserves the old contents.
    let s = a.alloc(64);
    assert!(!s.is_null());
    // SAFETY: `s.ptr` is a live 64-byte payload in our static heap.
    unsafe {
        for i in 0..64u64 {
            (s.ptr as *mut u8).add(i as usize).write(i as u8);
        }
    }
    let g = a.realloc(s, 2000);
    assert!(!g.is_null());
    assert!(g.len >= 2000);
    // SAFETY: `g.ptr` holds the relocated payload.
    unsafe {
        for i in 0..64u64 {
            assert_eq!((g.ptr as *const u8).add(i as usize).read(), i as u8);
        }
    }
    a.free(g);

    // Shrink within the fixed classes stays in place (default SPLIT_MIN keeps
    // the leftover as internal slack).
    let f = a.alloc(2000);
    let cap = f.len;
    let t = a.realloc(f, 100);
    assert_eq!(t.ptr, f.ptr, "fixed shrink should not move");
    assert_eq!(t.len, cap, "fixed shrink keeps the whole block");
    a.free(t);

    // Oversized shrink stays in place and shrinks; a follow-up allocation is
    // valid and disjoint from the retained block.
    let big = a.alloc(12000);
    assert!(big.len >= 12000);
    let sh = a.realloc(big, 5000);
    assert_eq!(sh.ptr, big.ptr, "oversized shrink stays in place");
    assert!(sh.len >= 5000 && sh.len < big.len, "block actually shrank");
    let other = a.alloc(3000);
    assert!(!other.is_null());
    assert!(
        !overlap(sh, other),
        "new allocation overlaps the retained block"
    );
    a.free(other);
    a.free(sh);
}

// --- fixed-range multi-carve (SPLIT_MIN < EXP_CEIL) ------------------------

#[repr(C, align(64))]
struct CarveHeap([u8; HEAP_N]);

static mut CARVE_HEAP: CarveHeap = CarveHeap([0; HEAP_N]);

fn carve_heap_base() -> u64 {
    &raw const CARVE_HEAP as u64
}

/// A low `SPLIT_MIN` so a fixed-range excess (`<= EXP_CEIL`) gets carved into
/// several class-exact blocks rather than kept as slack.
struct CarveConfig;

impl Config for CarveConfig {
    type Atomics = CoreAtomics;
    const MIN_ALIGN: u64 = 16;
    const LNR_FLOOR: u64 = 32;
    const EXP_FLOOR: u64 = 256;
    const EXP_CEIL: u64 = 4096;
    const SPLIT_MIN: u64 = 256; // < EXP_CEIL: carve fixed-range excess
    const CARVE_MAX: u64 = 8; // enough iterations to fully decompose it
    fn heap_base() -> u64 {
        carve_heap_base()
    }
    fn heap_size() -> u64 {
        HEAP_N as u64
    }
}

// Validates the low-SPLIT_MIN config at compile time.
const _: () = assert_config_valid::<CarveConfig>();

/// Returns `true` if the payloads `[ptr, ptr+len)` of `a` and `b` overlap.
fn overlap(a: Slice, b: Slice) -> bool {
    a.ptr < b.ptr + b.len && b.ptr < a.ptr + a.len
}

// Larger than the shared HEAP_N so several oversized (> EXP_CEIL = 65536)
// allocations, each with an up-to-EXP_CEIL alignment gap, fit.
const ALIGN_HEAP_N: usize = 1 << 20;

#[repr(C, align(64))]
struct AlignHeap([u8; ALIGN_HEAP_N]);

static mut ALIGN_HEAP: AlignHeap = AlignHeap([0; ALIGN_HEAP_N]);

fn align_heap_base() -> u64 {
    &raw const ALIGN_HEAP as u64
}

struct AlignConfig;

impl Config for AlignConfig {
    type Atomics = CoreAtomics;
    const MIN_ALIGN: u64 = 16;
    const LNR_FLOOR: u64 = 32;
    const EXP_FLOOR: u64 = 256;
    const EXP_CEIL: u64 = 65536;
    fn heap_base() -> u64 {
        align_heap_base()
    }
    fn heap_size() -> u64 {
        ALIGN_HEAP_N as u64
    }
}

#[test]
fn bump_aligns_exponential_blocks() {
    let a = CadAlloc::<AlignConfig>::new();
    a.init().expect("init");

    // Each exponential block's header must sit on its class alignment: half the
    // octave base (`2^(n-1)` for a block in the `2^n` octave).
    for &sz in &[300u64, 500, 2000, 5000, 9000] {
        let s = a.alloc(sz);
        assert!(!s.is_null());
        let header = s.ptr - 16; // block start
        let class = s.len + 16; // class-exact block size
        assert!(class >= AlignConfig::EXP_FLOOR, "not exponential: {class}");
        let align = 1u64 << (class.ilog2() - 1);
        assert_eq!(
            header % align,
            0,
            "class {class} header {header:#x} not {align}-aligned"
        );
        a.free(s);
    }

    // Oversized blocks (> EXP_CEIL) align to EXP_CEIL.
    for &sz in &[70_000u64, 100_000, 200_000] {
        let s = a.alloc(sz);
        assert!(!s.is_null());
        assert!(s.len + 16 > AlignConfig::EXP_CEIL, "not oversized");
        assert_eq!(
            (s.ptr - 16) % AlignConfig::EXP_CEIL,
            0,
            "oversized header {:#x} not EXP_CEIL-aligned",
            s.ptr - 16
        );
        a.free(s);
    }
}

#[test]
fn fixed_range_carve_is_non_overlapping() {
    let a = CadAlloc::<CarveConfig>::new();
    a.init().expect("init");

    // A max-class (4096) block shrunk to a small class leaves a ~3968-byte
    // fixed-range excess, which is carved into class-exact free blocks.
    let big = a.alloc(4080);
    assert!(!big.is_null());
    let small = a.realloc(big, 100);
    assert_eq!(small.ptr, big.ptr, "shrink stays in place");
    assert!(small.len >= 100 && small.len < 4080, "block shrank");

    // Allocate a spread of sizes: some reuse the carved pieces, some bump. None
    // may overlap the retained block or each other, and all must be MIN_ALIGN
    // aligned and inside the heap.
    let heap_lo = carve_heap_base();
    let heap_hi = heap_lo + HEAP_N as u64;
    let mut live = [Slice::NULL; 25];
    live[0] = small;
    for k in 1..live.len() {
        let sz = 16 + (k as u64 % 6) * 130;
        let s = a.alloc(sz);
        assert!(!s.is_null(), "alloc {k} failed");
        assert_eq!(s.ptr % 16, 0, "payload misaligned");
        assert!(s.ptr >= heap_lo && s.ptr + s.len <= heap_hi, "out of heap");
        for prev in &live[..k] {
            assert!(!overlap(s, *prev), "allocation {k} overlaps an earlier one");
        }
        live[k] = s;
    }
    for s in live {
        a.free(s);
    }
}

// --- GlobalAlloc adapter (feature `alloc`, over its own static heap) --------

#[cfg(feature = "alloc")]
mod global_alloc {
    use super::*;
    use core::alloc::{GlobalAlloc, Layout};

    const GA_HEAP_N: usize = 1 << 16;

    #[repr(C, align(64))]
    struct GaHeap([u8; GA_HEAP_N]);

    static mut GA_HEAP: GaHeap = GaHeap([0; GA_HEAP_N]);

    struct GaConfig;

    impl Config for GaConfig {
        type Atomics = CoreAtomics;
        const MIN_ALIGN: u64 = 16;
        const LNR_FLOOR: u64 = 32;
        const EXP_FLOOR: u64 = 256;
        const EXP_CEIL: u64 = 4096;
        fn heap_base() -> u64 {
            &raw const GA_HEAP as u64
        }
        fn heap_size() -> u64 {
            GA_HEAP_N as u64
        }
    }

    #[test]
    fn global_alloc_impl_round_trips() {
        let a = CadAlloc::<GaConfig>::new();
        a.init().expect("init");

        let layout = Layout::from_size_align(100, 8).unwrap();

        // Alignment beyond MIN_ALIGN is unsatisfiable and refused (null).
        let over = Layout::from_size_align(64, 256).unwrap();
        assert!(unsafe { GlobalAlloc::alloc(&a, over) }.is_null());

        // alloc_zeroed (the default impl) must zero even a *reused, dirty* block:
        // dirty one, free it, then the zeroed alloc reclaims it (LIFO) cleared.
        let d = unsafe { GlobalAlloc::alloc(&a, layout) };
        assert!(!d.is_null());
        unsafe { core::ptr::write_bytes(d, 0xFF, 100) };
        unsafe { GlobalAlloc::dealloc(&a, d, layout) };
        let z = unsafe { GlobalAlloc::alloc_zeroed(&a, layout) };
        assert_eq!(z, d, "freed block of the same class should be reused");
        for i in 0..100 {
            assert_eq!(unsafe { *z.add(i) }, 0, "alloc_zeroed left byte {i} dirty");
        }
        unsafe { GlobalAlloc::dealloc(&a, z, layout) };

        // A normal allocation is non-null, MIN_ALIGN-aligned, and writable.
        let p = unsafe { GlobalAlloc::alloc(&a, layout) };
        assert!(!p.is_null());
        assert_eq!(p as u64 % 16, 0, "not MIN_ALIGN-aligned");
        unsafe { core::ptr::write_bytes(p, 0xAB, 100) };

        // realloc grows (relocating past EXP_CEIL), preserving the old bytes.
        let p2 = unsafe { GlobalAlloc::realloc(&a, p, layout, 5000) };
        assert!(!p2.is_null());
        for i in 0..100 {
            assert_eq!(unsafe { *p2.add(i) }, 0xAB, "realloc lost byte {i}");
        }

        unsafe { GlobalAlloc::dealloc(&a, p2, Layout::from_size_align(5000, 8).unwrap()) };
        assert_eq!(a.verify(), Ok(()));
    }
}

// --- multithreaded stress test (real threads over a shared heap) -----------
//
// Serial unit tests can't observe a lock-free allocator's concurrency bugs. This
// hammers one heap from many threads doing alloc/free/realloc churn and detects
// corruption by fingerprinting: every live block is filled with a unique 64-bit
// token, verified before it is freed or reallocated. Two overlapping live blocks
// (a bad carve, a lost/duplicated free, an ABA slip) would clobber each other's
// tokens and trip an assertion. Complements the `loom` model check below, which
// is exhaustive but only over the isolated Treiber stack.
#[cfg(not(loom))]
mod stress {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::vec::Vec;

    const STRESS_HEAP_N: usize = 16 << 20; // 16 MiB

    #[repr(C, align(64))]
    struct StressHeap([u8; STRESS_HEAP_N]);

    static mut STRESS_HEAP: StressHeap = StressHeap([0; STRESS_HEAP_N]);

    fn stress_base() -> u64 {
        &raw const STRESS_HEAP as u64
    }

    struct StressConfig;

    impl Config for StressConfig {
        type Atomics = CoreAtomics;
        const MIN_ALIGN: u64 = 16;
        const LNR_FLOOR: u64 = 32;
        const EXP_FLOOR: u64 = 256;
        const EXP_CEIL: u64 = 8192; // low enough that the churn hits oversized too
        fn heap_base() -> u64 {
            stress_base()
        }
        fn heap_size() -> u64 {
            STRESS_HEAP_N as u64
        }
    }

    /// Hands out globally-unique, nonzero fill tokens.
    static NEXT_TOKEN: AtomicU64 = AtomicU64::new(1);

    /// Payloads are class-exact multiples of `MIN_ALIGN`, so `len` is a whole
    /// number of `u64` words. Fill every word with `token`.
    fn fill(s: Slice, token: u64) {
        let words = (s.len / 8) as usize;
        let p = s.ptr as *mut u64;
        for i in 0..words {
            // SAFETY: `s` is a live payload owned by this thread; no other thread
            // can touch these bytes until we free it. Volatile to defeat elision.
            unsafe { p.add(i).write_volatile(token) };
        }
    }

    /// Verifies the first `words` u64s of `s` all equal `token`.
    fn check_prefix(s: Slice, token: u64, words: usize) {
        let p = s.ptr as *const u64;
        for i in 0..words {
            // SAFETY: as `fill`; reading our own live payload.
            let v = unsafe { p.add(i).read_volatile() };
            assert_eq!(v, token, "corruption at word {i} of {:#x}", s.ptr);
        }
    }

    fn check(s: Slice, token: u64) {
        check_prefix(s, token, (s.len / 8) as usize);
    }

    #[test]
    fn concurrent_alloc_free_realloc_no_corruption() {
        let a = CadAlloc::<StressConfig>::new();
        a.init().expect("init");
        assert_eq!(a.verify(), Ok(()));

        const THREADS: usize = 8;
        const OPS: usize = 6000;

        let mut handles = Vec::new();
        for t in 0..THREADS {
            handles.push(std::thread::spawn(move || {
                // A fresh handle; state lives in the shared heap, not the handle.
                let a = CadAlloc::<StressConfig>::new();

                // Per-thread xorshift64 — no external RNG dependency.
                let mut rng =
                    0x9E37_79B9_7F4A_7C15u64 ^ (t as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03);
                let mut next = move || {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    rng
                };

                let mut live: Vec<(Slice, u64)> = Vec::new();
                for _ in 0..OPS {
                    let roll = next() % 100;
                    if roll < 45 || live.is_empty() {
                        // Allocate; sizes span linear, exponential, and oversized.
                        let size = 8 + next() % 12_000;
                        let s = a.alloc(size);
                        if !s.is_null() {
                            assert_eq!(s.ptr % 16, 0, "payload misaligned");
                            assert!(s.len + 16 >= size + 16 || s.len >= size);
                            let token = NEXT_TOKEN.fetch_add(1, Ordering::Relaxed);
                            fill(s, token);
                            live.push((s, token));
                        }
                    } else if roll < 80 {
                        // Free a random live block, verifying it first.
                        let i = (next() as usize) % live.len();
                        let (s, token) = live.swap_remove(i);
                        check(s, token);
                        a.free(s);
                    } else {
                        // Reallocate a random live block; the retained prefix must
                        // survive the (possibly relocating) resize.
                        let i = (next() as usize) % live.len();
                        let (s, token) = live[i];
                        check(s, token);
                        let new_size = 8 + next() % (s.len.max(16));
                        let g = a.realloc(s, new_size);
                        if g.is_null() {
                            // realloc failed: the original is left valid.
                            check(s, token);
                        } else {
                            let kept = core::cmp::min(s.len, g.len);
                            check_prefix(g, token, (kept / 8) as usize);
                            let token = NEXT_TOKEN.fetch_add(1, Ordering::Relaxed);
                            fill(g, token);
                            live[i] = (g, token);
                        }
                    }
                }

                // Return everything still held so the heap ends drained.
                for (s, token) in live {
                    check(s, token);
                    a.free(s);
                }
            }));
        }

        for h in handles {
            h.join()
                .expect("a stress thread panicked (corruption detected)");
        }

        // Metadata survived the churn, and the heap can still serve requests.
        assert_eq!(a.verify(), Ok(()));
        let s = a.alloc(64);
        assert!(!s.is_null(), "heap unusable after stress");
        a.free(s);
    }
}

// --- loom model check of the lock-free Treiber-stack bins -------------------
//
// Run with: `RUSTFLAGS="--cfg loom" cargo test --test ... treiber` (loom is a
// dev-dependency only under `--cfg loom`). Loom exhaustively explores thread
// interleavings and memory orderings for a *small* scenario, which is exactly
// the right tool for the bins' push/pop CAS protocol and its ABA tag. It cannot
// model the whole allocator: a real heap is far too many atomic locations for
// the state space, and the allocator addresses memory by raw `u64`, which loom's
// instrumented atomics cannot back directly. So we bridge the address-based
// `Atomics` trait to a tiny array of loom atomics and drive the real
// `push_fixed` / `pop_fixed` over it.
#[cfg(loom)]
mod loom_stack {
    use crate::allocator::CadAlloc;
    use crate::atomic::Atomics;
    use crate::config::Config;
    use loom::sync::Arc;
    use loom::sync::atomic::{AtomicU64, Ordering::SeqCst};
    use std::cell::Cell;
    use std::vec::Vec;

    /// Synthetic heap base; bin 0's head lives at `BASE + META_BYTES` (`+16`).
    const BASE: u64 = 0x1_0000;
    /// A handful of `u64` cells — head plus a couple of block `next` slots.
    const CELLS: usize = 32;

    /// A tiny "heap" of loom atomics, addressed by synthetic `u64` addresses.
    struct LoomHeap {
        words: Vec<AtomicU64>,
    }

    impl LoomHeap {
        fn new() -> Self {
            let mut words = Vec::with_capacity(CELLS);
            for _ in 0..CELLS {
                words.push(AtomicU64::new(0));
            }
            Self { words }
        }

        fn at(&self, addr: u64) -> &AtomicU64 {
            &self.words[((addr - BASE) / 8) as usize]
        }
    }

    std::thread_local! {
        /// Per-thread pointer to the model's shared heap, installed at entry.
        static HEAP: Cell<*const LoomHeap> = const { Cell::new(core::ptr::null()) };
    }

    fn set_heap(h: &Arc<LoomHeap>) {
        HEAP.with(|c| c.set(&**h as *const LoomHeap));
    }

    fn with_heap<R>(f: impl FnOnce(&LoomHeap) -> R) -> R {
        HEAP.with(|c| {
            // SAFETY: `set_heap` installed a pointer to an `Arc<LoomHeap>` that is
            // held alive (by this thread's closure, or by `main`) for the whole
            // model iteration in which any atomic op runs.
            let h = unsafe { &*c.get() };
            f(h)
        })
    }

    /// An [`Atomics`] backend over loom atomics, so loom can instrument every
    /// access the allocator makes to the bins.
    struct LoomAtomics;

    impl Atomics for LoomAtomics {
        unsafe fn atomic_load(addr: u64) -> u64 {
            with_heap(|h| h.at(addr).load(SeqCst))
        }
        unsafe fn atomic_store(addr: u64, val: u64) {
            with_heap(|h| h.at(addr).store(val, SeqCst));
        }
        unsafe fn atomic_cas(addr: u64, current: u64, new: u64) -> u64 {
            with_heap(
                |h| match h.at(addr).compare_exchange(current, new, SeqCst, SeqCst) {
                    Ok(p) | Err(p) => p,
                },
            )
        }
        unsafe fn atomic_cas_weak(addr: u64, current: u64, new: u64) -> (u64, bool) {
            with_heap(|h| {
                match h
                    .at(addr)
                    .compare_exchange_weak(current, new, SeqCst, SeqCst)
                {
                    Ok(p) => (p, true),
                    Err(p) => (p, false),
                }
            })
        }
    }

    struct LoomConfig;

    impl Config for LoomConfig {
        type Atomics = LoomAtomics;
        const MIN_ALIGN: u64 = 16;
        const LNR_FLOOR: u64 = 32;
        const EXP_FLOOR: u64 = 256;
        const EXP_CEIL: u64 = 65536;
        const HEAP_BASE: u64 = BASE;
        const HEAP_SIZE: u64 = (CELLS as u64) * 8;
    }

    #[test]
    fn treiber_stack_push_pop_conserves_blocks() {
        loom::model(|| {
            let heap = Arc::new(LoomHeap::new());
            set_heap(&heap);

            // Two distinct, MIN_ALIGN-aligned block headers, each with room for a
            // `next` word at `+8`, clear of bin 0's head at BASE+16.
            const A: u64 = BASE + 64;
            const B: u64 = BASE + 96;

            let h1 = heap.clone();
            let t1 = loom::thread::spawn(move || {
                set_heap(&h1);
                let a = CadAlloc::<LoomConfig>::new();
                a.push_fixed(0, A);
                a.pop_fixed(0)
            });
            let h2 = heap.clone();
            let t2 = loom::thread::spawn(move || {
                set_heap(&h2);
                let a = CadAlloc::<LoomConfig>::new();
                a.push_fixed(0, B);
                a.pop_fixed(0)
            });

            let r1 = t1.join().unwrap();
            let r2 = t2.join().unwrap();

            // Two pushes and two pops, so — for any interleaving — both pops must
            // succeed, together remove exactly {A, B}, and leave the stack empty.
            // A lost update, a duplicated pop, or an ABA slip breaks one of these.
            let a = CadAlloc::<LoomConfig>::new();
            assert_eq!(a.pop_fixed(0), 0, "stack not empty after both pops");
            assert!(r1 != 0 && r2 != 0, "a pop lost its block: {r1:#x}, {r2:#x}");
            assert_ne!(r1, r2, "the same block was popped twice");
            let mut got = [r1, r2];
            got.sort_unstable();
            assert_eq!(got, [A, B], "popped blocks are not {{A, B}}");
        });
    }
}
