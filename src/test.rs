//! Unit tests for the configuration surface and allocator core.

use crate::allocator::{bin_class, bin_index};
use crate::*;

/// A representative configuration used to exercise the config surface.
struct TestConfig;

impl Config for TestConfig {
    type Atomics = CoreAtomics;

    const MIN_ALIGN: u64 = 16;
    const LNR_FLOOR: u64 = 16;
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
    let s = Slice::new(0x1000, 0x100);
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
    // MIN_ALIGN=16, LNR_FLOOR=16, EXP_FLOOR=256, EXP_CEIL=65536:
    // linear = (256-16)/16 = 15; exp = 2*(16-8)+1 = 17; +1 oversized = 33.
    assert_eq!(TestConfig::NUM_LINEAR, 15);
    assert_eq!(TestConfig::NUM_EXP, 17);
    assert_eq!(TestConfig::NUM_BINS, 33);
    assert_eq!(CadAlloc::<TestConfig>::bin_count(), 33);
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
    // First/last known classes.
    assert_eq!(bin_class::<TestConfig>(0), 16);
    assert_eq!(bin_class::<TestConfig>(14), 240);
    assert_eq!(bin_class::<TestConfig>(15), 256);
    assert_eq!(bin_class::<TestConfig>(16), 384);
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
    const LNR_FLOOR: u64 = 16;
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
    const LNR_FLOOR: u64 = 16;
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
    let s1 = a.alloc(100, 16);
    assert!(!s1.is_null());
    assert_eq!(s1.ptr % 16, 0, "payload must be MIN_ALIGN-aligned");
    assert!(s1.len >= 100, "capacity {} < requested 100", s1.len);

    // A second allocation must not overlap the first.
    let s2 = a.alloc(100, 16);
    assert!(!s2.is_null());
    assert_ne!(s1.ptr, s2.ptr);
    assert!(
        s2.ptr >= s1.ptr + s1.len || s1.ptr >= s2.ptr + s2.len,
        "blocks overlap"
    );

    // Freeing then re-allocating the same class reuses the block (LIFO).
    let p1 = s1.ptr;
    a.free(s1);
    let s3 = a.alloc(100, 16);
    assert_eq!(s3.ptr, p1, "freed block should be reused");

    a.free(s2);
    a.free(s3);

    // Oversized path: > EXP_CEIL (4096).
    let big = a.alloc(5000, 16);
    assert!(!big.is_null());
    assert!(big.len >= 5000);
    let bp = big.ptr;
    a.free(big);
    // First-fit on the oversized list should recover the same block.
    let big2 = a.alloc(5000, 16);
    assert_eq!(big2.ptr, bp, "oversized block should be reused");
    a.free(big2);

    // Alignment beyond MIN_ALIGN is not yet supported.
    assert!(a.alloc(16, 32).is_null());
}

#[test]
fn init_rejects_bad_heap() {
    struct Misaligned;
    impl Config for Misaligned {
        type Atomics = CoreAtomics;
        const MIN_ALIGN: u64 = 16;
        const LNR_FLOOR: u64 = 16;
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
        const LNR_FLOOR: u64 = 16;
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
    const LNR_FLOOR: u64 = 16;
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

    let s = cadtest_alloc(100, 16);
    assert!(!s.is_null());
    assert!(s.len >= 100);
    cadtest_free(s);

    // Reuse after free (LIFO), exercising both alloc and free shims.
    let s2 = cadtest_alloc(100, 16);
    assert_eq!(s2.ptr, s.ptr);

    // realloc shim: grow, then free through the shim.
    let s3 = cadtest_realloc(s2, 2000, 16);
    assert!(!s3.is_null());
    assert!(s3.len >= 2000);
    cadtest_free(s3);

    // realloc(NULL, ...) behaves like alloc.
    let s4 = cadtest_realloc(Slice::NULL, 64, 16);
    assert!(!s4.is_null());
    cadtest_free(s4);

    // Alignment beyond MIN_ALIGN returns the null slice.
    assert!(cadtest_alloc(16, 32).is_null());
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
    const LNR_FLOOR: u64 = 16;
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
    let n = a.realloc(Slice::NULL, 50, 16);
    assert!(!n.is_null());
    a.free(n);

    // Grow relocates and preserves the old contents.
    let s = a.alloc(64, 16);
    assert!(!s.is_null());
    // SAFETY: `s.ptr` is a live 64-byte payload in our static heap.
    unsafe {
        for i in 0..64u64 {
            (s.ptr as *mut u8).add(i as usize).write(i as u8);
        }
    }
    let g = a.realloc(s, 2000, 16);
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
    let f = a.alloc(2000, 16);
    let cap = f.len;
    let t = a.realloc(f, 100, 16);
    assert_eq!(t.ptr, f.ptr, "fixed shrink should not move");
    assert_eq!(t.len, cap, "fixed shrink keeps the whole block");
    a.free(t);

    // Oversized shrink splits off the tail, which becomes a reusable free block.
    let big = a.alloc(12000, 16);
    assert!(big.len >= 12000);
    let sh = a.realloc(big, 5000, 16);
    assert_eq!(sh.ptr, big.ptr, "oversized shrink stays in place");
    assert!(sh.len >= 5000 && sh.len < big.len, "block actually shrank");
    // The tail sits immediately after the shrunk front block and is recovered
    // by the next oversized allocation (first-fit).
    let reuse = a.alloc(6000, 16);
    assert_eq!(reuse.ptr, big.ptr + sh.len + 16, "tail should be reused");
    a.free(sh);
    a.free(reuse);
}
