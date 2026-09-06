//! Unit tests for the configuration surface.

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
fn version_is_reported() {
    assert_eq!(version(), env!("CARGO_PKG_VERSION"));
}

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
    }
}
