//! Configuring a `cadalloc` allocator. Run with `cargo run --example basic`.
//!
//! The whole configuration step is implementing [`Config`] once: name an
//! atomics backend and the four power-of-two size-class constants.

use cadalloc::{CadAlloc, Config, CoreAtomics};

/// A configuration for a 1 MiB statically-known heap.
struct ExampleConfig;

impl Config for ExampleConfig {
    // Coordinate the free lists with the standard 64-bit atomics.
    type Atomics = CoreAtomics;

    // Size classes (all powers of two, MIN_ALIGN <= LNR <= EXP_FLOOR <= CEIL):
    const MIN_ALIGN: u64 = 16; // every allocation is 16-byte aligned
    const LNR_FLOOR: u64 = 16; // smallest class, linear step of MIN_ALIGN
    const EXP_FLOOR: u64 = 256; // linear below here, exponential above
    const EXP_CEIL: u64 = 65536; // largest class; larger requests go oversized

    // A compile-time heap region (leave at 0 and override `heap_base`/
    // `heap_size` when the region is only known at runtime).
    const HEAP_BASE: u64 = 0x2000_0000;
    const HEAP_SIZE: u64 = 1 << 20; // 1 MiB
}

// Reject an invalid configuration at compile time.
const _: () = cadalloc::assert_config_valid::<ExampleConfig>();

fn main() {
    println!("bins: {}", CadAlloc::<ExampleConfig>::bin_count());
    println!(
        "heap: base = {:#x}, size = {} bytes",
        cadalloc::const_heap_base::<ExampleConfig>(),
        cadalloc::const_heap_size::<ExampleConfig>(),
    );
    println!(
        "classes: MIN_ALIGN={}, LNR_FLOOR={}, EXP_FLOOR={}, EXP_CEIL={}",
        ExampleConfig::MIN_ALIGN,
        ExampleConfig::LNR_FLOOR,
        ExampleConfig::EXP_FLOOR,
        ExampleConfig::EXP_CEIL,
    );
}
