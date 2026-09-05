//! `cadalloc` — a minimal, configurable, embeddable allocator.
//!
//! `cadalloc` is a general-purpose allocator built around segregated free
//! lists. It targets multithreaded environments but deliberately does *not*
//! use thread-local free lists: the design goal is a small, predictable core
//! that is easy to embed and reason about, rather than the last few percent of
//! contended-allocation throughput.
//!
//! The public API is still being built out. See `PLANNED.md` for the design
//! surface and `README.md` for an overview.

/// Returns the version of this crate, as recorded in `Cargo.toml`.
///
/// # Examples
///
/// ```
/// assert!(!cadalloc::version().is_empty());
/// ```
#[must_use]
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_reported() {
        assert_eq!(version(), env!("CARGO_PKG_VERSION"));
    }
}
