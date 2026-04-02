//! Traits for composition algebra

use ternary_signal::Signal;

/// Trait for a binary equality checker.
///
/// Implement this to create composed operations like `contains` and `has_duplicate`.
/// Returns a signed similarity score: positive = similar, negative = different.
pub trait EqualityChecker {
    /// Check if two signals are similar.
    /// Returns positive for similar, negative for different.
    fn check_equal(&self, a: &Signal, b: &Signal) -> i32;
}

// Allow references to implement the trait
impl<E: EqualityChecker> EqualityChecker for &E {
    fn check_equal(&self, a: &Signal, b: &Signal) -> i32 {
        (*self).check_equal(a, b)
    }
}

/// Perfect equality checker (ground truth, no ANT).
///
/// Useful for testing and as a baseline.
#[derive(Clone, Copy, Debug, Default)]
pub struct PerfectEquality;

impl EqualityChecker for PerfectEquality {
    fn check_equal(&self, a: &Signal, b: &Signal) -> i32 {
        if a.current() == b.current() { 127 } else { -127 }
    }
}
