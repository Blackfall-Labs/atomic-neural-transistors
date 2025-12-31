//! Traits for composition algebra

/// Trait for a binary equality checker
///
/// Implement this to create composed operations like `contains` and `has_duplicate`
pub trait EqualityChecker {
    /// Check if two values are equal, returns probability [0, 1]
    fn check_equal(&self, a: u32, b: u32) -> f32;
}

// Allow references to implement the trait
impl<E: EqualityChecker> EqualityChecker for &E {
    fn check_equal(&self, a: u32, b: u32) -> f32 {
        (*self).check_equal(a, b)
    }
}

/// Perfect equality checker (ground truth, no neural network)
///
/// Useful for testing and as a baseline
#[derive(Clone, Copy, Debug, Default)]
pub struct PerfectEquality;

impl EqualityChecker for PerfectEquality {
    fn check_equal(&self, a: u32, b: u32) -> f32 {
        if a == b { 1.0 } else { 0.0 }
    }
}
