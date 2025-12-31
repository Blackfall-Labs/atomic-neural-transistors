//! DiffTRM - Difference embedding computation

use candle_core::{Result, Tensor, D};
use candle_nn::VarBuilder;

use crate::config::AtomicConfig;
use crate::core::AtomicTRM;

/// Computes a learned difference embedding between two inputs
///
/// Unlike simple subtraction, learns what aspects of difference matter
pub struct DiffTRM(AtomicTRM);

impl DiffTRM {
    /// Create a new DiffTRM for given embedding dimension
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self(AtomicTRM::new(
            &AtomicConfig::small(dim * 2, dim),
            vb,
        )?))
    }

    /// Compute difference embedding between a and b
    pub fn diff(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let combined = Tensor::cat(&[a, b], D::Minus1)?;
        self.0.forward(&combined)
    }

    /// Get parameter count
    pub fn param_count(&self) -> usize {
        self.0.param_count()
    }

    /// Alias for param_count (for compatibility)
    pub fn params(&self) -> usize {
        self.param_count()
    }
}
