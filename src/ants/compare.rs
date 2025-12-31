//! CompareTRM - Binary similarity comparison

use candle_core::{Result, Tensor, D};
use candle_nn::VarBuilder;

use crate::config::AtomicConfig;
use crate::core::AtomicTRM;

/// Compares two embeddings and returns similarity score
///
/// Output: [0, 1] probability that inputs are equal/similar
pub struct CompareTRM(AtomicTRM);

impl CompareTRM {
    /// Create a new CompareTRM for given embedding dimension
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self(AtomicTRM::new(
            &AtomicConfig::tiny(dim * 2, 1),
            vb,
        )?))
    }

    /// Compare two tensors, returns similarity in [0, 1]
    pub fn compare(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let combined = Tensor::cat(&[a, b], D::Minus1)?;
        candle_nn::ops::sigmoid(&self.0.forward(&combined)?)
    }

    /// Get parameter count
    pub fn param_count(&self) -> usize {
        self.0.param_count()
    }
}
