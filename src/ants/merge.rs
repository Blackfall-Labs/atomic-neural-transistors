//! MergeTRM - Combine multiple signals into one

use candle_core::{Result, Tensor, D};
use candle_nn::VarBuilder;

use crate::config::AtomicConfig;
use crate::core::AtomicTRM;

/// Merges multiple input signals into a single embedding
///
/// Learns optimal combination rather than simple average
pub struct MergeTRM(AtomicTRM);

impl MergeTRM {
    /// Create a new MergeTRM
    ///
    /// - `dim`: embedding dimension for each input
    /// - `n`: number of inputs to merge
    pub fn new(dim: usize, n: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self(AtomicTRM::new(
            &AtomicConfig::small(dim * n, dim),
            vb,
        )?))
    }

    /// Merge multiple signals into one
    pub fn merge(&self, signals: &[Tensor]) -> Result<Tensor> {
        let combined = Tensor::cat(signals, D::Minus1)?;
        self.0.forward(&combined)
    }

    /// Get parameter count
    pub fn param_count(&self) -> usize {
        self.0.param_count()
    }
}
