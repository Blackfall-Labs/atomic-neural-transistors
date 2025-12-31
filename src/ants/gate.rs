//! GateTRM - Attention-based signal routing

use candle_core::{Module, Result, Tensor, D};
use candle_nn::{linear, Linear, VarBuilder};

use crate::config::AtomicConfig;
use crate::core::AtomicTRM;

/// Gates a signal based on a control input
///
/// Used for conditional computation and attention-like routing
pub struct GateTRM {
    trm: AtomicTRM,
    gate: Linear,
}

impl GateTRM {
    /// Create a new GateTRM for given embedding dimension
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            trm: AtomicTRM::new(&AtomicConfig::tiny(dim * 2, dim), vb.pp("t"))?,
            gate: linear(dim, dim, vb.pp("g"))?,
        })
    }

    /// Gate a signal based on control
    ///
    /// - `signal`: the signal to potentially pass through
    /// - `control`: determines how much of signal passes
    pub fn gate(&self, signal: &Tensor, control: &Tensor) -> Result<Tensor> {
        let g = candle_nn::ops::sigmoid(&self.gate.forward(control)?)?;
        let combined = Tensor::cat(&[signal, control], D::Minus1)?;
        self.trm.forward(&combined)?.mul(&g)
    }

    /// Get parameter count
    pub fn param_count(&self) -> usize {
        let gate_params = self.gate.weight().elem_count();
        self.trm.param_count() + gate_params
    }
}
