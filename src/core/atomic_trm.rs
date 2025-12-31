//! AtomicTRM - The fundamental Atomic Neural Transistor
//!
//! This is the smallest meaningful unit of neural computation.
//! Like a transistor in silicon, it does one thing well and composes
//! with other ANTs to form complex behaviors.

use candle_core::{Module, Result, Tensor, D};
use candle_nn::{linear, Linear, VarBuilder};

use crate::config::AtomicConfig;

/// Atomic Neural Transistor - the fundamental building block
///
/// Architecture:
/// ```text
/// input → w_in → GELU → [recurrent iterations] → w_out → output
///                              ↓
///                     h' = gate * update + (1-gate) * h
/// ```
///
/// Typical size: 1-5K parameters
/// Typical inference: < 1ms
pub struct AtomicTRM {
    w_in: Linear,
    w_rec: Linear,
    w_gate: Linear,
    w_out: Linear,
    cfg: AtomicConfig,
}

impl AtomicTRM {
    /// Create a new AtomicTRM
    pub fn new(cfg: &AtomicConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_dim;
        Ok(Self {
            w_in: linear(cfg.input_dim, h, vb.pp("i"))?,
            w_rec: linear(h, h, vb.pp("r"))?,
            w_gate: linear(h * 2, h, vb.pp("g"))?,
            w_out: linear(h, cfg.output_dim, vb.pp("o"))?,
            cfg: cfg.clone(),
        })
    }

    /// Forward pass through the ANT
    ///
    /// Input shape: (batch, input_dim) or (batch, seq, input_dim)
    /// Output shape: (batch, output_dim) or (batch, seq, output_dim)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.w_in.forward(x)?.gelu_erf()?;

        for _ in 0..self.cfg.iterations {
            let update = self.w_rec.forward(&h)?.gelu_erf()?;
            let cat = Tensor::cat(&[&h, &update], D::Minus1)?;
            let gate = candle_nn::ops::sigmoid(&self.w_gate.forward(&cat)?)?;
            let inv_gate = Tensor::ones_like(&gate)?.sub(&gate)?;
            let gu = gate.mul(&update)?;
            let inv_gh = inv_gate.mul(&h)?;
            h = gu.add(&inv_gh)?;
        }

        self.w_out.forward(&h)
    }

    /// Get parameter count
    pub fn param_count(&self) -> usize {
        self.cfg.param_count()
    }

    /// Alias for param_count (for compatibility)
    pub fn params(&self) -> usize {
        self.param_count()
    }

    /// Get the configuration
    pub fn config(&self) -> &AtomicConfig {
        &self.cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn test_atomic_trm_is_tiny() -> Result<()> {
        let device = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);

        let trm = AtomicTRM::new(&AtomicConfig::tiny(32, 32), vb)?;
        println!("AtomicTRM (tiny): {} params", trm.param_count());
        assert!(trm.param_count() < 5000, "Should be under 5K params");
        Ok(())
    }

    #[test]
    fn test_forward_pass() -> Result<()> {
        let device = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);

        let trm = AtomicTRM::new(&AtomicConfig::tiny(32, 1), vb)?;
        let input = Tensor::randn(0f32, 1f32, (4, 32), &device)?;
        let output = trm.forward(&input)?;

        assert_eq!(output.dims(), &[4, 1]);
        Ok(())
    }
}
