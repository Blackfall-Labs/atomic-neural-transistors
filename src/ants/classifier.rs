//! ClassifierTRM - Multi-class classification

use candle_core::{Module, Result, Tensor, D};
use candle_nn::{linear, Linear, VarBuilder};

use crate::config::AtomicConfig;
use crate::core::AtomicTRM;

/// Multi-class classifier ANT
///
/// Typical accuracy: 98%+ on targeted classification tasks
pub struct ClassifierTRM {
    encoder: AtomicTRM,
    head: Linear,
    n_classes: usize,
}

impl ClassifierTRM {
    /// Create a new ClassifierTRM
    ///
    /// - `input_dim`: input embedding dimension
    /// - `hidden`: hidden dimension (24-48 typical)
    /// - `n_classes`: number of output classes
    pub fn new(input_dim: usize, hidden: usize, n_classes: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = AtomicConfig {
            input_dim,
            hidden_dim: hidden,
            output_dim: hidden,
            iterations: 3,
        };
        Ok(Self {
            encoder: AtomicTRM::new(&cfg, vb.pp("e"))?,
            head: linear(hidden, n_classes, vb.pp("h"))?,
            n_classes,
        })
    }

    /// Softmax classification probabilities
    pub fn classify(&self, x: &Tensor) -> Result<Tensor> {
        let enc = self.encoder.forward(x)?;
        let pooled = if enc.dims().len() == 3 {
            enc.mean(1)?
        } else {
            enc
        };
        candle_nn::ops::softmax(&self.head.forward(&pooled)?, D::Minus1)
    }

    /// Sigmoid activations (for multi-label or gating)
    pub fn activations(&self, x: &Tensor) -> Result<Tensor> {
        let enc = self.encoder.forward(x)?;
        let pooled = if enc.dims().len() == 3 {
            enc.mean(1)?
        } else {
            enc
        };
        candle_nn::ops::sigmoid(&self.head.forward(&pooled)?)
    }

    /// Get predicted class index
    pub fn predict(&self, x: &Tensor) -> Result<Tensor> {
        let probs = self.classify(x)?;
        probs.argmax(D::Minus1)
    }

    /// Get parameter count
    pub fn param_count(&self) -> usize {
        self.encoder.param_count() + self.head.weight().elem_count() + self.n_classes
    }

    /// Alias for param_count (for compatibility)
    pub fn params(&self) -> usize {
        self.param_count()
    }
}
