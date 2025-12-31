//! Configuration for AtomicTRM - the fundamental ANT

/// Configuration for an Atomic Neural Transistor
///
/// AtomicTRM is the smallest meaningful unit of neural computation.
/// Most configurations result in < 5K parameters.
#[derive(Clone, Debug)]
pub struct AtomicConfig {
    /// Input dimension
    pub input_dim: usize,
    /// Hidden dimension (typically 16-32 for tiny, 24-48 for small)
    pub hidden_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Number of recurrent iterations (2-3 typical)
    pub iterations: usize,
}

impl AtomicConfig {
    /// Tiny configuration (~1-2K parameters)
    ///
    /// Best for: binary comparison, equality checks
    pub fn tiny(input: usize, output: usize) -> Self {
        Self {
            input_dim: input,
            hidden_dim: 16,
            output_dim: output,
            iterations: 2,
        }
    }

    /// Small configuration (~3-5K parameters)
    ///
    /// Best for: pattern matching, simple classification
    pub fn small(input: usize, output: usize) -> Self {
        Self {
            input_dim: input,
            hidden_dim: 24,
            output_dim: output,
            iterations: 2,
        }
    }

    /// Estimate parameter count
    pub fn param_count(&self) -> usize {
        let (i, h, o) = (self.input_dim, self.hidden_dim, self.output_dim);
        // w_in + w_rec + w_gate + w_out (with biases)
        (i * h + h) + (h * h + h) + (2 * h * h + h) + (h * o + o)
    }
}
