//! Synaptic strength matrix storage using Signal.

use ternary_signal::Signal;

/// A matrix of Signal synaptic strengths with shape metadata.
#[derive(Clone, Debug)]
pub struct WeightMatrix {
    pub data: Vec<Signal>,
    pub rows: usize,
    pub cols: usize,
}

impl WeightMatrix {
    /// Create a new matrix filled with zero signals.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![Signal::ZERO; rows * cols],
            rows,
            cols,
        }
    }

    /// Create a frozen random projection matrix.
    ///
    /// Per MASTERY.md: frozen hidden regions use `s = p × m × k` where
    /// p = ±1 (random), m = 20-40 (random), k = random from K_LEVELS.
    pub fn random_frozen(rows: usize, cols: usize, seed: u64) -> Self {
        const K_LEVELS: [u8; 7] = [1, 4, 16, 32, 64, 128, 255];
        let mut state = if seed == 0 { 0xDEAD_BEEF_CAFE_1234 } else { seed };
        let mut data = Vec::with_capacity(rows * cols);
        for _ in 0..rows * cols {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let pol: i8 = if state & 1 == 0 { 1 } else { -1 };
            let mag = 20 + ((state >> 8) as u8 % 21);
            let k = K_LEVELS[((state >> 16) as usize) % K_LEVELS.len()];
            data.push(Signal::new_raw(pol, mag, k));
        }
        Self { data, rows, cols }
    }

    /// Create from raw Signal data with shape validation.
    pub fn from_data(data: Vec<Signal>, rows: usize, cols: usize) -> Option<Self> {
        if data.len() != rows * cols {
            return None;
        }
        Some(Self { data, rows, cols })
    }

    /// Get signal at (row, col).
    pub fn get(&self, row: usize, col: usize) -> Signal {
        self.data[row * self.cols + col]
    }

    /// Set signal at (row, col).
    pub fn set(&mut self, row: usize, col: usize, value: Signal) {
        self.data[row * self.cols + col] = value;
    }

    /// Ternary matrix-vector multiply: output[i] = Σ_j(strength[i,j].current() * input[j].current())
    pub fn matmul(&self, input: &[Signal]) -> Vec<Signal> {
        assert_eq!(input.len(), self.cols, "matmul input dimension mismatch");
        let mut output = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            let mut acc: i64 = 0;
            for j in 0..self.cols {
                let w = self.data[i * self.cols + j].current() as i64;
                let x = input[j].current() as i64;
                acc += w * x;
            }
            let clamped = acc.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
            output.push(Signal::from_current(clamped));
        }
        output
    }
}

/// ReLU on signals: zero out negative values, pass positive and zero through.
pub fn relu(signals: &[Signal]) -> Vec<Signal> {
    signals.iter().map(|s| {
        if s.current() < 0 { Signal::ZERO } else { *s }
    }).collect()
}

/// Integer softmax on signals.
///
/// Shifts all currents so the minimum is 0, then normalizes magnitudes
/// proportionally to sum to 255. Returns positive-polarity Signals.
pub fn softmax(signals: &[Signal]) -> Vec<Signal> {
    if signals.is_empty() {
        return Vec::new();
    }
    let currents: Vec<i32> = signals.iter().map(|s| s.current()).collect();
    let min_c = *currents.iter().min().unwrap();
    let shifted: Vec<u64> = currents.iter().map(|&c| (c as i64 - min_c as i64) as u64).collect();
    let total: u64 = shifted.iter().sum();

    if total == 0 {
        let uniform = (255 / signals.len() as u8).max(1);
        signals.iter().map(|_| Signal::new_raw(1, uniform, 1)).collect()
    } else {
        shifted.iter().map(|&s| {
            let mag = ((s * 255) / total) as u8;
            Signal::new_raw(1, mag, 1)
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let wm = WeightMatrix::zeros(4, 8);
        assert_eq!(wm.data.len(), 32);
        assert_eq!(wm.rows, 4);
        assert_eq!(wm.cols, 8);
        for s in &wm.data {
            assert_eq!(s.current(), 0);
        }
    }

    #[test]
    fn test_matmul_identity_like() {
        let mut wm = WeightMatrix::zeros(2, 2);
        wm.set(0, 0, Signal::new_raw(1, 1, 1)); // +1
        wm.set(1, 1, Signal::new_raw(1, 1, 1)); // +1
        let input = vec![
            Signal::new_raw(1, 64, 1),  // +64
            Signal::new_raw(-1, 32, 1), // -32
        ];
        let output = wm.matmul(&input);
        assert_eq!(output.len(), 2);
        assert!(output[0].current() > 0);
        assert!(output[1].current() < 0);
    }

    #[test]
    fn test_relu() {
        let signals = vec![
            Signal::new_raw(1, 42, 1),
            Signal::new_raw(-1, 17, 1),
            Signal::ZERO,
        ];
        let result = relu(&signals);
        assert!(result[0].current() > 0);
        assert_eq!(result[1].current(), 0);
        assert_eq!(result[2].current(), 0);
    }
}
