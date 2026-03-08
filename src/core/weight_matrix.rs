//! Weight matrix storage using PackedSignal bytes.
//!
//! Binary `.ant` format:
//! ```text
//! Header (8 bytes):
//!   magic: [u8; 4] = b"ANT\x01"
//!   rows:  u16 (LE)
//!   cols:  u16 (LE)
//! Body:
//!   data: [u8; rows * cols]  // each byte is a PackedSignal
//! ```

use std::path::Path;
use ternary_signal::PackedSignal;

const MAGIC: [u8; 4] = *b"ANT\x01";

/// A matrix of PackedSignal weights with shape metadata.
#[derive(Clone, Debug)]
pub struct WeightMatrix {
    pub data: Vec<PackedSignal>,
    pub rows: usize,
    pub cols: usize,
}

impl WeightMatrix {
    /// Create a new weight matrix filled with zero signals.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![PackedSignal::ZERO; rows * cols],
            rows,
            cols,
        }
    }

    /// Create from raw PackedSignal data with shape validation.
    pub fn from_data(data: Vec<PackedSignal>, rows: usize, cols: usize) -> Option<Self> {
        if data.len() != rows * cols {
            return None;
        }
        Some(Self { data, rows, cols })
    }

    /// Load from `.ant` binary file.
    #[deprecated(note = "Use Thermogram load_synaptic verb instead of .ant files")]
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Save to `.ant` binary file.
    #[deprecated(note = "Use Thermogram save_synaptic verb instead of .ant files")]
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let bytes = self.to_bytes();
        std::fs::write(path, bytes)
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() < 8 {
            return Err("file too short for header");
        }
        if bytes[0..4] != MAGIC {
            return Err("invalid magic bytes");
        }
        let rows = u16::from_le_bytes([bytes[4], bytes[5]]) as usize;
        let cols = u16::from_le_bytes([bytes[6], bytes[7]]) as usize;
        let expected = rows * cols;
        if bytes.len() - 8 != expected {
            return Err("data length does not match rows * cols");
        }
        let data: Vec<PackedSignal> = bytes[8..].iter().map(|&b| PackedSignal::from_raw(b)).collect();
        Ok(Self { data, rows, cols })
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + self.data.len());
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&(self.rows as u16).to_le_bytes());
        out.extend_from_slice(&(self.cols as u16).to_le_bytes());
        for ps in &self.data {
            out.push(ps.as_u8());
        }
        out
    }

    /// Get weight at (row, col).
    pub fn get(&self, row: usize, col: usize) -> PackedSignal {
        self.data[row * self.cols + col]
    }

    /// Set weight at (row, col).
    pub fn set(&mut self, row: usize, col: usize, value: PackedSignal) {
        self.data[row * self.cols + col] = value;
    }

    /// Ternary matrix-vector multiply: output[i] = Σ_j(weight[i,j].current() * input[j].current())
    /// Result is clamped to PackedSignal range.
    pub fn matmul(&self, input: &[PackedSignal]) -> Vec<PackedSignal> {
        assert_eq!(input.len(), self.cols, "matmul input dimension mismatch");
        let mut output = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            let mut acc: i64 = 0;
            for j in 0..self.cols {
                let w = self.data[i * self.cols + j].current() as i64;
                let x = input[j].current() as i64;
                acc += w * x;
            }
            // Clamp accumulator to i32 range, then convert to PackedSignal
            let clamped = acc.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
            output.push(packed_from_current(clamped));
        }
        output
    }
}

/// Convert a raw current value (signed i32) to the nearest PackedSignal.
pub fn packed_from_current(current: i32) -> PackedSignal {
    if current == 0 {
        return PackedSignal::ZERO;
    }
    let polarity: i8 = if current > 0 { 1 } else { -1 };
    let abs_val = (current as i64).unsigned_abs();

    // Find best (mag_code, mul_code) pair whose product is closest to abs_val
    const LOG_LUT: [u64; 8] = [0, 1, 4, 16, 32, 64, 128, 255];
    let mut best_code: u8 = 0;
    let mut best_dist: u64 = abs_val; // distance from 0

    for mc in 0u8..8 {
        for uc in 0u8..8 {
            let product = LOG_LUT[mc as usize] * LOG_LUT[uc as usize];
            let dist = if product > abs_val {
                product - abs_val
            } else {
                abs_val - product
            };
            if dist < best_dist {
                best_dist = dist;
                let pol_bits = if polarity > 0 { 0b01 } else { 0b10 };
                best_code = (pol_bits << 6) | (mc << 3) | uc;
            }
        }
    }
    PackedSignal::from_raw(best_code)
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
        for ps in &wm.data {
            assert_eq!(ps.current(), 0);
        }
    }

    #[test]
    fn test_roundtrip_bytes() {
        let mut wm = WeightMatrix::zeros(3, 5);
        wm.set(0, 0, PackedSignal::MAX_POSITIVE);
        wm.set(2, 4, PackedSignal::MAX_NEGATIVE);
        let bytes = wm.to_bytes();
        let wm2 = WeightMatrix::from_bytes(&bytes).unwrap();
        assert_eq!(wm2.rows, 3);
        assert_eq!(wm2.cols, 5);
        assert_eq!(wm2.get(0, 0).as_u8(), PackedSignal::MAX_POSITIVE.as_u8());
        assert_eq!(wm2.get(2, 4).as_u8(), PackedSignal::MAX_NEGATIVE.as_u8());
    }

    #[test]
    #[allow(deprecated)]
    fn test_roundtrip_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.ant");
        let mut wm = WeightMatrix::zeros(2, 3);
        wm.set(1, 2, PackedSignal::pack(1, 64, 32));
        wm.save(&path).unwrap();
        let loaded = WeightMatrix::load(&path).unwrap();
        assert_eq!(loaded.rows, 2);
        assert_eq!(loaded.cols, 3);
        assert_eq!(loaded.get(1, 2).as_u8(), wm.get(1, 2).as_u8());
    }

    #[test]
    fn test_matmul_identity_like() {
        // 2x2 weight matrix with positive diagonal
        let mut wm = WeightMatrix::zeros(2, 2);
        wm.set(0, 0, PackedSignal::pack(1, 1, 1)); // +1
        wm.set(1, 1, PackedSignal::pack(1, 1, 1)); // +1
        let input = vec![
            PackedSignal::pack(1, 64, 1), // +64
            PackedSignal::pack(-1, 32, 1), // -32
        ];
        let output = wm.matmul(&input);
        assert_eq!(output.len(), 2);
        // output[0] should be positive (64*1 + 0)
        assert!(output[0].is_positive());
        // output[1] should be negative (0 + -32*1)
        assert!(output[1].is_negative());
    }

    #[test]
    fn test_packed_from_current() {
        let zero = packed_from_current(0);
        assert_eq!(zero.current(), 0);

        let pos = packed_from_current(100);
        assert!(pos.is_positive());

        let neg = packed_from_current(-100);
        assert!(neg.is_negative());
    }
}
