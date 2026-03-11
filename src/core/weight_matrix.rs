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

    /// Create a frozen random projection matrix.
    ///
    /// Per MASTERY.md: frozen hidden layers use `s = p × m × k` where
    /// p = ±1 (random), m = 20-40 (random), k = random (1-255).
    /// All three components vary — k controls signal gain:
    ///   k=1 attenuates (>>2), k=16-32 neutral, k=128-255 amplifies (<<2-3).
    /// These are NOT learned — they provide fixed random projections
    /// that create class separation for the output layer to learn from.
    ///
    /// Uses xorshift64 PRNG seeded by `seed`.
    pub fn random_frozen(rows: usize, cols: usize, seed: u64) -> Self {
        // The 8 representable multiplier levels from LOG_LUT
        const K_LEVELS: [u8; 7] = [1, 4, 16, 32, 64, 128, 255];

        let mut state = if seed == 0 { 0xDEAD_BEEF_CAFE_1234 } else { seed };
        let mut data = Vec::with_capacity(rows * cols);
        for _ in 0..rows * cols {
            // xorshift64
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let pol: i8 = if state & 1 == 0 { 1 } else { -1 };
            let mag = 20 + ((state >> 8) as u8 % 21); // 20-40 inclusive
            let k = K_LEVELS[((state >> 16) as usize) % K_LEVELS.len()];
            data.push(PackedSignal::pack(pol, mag, k));
        }
        Self { data, rows, cols }
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

/// Precomputed sorted table of (product, positive_code, negative_code) for packed_from_current.
///
/// Built from the LOG_LUT [0, 1, 4, 16, 32, 64, 128, 255]. For each (mag_code, mul_code)
/// pair, product = LOG_LUT[mag_code] * LOG_LUT[mul_code]. Entries are sorted by product,
/// deduplicated (keeping the last code for each product, matching the original loop order).
/// This turns the 64-iteration brute force search into a binary search over ~30 entries.
#[derive(Clone, Copy)]
struct PackedEntry {
    product: u64,
    pos_code: u8,
    neg_code: u8,
}

/// Build the lookup table at compile time.
const fn build_packed_table() -> ([PackedEntry; 64], usize) {
    const LOG_LUT: [u64; 8] = [0, 1, 4, 16, 32, 64, 128, 255];
    let mut entries = [PackedEntry { product: 0, pos_code: 0, neg_code: 0 }; 64];
    let mut count = 0usize;

    // Generate all 64 products with their codes
    let mut mc = 0u8;
    while mc < 8 {
        let mut uc = 0u8;
        while uc < 8 {
            let product = LOG_LUT[mc as usize] * LOG_LUT[uc as usize];
            let pos_code = (0b01 << 6) | (mc << 3) | uc;
            let neg_code = (0b10 << 6) | (mc << 3) | uc;
            entries[count] = PackedEntry { product, pos_code, neg_code };
            count += 1;
            uc += 1;
        }
        mc += 1;
    }

    // Insertion sort by product (const fn can't use sort)
    let mut i = 1;
    while i < count {
        let mut j = i;
        while j > 0 && entries[j - 1].product > entries[j].product {
            let tmp = PackedEntry {
                product: entries[j].product,
                pos_code: entries[j].pos_code,
                neg_code: entries[j].neg_code,
            };
            entries[j] = PackedEntry {
                product: entries[j - 1].product,
                pos_code: entries[j - 1].pos_code,
                neg_code: entries[j - 1].neg_code,
            };
            entries[j - 1] = tmp;
            j -= 1;
        }
        i += 1;
    }

    // Deduplicate: keep last entry for each product (matches original loop's "last wins")
    let mut deduped = [PackedEntry { product: 0, pos_code: 0, neg_code: 0 }; 64];
    let mut dcount = 0usize;
    let mut k = 0;
    while k < count {
        // Find the last entry with this product value
        let mut last = k;
        while last + 1 < count && entries[last + 1].product == entries[k].product {
            last += 1;
        }
        deduped[dcount] = PackedEntry {
            product: entries[last].product,
            pos_code: entries[last].pos_code,
            neg_code: entries[last].neg_code,
        };
        dcount += 1;
        k = last + 1;
    }

    (deduped, dcount)
}

const PACKED_TABLE_DATA: ([PackedEntry; 64], usize) = build_packed_table();
const PACKED_TABLE: &[PackedEntry] = {
    // We can't slice a const array directly, but we know the count.
    // Use the full 64-entry array; entries beyond count are unused (product=0, duplicates of first).
    &PACKED_TABLE_DATA.0
};
const PACKED_TABLE_LEN: usize = PACKED_TABLE_DATA.1;

/// Convert a raw current value (signed i32) to the nearest PackedSignal.
///
/// Uses a precomputed sorted table with binary search instead of brute-forcing
/// all 64 (magnitude, multiplier) combinations.
pub fn packed_from_current(current: i32) -> PackedSignal {
    if current == 0 {
        return PackedSignal::ZERO;
    }
    let is_positive = current > 0;
    let abs_val = (current as i64).unsigned_abs();
    let table = &PACKED_TABLE[..PACKED_TABLE_LEN];

    // Binary search for the closest product
    let mut lo = 0usize;
    let mut hi = table.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if table[mid].product < abs_val {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    // Check lo and lo-1 for the closest match
    let best = if lo >= table.len() {
        table.len() - 1
    } else if lo == 0 {
        0
    } else {
        let dist_lo = if table[lo].product >= abs_val {
            table[lo].product - abs_val
        } else {
            abs_val - table[lo].product
        };
        let dist_prev = abs_val - table[lo - 1].product;
        if dist_prev <= dist_lo { lo - 1 } else { lo }
    };

    let code = if is_positive {
        table[best].pos_code
    } else {
        table[best].neg_code
    };
    PackedSignal::from_raw(code)
}

/// ReLU on packed signals: zero out negative values, pass positive and zero through.
pub fn relu_packed(signals: &[PackedSignal]) -> Vec<PackedSignal> {
    signals.iter().map(|s| {
        if s.is_negative() { PackedSignal::ZERO } else { *s }
    }).collect()
}

/// Integer softmax on packed signals.
///
/// Shifts all currents so the minimum is 0, then normalizes magnitudes
/// proportionally to sum to 255. Returns positive-polarity PackedSignals.
pub fn softmax_packed(signals: &[PackedSignal]) -> Vec<PackedSignal> {
    if signals.is_empty() {
        return Vec::new();
    }
    let currents: Vec<i32> = signals.iter().map(|s| s.current()).collect();
    let min_c = *currents.iter().min().unwrap();
    let shifted: Vec<u64> = currents.iter().map(|&c| (c as i64 - min_c as i64) as u64).collect();
    let total: u64 = shifted.iter().sum();

    if total == 0 {
        let uniform = (255 / signals.len() as u8).max(1);
        signals.iter().map(|_| PackedSignal::pack(1, uniform, 1)).collect()
    } else {
        shifted.iter().map(|&s| {
            let mag = ((s * 255) / total) as u8;
            PackedSignal::pack(1, mag, 1)
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
