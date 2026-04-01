//! Thermal weights — per-weight plasticity gating.
//!
//! Each weight carries its own temperature. HOT weights change freely,
//! COLD weights are locked. Hits (correct participations) cool weights down.
//! Consistent reinforcement across diverse inputs → weight locks in.
//! Inconsistent performance → weight stays hot and keeps searching.
//!
//! ## Format
//!
//! Per weight: 6 bytes
//!   signal:      PackedSignal (1 byte) — s = p × m × k
//!   temperature: u8 — 255=HOT → 0=COLD
//!   hits:        u16 (LE) — reinforcement count
//!   pressure:    i16 (LE) — accumulated mastery pressure
//!
//! ## .ant v2 binary format
//!
//! ```text
//! Header (16 bytes):
//!   magic:     [u8; 4] = b"ANT\x02"
//!   n_layers:  u16 (LE)
//!   flags:     u16 (LE) — reserved
//!   checksum:  u32 (LE) — CRC32 of all layer data
//!   reserved:  u32
//!
//! Per layer (8 bytes header + 6 bytes × rows × cols):
//!   rows:      u16 (LE)
//!   cols:      u16 (LE)
//!   reserved:  u32
//!   weights:   [ThermalWeight; rows × cols]
//! ```

use ternary_signal::PackedSignal;

/// Temperature band thresholds.
pub const TEMP_HOT: u8 = 192;
pub const TEMP_WARM: u8 = 128;
pub const TEMP_COOL: u8 = 64;
// Below TEMP_COOL = COLD

/// A single weight with thermal state.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct ThermalWeight {
    /// The signal value: s = p × m × k.
    pub signal: PackedSignal,
    /// Temperature: 255 = HOT (fully plastic) → 0 = COLD (frozen).
    pub temperature: u8,
    /// Hit count: incremented on correct participation.
    pub hits: u16,
    /// Accumulated mastery pressure (signed).
    pub pressure: i16,
}

impl ThermalWeight {
    /// Zero weight, fully hot, no history.
    pub const ZERO_HOT: Self = Self {
        signal: PackedSignal::ZERO,
        temperature: 255,
        hits: 0,
        pressure: 0,
    };

    /// Create from a PackedSignal at full temperature.
    pub fn hot(signal: PackedSignal) -> Self {
        Self { signal, temperature: 255, hits: 0, pressure: 0 }
    }

    /// Is this weight frozen?
    pub fn is_cold(&self) -> bool { self.temperature < TEMP_COOL }

    /// Is this weight fully plastic?
    pub fn is_hot(&self) -> bool { self.temperature >= TEMP_HOT }

    /// Temperature band name.
    pub fn band(&self) -> &'static str {
        if self.temperature >= TEMP_HOT { "HOT" }
        else if self.temperature >= TEMP_WARM { "WARM" }
        else if self.temperature >= TEMP_COOL { "COOL" }
        else { "COLD" }
    }

    /// Pressure threshold multiplier based on temperature.
    /// HOT = 1×, WARM = 2×, COOL = 4×, COLD = blocked.
    pub fn pressure_multiplier(&self) -> u32 {
        if self.temperature >= TEMP_HOT { 1 }
        else if self.temperature >= TEMP_WARM { 2 }
        else if self.temperature >= TEMP_COOL { 4 }
        else { 0 } // COLD = no transitions
    }

    /// Record a correct participation. Cools the weight.
    pub fn hit(&mut self, cooling_rate: u16) {
        self.hits = self.hits.saturating_add(1);
        if cooling_rate > 0 && self.hits % cooling_rate == 0 && self.temperature > 0 {
            self.temperature = self.temperature.saturating_sub(1);
        }
    }

    /// Warm up (on sustained errors despite being cool/cold).
    pub fn warm(&mut self, amount: u8) {
        self.temperature = self.temperature.saturating_add(amount);
    }

    /// Serialize to 6 bytes.
    pub fn to_bytes(&self) -> [u8; 6] {
        let hits = self.hits.to_le_bytes();
        let pressure = self.pressure.to_le_bytes();
        [self.signal.as_u8(), self.temperature, hits[0], hits[1], pressure[0], pressure[1]]
    }

    /// Deserialize from 6 bytes.
    pub fn from_bytes(b: &[u8; 6]) -> Self {
        Self {
            signal: PackedSignal::from_raw(b[0]),
            temperature: b[1],
            hits: u16::from_le_bytes([b[2], b[3]]),
            pressure: i16::from_le_bytes([b[4], b[5]]),
        }
    }
}

impl Default for ThermalWeight {
    fn default() -> Self { Self::ZERO_HOT }
}

/// A matrix of thermal weights.
#[derive(Clone, Debug)]
pub struct ThermalWeightMatrix {
    pub data: Vec<ThermalWeight>,
    pub rows: usize,
    pub cols: usize,
}

/// Configuration for thermal mastery updates.
#[derive(Clone, Debug)]
pub struct ThermalMasteryConfig {
    /// Base pressure threshold (scaled by temperature band).
    pub pressure_threshold: i32,
    /// Pressure decay per epoch.
    pub decay_rate: i32,
    /// Minimum participations before learning.
    pub participation_gate: u32,
    /// Hits needed per 1 degree of cooling.
    pub cooling_rate: u16,
    /// Temperature increase on sustained wrong detection.
    pub warming_step: u8,
    /// Pressure magnitude that triggers warming.
    pub warming_threshold: i32,
}

impl Default for ThermalMasteryConfig {
    fn default() -> Self {
        Self {
            pressure_threshold: 3,
            decay_rate: 1,
            participation_gate: 5,
            cooling_rate: 100,
            warming_step: 10,
            warming_threshold: 20,
        }
    }
}

const MAGIC_V2: [u8; 4] = *b"ANT\x02";
const BYTES_PER_WEIGHT: usize = 6;
const LAYER_HEADER: usize = 8;
const FILE_HEADER: usize = 16;

impl ThermalWeightMatrix {
    /// All weights zero and HOT.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![ThermalWeight::ZERO_HOT; rows * cols],
            rows,
            cols,
        }
    }

    /// Random-initialized, all HOT.
    pub fn random_hot(rows: usize, cols: usize, seed: u64) -> Self {
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
            data.push(ThermalWeight::hot(PackedSignal::pack(pol, mag, k)));
        }
        Self { data, rows, cols }
    }

    /// Get weight at (row, col).
    pub fn get(&self, row: usize, col: usize) -> &ThermalWeight {
        &self.data[row * self.cols + col]
    }

    /// Get mutable weight at (row, col).
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut ThermalWeight {
        &mut self.data[row * self.cols + col]
    }

    /// Matmul using only the signal values (temperature doesn't affect computation).
    pub fn matmul(&self, input: &[PackedSignal]) -> Vec<PackedSignal> {
        assert_eq!(input.len(), self.cols, "matmul input dimension mismatch");
        let mut output = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            let mut acc: i64 = 0;
            for j in 0..self.cols {
                let w = self.data[i * self.cols + j].signal.current() as i64;
                let x = input[j].current() as i64;
                acc += w * x;
            }
            let clamped = acc.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
            output.push(super::weight_matrix::packed_from_current(clamped));
        }
        output
    }

    /// Extract just the PackedSignal values (for compatibility with existing matmul code).
    pub fn signals(&self) -> Vec<PackedSignal> {
        self.data.iter().map(|tw| tw.signal).collect()
    }

    /// Temperature distribution summary.
    pub fn temp_summary(&self) -> (usize, usize, usize, usize) {
        let mut hot = 0; let mut warm = 0; let mut cool = 0; let mut cold = 0;
        for tw in &self.data {
            if tw.temperature >= TEMP_HOT { hot += 1; }
            else if tw.temperature >= TEMP_WARM { warm += 1; }
            else if tw.temperature >= TEMP_COOL { cool += 1; }
            else { cold += 1; }
        }
        (hot, warm, cool, cold)
    }

    /// Save to .ant v2 binary format.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        // Body = layer header + weight data (everything after file header)
        let mut body = Vec::new();
        body.extend_from_slice(&(self.rows as u16).to_le_bytes());
        body.extend_from_slice(&(self.cols as u16).to_le_bytes());
        body.extend_from_slice(&0u32.to_le_bytes());
        body.extend_from_slice(&self.to_bytes_body());

        let checksum = crc32fast::hash(&body);

        let mut out = Vec::with_capacity(FILE_HEADER + body.len());
        out.extend_from_slice(&MAGIC_V2);
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&checksum.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&body);

        std::fs::write(path, out)
    }

    /// Load from .ant v2 binary format.
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes_full(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Save multiple layers to one .ant v2 file.
    pub fn save_multi(layers: &[&ThermalWeightMatrix], path: &std::path::Path) -> std::io::Result<()> {
        let mut body = Vec::new();
        for layer in layers {
            body.extend_from_slice(&(layer.rows as u16).to_le_bytes());
            body.extend_from_slice(&(layer.cols as u16).to_le_bytes());
            body.extend_from_slice(&0u32.to_le_bytes());
            body.extend_from_slice(&layer.to_bytes_body());
        }
        let checksum = crc32fast::hash(&body);

        let mut out = Vec::with_capacity(FILE_HEADER + body.len());
        out.extend_from_slice(&MAGIC_V2);
        out.extend_from_slice(&(layers.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&checksum.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&body);

        std::fs::write(path, out)
    }

    /// Load multiple layers from one .ant v2 file.
    pub fn load_multi(path: &std::path::Path) -> std::io::Result<Vec<Self>> {
        let bytes = std::fs::read(path)?;
        if bytes.len() < FILE_HEADER || bytes[0..4] != MAGIC_V2 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "not ANT v2"));
        }
        let n_layers = u16::from_le_bytes([bytes[4], bytes[5]]) as usize;
        let stored_crc = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);

        let body = &bytes[FILE_HEADER..];
        if crc32fast::hash(body) != stored_crc {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "checksum mismatch"));
        }

        let mut layers = Vec::with_capacity(n_layers);
        let mut offset = 0;
        for _ in 0..n_layers {
            if offset + LAYER_HEADER > body.len() { break; }
            let rows = u16::from_le_bytes([body[offset], body[offset + 1]]) as usize;
            let cols = u16::from_le_bytes([body[offset + 2], body[offset + 3]]) as usize;
            offset += LAYER_HEADER;

            let n_weights = rows * cols;
            let weight_bytes = n_weights * BYTES_PER_WEIGHT;
            if offset + weight_bytes > body.len() { break; }

            let mut data = Vec::with_capacity(n_weights);
            for i in 0..n_weights {
                let base = offset + i * BYTES_PER_WEIGHT;
                let b: [u8; 6] = [
                    body[base], body[base + 1], body[base + 2],
                    body[base + 3], body[base + 4], body[base + 5],
                ];
                data.push(ThermalWeight::from_bytes(&b));
            }
            offset += weight_bytes;

            layers.push(Self { data, rows, cols });
        }

        Ok(layers)
    }

    fn to_bytes_body(&self) -> Vec<u8> {
        let mut body = Vec::with_capacity(self.data.len() * BYTES_PER_WEIGHT);
        for tw in &self.data {
            body.extend_from_slice(&tw.to_bytes());
        }
        body
    }

    fn from_bytes_full(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() < FILE_HEADER + LAYER_HEADER { return Err("too short"); }
        if bytes[0..4] != MAGIC_V2 { return Err("not ANT v2"); }

        let stored_crc = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let body = &bytes[FILE_HEADER..];
        if crc32fast::hash(body) != stored_crc { return Err("checksum mismatch"); }

        let rows = u16::from_le_bytes([body[0], body[1]]) as usize;
        let cols = u16::from_le_bytes([body[2], body[3]]) as usize;
        let weight_start = LAYER_HEADER;
        let n_weights = rows * cols;
        let expected = weight_start + n_weights * BYTES_PER_WEIGHT;
        if body.len() < expected { return Err("truncated data"); }

        let mut data = Vec::with_capacity(n_weights);
        for i in 0..n_weights {
            let base = weight_start + i * BYTES_PER_WEIGHT;
            let b: [u8; 6] = [
                body[base], body[base + 1], body[base + 2],
                body[base + 3], body[base + 4], body[base + 5],
            ];
            data.push(ThermalWeight::from_bytes(&b));
        }

        Ok(Self { data, rows, cols })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thermal_weight_zero_hot() {
        let tw = ThermalWeight::ZERO_HOT;
        assert_eq!(tw.signal.current(), 0);
        assert_eq!(tw.temperature, 255);
        assert!(tw.is_hot());
        assert!(!tw.is_cold());
        assert_eq!(tw.hits, 0);
        assert_eq!(tw.pressure, 0);
    }

    #[test]
    fn thermal_weight_cooling() {
        let mut tw = ThermalWeight::ZERO_HOT;
        for _ in 0..100 {
            tw.hit(100); // cooling_rate = 100
        }
        assert_eq!(tw.hits, 100);
        assert_eq!(tw.temperature, 254); // dropped by 1
    }

    #[test]
    fn thermal_weight_bands() {
        let mut tw = ThermalWeight::ZERO_HOT;
        assert_eq!(tw.band(), "HOT");
        tw.temperature = 150;
        assert_eq!(tw.band(), "WARM");
        tw.temperature = 80;
        assert_eq!(tw.band(), "COOL");
        tw.temperature = 10;
        assert_eq!(tw.band(), "COLD");
        assert!(tw.is_cold());
    }

    #[test]
    fn thermal_weight_pressure_multiplier() {
        let mut tw = ThermalWeight::ZERO_HOT;
        assert_eq!(tw.pressure_multiplier(), 1); // HOT
        tw.temperature = 150;
        assert_eq!(tw.pressure_multiplier(), 2); // WARM
        tw.temperature = 80;
        assert_eq!(tw.pressure_multiplier(), 4); // COOL
        tw.temperature = 10;
        assert_eq!(tw.pressure_multiplier(), 0); // COLD = blocked
    }

    #[test]
    fn thermal_weight_serialization() {
        let tw = ThermalWeight {
            signal: PackedSignal::pack(1, 64, 32),
            temperature: 180,
            hits: 1234,
            pressure: -42,
        };
        let bytes = tw.to_bytes();
        let tw2 = ThermalWeight::from_bytes(&bytes);
        assert_eq!(tw.signal.as_u8(), tw2.signal.as_u8());
        assert_eq!(tw.temperature, tw2.temperature);
        assert_eq!(tw.hits, tw2.hits);
        assert_eq!(tw.pressure, tw2.pressure);
    }

    #[test]
    fn thermal_matrix_zeros() {
        let m = ThermalWeightMatrix::zeros(4, 8);
        assert_eq!(m.data.len(), 32);
        assert!(m.data.iter().all(|tw| tw.is_hot()));
        assert!(m.data.iter().all(|tw| tw.signal.current() == 0));
    }

    #[test]
    fn thermal_matrix_random() {
        let m = ThermalWeightMatrix::random_hot(4, 8, 42);
        assert_eq!(m.data.len(), 32);
        assert!(m.data.iter().all(|tw| tw.is_hot()));
        assert!(m.data.iter().any(|tw| tw.signal.current() != 0));
    }

    #[test]
    fn thermal_matrix_matmul() {
        let mut m = ThermalWeightMatrix::zeros(2, 2);
        m.data[0] = ThermalWeight::hot(PackedSignal::pack(1, 1, 1)); // +1
        m.data[3] = ThermalWeight::hot(PackedSignal::pack(1, 1, 1)); // +1
        let input = vec![
            PackedSignal::pack(1, 64, 1),
            PackedSignal::pack(-1, 32, 1),
        ];
        let output = m.matmul(&input);
        assert!(output[0].is_positive());
        assert!(output[1].is_negative());
    }

    #[test]
    fn thermal_matrix_save_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.ant");

        let mut m = ThermalWeightMatrix::random_hot(3, 5, 42);
        m.data[7].temperature = 50; // set one to COLD
        m.data[7].hits = 9999;
        m.data[7].pressure = -100;
        m.save(&path).unwrap();

        let m2 = ThermalWeightMatrix::load(&path).unwrap();
        assert_eq!(m2.rows, 3);
        assert_eq!(m2.cols, 5);
        assert_eq!(m2.data.len(), 15);
        assert_eq!(m2.data[7].temperature, 50);
        assert_eq!(m2.data[7].hits, 9999);
        assert_eq!(m2.data[7].pressure, -100);

        for i in 0..15 {
            assert_eq!(m.data[i].signal.as_u8(), m2.data[i].signal.as_u8());
            assert_eq!(m.data[i].temperature, m2.data[i].temperature);
        }
    }

    #[test]
    fn thermal_matrix_multi_layer_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("multi.ant");

        let hidden = ThermalWeightMatrix::random_hot(48, 80, 42);
        let output = ThermalWeightMatrix::zeros(1, 48);

        ThermalWeightMatrix::save_multi(&[&hidden, &output], &path).unwrap();
        let loaded = ThermalWeightMatrix::load_multi(&path).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].rows, 48);
        assert_eq!(loaded[0].cols, 80);
        assert_eq!(loaded[1].rows, 1);
        assert_eq!(loaded[1].cols, 48);

        for i in 0..hidden.data.len() {
            assert_eq!(hidden.data[i].signal.as_u8(), loaded[0].data[i].signal.as_u8());
        }
    }

    #[test]
    fn temp_summary() {
        let mut m = ThermalWeightMatrix::zeros(2, 2);
        m.data[0].temperature = 255; // HOT
        m.data[1].temperature = 150; // WARM
        m.data[2].temperature = 80;  // COOL
        m.data[3].temperature = 10;  // COLD
        let (h, w, c, d) = m.temp_summary();
        assert_eq!((h, w, c, d), (1, 1, 1, 1));
    }
}
