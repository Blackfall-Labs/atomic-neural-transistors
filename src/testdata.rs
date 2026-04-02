//! Test data generation for runners and examples.
//!
//! Provides a deterministic PRNG and Hadamard-pattern class prototype generation.
//! Used by all runner binaries for reproducible experiments.

use ternary_signal::Signal;

/// Simple deterministic PRNG (xorshift64).
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0xDEAD_BEEF } else { seed })
    }

    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    pub fn next_u8(&mut self) -> u8 {
        (self.next() & 0xFF) as u8
    }

    pub fn next_i32(&mut self, range: i32) -> i32 {
        (self.next() % range as u64) as i32
    }
}

/// A labeled sample for classification.
pub struct Sample {
    pub signal: Vec<Signal>,
    pub class: usize,
}

/// Generate N class prototypes using Hadamard-like polarity signatures.
/// Each class has full `dim`-dim coverage with a unique polarity pattern.
pub fn generate_class_prototypes(rng: &mut Rng, n_classes: usize, dim: usize) -> Vec<Vec<Signal>> {
    (0..n_classes)
        .map(|class| {
            (0..dim)
                .map(|d| {
                    let bits = ((class & d) as u32).count_ones();
                    let pol: i8 = if bits % 2 == 0 { 1 } else { -1 };
                    let mag = 128 + rng.next_u8() % 64;
                    Signal::new_raw(pol, mag, 1)
                })
                .collect()
        })
        .collect()
}

/// Add noise to a prototype signal.
pub fn add_noise(rng: &mut Rng, proto: &[Signal]) -> Vec<Signal> {
    proto
        .iter()
        .map(|s| {
            let c = s.current();
            let noise = (rng.next() % 81) as i32 - 40;
            Signal::from_current(c.saturating_add(noise))
        })
        .collect()
}

/// Add domain-shifted noise (simulates cross-speaker variation).
/// Rotates adjacent dimension pairs and applies spectral tilt scaling.
pub fn add_shifted_noise(rng: &mut Rng, proto: &[Signal]) -> Vec<Signal> {
    let mut result: Vec<Signal> = add_noise(rng, proto);

    // Rotate adjacent dimension pairs
    for i in (0..result.len().saturating_sub(1)).step_by(2) {
        let a = result[i].current() as i64;
        let b = result[i + 1].current() as i64;
        let a_new = (a * 7 + b * 3) / 10;
        let b_new = (-a * 3 + b * 7) / 10;
        result[i] = Signal::from_current(a_new as i32);
        result[i + 1] = Signal::from_current(b_new as i32);
    }

    // Scale odd dimensions by 1.5×, even by 0.7× (spectral tilt)
    for (i, s) in result.iter_mut().enumerate() {
        let c = s.current() as i64;
        let scaled = if i % 2 == 0 { c * 7 / 10 } else { c * 15 / 10 };
        *s = Signal::from_current(scaled as i32);
    }

    result
}

/// Generate a dataset of labeled samples from class prototypes.
pub fn generate_dataset(rng: &mut Rng, n: usize, prototypes: &[Vec<Signal>]) -> Vec<Sample> {
    let num_classes = prototypes.len();
    (0..n)
        .map(|i| {
            let class = i % num_classes;
            let signal = add_noise(rng, &prototypes[class]);
            Sample { signal, class }
        })
        .collect()
}

/// Generate product features: element-wise (a[i] * b[i]) >> 8.
pub fn product_features(a: &[Signal], b: &[Signal]) -> Vec<Signal> {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let product = (x.current() as i64 * y.current() as i64) >> 8;
            Signal::from_current(product as i32)
        })
        .collect()
}

/// Evaluate classification accuracy.
pub fn evaluate_accuracy(
    predict: &mut dyn FnMut(&[Signal]) -> usize,
    data: &[Sample],
) -> f64 {
    let correct = data.iter().filter(|s| predict(&s.signal) == s.class).count();
    correct as f64 / data.len() as f64 * 100.0
}
