//! CompareANT Mastery Training — Example 1
//!
//! Trains a similarity detector from scratch on 32-dim ternary signal pairs
//! using mastery learning (integer-only, pressure-based, no gradients).
//!
//! Architecture: element-wise product features → single trainable output synaptic matrix.
//! Product features encode sign agreement: identical pairs produce all-positive products,
//! different pairs produce mixed-sign products (via Hadamard polarity patterns).
//!
//! Training data:
//!   - 1,000 random pairs (50% identical, 50% different) from 8 prototype patterns
//!   - Hold out 200 for evaluation
//!
//! Proves: mastery convergence, thermogram persistence, accuracy after reload.

use atomic_neural_transistors::core::weight_matrix::{packed_from_current, WeightMatrix};
use atomic_neural_transistors::learning::{MasteryConfig, MasteryState};
use atomic_neural_transistors::PackedSignal;
use thermogram::{Delta, PlasticityRule, Thermogram};

// ---------------------------------------------------------------------------
// Simple deterministic PRNG (xorshift64)
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn next_u8(&mut self) -> u8 {
        (self.next() & 0xFF) as u8
    }
    #[allow(dead_code)]
    fn next_bool(&mut self) -> bool {
        self.next() & 1 == 0
    }
}

// ---------------------------------------------------------------------------
// Data generation
// ---------------------------------------------------------------------------

struct Sample {
    vec_a: Vec<PackedSignal>,
    vec_b: Vec<PackedSignal>,
    identical: bool,
}

/// Generate prototypes: distinct 32-dim patterns using Hadamard-like polarity signatures.
/// Each prototype has high magnitude across ALL dimensions. Polarity patterns are
/// designed so that any two different prototypes disagree on exactly half the dimensions,
/// making product features cleanly discriminative via element-wise sign analysis.
fn generate_prototypes(rng: &mut Rng, count: usize) -> Vec<Vec<PackedSignal>> {
    // Hadamard-like rows — each pair disagrees on ~50% of dimensions
    let patterns: Vec<Vec<i8>> = (0..count)
        .map(|p| {
            (0..32)
                .map(|d| {
                    // Use bit-counting parity: sign = (-1)^popcount(p & d)
                    let bits = (p & d).count_ones();
                    if bits % 2 == 0 { 1i8 } else { -1i8 }
                })
                .collect()
        })
        .collect();

    patterns
        .iter()
        .map(|pat| {
            pat.iter()
                .map(|&pol| {
                    let mag = 128 + rng.next_u8() % 64;
                    PackedSignal::pack(pol, mag, 1)
                })
                .collect()
        })
        .collect()
}

/// Add noise to a prototype signal.
fn add_noise(rng: &mut Rng, proto: &[PackedSignal]) -> Vec<PackedSignal> {
    proto
        .iter()
        .map(|s| {
            let c = s.current();
            let noise = (rng.next() % 81) as i32 - 40;
            packed_from_current(c.saturating_add(noise))
        })
        .collect()
}

fn generate_dataset(rng: &mut Rng, n: usize, prototypes: &[Vec<PackedSignal>]) -> Vec<Sample> {
    let num_protos = prototypes.len();
    let mut samples = Vec::with_capacity(n);
    for i in 0..n {
        let proto_a = (rng.next() as usize) % num_protos;
        let vec_a = add_noise(rng, &prototypes[proto_a]);
        let identical = i % 2 == 0;
        let vec_b = if identical {
            add_noise(rng, &prototypes[proto_a]) // same proto, different noise
        } else {
            let mut proto_b = (rng.next() as usize) % num_protos;
            while proto_b == proto_a {
                proto_b = (rng.next() as usize) % num_protos;
            }
            add_noise(rng, &prototypes[proto_b])
        };
        samples.push(Sample { vec_a, vec_b, identical });
    }
    samples
}

// ---------------------------------------------------------------------------
// Forward pass (Rust-native, matching compare.rune architecture)
// ---------------------------------------------------------------------------

fn matmul(input: &[PackedSignal], w: &WeightMatrix) -> Vec<PackedSignal> {
    let mut output = Vec::with_capacity(w.rows);
    for i in 0..w.rows {
        let mut sum: i64 = 0;
        for j in 0..w.cols {
            sum += input[j].current() as i64 * w.data[i * w.cols + j].current() as i64;
        }
        // Clamp to i32 range
        let clamped = sum.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        output.push(packed_from_current(clamped));
    }
    output
}

/// Compute element-wise product features: feature[j] = a[j] * b[j]
/// For identical pairs (same prototype): products are positive (sign agreement)
/// For different pairs (opposite prototypes): products are negative (sign conflict)
fn comparison_features(a: &[PackedSignal], b: &[PackedSignal]) -> Vec<PackedSignal> {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let product = x.current() as i64 * y.current() as i64;
            packed_from_current((product / 256) as i32)
        })
        .collect()
}

struct CompareNetwork {
    w_out: WeightMatrix,   // 1 × 32 (trained by mastery on product features)
    ms_out: MasteryState,
}

impl CompareNetwork {
    fn new(_rng: &mut Rng) -> Self {
        let config = MasteryConfig {
            pressure_threshold: 3,
            decay_rate: 1,
            participation_gate: 5,
        };

        // Output layer starts at zeros — will be learned
        // Input is 32-dim product features: a[j]*b[j]
        let w_out = WeightMatrix::zeros(1, 32);

        Self {
            ms_out: MasteryState::new(w_out.data.len(), config),
            w_out,
        }
    }

    fn forward(&self, a: &[PackedSignal], b: &[PackedSignal]) -> PackedSignal {
        let features = comparison_features(a, b);
        let out = matmul(&features, &self.w_out);
        out[0]
    }

    fn train_step(&mut self, a: &[PackedSignal], b: &[PackedSignal], target: PackedSignal) {
        let features = comparison_features(a, b);
        let raw_out = matmul(&features, &self.w_out);
        // Clamp output to prevent overshoot oscillation
        let clamped = packed_from_current(raw_out[0].current().clamp(-127, 127));
        self.ms_out.update(&mut self.w_out, &features, &[clamped], &[target]);
    }

    fn predict(&self, a: &[PackedSignal], b: &[PackedSignal]) -> bool {
        self.forward(a, b).current() > 0
    }

    fn save_to_thermogram(&self) -> Thermogram {
        let mut thermo = Thermogram::new("compare", PlasticityRule::stdp_like());

        let delta_out = Delta::create("compare.w_out", self.w_out.data.clone(), "mastery");
        thermo.apply_delta(delta_out).unwrap();

        thermo
    }
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

struct EvalResult {
    accuracy: f64,
    false_positive_rate: f64,
    false_negative_rate: f64,
    correct: usize,
    total: usize,
}

impl std::fmt::Display for EvalResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}/{} ({:.1}%) | FP: {:.1}% | FN: {:.1}%",
            self.correct, self.total, self.accuracy,
            self.false_positive_rate, self.false_negative_rate
        )
    }
}

fn evaluate_network(net: &CompareNetwork, test_set: &[Sample]) -> EvalResult {
    let mut correct = 0;
    let mut false_positives = 0;
    let mut false_negatives = 0;

    for sample in test_set {
        let predicted_same = net.predict(&sample.vec_a, &sample.vec_b);
        if predicted_same == sample.identical {
            correct += 1;
        } else if predicted_same && !sample.identical {
            false_positives += 1;
        } else {
            false_negatives += 1;
        }
    }

    let total = test_set.len();
    EvalResult {
        accuracy: correct as f64 / total as f64 * 100.0,
        false_positive_rate: false_positives as f64 / total as f64 * 100.0,
        false_negative_rate: false_negatives as f64 / total as f64 * 100.0,
        correct,
        total,
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("=== CompareANT Mastery Training ===\n");

    let mut rng = Rng::new(42);

    // Generate prototypes and data
    let prototypes = generate_prototypes(&mut rng, 8);
    println!("Prototypes: {} distinct patterns", prototypes.len());
    let all_data = generate_dataset(&mut rng, 1000, &prototypes);
    let (test_set, train_set) = all_data.split_at(200);
    println!("Data: {} training, {} test", train_set.len(), test_set.len());

    // Create network with random init
    let mut net = CompareNetwork::new(&mut rng);

    // Initial accuracy
    let initial = evaluate_network(&net, test_set);
    println!("Cycle 0 (random init): {initial}");

    // Training loop
    let max_epochs = 50;
    let mut first_99 = None;
    let mut best_accuracy = 0.0f64;

    for epoch in 1..=max_epochs {
        for sample in train_set.iter() {
            let target = if sample.identical {
                PackedSignal::pack(1, 127, 1)   // positive = identical
            } else {
                PackedSignal::pack(-1, 127, 1)   // negative = different
            };
            net.train_step(&sample.vec_a, &sample.vec_b, target);
        }

        // Pressure decay once per epoch (output layer only)
        net.ms_out.decay();

        let result = evaluate_network(&net, test_set);
        if result.accuracy > best_accuracy {
            best_accuracy = result.accuracy;
        }
        println!("Cycle {epoch}: {result}");

        if first_99.is_none() && result.accuracy >= 99.0 {
            first_99 = Some(epoch);
        }

        if result.accuracy >= 99.5 {
            println!("\nTarget accuracy 99.5% reached at cycle {epoch}!");
            break;
        }
    }

    if let Some(e) = first_99 {
        println!("First 99%+ accuracy at cycle {e}");
    }
    println!("Best accuracy: {best_accuracy:.1}%");

    // Print mastery statistics (output layer only — hidden layers are fixed projections)
    println!("  w_out: {} steps, {} transitions", net.ms_out.steps, net.ms_out.transitions);

    // Save thermogram
    let trained_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("trained");
    std::fs::create_dir_all(&trained_dir).unwrap();
    let thermo_path = trained_dir.join("compare.thermo");
    println!("\nSaving thermogram to {:?}...", thermo_path);
    let thermo = net.save_to_thermogram();
    thermo.save(&thermo_path).unwrap();

    // Pre-save accuracy
    let pre_save = evaluate_network(&net, test_set);
    println!("Pre-save accuracy: {pre_save}");

    // Load thermogram and rebuild network
    println!("\nLoading thermogram and verifying...");
    let loaded_thermo = Thermogram::load(&thermo_path).unwrap();
    let loaded_delta = loaded_thermo.dirty_chain.deltas.iter()
        .find(|d| d.key == "compare.w_out")
        .expect("w_out delta not found in thermogram");
    let loaded_w_out = WeightMatrix::from_data(
        loaded_delta.value.clone(), 1, 32,
    ).unwrap();
    let loaded_net = CompareNetwork {
        w_out: loaded_w_out,
        ms_out: MasteryState::new(32, MasteryConfig::default()),
    };

    let post_load = evaluate_network(&loaded_net, test_set);
    println!("Post-load accuracy: {post_load}");

    // Verify byte-identical outputs
    println!("\nVerifying byte-identical outputs...");
    let mut mismatches = 0;
    for sample in test_set.iter().take(50) {
        let net_out = net.forward(&sample.vec_a, &sample.vec_b);
        let loaded_out = loaded_net.forward(&sample.vec_a, &sample.vec_b);
        if net_out.as_u8() != loaded_out.as_u8() {
            mismatches += 1;
        }
    }
    println!(
        "Byte-identical after reload: {} (mismatches: {}/50)",
        if mismatches == 0 { "YES" } else { "NO" },
        mismatches
    );

    println!("\n=== CompareANT Mastery Training Complete ===");
}
