//! ClassifierANT Mastery Training — Example 2
//!
//! Trains a 4-class classifier from scratch on 32-dim ternary signal patterns
//! using mastery learning (integer-only, pressure-based, no gradients).
//!
//! Architecture (from astromind-archive production system):
//!   - Frozen random hidden projection: 32 → 24 (fixed random synaptic strengths)
//!   - Learned output synaptic matrix: 24 → 4 (mastery-trained from zero)
//!
//! Training data:
//!   - 1,000 random samples, 4 classes (250 each)
//!   - Each class has a distinct Hadamard polarity signature across 32 dims
//!   - Hold out 200 for evaluation
//!
//! Proves: multi-class mastery convergence, per-class accuracy, thermogram persistence.

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
}

// ---------------------------------------------------------------------------
// Data generation
// ---------------------------------------------------------------------------

struct Sample {
    signal: Vec<PackedSignal>,
    class: usize,
}

/// Generate 4 class prototypes using Hadamard-like polarity signatures.
/// Each class has full 32-dim coverage with a unique polarity pattern.
fn generate_class_prototypes(rng: &mut Rng) -> Vec<Vec<PackedSignal>> {
    (0..4)
        .map(|class| {
            (0..32)
                .map(|d| {
                    let bits = ((class & d) as u32).count_ones();
                    let pol: i8 = if bits % 2 == 0 { 1 } else { -1 };
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
    let num_classes = prototypes.len();
    (0..n)
        .map(|i| {
            let class = i % num_classes;
            let signal = add_noise(rng, &prototypes[class]);
            Sample { signal, class }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Network operations
// ---------------------------------------------------------------------------

fn matmul(input: &[PackedSignal], w: &WeightMatrix) -> Vec<PackedSignal> {
    let mut output = Vec::with_capacity(w.rows);
    for i in 0..w.rows {
        let mut sum: i64 = 0;
        for j in 0..w.cols {
            sum += input[j].current() as i64 * w.data[i * w.cols + j].current() as i64;
        }
        let clamped = sum.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        output.push(packed_from_current(clamped));
    }
    output
}

fn relu(signals: &[PackedSignal]) -> Vec<PackedSignal> {
    signals
        .iter()
        .map(|s| {
            if s.current() > 0 {
                *s
            } else {
                PackedSignal::ZERO
            }
        })
        .collect()
}

struct ClassifierNetwork {
    w_hidden: WeightMatrix,  // 24 × 32 (frozen random projection)
    w_out: WeightMatrix,     // 4 × 24 (trained by mastery)
    ms_out: MasteryState,
}

impl ClassifierNetwork {
    fn new(rng: &mut Rng) -> Self {
        let config = MasteryConfig {
            pressure_threshold: 3,
            decay_rate: 1,
            participation_gate: 5,
        };

        // Frozen random projection (production: ±1 polarity, 20-40 magnitude)
        let hidden_data: Vec<PackedSignal> = (0..24 * 32)
            .map(|_| {
                let mag = 20 + (rng.next_u8() % 21); // 20-40
                let pol: i8 = if rng.next() & 1 == 0 { 1 } else { -1 };
                PackedSignal::pack(pol, mag, 1)
            })
            .collect();
        let w_hidden = WeightMatrix::from_data(hidden_data, 24, 32).unwrap();

        // Output layer starts at zero — will be learned
        let w_out = WeightMatrix::zeros(4, 24);

        Self {
            w_hidden,
            ms_out: MasteryState::new(w_out.data.len(), config),
            w_out,
        }
    }

    fn forward(&self, input: &[PackedSignal]) -> Vec<PackedSignal> {
        let h = relu(&matmul(input, &self.w_hidden));
        matmul(&h, &self.w_out)
    }

    fn train_step(&mut self, input: &[PackedSignal], target_class: usize) {
        let h = relu(&matmul(input, &self.w_hidden));
        let out = matmul(&h, &self.w_out);

        // One-hot target: correct class gets +127, others get -127
        let target: Vec<PackedSignal> = (0..4)
            .map(|c| {
                if c == target_class {
                    PackedSignal::pack(1, 127, 1)
                } else {
                    PackedSignal::pack(-1, 127, 1)
                }
            })
            .collect();

        // Clamp output before error computation to prevent overshoot
        let clamped: Vec<PackedSignal> = out
            .iter()
            .map(|s| packed_from_current(s.current().clamp(-127, 127)))
            .collect();

        self.ms_out.update(&mut self.w_out, &h, &clamped, &target);
    }

    fn predict(&self, input: &[PackedSignal]) -> usize {
        let out = self.forward(input);
        out.iter()
            .enumerate()
            .max_by_key(|(_, s)| s.current())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn save_to_thermogram(&self) -> Thermogram {
        let mut thermo = Thermogram::new("classifier", PlasticityRule::stdp_like());

        let delta_hidden = Delta::create(
            "classifier.w_hidden",
            self.w_hidden.data.clone(),
            "mastery",
        );
        thermo.apply_delta(delta_hidden).unwrap();

        let prev = thermo.dirty_chain.head_hash.clone();
        let delta_out = Delta::update(
            "classifier.w_out",
            self.w_out.data.clone(),
            "mastery",
            thermogram::Signal::positive(255),
            prev,
        );
        thermo.apply_delta(delta_out).unwrap();

        thermo
    }
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

fn evaluate(net: &ClassifierNetwork, test_set: &[Sample]) -> (f64, [usize; 4], [usize; 4]) {
    let mut correct = 0;
    let mut per_class_correct = [0usize; 4];
    let mut per_class_total = [0usize; 4];

    for sample in test_set {
        let predicted = net.predict(&sample.signal);
        per_class_total[sample.class] += 1;
        if predicted == sample.class {
            correct += 1;
            per_class_correct[sample.class] += 1;
        }
    }

    let accuracy = correct as f64 / test_set.len() as f64 * 100.0;
    (accuracy, per_class_correct, per_class_total)
}

fn print_eval(label: &str, accuracy: f64, per_class_correct: &[usize; 4], per_class_total: &[usize; 4]) {
    println!(
        "{label}: {:.1}% ({}/{} correct)",
        accuracy,
        per_class_correct.iter().sum::<usize>(),
        per_class_total.iter().sum::<usize>(),
    );
    for c in 0..4 {
        println!(
            "  Class {c}: {}/{} ({:.0}%)",
            per_class_correct[c],
            per_class_total[c],
            if per_class_total[c] > 0 {
                per_class_correct[c] as f64 / per_class_total[c] as f64 * 100.0
            } else {
                0.0
            }
        );
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("=== ClassifierANT Mastery Training ===\n");

    let mut rng = Rng::new(0xDEAD);

    // Generate prototypes and data
    let prototypes = generate_class_prototypes(&mut rng);
    println!("Classes: {} with Hadamard polarity signatures", prototypes.len());
    let all_data = generate_dataset(&mut rng, 1000, &prototypes);
    let (test_set, train_set) = all_data.split_at(200);
    println!("Data: {} training, {} test\n", train_set.len(), test_set.len());

    // Create network
    let mut net = ClassifierNetwork::new(&mut rng);

    // Initial accuracy (should be ~25% random)
    let (acc, correct, total) = evaluate(&net, test_set);
    print_eval("Cycle 0 (random init)", acc, &correct, &total);

    // Training loop
    let max_epochs = 50;
    let mut first_98 = None;
    let mut best_accuracy = 0.0f64;

    for epoch in 1..=max_epochs {
        for sample in train_set.iter() {
            net.train_step(&sample.signal, sample.class);
        }

        // Pressure decay once per epoch
        net.ms_out.decay();

        let (acc, correct, total) = evaluate(&net, test_set);
        if acc > best_accuracy {
            best_accuracy = acc;
        }
        println!("Cycle {epoch}: {acc:.1}%");

        if first_98.is_none() && acc >= 98.0 {
            first_98 = Some(epoch);
        }

        if acc >= 99.0 {
            println!("\nTarget accuracy 99%+ reached at cycle {epoch}!");
            print_eval(&format!("Cycle {epoch}"), acc, &correct, &total);
            break;
        }
    }

    if let Some(e) = first_98 {
        println!("First 98%+ accuracy at cycle {e}");
    }
    println!("Best accuracy: {best_accuracy:.1}%");
    println!("  w_out: {} steps, {} transitions", net.ms_out.steps, net.ms_out.transitions);

    // Save thermogram
    let trained_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("trained");
    std::fs::create_dir_all(&trained_dir).unwrap();
    let thermo_path = trained_dir.join("classifier.thermo");
    println!("\nSaving thermogram to {:?}...", thermo_path);
    let thermo = net.save_to_thermogram();
    thermo.save(&thermo_path).unwrap();

    let (pre_acc, pre_correct, pre_total) = evaluate(&net, test_set);
    print_eval("Pre-save accuracy", pre_acc, &pre_correct, &pre_total);

    // Load thermogram and rebuild network
    println!("\nLoading thermogram and verifying...");
    let loaded_thermo = Thermogram::load(&thermo_path).unwrap();

    let loaded_hidden = loaded_thermo
        .dirty_chain
        .deltas
        .iter()
        .find(|d| d.key == "classifier.w_hidden")
        .expect("w_hidden delta not found");
    let loaded_out = loaded_thermo
        .dirty_chain
        .deltas
        .iter()
        .find(|d| d.key == "classifier.w_out")
        .expect("w_out delta not found");

    let loaded_net = ClassifierNetwork {
        w_hidden: WeightMatrix::from_data(loaded_hidden.value.clone(), 24, 32).unwrap(),
        w_out: WeightMatrix::from_data(loaded_out.value.clone(), 4, 24).unwrap(),
        ms_out: MasteryState::new(96, MasteryConfig::default()),
    };

    let (post_acc, post_correct, post_total) = evaluate(&loaded_net, test_set);
    print_eval("Post-load accuracy", post_acc, &post_correct, &post_total);

    // Verify byte-identical outputs
    println!("\nVerifying byte-identical outputs...");
    let mut mismatches = 0;
    for sample in test_set.iter().take(50) {
        let net_out = net.forward(&sample.signal);
        let loaded_out = loaded_net.forward(&sample.signal);
        for (a, b) in net_out.iter().zip(loaded_out.iter()) {
            if a.as_u8() != b.as_u8() {
                mismatches += 1;
            }
        }
    }
    println!(
        "Byte-identical after reload: {} (mismatches: {}/200)",
        if mismatches == 0 { "YES" } else { "NO" },
        mismatches
    );

    println!("\n=== ClassifierANT Mastery Training Complete ===");
}
