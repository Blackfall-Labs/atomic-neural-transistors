//! MergeANT Signal Fusion — Example 5
//!
//! Trains a signal merger that learns to combine two 32-dim ternary signals
//! into a single 32-dim output that preserves class identity from both inputs.
//!
//! Architecture: frozen 24×64 hidden → ReLU → mastery-trained 32×24 output
//!
//! Task: Given two signals from known classes (A, B), learn to produce a
//! merged output whose polarity pattern encodes the identity of both inputs.
//! The merged output is a 4-class target (class_a * 2 + class_b), evaluated
//! by comparing merge output polarity to expected merged prototype.
//!
//! Proves: mastery learning can solve multi-input fusion with frozen projections.

use atomic_neural_transistors::core::weight_matrix::{packed_from_current, WeightMatrix};
use atomic_neural_transistors::learning::{MasteryConfig, MasteryState};
use atomic_neural_transistors::PackedSignal;
use thermogram::{Delta, PlasticityRule, Thermogram};

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Self(seed) }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn next_u8(&mut self) -> u8 { (self.next() & 0xFF) as u8 }
}

fn matmul(input: &[PackedSignal], w: &WeightMatrix) -> Vec<PackedSignal> {
    let mut output = Vec::with_capacity(w.rows);
    for i in 0..w.rows {
        let mut sum: i64 = 0;
        for j in 0..w.cols {
            sum += input[j].current() as i64 * w.data[i * w.cols + j].current() as i64;
        }
        output.push(packed_from_current(sum.clamp(i32::MIN as i64, i32::MAX as i64) as i32));
    }
    output
}

fn relu(signals: &[PackedSignal]) -> Vec<PackedSignal> {
    signals.iter().map(|s| if s.current() > 0 { *s } else { PackedSignal::ZERO }).collect()
}

fn generate_prototypes(rng: &mut Rng, count: usize) -> Vec<Vec<PackedSignal>> {
    (0..count).map(|p| {
        (0..32).map(|d| {
            let bits = ((p & d) as u32).count_ones();
            let pol: i8 = if bits % 2 == 0 { 1 } else { -1 };
            PackedSignal::pack(pol, 128 + rng.next_u8() % 64, 1)
        }).collect()
    }).collect()
}

fn add_noise(rng: &mut Rng, proto: &[PackedSignal]) -> Vec<PackedSignal> {
    proto.iter().map(|s| {
        packed_from_current(s.current().saturating_add((rng.next() % 81) as i32 - 40))
    }).collect()
}

/// Generate merged target prototypes for each (class_a, class_b) pair.
/// Uses Hadamard patterns with a combined index.
fn generate_merge_targets(_rng: &mut Rng, n_classes: usize) -> Vec<Vec<PackedSignal>> {
    let n_combos = n_classes * n_classes;
    (0..n_combos).map(|combo| {
        (0..32).map(|d| {
            let bits = ((combo & d) as u32).count_ones();
            let pol: i8 = if bits % 2 == 0 { 1 } else { -1 };
            PackedSignal::pack(pol, 127, 1)
        }).collect()
    }).collect()
}

struct MergeNetwork {
    w_hidden: WeightMatrix,  // 24 × 64 (frozen)
    w_out: WeightMatrix,     // 32 × 24 (learned)
    ms_out: MasteryState,
}

impl MergeNetwork {
    fn new(rng: &mut Rng) -> Self {
        let config = MasteryConfig { pressure_threshold: 3, decay_rate: 1, participation_gate: 5 };
        let hidden_data: Vec<PackedSignal> = (0..24 * 64).map(|_| {
            let mag = 20 + (rng.next_u8() % 21);
            let pol: i8 = if rng.next() & 1 == 0 { 1 } else { -1 };
            PackedSignal::pack(pol, mag, 1)
        }).collect();
        let w_hidden = WeightMatrix::from_data(hidden_data, 24, 64).unwrap();
        let w_out = WeightMatrix::zeros(32, 24);
        Self { w_hidden, ms_out: MasteryState::new(32 * 24, config), w_out }
    }

    fn forward(&self, a: &[PackedSignal], b: &[PackedSignal]) -> Vec<PackedSignal> {
        let mut input = Vec::with_capacity(64);
        input.extend_from_slice(a);
        input.extend_from_slice(b);
        let h = relu(&matmul(&input, &self.w_hidden));
        matmul(&h, &self.w_out)
    }

    fn train_step(&mut self, a: &[PackedSignal], b: &[PackedSignal], target: &[PackedSignal]) {
        let mut input = Vec::with_capacity(64);
        input.extend_from_slice(a);
        input.extend_from_slice(b);
        let h = relu(&matmul(&input, &self.w_hidden));
        let out = matmul(&h, &self.w_out);
        let clamped: Vec<PackedSignal> = out.iter()
            .map(|s| packed_from_current(s.current().clamp(-127, 127)))
            .collect();
        self.ms_out.update(&mut self.w_out, &h, &clamped, target);
    }

    /// Classify merge output by comparing polarity pattern to all merge targets
    fn classify(&self, a: &[PackedSignal], b: &[PackedSignal], targets: &[Vec<PackedSignal>]) -> usize {
        let out = self.forward(a, b);
        targets.iter().enumerate()
            .max_by_key(|(_, target)| {
                // Dot product as similarity
                out.iter().zip(target.iter())
                    .map(|(o, t)| o.current() as i64 * t.current() as i64)
                    .sum::<i64>()
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

struct Sample {
    sig_a: Vec<PackedSignal>,
    sig_b: Vec<PackedSignal>,
    combo: usize,
}

fn generate_dataset(rng: &mut Rng, n: usize, protos: &[Vec<PackedSignal>], n_classes: usize) -> Vec<Sample> {
    (0..n).map(|_| {
        let class_a = (rng.next() as usize) % n_classes;
        let class_b = (rng.next() as usize) % n_classes;
        let sig_a = add_noise(rng, &protos[class_a]);
        let sig_b = add_noise(rng, &protos[class_b]);
        let combo = class_a * n_classes + class_b;
        Sample { sig_a, sig_b, combo }
    }).collect()
}

fn main() {
    println!("=== MergeANT Mastery Training ===\n");
    let mut rng = Rng::new(0x3E26E);
    let n_classes = 2; // 2 input classes → 4 merge combinations

    let protos = generate_prototypes(&mut rng, n_classes);
    let merge_targets = generate_merge_targets(&mut rng, n_classes);
    println!("Input classes: {}, Merge combos: {}\n", n_classes, merge_targets.len());

    let all_data = generate_dataset(&mut rng, 1000, &protos, n_classes);
    let (test_set, train_set) = all_data.split_at(200);
    println!("Data: {} training, {} test\n", train_set.len(), test_set.len());

    let mut net = MergeNetwork::new(&mut rng);

    let max_epochs = 30;
    let mut best_acc = 0.0f64;

    for epoch in 1..=max_epochs {
        for s in train_set {
            net.train_step(&s.sig_a, &s.sig_b, &merge_targets[s.combo]);
        }
        net.ms_out.decay();

        let correct = test_set.iter().filter(|s| {
            net.classify(&s.sig_a, &s.sig_b, &merge_targets) == s.combo
        }).count();
        let acc = correct as f64 / test_set.len() as f64 * 100.0;
        if acc > best_acc { best_acc = acc; }
        println!("Cycle {epoch}: {correct}/200 ({acc:.1}%)");
        if acc >= 99.0 {
            println!("\nTarget 99%+ reached at cycle {epoch}!");
            break;
        }
    }
    println!("Best accuracy: {best_acc:.1}%");
    println!("w_out: {} steps, {} transitions", net.ms_out.steps, net.ms_out.transitions);

    // Per-combo breakdown
    println!("\nPer-combo accuracy:");
    for combo in 0..merge_targets.len() {
        let combo_samples: Vec<&Sample> = test_set.iter().filter(|s| s.combo == combo).collect();
        if combo_samples.is_empty() { continue; }
        let correct = combo_samples.iter().filter(|s| {
            net.classify(&s.sig_a, &s.sig_b, &merge_targets) == s.combo
        }).count();
        let ca = combo / n_classes;
        let cb = combo % n_classes;
        println!("  ({ca},{cb}): {correct}/{} ({:.1}%)",
            combo_samples.len(), correct as f64 / combo_samples.len() as f64 * 100.0);
    }

    // Save thermogram
    let trained_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("trained");
    std::fs::create_dir_all(&trained_dir).unwrap();
    let thermo_path = trained_dir.join("merge.thermo");
    let mut thermo = Thermogram::new("merge", PlasticityRule::stdp_like());
    thermo.apply_delta(Delta::create("merge.w_hidden", net.w_hidden.data.clone(), "mastery")).unwrap();
    let prev = thermo.dirty_chain.head_hash.clone();
    thermo.apply_delta(Delta::update("merge.w_out", net.w_out.data.clone(), "mastery",
        thermogram::Signal::positive(255), prev)).unwrap();
    thermo.save(&thermo_path).unwrap();
    println!("\nSaved thermogram to {:?}", thermo_path);

    println!("\n=== MergeANT Mastery Training Complete ===");
}
