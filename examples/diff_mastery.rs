//! DiffANT Change Detection — Example 3
//!
//! Trains a difference detector on 32-dim ternary signal pairs using
//! mastery learning. Detects whether two signals come from different
//! prototypes — the inverse of CompareANT's similarity detection.
//!
//! Architecture: product_features(a, b) → mastery-trained 1×32 output
//!
//! Key difference from CompareANT (Ex 1): Same product feature architecture,
//! but with inverted targets. CompareANT: identical→positive, different→negative.
//! DiffANT: changed→positive, unchanged→negative. This proves the mastery
//! mechanism works for both polarity directions on the same feature space.
//!
//! Evaluation: output > 0 = "changed"

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

/// Product features: p[j] = a[j] * b[j] / 256
/// Same prototype → mostly positive (same polarities)
/// Different prototype → mixed signs (polarity disagreements)
fn product_features(a: &[PackedSignal], b: &[PackedSignal]) -> Vec<PackedSignal> {
    a.iter().zip(b.iter()).map(|(x, y)| {
        let product = x.current() as i64 * y.current() as i64;
        packed_from_current((product / 256) as i32)
    }).collect()
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

struct DiffNetwork {
    w_out: WeightMatrix, // 1 × 32 (learned)
    ms_out: MasteryState,
}

impl DiffNetwork {
    fn new() -> Self {
        let config = MasteryConfig { pressure_threshold: 3, decay_rate: 1, participation_gate: 5 };
        let w_out = WeightMatrix::zeros(1, 32);
        Self { ms_out: MasteryState::new(32, config), w_out }
    }

    fn forward(&self, a: &[PackedSignal], b: &[PackedSignal]) -> PackedSignal {
        let features = product_features(a, b);
        matmul(&features, &self.w_out)[0]
    }

    fn train_step(&mut self, a: &[PackedSignal], b: &[PackedSignal], target: PackedSignal) {
        let features = product_features(a, b);
        let raw_out = matmul(&features, &self.w_out);
        let clamped = packed_from_current(raw_out[0].current().clamp(-127, 127));
        self.ms_out.update(&mut self.w_out, &features, &[clamped], &[target]);
    }

    fn predict_changed(&self, a: &[PackedSignal], b: &[PackedSignal]) -> bool {
        self.forward(a, b).current() > 0
    }
}

struct Sample {
    vec_a: Vec<PackedSignal>,
    vec_b: Vec<PackedSignal>,
    changed: bool,
}

fn generate_dataset(rng: &mut Rng, n: usize, prototypes: &[Vec<PackedSignal>]) -> Vec<Sample> {
    let np = prototypes.len();
    (0..n).map(|i| {
        let pa = (rng.next() as usize) % np;
        let vec_a = add_noise(rng, &prototypes[pa]);
        let changed = i % 2 != 0;
        let vec_b = if changed {
            let mut pb = (rng.next() as usize) % np;
            while pb == pa { pb = (rng.next() as usize) % np; }
            add_noise(rng, &prototypes[pb])
        } else {
            add_noise(rng, &prototypes[pa])
        };
        Sample { vec_a, vec_b, changed }
    }).collect()
}

fn main() {
    println!("=== DiffANT Mastery Training ===\n");
    let mut rng = Rng::new(0xD1FF);
    let prototypes = generate_prototypes(&mut rng, 8);
    let all_data = generate_dataset(&mut rng, 1000, &prototypes);
    let (test_set, train_set) = all_data.split_at(200);
    println!("Data: {} training, {} test\n", train_set.len(), test_set.len());

    let mut net = DiffNetwork::new();

    let max_epochs = 20;
    let mut best_acc = 0.0f64;

    for epoch in 1..=max_epochs {
        for s in train_set {
            // Inverted targets vs CompareANT:
            // changed → positive (different prototype = positive output)
            // unchanged → negative (same prototype = negative output)
            let target = if s.changed {
                PackedSignal::pack(1, 127, 1)
            } else {
                PackedSignal::pack(-1, 127, 1)
            };
            net.train_step(&s.vec_a, &s.vec_b, target);
        }
        net.ms_out.decay();

        let correct = test_set.iter().filter(|s| {
            net.predict_changed(&s.vec_a, &s.vec_b) == s.changed
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

    // Save thermogram
    let trained_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("trained");
    std::fs::create_dir_all(&trained_dir).unwrap();
    let thermo_path = trained_dir.join("diff.thermo");
    let mut thermo = Thermogram::new("diff", PlasticityRule::stdp_like());
    thermo.apply_delta(Delta::create("diff.w_out", net.w_out.data.clone(), "mastery")).unwrap();
    thermo.save(&thermo_path).unwrap();
    println!("\nSaved thermogram to {:?}", thermo_path);

    println!("\n=== DiffANT Mastery Training Complete ===");
}
