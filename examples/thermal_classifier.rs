//! Thermal classifier — proves per-weight temperature gating works.
//!
//! Same 4-class problem as classifier_mastery.rs, but with thermal weights
//! on BOTH layers. Compares:
//!   A) Frozen hidden + learned output (existing approach)
//!   B) All-thermal: both layers learn, temperature gates plasticity
//!
//! Test includes a "domain shift" — test data has different noise characteristics
//! than training data, simulating cross-speaker variation. Frozen hidden should
//! degrade on shifted data. Thermal should adapt because hidden weights that
//! capture domain-invariant features cool down while domain-specific ones stay hot.

use atomic_neural_transistors::core::weight_matrix::{packed_from_current, WeightMatrix, relu_packed};
use atomic_neural_transistors::core::thermal::{ThermalWeightMatrix, ThermalMasteryConfig};
use atomic_neural_transistors::thermal_mastery::ThermalMasteryState;
use atomic_neural_transistors::learning::{MasteryConfig, MasteryState};
use atomic_neural_transistors::PackedSignal;

// ---------------------------------------------------------------------------
// PRNG
// ---------------------------------------------------------------------------

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Self(seed) }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0
    }
    fn next_u8(&mut self) -> u8 { (self.next() & 0xFF) as u8 }
    fn next_i32(&mut self, range: i32) -> i32 { (self.next() % range as u64) as i32 }
}

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

struct Sample { signal: Vec<PackedSignal>, class: usize }

fn generate_prototypes(rng: &mut Rng) -> Vec<Vec<PackedSignal>> {
    (0..4).map(|class| {
        (0..32).map(|d| {
            let bits = ((class & d) as u32).count_ones();
            let pol: i8 = if bits % 2 == 0 { 1 } else { -1 };
            let mag = 128 + rng.next_u8() % 64;
            PackedSignal::pack(pol, mag, 1)
        }).collect()
    }).collect()
}

/// Normal noise (small perturbation).
fn add_noise(rng: &mut Rng, proto: &[PackedSignal]) -> Vec<PackedSignal> {
    proto.iter().map(|s| {
        let noise = rng.next_i32(81) - 40; // [-40, 40]
        packed_from_current(s.current().saturating_add(noise))
    }).collect()
}

/// Domain-shifted noise — simulates cross-speaker variation.
/// Rotates feature pairs (swaps adjacent dimensions with mixing) AND
/// applies different magnitude scaling. This breaks frozen random projections
/// because the hidden layer sees a fundamentally different input distribution.
fn add_shifted_noise(rng: &mut Rng, proto: &[PackedSignal]) -> Vec<PackedSignal> {
    let mut result: Vec<PackedSignal> = proto.iter().map(|s| {
        let noise = rng.next_i32(81) - 40;
        packed_from_current(s.current().saturating_add(noise))
    }).collect();

    // Rotate adjacent dimension pairs: mix dim[i] and dim[i+1]
    // This simulates speaker-dependent spectral tilt / formant shift
    for i in (0..result.len() - 1).step_by(2) {
        let a = result[i].current() as i64;
        let b = result[i + 1].current() as i64;
        // Rotation: a' = 0.7a + 0.3b, b' = -0.3a + 0.7b (approximated in integer)
        let a_new = (a * 7 + b * 3) / 10;
        let b_new = (-a * 3 + b * 7) / 10;
        result[i] = packed_from_current(a_new as i32);
        result[i + 1] = packed_from_current(b_new as i32);
    }

    // Scale odd dimensions by 1.5×, even by 0.7× (spectral tilt)
    for (i, s) in result.iter_mut().enumerate() {
        let c = s.current() as i64;
        let scaled = if i % 2 == 0 { c * 7 / 10 } else { c * 15 / 10 };
        *s = packed_from_current(scaled as i32);
    }

    result
}

fn generate_train(rng: &mut Rng, n: usize, protos: &[Vec<PackedSignal>]) -> Vec<Sample> {
    (0..n).map(|i| {
        let class = i % protos.len();
        Sample { signal: add_noise(rng, &protos[class]), class }
    }).collect()
}

fn generate_test_normal(rng: &mut Rng, n: usize, protos: &[Vec<PackedSignal>]) -> Vec<Sample> {
    (0..n).map(|i| {
        let class = i % protos.len();
        Sample { signal: add_noise(rng, &protos[class]), class }
    }).collect()
}

fn generate_test_shifted(rng: &mut Rng, n: usize, protos: &[Vec<PackedSignal>]) -> Vec<Sample> {
    (0..n).map(|i| {
        let class = i % protos.len();
        Sample { signal: add_shifted_noise(rng, &protos[class]), class }
    }).collect()
}

// ---------------------------------------------------------------------------
// Network A: Frozen hidden + learned output (baseline)
// ---------------------------------------------------------------------------

fn matmul(input: &[PackedSignal], w: &WeightMatrix) -> Vec<PackedSignal> {
    w.matmul(input)
}

struct FrozenNetwork {
    w_hidden: WeightMatrix,
    w_out: WeightMatrix,
    ms_out: MasteryState,
}

impl FrozenNetwork {
    fn new(rng: &mut Rng) -> Self {
        let hidden_data: Vec<PackedSignal> = (0..24 * 32).map(|_| {
            let mag = 20 + (rng.next_u8() % 21);
            let pol: i8 = if rng.next() & 1 == 0 { 1 } else { -1 };
            PackedSignal::pack(pol, mag, 1)
        }).collect();
        Self {
            w_hidden: WeightMatrix::from_data(hidden_data, 24, 32).unwrap(),
            w_out: WeightMatrix::zeros(4, 24),
            ms_out: MasteryState::new(96, MasteryConfig::default()),
        }
    }

    fn forward(&self, input: &[PackedSignal]) -> Vec<PackedSignal> {
        let h = relu_packed(&matmul(input, &self.w_hidden));
        matmul(&h, &self.w_out)
    }

    fn predict(&self, input: &[PackedSignal]) -> usize {
        self.forward(input).iter().enumerate()
            .max_by_key(|(_, s)| s.current()).map(|(i, _)| i).unwrap_or(0)
    }

    fn train_step(&mut self, input: &[PackedSignal], class: usize) {
        let h = relu_packed(&matmul(input, &self.w_hidden));
        let out = matmul(&h, &self.w_out);
        let target: Vec<PackedSignal> = (0..4).map(|c| {
            if c == class { PackedSignal::pack(1, 127, 1) }
            else { PackedSignal::pack(-1, 127, 1) }
        }).collect();
        let clamped: Vec<PackedSignal> = out.iter()
            .map(|s| packed_from_current(s.current().clamp(-127, 127))).collect();
        self.ms_out.update(&mut self.w_out, &h, &clamped, &target);
    }
}

// ---------------------------------------------------------------------------
// Network B: All-thermal (both layers learn)
// ---------------------------------------------------------------------------

struct ThermalNetwork {
    w_hidden: ThermalWeightMatrix,
    w_out: ThermalWeightMatrix,
    ms_hidden: ThermalMasteryState,
    ms_out: ThermalMasteryState,
}

impl ThermalNetwork {
    fn new(seed: u64) -> Self {
        let hidden_config = ThermalMasteryConfig {
            pressure_threshold: 5,  // higher for hidden stability
            participation_gate: 10,
            cooling_rate: 50,       // cools faster for this small problem
            ..Default::default()
        };
        let out_config = ThermalMasteryConfig {
            pressure_threshold: 3,
            participation_gate: 5,
            cooling_rate: 50,
            ..Default::default()
        };

        Self {
            w_hidden: ThermalWeightMatrix::random_hot(24, 32, seed),
            w_out: ThermalWeightMatrix::zeros(4, 24),
            ms_hidden: ThermalMasteryState::new(hidden_config),
            ms_out: ThermalMasteryState::new(out_config),
        }
    }

    fn forward(&self, input: &[PackedSignal]) -> Vec<PackedSignal> {
        let h = relu_packed(&self.w_hidden.matmul(input));
        self.w_out.matmul(&h)
    }

    fn predict(&self, input: &[PackedSignal]) -> usize {
        self.forward(input).iter().enumerate()
            .max_by_key(|(_, s)| s.current()).map(|(i, _)| i).unwrap_or(0)
    }

    fn train_step(&mut self, input: &[PackedSignal], class: usize) {
        let h = relu_packed(&self.w_hidden.matmul(input));
        let out = self.w_out.matmul(&h);

        let target_out: Vec<PackedSignal> = (0..4).map(|c| {
            if c == class { PackedSignal::pack(1, 127, 1) }
            else { PackedSignal::pack(-1, 127, 1) }
        }).collect();
        let clamped_out: Vec<PackedSignal> = out.iter()
            .map(|s| packed_from_current(s.current().clamp(-127, 127))).collect();

        let predicted = out.iter().enumerate()
            .max_by_key(|(_, s)| s.current()).map(|(i, _)| i).unwrap_or(0);
        let correct = predicted == class;

        // Output layer mastery
        self.ms_out.update(&mut self.w_out, &h, &clamped_out, &target_out, correct);

        // Hidden layer mastery: target derived from output ERROR × output weights.
        // Only adjust hidden weights in proportion to how wrong the output is.
        // If output is already correct (error≈0), hidden target ≈ current hidden → no change.
        let errors: Vec<i64> = (0..4).map(|c| {
            target_out[c].current() as i64 - clamped_out[c].current() as i64
        }).collect();

        let target_hidden: Vec<PackedSignal> = (0..24).map(|i| {
            let h_cur = h[i].current() as i64;
            // Nudge = sum of (error × output_weight_sign) / num_outputs.
            // Proportional to how wrong we are. Larger errors push harder.
            let mut nudge: i64 = 0;
            for c in 0..4 {
                let w_sign = self.w_out.get(c, i).signal.current().signum() as i64;
                // Scale error to a manageable range: divide by 4 to dampen
                let scaled_error = errors[c] / 4;
                nudge += scaled_error * w_sign;
            }
            // Clamp nudge so hidden layer moves steadily, not violently
            let target = h_cur + nudge.clamp(-32, 32);
            packed_from_current(target.clamp(-127, 127) as i32)
        }).collect();
        let clamped_hidden: Vec<PackedSignal> = h.iter()
            .map(|s| packed_from_current(s.current().clamp(-127, 127))).collect();

        self.ms_hidden.update(&mut self.w_hidden, input, &clamped_hidden, &target_hidden, correct);
    }
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

fn eval_accuracy(predict: &dyn Fn(&[PackedSignal]) -> usize, data: &[Sample]) -> f64 {
    let correct = data.iter().filter(|s| predict(&s.signal) == s.class).count();
    correct as f64 / data.len() as f64 * 100.0
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("=== Thermal Classifier: Frozen vs Thermal Weights ===\n");

    let mut rng = Rng::new(0xDEAD);
    let protos = generate_prototypes(&mut rng);

    let train = generate_train(&mut rng, 800, &protos);
    let test_normal = generate_test_normal(&mut rng, 200, &protos);
    let test_shifted = generate_test_shifted(&mut rng, 200, &protos);

    println!("Data: {} train, {} test (normal), {} test (shifted)\n",
        train.len(), test_normal.len(), test_shifted.len());

    // Network A: Frozen
    let mut frozen = FrozenNetwork::new(&mut rng);
    // Network B: Thermal
    let mut thermal = ThermalNetwork::new(42);

    println!("{:>6}  {:>10}  {:>10}  {:>10}  {:>10}  {:>12}",
        "Cycle", "Frz-Norm", "Frz-Shift", "Thm-Norm", "Thm-Shift", "Thm-Temp");

    // Phase 1: Train on normal data
    println!("\n--- Phase 1: Training on normal noise ---");
    for cycle in 0..10 {
        for s in &train {
            frozen.train_step(&s.signal, s.class);
            thermal.train_step(&s.signal, s.class);
        }
        frozen.ms_out.decay();
        thermal.ms_out.decay(&mut thermal.w_out);
        thermal.ms_hidden.decay(&mut thermal.w_hidden);

        let fn_acc = eval_accuracy(&|x| frozen.predict(x), &test_normal);
        let fs_acc = eval_accuracy(&|x| frozen.predict(x), &test_shifted);
        let tn_acc = eval_accuracy(&|x| thermal.predict(x), &test_normal);
        let ts_acc = eval_accuracy(&|x| thermal.predict(x), &test_shifted);
        let (h, w, c, d) = thermal.w_hidden.temp_summary();

        println!("{:>6}  {:>9.1}%  {:>9.1}%  {:>9.1}%  {:>9.1}%  H={} W={} C={} D={}",
            cycle + 1, fn_acc, fs_acc, tn_acc, ts_acc, h, w, c, d);
    }

    // Phase 2: Train on SHIFTED data (domain adaptation)
    println!("\n--- Phase 2: Adapting to shifted noise ---");
    let train_shifted = generate_test_shifted(&mut rng, 800, &protos);
    for cycle in 0..10 {
        for s in &train_shifted {
            frozen.train_step(&s.signal, s.class);
            thermal.train_step(&s.signal, s.class);
        }
        frozen.ms_out.decay();
        thermal.ms_out.decay(&mut thermal.w_out);
        thermal.ms_hidden.decay(&mut thermal.w_hidden);

        let fn_acc = eval_accuracy(&|x| frozen.predict(x), &test_normal);
        let fs_acc = eval_accuracy(&|x| frozen.predict(x), &test_shifted);
        let tn_acc = eval_accuracy(&|x| thermal.predict(x), &test_normal);
        let ts_acc = eval_accuracy(&|x| thermal.predict(x), &test_shifted);
        let (h, w, c, d) = thermal.w_hidden.temp_summary();

        println!("{:>6}  {:>9.1}%  {:>9.1}%  {:>9.1}%  {:>9.1}%  H={} W={} C={} D={}",
            cycle + 11, fn_acc, fs_acc, tn_acc, ts_acc, h, w, c, d);
    }

    // Phase 3: Save/load roundtrip
    println!("\n--- Phase 3: Persistence ---");
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("trained");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("thermal_classifier.ant");

    ThermalWeightMatrix::save_multi(
        &[&thermal.w_hidden, &thermal.w_out], &path
    ).unwrap();
    println!("Saved to {:?}", path);

    let loaded = ThermalWeightMatrix::load_multi(&path).unwrap();
    let loaded_net = ThermalNetwork {
        w_hidden: loaded[0].clone(),
        w_out: loaded[1].clone(),
        ms_hidden: ThermalMasteryState::new(ThermalMasteryConfig::default()),
        ms_out: ThermalMasteryState::new(ThermalMasteryConfig::default()),
    };

    let pre = eval_accuracy(&|x| thermal.predict(x), &test_shifted);
    let post = eval_accuracy(&|x| loaded_net.predict(x), &test_shifted);
    println!("Pre-save:  {:.1}%", pre);
    println!("Post-load: {:.1}%", post);
    println!("Match: {}", if (pre - post).abs() < 0.1 { "YES" } else { "NO" });

    // Summary
    println!("\n--- Summary ---");
    let (h, w, c, d) = thermal.w_hidden.temp_summary();
    println!("Thermal hidden weights: HOT={} WARM={} COOL={} COLD={}", h, w, c, d);
    println!("Thermal hidden transitions: {}", thermal.ms_hidden.transitions);
    println!("Thermal output transitions: {}", thermal.ms_out.transitions);
    println!("Frozen output transitions:  {}", frozen.ms_out.transitions);

    println!("\n=== Done ===");
}
