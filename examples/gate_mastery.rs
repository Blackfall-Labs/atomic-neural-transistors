//! GateANT Signal Gating — Example 4
//!
//! Trains a signal gate that learns to pass or block individual signal
//! dimensions based on a control context. The gate learns which dimensions
//! of the signal to preserve and which to suppress.
//!
//! Architecture: frozen 16×64 hidden → ReLU → mastery-trained 32×16 output → sigmoid
//!
//! Task: Given a signal and a control pattern, learn to gate (multiply)
//! the signal by learned gate values. The control selects which "mode"
//! of gating to apply.
//!
//! Evaluation: gated output should preserve signal where gate ≈ 1 and
//! suppress where gate ≈ 0, measured by correlation with expected output.

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

/// Sigmoid approximation: clamp to [0, 255] range
/// Maps negative → 0, zero → 128, positive → 255
fn sigmoid(signals: &[PackedSignal]) -> Vec<PackedSignal> {
    signals.iter().map(|s| {
        let v = s.current();
        let out = if v <= -128 { 0 }
            else if v >= 127 { 255 }
            else { (v + 128) as i32 };
        packed_from_current(out)
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

/// Generate gate masks — each control pattern selects which dims to pass.
/// Mask is binary: pass (127) or block (0) for each of 32 dims.
fn generate_gate_masks(count: usize) -> Vec<Vec<PackedSignal>> {
    (0..count).map(|pattern| {
        (0..32).map(|d| {
            // Hadamard-like: pass if popcount(pattern & d) is even
            let bits = ((pattern & d) as u32).count_ones();
            if bits % 2 == 0 {
                PackedSignal::pack(1, 127, 1) // pass
            } else {
                PackedSignal::ZERO // block
            }
        }).collect()
    }).collect()
}

struct GateNetwork {
    w_hidden: WeightMatrix,  // 16 × 64 (frozen)
    w_out: WeightMatrix,     // 32 × 16 (learned)
    ms_out: MasteryState,
}

impl GateNetwork {
    fn new(rng: &mut Rng) -> Self {
        let config = MasteryConfig { pressure_threshold: 3, decay_rate: 1, participation_gate: 5 };
        let hidden_data: Vec<PackedSignal> = (0..16 * 64).map(|_| {
            let mag = 20 + (rng.next_u8() % 21);
            let pol: i8 = if rng.next() & 1 == 0 { 1 } else { -1 };
            PackedSignal::pack(pol, mag, 1)
        }).collect();
        let w_hidden = WeightMatrix::from_data(hidden_data, 16, 64).unwrap();
        let w_out = WeightMatrix::zeros(32, 16);
        Self { w_hidden, ms_out: MasteryState::new(32 * 16, config), w_out }
    }

    fn forward(&self, signal: &[PackedSignal], control: &[PackedSignal]) -> Vec<PackedSignal> {
        let mut input = Vec::with_capacity(64);
        input.extend_from_slice(signal);
        input.extend_from_slice(control);
        let h = relu(&matmul(&input, &self.w_hidden));
        let raw = matmul(&h, &self.w_out);
        sigmoid(&raw)
    }

    fn train_step(&mut self, signal: &[PackedSignal], control: &[PackedSignal], target: &[PackedSignal]) {
        let mut input = Vec::with_capacity(64);
        input.extend_from_slice(signal);
        input.extend_from_slice(control);
        let h = relu(&matmul(&input, &self.w_hidden));
        let raw = matmul(&h, &self.w_out);
        let gate = sigmoid(&raw);
        let clamped: Vec<PackedSignal> = gate.iter()
            .map(|s| packed_from_current(s.current().clamp(0, 255)))
            .collect();
        self.ms_out.update(&mut self.w_out, &h, &clamped, target);
    }
}

struct Sample {
    signal: Vec<PackedSignal>,
    control: Vec<PackedSignal>,
    target_gate: Vec<PackedSignal>, // expected gate output (0 or 255)
}

fn generate_dataset(
    rng: &mut Rng, n: usize,
    signal_protos: &[Vec<PackedSignal>],
    gate_masks: &[Vec<PackedSignal>],
    control_protos: &[Vec<PackedSignal>],
) -> Vec<Sample> {
    let n_sig = signal_protos.len();
    let n_ctrl = control_protos.len();
    (0..n).map(|_| {
        let sig_idx = (rng.next() as usize) % n_sig;
        let ctrl_idx = (rng.next() as usize) % n_ctrl;
        let signal = add_noise(rng, &signal_protos[sig_idx]);
        let control = add_noise(rng, &control_protos[ctrl_idx]);
        // Target gate: the mask for this control pattern
        // Map to sigmoid range: pass → 255, block → 0
        let target_gate: Vec<PackedSignal> = gate_masks[ctrl_idx].iter().map(|m| {
            if m.current() > 0 {
                packed_from_current(255) // pass
            } else {
                packed_from_current(0) // block
            }
        }).collect();
        Sample { signal, control, target_gate }
    }).collect()
}

fn evaluate(net: &GateNetwork, test_set: &[Sample]) -> f64 {
    let mut total_correct = 0;
    let mut total_dims = 0;
    for s in test_set {
        let gate = net.forward(&s.signal, &s.control);
        for (g, t) in gate.iter().zip(s.target_gate.iter()) {
            // Correct if both high (> 128) or both low (<= 128)
            let g_pass = g.current() > 128;
            let t_pass = t.current() > 128;
            if g_pass == t_pass { total_correct += 1; }
            total_dims += 1;
        }
    }
    total_correct as f64 / total_dims as f64 * 100.0
}

fn main() {
    println!("=== GateANT Mastery Training ===\n");
    let mut rng = Rng::new(0x6A7E);

    // 4 control patterns → 4 different gate masks
    let signal_protos = generate_prototypes(&mut rng, 8);
    let control_protos = generate_prototypes(&mut rng, 4);
    let gate_masks = generate_gate_masks(4);

    let all_data = generate_dataset(&mut rng, 1000, &signal_protos, &gate_masks, &control_protos);
    let (test_set, train_set) = all_data.split_at(200);
    println!("Data: {} training, {} test\n", train_set.len(), test_set.len());

    let mut net = GateNetwork::new(&mut rng);

    let max_epochs = 30;
    let mut best_acc = 0.0f64;

    for epoch in 1..=max_epochs {
        for s in train_set {
            net.train_step(&s.signal, &s.control, &s.target_gate);
        }
        net.ms_out.decay();

        let acc = evaluate(&net, test_set);
        if acc > best_acc { best_acc = acc; }
        println!("Cycle {epoch}: {acc:.1}% per-dim gate accuracy");
        if acc >= 95.0 {
            println!("\nTarget 95%+ reached at cycle {epoch}!");
            break;
        }
    }
    println!("Best accuracy: {best_acc:.1}%");
    println!("w_out: {} steps, {} transitions", net.ms_out.steps, net.ms_out.transitions);

    // Save thermogram
    let trained_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("trained");
    std::fs::create_dir_all(&trained_dir).unwrap();
    let thermo_path = trained_dir.join("gate.thermo");
    let mut thermo = Thermogram::new("gate", PlasticityRule::stdp_like());
    thermo.apply_delta(Delta::create("gate.w_hidden", net.w_hidden.data.clone(), "mastery")).unwrap();
    let prev = thermo.dirty_chain.head_hash.clone();
    thermo.apply_delta(Delta::update("gate.w_out", net.w_out.data.clone(), "mastery",
        thermogram::Signal::positive(255), prev)).unwrap();
    thermo.save(&thermo_path).unwrap();
    println!("\nSaved thermogram to {:?}", thermo_path);

    println!("\n=== GateANT Mastery Training Complete ===");
}
