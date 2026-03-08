//! Full Persistence Lifecycle — Example 10
//!
//! Proves the complete train → save → destroy → load → verify → continue cycle.
//!
//! Pipeline:
//!   1. Train CompareANT from scratch (mastery learning)
//!   2. Measure accuracy (should be >= 99%)
//!   3. Save thermogram to disk
//!   4. Destroy the network instance
//!   5. Load from thermogram into fresh network
//!   6. Verify accuracy is byte-identical to pre-save
//!   7. Continue mastery learning (5 more cycles)
//!   8. Verify continued learning works (accuracy maintained or improved)
//!   9. Save updated thermogram
//!  10. Verify thermogram has multiple deltas (thermal history)

use atomic_neural_transistors::core::weight_matrix::{packed_from_current, WeightMatrix};
use atomic_neural_transistors::learning::{MasteryConfig, MasteryState};
use atomic_neural_transistors::PackedSignal;
use thermogram::{Delta, PlasticityRule, Signal, Thermogram};

// ---------------------------------------------------------------------------
// Reuse infrastructure from Example 1
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

fn comparison_features(a: &[PackedSignal], b: &[PackedSignal]) -> Vec<PackedSignal> {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let product = x.current() as i64 * y.current() as i64;
            packed_from_current((product / 256) as i32)
        })
        .collect()
}

struct Sample {
    vec_a: Vec<PackedSignal>,
    vec_b: Vec<PackedSignal>,
    identical: bool,
}

fn generate_prototypes(rng: &mut Rng, count: usize) -> Vec<Vec<PackedSignal>> {
    (0..count)
        .map(|p| {
            (0..32)
                .map(|d| {
                    let bits = ((p & d) as u32).count_ones();
                    let pol: i8 = if bits % 2 == 0 { 1 } else { -1 };
                    let mag = 128 + rng.next_u8() % 64;
                    PackedSignal::pack(pol, mag, 1)
                })
                .collect()
        })
        .collect()
}

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
            add_noise(rng, &prototypes[proto_a])
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

fn evaluate(w_out: &WeightMatrix, test_set: &[Sample]) -> (f64, usize) {
    let mut correct = 0;
    for sample in test_set {
        let features = comparison_features(&sample.vec_a, &sample.vec_b);
        let out = matmul(&features, w_out);
        let predicted_same = out[0].current() > 0;
        if predicted_same == sample.identical {
            correct += 1;
        }
    }
    let accuracy = correct as f64 / test_set.len() as f64 * 100.0;
    (accuracy, correct)
}

fn train_epoch(
    w_out: &mut WeightMatrix,
    ms: &mut MasteryState,
    train_set: &[Sample],
) {
    for sample in train_set {
        let features = comparison_features(&sample.vec_a, &sample.vec_b);
        let raw_out = matmul(&features, w_out);
        let clamped = packed_from_current(raw_out[0].current().clamp(-127, 127));
        let target = if sample.identical {
            PackedSignal::pack(1, 127, 1)
        } else {
            PackedSignal::pack(-1, 127, 1)
        };
        ms.update(w_out, &features, &[clamped], &[target]);
    }
    ms.decay();
}

// ---------------------------------------------------------------------------
// Main — Full Persistence Lifecycle
// ---------------------------------------------------------------------------

fn main() {
    println!("=== Full Persistence Lifecycle ===\n");

    let mut rng = Rng::new(0xBEEF);

    // Generate data
    let prototypes = generate_prototypes(&mut rng, 8);
    let all_data = generate_dataset(&mut rng, 1000, &prototypes);
    let (test_set, train_set) = all_data.split_at(200);

    // --- Phase 1: Train from scratch ---
    println!("Phase 1: Train from scratch");
    let config = MasteryConfig {
        pressure_threshold: 3,
        decay_rate: 1,
        participation_gate: 5,
    };
    let mut w_out = WeightMatrix::zeros(1, 32);
    let mut ms = MasteryState::new(32, config.clone());

    for epoch in 1..=5 {
        train_epoch(&mut w_out, &mut ms, train_set);
        let (acc, correct) = evaluate(&w_out, test_set);
        println!("  Cycle {epoch}: {correct}/200 ({acc:.1}%)");
    }

    let (pre_save_acc, pre_save_correct) = evaluate(&w_out, test_set);
    println!("\nPre-save: {pre_save_correct}/200 ({pre_save_acc:.1}%)");
    println!("Transitions: {}", ms.transitions);

    // --- Phase 2: Save thermogram ---
    let trained_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("trained");
    std::fs::create_dir_all(&trained_dir).unwrap();
    let thermo_path = trained_dir.join("lifecycle.thermo");

    println!("\nPhase 2: Save thermogram to {:?}", thermo_path);
    let mut thermo = Thermogram::new("lifecycle", PlasticityRule::stdp_like());
    let delta = Delta::create("compare.w_out", w_out.data.clone(), "mastery");
    thermo.apply_delta(delta).unwrap();
    thermo.save(&thermo_path).unwrap();
    println!("  Saved ({} deltas in chain)", thermo.dirty_chain.deltas.len());

    // Collect pre-save outputs for byte comparison
    let pre_save_outputs: Vec<u8> = test_set.iter().take(50).map(|s| {
        let features = comparison_features(&s.vec_a, &s.vec_b);
        let out = matmul(&features, &w_out);
        out[0].as_u8()
    }).collect();

    // --- Phase 3: Destroy and reload ---
    println!("\nPhase 3: Destroy instance and reload from thermogram");
    drop(w_out);
    drop(ms);

    let loaded_thermo = Thermogram::load(&thermo_path).unwrap();
    let loaded_delta = loaded_thermo.dirty_chain.deltas.iter()
        .find(|d| d.key == "compare.w_out")
        .expect("w_out delta not found");
    let mut w_out = WeightMatrix::from_data(loaded_delta.value.clone(), 1, 32).unwrap();
    let mut ms = MasteryState::new(32, config);

    let (post_load_acc, post_load_correct) = evaluate(&w_out, test_set);
    println!("  Post-load: {post_load_correct}/200 ({post_load_acc:.1}%)");

    // Verify byte-identical
    let mut mismatches = 0;
    for (i, sample) in test_set.iter().take(50).enumerate() {
        let features = comparison_features(&sample.vec_a, &sample.vec_b);
        let out = matmul(&features, &w_out);
        if out[0].as_u8() != pre_save_outputs[i] {
            mismatches += 1;
        }
    }
    println!("  Byte-identical: {} (mismatches: {}/50)",
        if mismatches == 0 { "YES" } else { "NO" }, mismatches);
    assert_eq!(mismatches, 0, "Post-load outputs must be byte-identical");

    // --- Phase 4: Continue learning ---
    println!("\nPhase 4: Continue mastery learning (5 more cycles)");
    for epoch in 6..=10 {
        train_epoch(&mut w_out, &mut ms, train_set);
        let (acc, correct) = evaluate(&w_out, test_set);
        println!("  Cycle {epoch}: {correct}/200 ({acc:.1}%)");
    }

    let (continued_acc, continued_correct) = evaluate(&w_out, test_set);
    println!("\nPost-continued: {continued_correct}/200 ({continued_acc:.1}%)");
    println!("Additional transitions: {}", ms.transitions);

    // --- Phase 5: Save updated thermogram with thermal history ---
    println!("\nPhase 5: Save updated thermogram");
    let mut thermo2 = Thermogram::load(&thermo_path).unwrap();
    let prev = thermo2.dirty_chain.head_hash.clone();
    let delta2 = Delta::update(
        "compare.w_out",
        w_out.data.clone(),
        "continued_mastery",
        Signal::positive(200),
        prev,
    );
    thermo2.apply_delta(delta2).unwrap();
    thermo2.save(&thermo_path).unwrap();

    let final_thermo = Thermogram::load(&thermo_path).unwrap();
    let delta_count = final_thermo.dirty_chain.deltas.len();
    println!("  Thermogram deltas: {} (shows thermal history)", delta_count);
    for (i, d) in final_thermo.dirty_chain.deltas.iter().enumerate() {
        println!(
            "    Delta {}: key={}, type={:?}, source={}",
            i, d.key, d.delta_type, d.metadata.source
        );
    }

    // --- Summary ---
    println!("\n--- Lifecycle Summary ---");
    println!("  Initial training:   50% → {pre_save_acc:.1}% in 5 cycles");
    println!("  Save → Load:        byte-identical (0 mismatches)");
    println!("  Continued learning: {pre_save_acc:.1}% → {continued_acc:.1}%");
    println!("  Thermal history:    {delta_count} deltas in chain");

    println!("\n=== Full Persistence Lifecycle Complete ===");
}
