//! Composition: has_duplicate — Example 6
//!
//! Demonstrates ANT composition without retraining. A trained CompareANT
//! (from Example 1) is composed into a has_duplicate detector that checks
//! all pairs in a sequence for equality.
//!
//! Pipeline:
//!   has_duplicate(seq) = OR(CompareANT(seq[i], seq[j]) for all i < j)
//!
//! This proves the algebraic composition property: trained accuracy preserves
//! through composition without any additional mastery learning.

use atomic_neural_transistors::core::weight_matrix::{packed_from_current, WeightMatrix};
use atomic_neural_transistors::learning::{MasteryConfig, MasteryState};
use atomic_neural_transistors::PackedSignal;
use thermogram::{Delta, PlasticityRule, Thermogram};

// ---------------------------------------------------------------------------
// Reuse CompareNetwork from Example 1
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

struct CompareNetwork {
    w_out: WeightMatrix,
}

impl CompareNetwork {
    fn compare(&self, a: &[PackedSignal], b: &[PackedSignal]) -> bool {
        let features = comparison_features(a, b);
        let out = matmul(&features, &self.w_out);
        out[0].current() > 0
    }
}

// ---------------------------------------------------------------------------
// Train CompareANT (replicating Example 1 setup)
// ---------------------------------------------------------------------------

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

fn train_compare_network(rng: &mut Rng) -> CompareNetwork {
    let prototypes = generate_prototypes(rng, 8);
    let config = MasteryConfig {
        pressure_threshold: 3,
        decay_rate: 1,
        participation_gate: 5,
    };
    let mut w_out = WeightMatrix::zeros(1, 32);
    let mut ms = MasteryState::new(32, config);

    // Train for 2 epochs (converges in 1-2 with Hadamard prototypes)
    for _ in 0..2 {
        for i in 0..800 {
            let proto_a = (rng.next() as usize) % prototypes.len();
            let a = add_noise(rng, &prototypes[proto_a]);
            let identical = i % 2 == 0;
            let b = if identical {
                add_noise(rng, &prototypes[proto_a])
            } else {
                let mut proto_b = (rng.next() as usize) % prototypes.len();
                while proto_b == proto_a {
                    proto_b = (rng.next() as usize) % prototypes.len();
                }
                add_noise(rng, &prototypes[proto_b])
            };

            let features = comparison_features(&a, &b);
            let raw_out = matmul(&features, &w_out);
            let clamped = packed_from_current(raw_out[0].current().clamp(-127, 127));

            let target = if identical {
                PackedSignal::pack(1, 127, 1)
            } else {
                PackedSignal::pack(-1, 127, 1)
            };
            ms.update(&mut w_out, &features, &[clamped], &[target]);
        }
        ms.decay();
    }

    println!("CompareANT trained: {} transitions", ms.transitions);
    CompareNetwork { w_out }
}

// ---------------------------------------------------------------------------
// has_duplicate composition
// ---------------------------------------------------------------------------

fn has_duplicate(net: &CompareNetwork, sequence: &[Vec<PackedSignal>]) -> bool {
    for i in 0..sequence.len() {
        for j in (i + 1)..sequence.len() {
            if net.compare(&sequence[i], &sequence[j]) {
                return true;
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Test data generation
// ---------------------------------------------------------------------------

struct DupSample {
    sequence: Vec<Vec<PackedSignal>>,
    has_dup: bool,
}

fn generate_dup_samples(rng: &mut Rng, n: usize, prototypes: &[Vec<PackedSignal>]) -> Vec<DupSample> {
    let num_protos = prototypes.len();
    let mut samples = Vec::with_capacity(n);

    for i in 0..n {
        // Max unique length = number of prototypes (can't have more unique elements than protos)
        let max_len = num_protos.min(8);
        let seq_len = 4 + (rng.next() as usize % (max_len - 3)); // 4 to max_len
        let has_dup = i % 2 == 0;

        let mut sequence = Vec::with_capacity(seq_len);

        if has_dup {
            // Generate unique elements, then duplicate one
            let mut used_protos = Vec::new();
            for _ in 0..seq_len {
                let mut proto = (rng.next() as usize) % num_protos;
                // Allow some proto reuse for variety, but guarantee at least one dup
                used_protos.push(proto);
                sequence.push(add_noise(rng, &prototypes[proto]));
            }
            // Force a duplicate: copy element 0's prototype to a random position
            let dup_pos = 1 + (rng.next() as usize % (seq_len - 1));
            sequence[dup_pos] = add_noise(rng, &prototypes[used_protos[0]]);
        } else {
            // All unique prototypes
            let mut available: Vec<usize> = (0..num_protos).collect();
            for _ in 0..seq_len {
                if available.is_empty() {
                    // More elements than prototypes — use a random one
                    // (this shouldn't happen with 8 protos and max 9 elements)
                    let proto = (rng.next() as usize) % num_protos;
                    sequence.push(add_noise(rng, &prototypes[proto]));
                } else {
                    let idx = (rng.next() as usize) % available.len();
                    let proto = available.remove(idx);
                    sequence.push(add_noise(rng, &prototypes[proto]));
                }
            }
        }

        samples.push(DupSample { sequence, has_dup });
    }

    samples
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("=== Composition: has_duplicate ===\n");

    let mut rng = Rng::new(0xCAFE);

    // Train CompareANT
    println!("Training CompareANT...");
    let net = train_compare_network(&mut rng);

    // Generate test data
    let prototypes = generate_prototypes(&mut rng, 8);
    let samples = generate_dup_samples(&mut rng, 500, &prototypes);
    println!("Test samples: {} ({} with duplicates, {} unique)\n",
        samples.len(),
        samples.iter().filter(|s| s.has_dup).count(),
        samples.iter().filter(|s| !s.has_dup).count(),
    );

    // Evaluate
    let mut correct = 0;
    let mut false_positives = 0;
    let mut false_negatives = 0;

    for sample in &samples {
        let predicted = has_duplicate(&net, &sample.sequence);
        if predicted == sample.has_dup {
            correct += 1;
        } else if predicted && !sample.has_dup {
            false_positives += 1;
        } else {
            false_negatives += 1;
        }
    }

    let total = samples.len();
    let accuracy = correct as f64 / total as f64 * 100.0;
    println!("Results:");
    println!("  Accuracy:  {correct}/{total} ({accuracy:.1}%)");
    println!("  False positives: {false_positives} (predicted dup when unique)");
    println!("  False negatives: {false_negatives} (missed dup)");

    // Breakdown by sequence length
    println!("\nBy sequence length:");
    for len in 4..=9 {
        let len_samples: Vec<&DupSample> = samples.iter()
            .filter(|s| s.sequence.len() == len)
            .collect();
        if len_samples.is_empty() {
            continue;
        }
        let len_correct = len_samples.iter()
            .filter(|s| has_duplicate(&net, &s.sequence) == s.has_dup)
            .count();
        println!(
            "  Length {len}: {len_correct}/{} ({:.1}%)",
            len_samples.len(),
            len_correct as f64 / len_samples.len() as f64 * 100.0
        );
    }

    // Save trained CompareANT thermogram for other composition examples
    let trained_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("trained");
    std::fs::create_dir_all(&trained_dir).unwrap();
    let thermo_path = trained_dir.join("compare_composed.thermo");
    let mut thermo = Thermogram::new("compare", PlasticityRule::stdp_like());
    let delta = Delta::create("compare.w_out", net.w_out.data.clone(), "mastery");
    thermo.apply_delta(delta).unwrap();
    thermo.save(&thermo_path).unwrap();
    println!("\nSaved trained CompareANT to {:?}", thermo_path);

    println!("\n=== Composition: has_duplicate Complete ===");
}
