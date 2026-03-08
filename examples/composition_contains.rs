//! Composition: contains — Example 7
//!
//! Composes a trained CompareANT into a "contains" detector that checks
//! whether a query element exists in a sequence, without any additional
//! mastery learning.
//!
//! Pipeline:
//!   contains(query, seq) = OR(CompareANT(query, seq[i]) for all i)
//!
//! This extends Example 6 (has_duplicate) to show that the same trained
//! CompareANT can be composed into multiple different higher-order
//! operations.

use atomic_neural_transistors::core::weight_matrix::{packed_from_current, WeightMatrix};
use atomic_neural_transistors::learning::{MasteryConfig, MasteryState};
use atomic_neural_transistors::PackedSignal;

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

fn comparison_features(a: &[PackedSignal], b: &[PackedSignal]) -> Vec<PackedSignal> {
    a.iter().zip(b.iter()).map(|(x, y)| {
        let product = x.current() as i64 * y.current() as i64;
        packed_from_current((product / 256) as i32)
    }).collect()
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
// Train CompareANT (same as Example 1 and 6)
// ---------------------------------------------------------------------------

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

fn train_compare_network(rng: &mut Rng) -> CompareNetwork {
    let prototypes = generate_prototypes(rng, 8);
    let config = MasteryConfig { pressure_threshold: 3, decay_rate: 1, participation_gate: 5 };
    let mut w_out = WeightMatrix::zeros(1, 32);
    let mut ms = MasteryState::new(32, config);

    for _ in 0..2 {
        for i in 0..800 {
            let proto_a = (rng.next() as usize) % prototypes.len();
            let a = add_noise(rng, &prototypes[proto_a]);
            let identical = i % 2 == 0;
            let b = if identical {
                add_noise(rng, &prototypes[proto_a])
            } else {
                let mut proto_b = (rng.next() as usize) % prototypes.len();
                while proto_b == proto_a { proto_b = (rng.next() as usize) % prototypes.len(); }
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
// contains composition
// ---------------------------------------------------------------------------

fn contains(net: &CompareNetwork, query: &[PackedSignal], sequence: &[Vec<PackedSignal>]) -> bool {
    sequence.iter().any(|elem| net.compare(query, elem))
}

// ---------------------------------------------------------------------------
// Test data
// ---------------------------------------------------------------------------

struct ContainsSample {
    query: Vec<PackedSignal>,
    sequence: Vec<Vec<PackedSignal>>,
    expected: bool,
}

fn generate_contains_samples(
    rng: &mut Rng, n: usize, prototypes: &[Vec<PackedSignal>],
) -> Vec<ContainsSample> {
    let np = prototypes.len();
    (0..n).map(|i| {
        let seq_len = 3 + (rng.next() as usize % 6); // 3-8 elements
        let query_proto = (rng.next() as usize) % np;
        let query = add_noise(rng, &prototypes[query_proto]);
        let should_contain = i % 2 == 0;

        let mut sequence = Vec::with_capacity(seq_len);
        if should_contain {
            // Build sequence with one matching element
            let match_pos = (rng.next() as usize) % seq_len;
            for pos in 0..seq_len {
                if pos == match_pos {
                    sequence.push(add_noise(rng, &prototypes[query_proto]));
                } else {
                    let mut p = (rng.next() as usize) % np;
                    while p == query_proto { p = (rng.next() as usize) % np; }
                    sequence.push(add_noise(rng, &prototypes[p]));
                }
            }
        } else {
            // All different from query
            for _ in 0..seq_len {
                let mut p = (rng.next() as usize) % np;
                while p == query_proto { p = (rng.next() as usize) % np; }
                sequence.push(add_noise(rng, &prototypes[p]));
            }
        }

        ContainsSample { query, sequence, expected: should_contain }
    }).collect()
}

fn main() {
    println!("=== Composition: contains ===\n");

    let mut rng = Rng::new(0xC047);

    // Train CompareANT
    println!("Training CompareANT...");
    let net = train_compare_network(&mut rng);

    // Generate test data
    let prototypes = generate_prototypes(&mut rng, 8);
    let samples = generate_contains_samples(&mut rng, 500, &prototypes);
    println!("Test samples: {} ({} with match, {} without)\n",
        samples.len(),
        samples.iter().filter(|s| s.expected).count(),
        samples.iter().filter(|s| !s.expected).count(),
    );

    // Evaluate
    let mut correct = 0;
    let mut false_positives = 0;
    let mut false_negatives = 0;

    for sample in &samples {
        let predicted = contains(&net, &sample.query, &sample.sequence);
        if predicted == sample.expected {
            correct += 1;
        } else if predicted && !sample.expected {
            false_positives += 1;
        } else {
            false_negatives += 1;
        }
    }

    let accuracy = correct as f64 / samples.len() as f64 * 100.0;
    println!("Results:");
    println!("  Accuracy:  {correct}/{} ({accuracy:.1}%)", samples.len());
    println!("  False positives: {false_positives} (found match when none exists)");
    println!("  False negatives: {false_negatives} (missed existing match)");

    // Breakdown by sequence length
    println!("\nBy sequence length:");
    for len in 3..=8 {
        let len_samples: Vec<&ContainsSample> = samples.iter()
            .filter(|s| s.sequence.len() == len).collect();
        if len_samples.is_empty() { continue; }
        let len_correct = len_samples.iter()
            .filter(|s| contains(&net, &s.query, &s.sequence) == s.expected)
            .count();
        println!("  Length {len}: {len_correct}/{} ({:.1}%)",
            len_samples.len(),
            len_correct as f64 / len_samples.len() as f64 * 100.0);
    }

    println!("\n=== Composition: contains Complete ===");
}
