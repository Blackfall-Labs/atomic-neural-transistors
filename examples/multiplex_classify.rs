//! Multiplex Classification — Example 11
//!
//! Demonstrates the full multiplex encoding loop:
//!   - 3 ANTs process the same input in parallel (different feature extraction)
//!   - Salience router learns which ANT to trust per input pattern
//!   - Prediction engine tracks expected outputs
//!   - Surprise triggers learning moments
//!   - Positive surprise → reinforce (DA injection)
//!   - Negative surprise → anti-pattern pressure (inverted targets)
//!   - Neuromodulator chemicals self-balance (DA ↔ 5HT antagonism)
//!
//! Target: router learns to favor the right ANT per task. Specialization emerges.

use atomic_neural_transistors::core::weight_matrix::{packed_from_current, WeightMatrix};
use atomic_neural_transistors::multiplex::{AntSlot, MultiplexEncoder};
use atomic_neural_transistors::PackedSignal;

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
// Data generation: 4 classes, 32-dim signals
// ---------------------------------------------------------------------------

struct Sample {
    signal: Vec<PackedSignal>,
    class: usize,
}

/// Generate 4 class prototypes using Hadamard polarity signatures.
fn generate_prototypes(rng: &mut Rng) -> Vec<Vec<PackedSignal>> {
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

/// Add noise to a prototype.
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

/// Generate samples.
fn generate_samples(rng: &mut Rng, protos: &[Vec<PackedSignal>], count: usize) -> Vec<Sample> {
    (0..count)
        .map(|_| {
            let class = (rng.next() % 4) as usize;
            Sample {
                signal: add_noise(rng, &protos[class]),
                class,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// ANT feature extractors
// ---------------------------------------------------------------------------

/// Product features: element-wise product between input and a frozen prototype.
/// Different prototypes yield different feature extractions.
fn make_product_ant(prototype: Vec<PackedSignal>) -> Box<dyn Fn(&[PackedSignal]) -> Vec<PackedSignal>> {
    Box::new(move |input: &[PackedSignal]| {
        input
            .iter()
            .zip(prototype.iter())
            .map(|(x, p)| {
                let product = x.current() as i64 * p.current() as i64;
                packed_from_current((product / 256) as i32)
            })
            .collect()
    })
}

/// Frozen random projection: matmul through random synaptic strengths + ReLU.
fn make_projection_ant(rng: &mut Rng, input_dim: usize, output_dim: usize) -> Box<dyn Fn(&[PackedSignal]) -> Vec<PackedSignal>> {
    // Build frozen random synaptic strengths
    let mut data = Vec::with_capacity(output_dim * input_dim);
    for _ in 0..output_dim * input_dim {
        let pol: i8 = if rng.next() % 2 == 0 { 1 } else { -1 };
        let mag = 20 + (rng.next_u8() % 20);
        data.push(PackedSignal::pack(pol, mag, 1));
    }
    let wm = WeightMatrix::from_data(data, output_dim, input_dim).unwrap();

    Box::new(move |input: &[PackedSignal]| {
        let raw = wm.matmul(input);
        // ReLU
        raw.iter()
            .map(|s| {
                if s.current() > 0 {
                    *s
                } else {
                    PackedSignal::ZERO
                }
            })
            .collect()
    })
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("=== Multiplex Classification — Example 11 ===");
    println!();

    let mut rng = Rng::new(0xABC_DEF_123);

    // Generate prototypes and data
    let protos = generate_prototypes(&mut rng);
    let train_data = generate_samples(&mut rng, &protos, 500);
    let test_data = generate_samples(&mut rng, &protos, 200);

    // Create target signals for each class (polarity encodes class)
    let class_targets: Vec<Vec<PackedSignal>> = (0..4)
        .map(|c| {
            (0..32)
                .map(|d| {
                    let bits = ((c ^ d) as u32).count_ones();
                    let pol: i8 = if bits % 2 == 0 { 1 } else { -1 };
                    packed_from_current(pol as i32 * 100)
                })
                .collect()
        })
        .collect();

    // Create multiplex encoder with 3 ANTs
    let output_dim = 32;
    let mut mux = MultiplexEncoder::new(output_dim, 3, 40);

    // ANT 0: Product features against prototype 0
    let ant0 = AntSlot::with_passthrough(
        "product_p0",
        make_product_ant(protos[0].clone()),
    );

    // ANT 1: Product features against prototype 2
    let ant1 = AntSlot::with_passthrough(
        "product_p2",
        make_product_ant(protos[2].clone()),
    );

    // ANT 2: Random projection
    let ant2 = AntSlot::with_passthrough(
        "projection",
        make_projection_ant(&mut rng, 32, 32),
    );

    mux.add_slot(ant0);
    mux.add_slot(ant1);
    mux.add_slot(ant2);
    mux.finalize();

    println!("Architecture:");
    println!("  ANT 0: product features vs prototype 0");
    println!("  ANT 1: product features vs prototype 2");
    println!("  ANT 2: frozen random projection + ReLU");
    println!("  Salience: gate-based fusion (learned)");
    println!("  Prediction: EMA (shift=3, threshold=40)");
    println!("  Neuromod: DA/NE/5HT (baseline=128, gate=77)");
    println!();

    // Track DA over time for oscillation report
    let mut da_history: Vec<u8> = Vec::new();
    let mut surprise_count = 0u32;
    let mut learning_count = 0u32;
    let mut positive_surprises = 0u32;
    let mut negative_surprises = 0u32;

    // Process training data
    println!("--- Processing {} training samples ---", train_data.len());
    for sample in &train_data {
        let target = &class_targets[sample.class];
        let result = mux.process(&sample.signal, Some(target));

        da_history.push(result.dopamine);
        if result.surprise.is_surprising {
            surprise_count += 1;
        }
        if result.learning_occurred {
            learning_count += 1;
            match result.surprise.direction {
                1 => positive_surprises += 1,
                -1 => negative_surprises += 1,
                _ => {}
            }
        }
    }

    println!("  Surprises detected: {}", surprise_count);
    println!("  Learning moments: {}", learning_count);
    println!("    Positive (reinforce): {}", positive_surprises);
    println!("    Negative (anti-pattern): {}", negative_surprises);

    // DA oscillation analysis
    let da_min = da_history.iter().copied().min().unwrap_or(128);
    let da_max = da_history.iter().copied().max().unwrap_or(128);
    let da_final = *da_history.last().unwrap_or(&128);
    println!();
    println!("--- Neuromodulator Dynamics ---");
    println!("  DA range: {} - {} (baseline=128, gate=77)", da_min, da_max);
    println!("  DA final: {}", da_final);
    println!(
        "  DA gate status: {} (DA {} > gate 77)",
        if da_final > 77 { "OPEN" } else { "CLOSED" },
        da_final,
    );

    // Evaluate: measure which ANT wins for each class
    println!();
    println!("--- Evaluation on {} test samples ---", test_data.len());

    let mut winner_by_class = vec![vec![0u32; 3]; 4]; // [class][ant] = count
    let mut correct = 0u32;
    let total = test_data.len() as u32;

    for sample in &test_data {
        let target = &class_targets[sample.class];
        let result = mux.process(&sample.signal, None); // no learning during eval

        winner_by_class[sample.class][result.route.winner] += 1;

        // Check if output is closer to correct class target than others
        let output_dist_to_correct: i64 = result
            .output
            .iter()
            .zip(target.iter())
            .map(|(o, t)| (o.current() as i64 - t.current() as i64).abs())
            .sum();

        let mut is_closest = true;
        for other_class in 0..4 {
            if other_class == sample.class {
                continue;
            }
            let dist: i64 = result
                .output
                .iter()
                .zip(class_targets[other_class].iter())
                .map(|(o, t)| (o.current() as i64 - t.current() as i64).abs())
                .sum();
            if dist < output_dist_to_correct {
                is_closest = false;
                break;
            }
        }
        if is_closest {
            correct += 1;
        }
    }

    let accuracy = correct as f64 / total as f64 * 100.0;
    println!("  Accuracy: {:.1}% ({}/{})", accuracy, correct, total);

    println!();
    println!("  Winner distribution by class:");
    for c in 0..4 {
        let total_c: u32 = winner_by_class[c].iter().sum();
        if total_c == 0 {
            continue;
        }
        print!("    Class {}: ", c);
        for a in 0..3 {
            let pct = winner_by_class[c][a] as f64 / total_c as f64 * 100.0;
            print!("ANT{}={:.0}%  ", a, pct);
        }
        println!();
    }

    // Router gate mastery stats
    let gate_ms = mux.router().gate_mastery();
    println!();
    println!("--- Router Stats ---");
    println!("  Gate mastery steps: {}", gate_ms.steps);
    println!("  Gate transitions: {}", gate_ms.transitions);

    println!();
    println!("=== Multiplex classification complete ===");
}
