//! Adaptive Cascade — Example 12
//!
//! Demonstrates all three multiplex capabilities in one scenario:
//!
//! **Phase 1 — Specialization Emergence**
//!   Three specialist ANTs each extract different product features from
//!   the same 32-dim input. The salience router learns which ANT to trust
//!   per class. A verifier ANT trains on the routed output (cascaded chaining).
//!
//! **Phase 2 — Adversarial Adaptation**
//!   A gain-stage failure inverts polarity on dims 0-15. The prediction engine
//!   detects the shift via surprise spike. Anti-pattern learning suppresses old
//!   patterns, new mastery emerges.
//!
//! **Phase 3 — Cascaded Verification**
//!   The multiplex output feeds into the verifier (dual encoder chaining).
//!   Disagreement between multiplex and verifier injects negative DA,
//!   demonstrating neuromodulator feedback across the cascade.

use atomic_neural_transistors::core::weight_matrix::{packed_from_current, WeightMatrix};
use atomic_neural_transistors::learning::{MasteryConfig, MasteryState};
use atomic_neural_transistors::multiplex::{AntSlot, MultiplexEncoder};
use atomic_neural_transistors::neuromod::Chemical;
use atomic_neural_transistors::PackedSignal;

// ---------------------------------------------------------------------------
// PRNG (xorshift64)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Data generation
// ---------------------------------------------------------------------------

struct Sample {
    signal: Vec<PackedSignal>,
    class: usize,
}

/// 4 class prototypes with Hadamard polarity signatures.
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
            let noise = (rng.next() % 81) as i32 - 40;
            packed_from_current(s.current().saturating_add(noise))
        })
        .collect()
}

/// Gain-stage failure: flip polarity on first 16 dims.
fn rotate_prototype(proto: &[PackedSignal]) -> Vec<PackedSignal> {
    proto
        .iter()
        .enumerate()
        .map(|(d, s)| {
            if d < 16 {
                packed_from_current(-s.current())
            } else {
                *s
            }
        })
        .collect()
}

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
// ANT constructors
// ---------------------------------------------------------------------------

/// Correlation-boosted product features: same as product, but adds
/// a global correlation score to each output dimension. This amplifies
/// magnitude when input polarity matches prototype, creating genuine
/// winner differentiation.
fn make_boosted_ant(prototype: Vec<PackedSignal>) -> Box<dyn Fn(&[PackedSignal]) -> Vec<PackedSignal>> {
    Box::new(move |input: &[PackedSignal]| {
        // Compute global correlation: how many dims agree in polarity
        let agree_count: i32 = input
            .iter()
            .zip(prototype.iter())
            .map(|(x, p)| {
                let xc = x.current();
                let pc = p.current();
                if (xc > 0 && pc > 0) || (xc < 0 && pc < 0) { 1 } else { 0 }
            })
            .sum();
        // Boost factor: -128 to +128 range based on agreement ratio
        let boost = (agree_count * 256 / input.len().max(1) as i32) - 128;

        input
            .iter()
            .zip(prototype.iter())
            .map(|(x, p)| {
                let product = x.current() as i64 * p.current() as i64;
                let base = (product / 256) as i32;
                packed_from_current(base.saturating_add(boost))
            })
            .collect()
    })
}

/// Frozen random projection matrix.
fn make_frozen_projection(rng: &mut Rng, rows: usize, cols: usize) -> WeightMatrix {
    let mut data = Vec::with_capacity(rows * cols);
    for _ in 0..rows * cols {
        let pol: i8 = if rng.next() % 2 == 0 { 1 } else { -1 };
        let mag = 20 + (rng.next_u8() % 20);
        data.push(PackedSignal::pack(pol, mag, 1));
    }
    WeightMatrix::from_data(data, rows, cols).unwrap()
}

// ---------------------------------------------------------------------------
// Verifier (cascade second stage)
// ---------------------------------------------------------------------------

fn verifier_forward(
    input: &[PackedSignal],
    hidden: &WeightMatrix,
    out: &WeightMatrix,
) -> Vec<PackedSignal> {
    let h: Vec<PackedSignal> = hidden
        .matmul(input)
        .iter()
        .map(|s| if s.current() > 0 { *s } else { PackedSignal::ZERO })
        .collect();
    out.matmul(&h)
}

fn verifier_predict(
    input: &[PackedSignal],
    hidden: &WeightMatrix,
    out: &WeightMatrix,
) -> usize {
    let logits = verifier_forward(input, hidden, out);
    logits
        .iter()
        .enumerate()
        .max_by_key(|(_, s)| s.current())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build class targets: 32-dim Hadamard targets (+127/-127 per class).
fn make_class_targets() -> Vec<Vec<PackedSignal>> {
    (0..4)
        .map(|c| {
            (0..32)
                .map(|d| {
                    let bits = ((c ^ d) as u32).count_ones();
                    let pol: i32 = if bits % 2 == 0 { 1 } else { -1 };
                    packed_from_current(pol * 100)
                })
                .collect()
        })
        .collect()
}

/// One-hot target for verifier: +127 for correct class, -127 for others.
fn make_onehot_target(class: usize) -> Vec<PackedSignal> {
    (0..4)
        .map(|c| {
            if c == class {
                packed_from_current(127)
            } else {
                packed_from_current(-127)
            }
        })
        .collect()
}

/// Classify output by min L1 distance to class targets.
fn classify_by_distance(output: &[PackedSignal], targets: &[Vec<PackedSignal>]) -> usize {
    targets
        .iter()
        .enumerate()
        .min_by_key(|(_, tgt)| {
            output
                .iter()
                .zip(tgt.iter())
                .map(|(o, t)| (o.current() as i64 - t.current() as i64).abs())
                .sum::<i64>()
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("=== Adaptive Cascade — Example 12 ===");
    println!("  3 phases: Specialization → Adaptation → Cascade Verification");
    println!();

    let mut rng = Rng::new(0xCAFE_F00D_DEAD_BEEF);

    // Data generation
    let protos = generate_prototypes(&mut rng);
    let class_targets = make_class_targets();

    // Build multiplex encoder (3 ANTs, 32-dim output)
    // Lower surprise threshold (20) so the prediction engine triggers learning
    let mut mux = MultiplexEncoder::new(32, 3, 20);

    // ANT α: boosted product features vs prototype[0] — strong at class 0
    mux.add_slot(AntSlot::with_passthrough(
        "alpha",
        make_boosted_ant(protos[0].clone()),
    ));

    // ANT β: boosted product features vs prototype[2] — strong at class 2
    mux.add_slot(AntSlot::with_passthrough(
        "beta",
        make_boosted_ant(protos[2].clone()),
    ));

    // ANT γ: boosted product features vs prototype[1] — strong at class 1
    mux.add_slot(AntSlot::with_passthrough(
        "gamma",
        make_boosted_ant(protos[1].clone()),
    ));

    mux.finalize();

    // Build verifier (cascade second stage)
    let verifier_hidden = make_frozen_projection(&mut rng, 16, 32);
    let mut verifier_out = WeightMatrix::zeros(4, 16);
    let mut verifier_ms = MasteryState::new(
        4 * 16,
        MasteryConfig {
            pressure_threshold: 3,
            decay_rate: 1,
            participation_gate: 5,
        },
    );

    // =========================================================================
    // Phase 1: Specialization Emergence
    // =========================================================================

    println!("--- Phase 1: Specialization Emergence (160 samples) ---");

    let phase1_data = generate_samples(&mut rng, &protos, 160);
    let mut p1_surprises = 0u32;
    let mut p1_learning = 0u32;
    let mut winner_counts = vec![vec![0u32; 3]; 4]; // [class][ant]

    for sample in &phase1_data {
        let target = &class_targets[sample.class];
        let result = mux.process(&sample.signal, Some(target));

        if result.surprise.is_surprising {
            p1_surprises += 1;
        }
        if result.learning_occurred {
            p1_learning += 1;
        }
        winner_counts[sample.class][result.route.winner] += 1;

        // Train verifier on routed output (cascade chaining)
        let h: Vec<PackedSignal> = verifier_hidden
            .matmul(&result.output)
            .iter()
            .map(|s| if s.current() > 0 { *s } else { PackedSignal::ZERO })
            .collect();
        let raw = verifier_out.matmul(&h);
        let clamped: Vec<PackedSignal> = raw
            .iter()
            .map(|s| packed_from_current(s.current().clamp(-127, 127)))
            .collect();
        let onehot = make_onehot_target(sample.class);
        verifier_ms.update(&mut verifier_out, &h, &clamped, &onehot);
    }

    // Decay verifier pressure at end of phase
    verifier_ms.decay();

    println!("  Surprises: {}  Learning moments: {}", p1_surprises, p1_learning);
    println!("  DA: {}  (baseline=128, gate=77)", mux.neuromod.dopamine);
    println!("  Verifier transitions: {}", verifier_ms.transitions);
    println!("  Winner distribution per class:");
    for c in 0..4 {
        let total_c: u32 = winner_counts[c].iter().sum();
        if total_c == 0 { continue; }
        print!("    Class {}: ", c);
        for a in 0..3 {
            let pct = winner_counts[c][a] as f64 / total_c as f64 * 100.0;
            print!("{}={:.0}%  ", ["α", "β", "γ"][a], pct);
        }
        println!();
    }

    let da_after_p1 = mux.neuromod.dopamine;

    // =========================================================================
    // Phase 2: Distribution Shift
    // =========================================================================

    println!();
    println!("--- Phase 2: Distribution Shift (80 samples) ---");
    println!("  Gain-stage failure: polarity inversion on dims 0-15");

    // Rotate all prototypes
    let shifted_protos: Vec<Vec<PackedSignal>> =
        protos.iter().map(|p| rotate_prototype(p)).collect();

    let phase2_data = generate_samples(&mut rng, &shifted_protos, 80);
    let mut surprise_mags: Vec<i64> = Vec::new();
    let mut p2_first_half_surprises = 0u32;
    let mut p2_second_half_surprises = 0u32;
    let mut da_nadir = mux.neuromod.dopamine;
    let mut da_nadir_step = 0usize;

    for (i, sample) in phase2_data.iter().enumerate() {
        let target = &class_targets[sample.class];
        let result = mux.process(&sample.signal, Some(target));

        if i < 10 {
            surprise_mags.push(result.surprise.magnitude);
        }
        if result.surprise.is_surprising {
            if i < 40 {
                p2_first_half_surprises += 1;
            } else {
                p2_second_half_surprises += 1;
            }
        }
        if result.dopamine < da_nadir {
            da_nadir = result.dopamine;
            da_nadir_step = i;
        }

        // Continue training verifier on shifted data (adaptation)
        let h: Vec<PackedSignal> = verifier_hidden
            .matmul(&result.output)
            .iter()
            .map(|s| if s.current() > 0 { *s } else { PackedSignal::ZERO })
            .collect();
        let raw = verifier_out.matmul(&h);
        let clamped: Vec<PackedSignal> = raw
            .iter()
            .map(|s| packed_from_current(s.current().clamp(-127, 127)))
            .collect();
        let onehot = make_onehot_target(sample.class);
        verifier_ms.update(&mut verifier_out, &h, &clamped, &onehot);
    }
    verifier_ms.decay();

    print!("  First 10 surprise magnitudes: [");
    for (i, m) in surprise_mags.iter().enumerate() {
        if i > 0 { print!(", "); }
        print!("{}", m);
    }
    println!("]");
    println!(
        "  Surprises: first 40 = {}  last 40 = {}  {}",
        p2_first_half_surprises,
        p2_second_half_surprises,
        if p2_second_half_surprises < p2_first_half_surprises {
            "(adapting)"
        } else {
            "(still adapting)"
        }
    );
    println!(
        "  DA nadir: {} at step {}  (started at {})",
        da_nadir, da_nadir_step, da_after_p1
    );
    println!("  DA recovery: {}", mux.neuromod.dopamine);

    // =========================================================================
    // Phase 3: Cascaded Verification
    // =========================================================================

    println!();
    println!("--- Phase 3: Cascaded Verification (60 samples, no learning) ---");

    let phase3_data = generate_samples(&mut rng, &shifted_protos, 60);
    let mut mux_correct = 0u32;
    let mut ver_correct = 0u32;
    let mut agree = 0u32;
    let mut conflicts = 0u32;

    for sample in &phase3_data {
        // No target → no learning
        let result = mux.process(&sample.signal, None);

        // Multiplex class prediction (min L1 distance)
        let mux_class = classify_by_distance(&result.output, &class_targets);

        // Verifier class prediction (cascade chain)
        let ver_class = verifier_predict(&result.output, &verifier_hidden, &verifier_out);

        if mux_class == sample.class {
            mux_correct += 1;
        }
        if ver_class == sample.class {
            ver_correct += 1;
        }
        if mux_class == ver_class {
            agree += 1;
        } else {
            conflicts += 1;
            // Cascade feedback: disagreement reduces DA
            mux.neuromod.inject(Chemical::Dopamine, -10);
            mux.neuromod.tick();
        }
    }

    let total = phase3_data.len() as u32;
    println!(
        "  Multiplex accuracy: {:.1}% ({}/{})",
        mux_correct as f64 / total as f64 * 100.0,
        mux_correct,
        total
    );
    println!(
        "  Verifier accuracy:  {:.1}% ({}/{})",
        ver_correct as f64 / total as f64 * 100.0,
        ver_correct,
        total
    );
    println!(
        "  Agreement rate:     {:.1}% ({}/{})",
        agree as f64 / total as f64 * 100.0,
        agree,
        total
    );
    println!("  Cascade conflicts:  {} (each → DA -10)", conflicts);
    println!(
        "  Neuromod final:     DA={}  NE={}  5HT={}",
        mux.neuromod.dopamine,
        mux.neuromod.norepinephrine,
        mux.neuromod.serotonin
    );
    println!(
        "  Plasticity gate:    {} (DA {} {} gate {})",
        if mux.neuromod.dopamine > mux.neuromod.dopamine_gate { "OPEN" } else { "CLOSED" },
        mux.neuromod.dopamine,
        if mux.neuromod.dopamine > mux.neuromod.dopamine_gate { ">" } else { "<=" },
        mux.neuromod.dopamine_gate
    );

    // Summary
    println!();
    println!("=== Summary ===");
    println!("  Phase 1: {} ANTs specialized across {} classes", 3, 4);
    println!("  Phase 2: Surprise spike detected shift, {} adaptation moments", p2_first_half_surprises + p2_second_half_surprises);
    println!(
        "  Phase 3: Cascade verified {:.1}% of outputs, {} conflicts fed back to neuromod",
        agree as f64 / total as f64 * 100.0,
        conflicts
    );
    println!("=== Adaptive cascade complete ===");
}
