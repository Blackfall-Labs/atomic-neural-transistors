//! MergeANT Mastery Runner
//!
//! Trains a merge network that fuses two 32-dim signals into one 32-dim output via Runes.
//! Input: signal_a(32) ++ signal_b(32) = 64-dim
//! Target: Hadamard merge prototype for (class_a, class_b)
//! Evaluation: dot-product similarity between output and expected merge target.

use std::sync::Arc;
use atomic_neural_transistors::{AtomicNeuralTransistor, Signal, Value};
use atomic_neural_transistors::testdata::{Rng, generate_class_prototypes, add_noise};

fn main() {
    println!("=== MergeANT Mastery Runner ===\n");

    let mut rng = Rng::new(0x3E26);

    let num_classes = 4usize;
    let dim = 32usize;

    // Generate class prototypes for inputs
    let prototypes = generate_class_prototypes(&mut rng, num_classes, dim);
    println!("Input prototypes: {} classes, {}-dim", num_classes, dim);

    // Generate Hadamard merge targets for each (class_a, class_b) pair
    // Merge target is a unique prototype derived from both classes
    let mut merge_targets: Vec<Vec<Signal>> = Vec::with_capacity(num_classes * num_classes);
    for ca in 0..num_classes {
        for cb in 0..num_classes {
            let pair_id = ca * num_classes + cb;
            let target: Vec<Signal> = (0..dim).map(|d| {
                // Hadamard-like pattern based on pair_id
                let bits = ((pair_id & d) as u32).count_ones();
                let pol: i8 = if bits % 2 == 0 { 1 } else { -1 };
                let mag = 100 + (rng.next_u8() % 28);
                Signal::new_raw(pol, mag, 1)
            }).collect();
            merge_targets.push(target);
        }
    }

    // Generate merge samples
    let n_total = 1000;
    let n_test = 200;

    struct MergeSample {
        a: Vec<Signal>,
        b: Vec<Signal>,
        class_a: usize,
        class_b: usize,
    }

    let mut samples: Vec<MergeSample> = Vec::with_capacity(n_total);
    for _ in 0..n_total {
        let class_a = (rng.next() as usize) % num_classes;
        let class_b = (rng.next() as usize) % num_classes;
        let a = add_noise(&mut rng, &prototypes[class_a]);
        let b = add_noise(&mut rng, &prototypes[class_b]);
        samples.push(MergeSample { a, b, class_a, class_b });
    }

    let (test_set, train_set) = samples.split_at(n_test);
    println!("Data: {} training, {} test\n", train_set.len(), test_set.len());

    // Load ANT from rune source
    let source = include_str!("../runes/merge_train.rune");
    let mut ant = AtomicNeuralTransistor::from_source(source)
        .expect("Failed to load merge_train.rune");

    let max_cycles = 50;
    let mut best_similarity = f64::NEG_INFINITY;

    for cycle in 1..=max_cycles {
        // Train on each sample
        for sample in train_set.iter() {
            let target_idx = sample.class_a * num_classes + sample.class_b;
            let target = &merge_targets[target_idx];

            // Pack [a(32) ++ b(32) | target(32)] = 96 values
            let mut values: Vec<Value> = sample.a.iter()
                .chain(sample.b.iter())
                .map(|s| Value::Integer(s.current() as i64))
                .collect();
            for t in target.iter() {
                values.push(Value::Integer(t.current() as i64));
            }

            ant.call_values("train", vec![Value::Array(Arc::new(values))])
                .expect("train call failed");
        }

        // Decay mastery pressure
        ant.call_values("decay", vec![])
            .expect("decay call failed");

        // Evaluate on test set: dot-product similarity to expected merge target
        let mut total_similarity = 0.0f64;
        let mut correct_polarity = 0usize;
        let total_dims = test_set.len() * dim;

        for sample in test_set.iter() {
            let target_idx = sample.class_a * num_classes + sample.class_b;
            let target = &merge_targets[target_idx];

            let input: Vec<Value> = sample.a.iter()
                .chain(sample.b.iter())
                .map(|s| Value::Integer(s.current() as i64))
                .collect();

            let result = ant.call_values("forward", vec![Value::Array(Arc::new(input))])
                .expect("forward call failed");

            let output_vals: Vec<i64> = match result {
                Value::Array(arr) => {
                    arr.iter().map(|v| {
                        if let Value::Integer(n) = v { *n } else { 0 }
                    }).collect()
                }
                _ => vec![0; dim],
            };

            // Dot product similarity (normalized)
            let mut dot: i64 = 0;
            let mut mag_out: i64 = 0;
            let mut mag_tgt: i64 = 0;
            for d in 0..dim {
                let o = if d < output_vals.len() { output_vals[d] } else { 0 };
                let t = target[d].current() as i64;
                dot += o * t;
                mag_out += o * o;
                mag_tgt += t * t;
            }

            let denom = ((mag_out as f64).sqrt() * (mag_tgt as f64).sqrt()).max(1.0);
            let cosine = dot as f64 / denom;
            total_similarity += cosine;

            // Count per-dimension polarity agreement
            for d in 0..dim {
                let o = if d < output_vals.len() { output_vals[d] } else { 0 };
                let t = target[d].current() as i64;
                if (o > 0 && t > 0) || (o < 0 && t < 0) || (o == 0 && t == 0) {
                    correct_polarity += 1;
                }
            }
        }

        let avg_sim = total_similarity / test_set.len() as f64;
        let pol_acc = correct_polarity as f64 / total_dims as f64 * 100.0;
        if avg_sim > best_similarity {
            best_similarity = avg_sim;
        }

        println!(
            "Cycle {:2}: avg cosine similarity {:.4}, polarity accuracy {:.1}%",
            cycle, avg_sim, pol_acc
        );

        if avg_sim >= 0.90 {
            println!("\nTarget similarity 0.90+ reached at cycle {}!", cycle);
            break;
        }
    }

    println!("\nBest avg cosine similarity: {:.4}", best_similarity);
    println!("\n=== MergeANT Mastery Runner Complete ===");
}
