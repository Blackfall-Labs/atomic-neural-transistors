//! Composition: Contains Runner
//!
//! Trains a CompareANT using compare_train.rune, then composes it into a
//! "contains" detector: contains(query, sequence) = OR(compare(query, seq[i]) for all i).
//! Evaluates accuracy on generated test sequences.

use std::sync::Arc;
use atomic_neural_transistors::{AtomicNeuralTransistor, Signal, Value};
use atomic_neural_transistors::testdata::{Rng, generate_class_prototypes, add_noise};

fn main() {
    println!("=== Composition: Contains Runner ===\n");

    let mut rng = Rng::new(0xC047);

    // Generate 8 class prototypes, 32-dim
    let prototypes = generate_class_prototypes(&mut rng, 8, 32);
    let num_classes = prototypes.len();

    // Load CompareANT
    let source = include_str!("../runes/compare_train.rune");
    let mut ant = AtomicNeuralTransistor::from_source(source)
        .expect("Failed to load compare_train.rune");

    // Train CompareANT
    println!("Training CompareANT...");
    for _cycle in 0..3 {
        for i in 0..800 {
            let class_a = (rng.next() as usize) % num_classes;
            let a = add_noise(&mut rng, &prototypes[class_a]);
            let same = i % 2 == 0;
            let b = if same {
                add_noise(&mut rng, &prototypes[class_a])
            } else {
                let mut class_b = (rng.next() as usize) % num_classes;
                while class_b == class_a {
                    class_b = (rng.next() as usize) % num_classes;
                }
                add_noise(&mut rng, &prototypes[class_b])
            };

            // Pack [a(32) ++ b(32) | target(1)]
            let mut values: Vec<Value> = a.iter().chain(b.iter())
                .map(|s| Value::Integer(s.current() as i64))
                .collect();
            let target = if same { 127i64 } else { -127i64 };
            values.push(Value::Integer(target));

            ant.call_values("train", vec![Value::Array(Arc::new(values))])
                .expect("train call failed");
        }
        ant.call_values("decay", vec![]).expect("decay failed");
    }
    println!("CompareANT trained.\n");

    // Helper: compare two signals via ANT
    let compare = |ant: &mut AtomicNeuralTransistor, a: &[Signal], b: &[Signal]| -> bool {
        let input: Vec<Value> = a.iter().chain(b.iter())
            .map(|s| Value::Integer(s.current() as i64))
            .collect();
        let result = ant.call_values("forward", vec![Value::Array(Arc::new(input))])
            .expect("forward failed");
        let val = match result {
            Value::Array(arr) => {
                if let Some(Value::Integer(v)) = arr.first() { *v } else { 0 }
            }
            Value::Integer(v) => v,
            _ => 0,
        };
        val > 0
    };

    // Generate test data for contains
    let n_samples = 500;
    let mut correct = 0usize;
    let mut false_positives = 0usize;
    let mut false_negatives = 0usize;

    for i in 0..n_samples {
        let query_proto = (rng.next() as usize) % num_classes;
        let query = add_noise(&mut rng, &prototypes[query_proto]);
        let seq_len = 3 + (rng.next() as usize % 6); // 3-8 elements
        let should_contain = i % 2 == 0;

        let mut sequence: Vec<Vec<Signal>> = Vec::with_capacity(seq_len);
        if should_contain {
            let match_pos = (rng.next() as usize) % seq_len;
            for pos in 0..seq_len {
                if pos == match_pos {
                    sequence.push(add_noise(&mut rng, &prototypes[query_proto]));
                } else {
                    let mut p = (rng.next() as usize) % num_classes;
                    while p == query_proto { p = (rng.next() as usize) % num_classes; }
                    sequence.push(add_noise(&mut rng, &prototypes[p]));
                }
            }
        } else {
            for _ in 0..seq_len {
                let mut p = (rng.next() as usize) % num_classes;
                while p == query_proto { p = (rng.next() as usize) % num_classes; }
                sequence.push(add_noise(&mut rng, &prototypes[p]));
            }
        }

        // contains = OR(compare(query, seq[i]) for all i)
        let predicted = sequence.iter().any(|elem| compare(&mut ant, &query, elem));

        if predicted == should_contain {
            correct += 1;
        } else if predicted && !should_contain {
            false_positives += 1;
        } else {
            false_negatives += 1;
        }
    }

    let accuracy = correct as f64 / n_samples as f64 * 100.0;
    println!("Results:");
    println!("  Accuracy:  {correct}/{n_samples} ({accuracy:.1}%)");
    println!("  False positives: {false_positives} (found match when none exists)");
    println!("  False negatives: {false_negatives} (missed existing match)");

    println!("\n=== Composition: Contains Runner Complete ===");
}
