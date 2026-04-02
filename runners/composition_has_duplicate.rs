//! Composition: Has Duplicate Runner
//!
//! Trains a CompareANT using compare_train.rune, then composes it into a
//! "has_duplicate" detector: for all pairs (i,j) where i<j, call forward
//! with [seq[i] ++ seq[j]], check if output > 0. OR results.

use std::sync::Arc;
use atomic_neural_transistors::{AtomicNeuralTransistor, Signal, Value};
use atomic_neural_transistors::testdata::{Rng, generate_class_prototypes, add_noise};

fn compare(ant: &mut AtomicNeuralTransistor, a: &[Signal], b: &[Signal]) -> bool {
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
}

fn has_duplicate(ant: &mut AtomicNeuralTransistor, sequence: &[Vec<Signal>]) -> bool {
    for i in 0..sequence.len() {
        for j in (i + 1)..sequence.len() {
            if compare(ant, &sequence[i], &sequence[j]) {
                return true;
            }
        }
    }
    false
}

fn main() {
    println!("=== Composition: Has Duplicate Runner ===\n");

    let mut rng = Rng::new(0xCAFE);

    let prototypes = generate_class_prototypes(&mut rng, 8, 32);
    let num_classes = prototypes.len();

    // Load and train CompareANT
    let source = include_str!("../runes/compare_train.rune");
    let mut ant = AtomicNeuralTransistor::from_source(source)
        .expect("Failed to load compare_train.rune");

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

    // Generate test sequences
    let n_samples = 500;
    let mut correct = 0usize;
    let mut false_positives = 0usize;
    let mut false_negatives = 0usize;

    for i in 0..n_samples {
        let max_len = num_classes.min(8);
        let seq_len = 4 + (rng.next() as usize % (max_len - 3));
        let has_dup = i % 2 == 0;

        let mut sequence: Vec<Vec<Signal>> = Vec::with_capacity(seq_len);

        if has_dup {
            let mut used_protos = Vec::new();
            for _ in 0..seq_len {
                let proto = (rng.next() as usize) % num_classes;
                used_protos.push(proto);
                sequence.push(add_noise(&mut rng, &prototypes[proto]));
            }
            // Force a duplicate
            let dup_pos = 1 + (rng.next() as usize % (seq_len - 1));
            sequence[dup_pos] = add_noise(&mut rng, &prototypes[used_protos[0]]);
        } else {
            let mut available: Vec<usize> = (0..num_classes).collect();
            for _ in 0..seq_len {
                if available.is_empty() {
                    let proto = (rng.next() as usize) % num_classes;
                    sequence.push(add_noise(&mut rng, &prototypes[proto]));
                } else {
                    let idx = (rng.next() as usize) % available.len();
                    let proto = available.remove(idx);
                    sequence.push(add_noise(&mut rng, &prototypes[proto]));
                }
            }
        }

        let predicted = has_duplicate(&mut ant, &sequence);
        if predicted == has_dup {
            correct += 1;
        } else if predicted && !has_dup {
            false_positives += 1;
        } else {
            false_negatives += 1;
        }
    }

    let accuracy = correct as f64 / n_samples as f64 * 100.0;
    println!("Results:");
    println!("  Accuracy:  {correct}/{n_samples} ({accuracy:.1}%)");
    println!("  False positives: {false_positives} (predicted dup when unique)");
    println!("  False negatives: {false_negatives} (missed dup)");

    // Breakdown by sequence length
    println!("\nNote: per-length breakdown omitted in runner (single-pass evaluation)");

    println!("\n=== Composition: Has Duplicate Runner Complete ===");
}
