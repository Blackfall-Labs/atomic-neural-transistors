//! CompareANT Mastery Runner
//!
//! Trains a binary similarity detector on 32-dim ternary signal pairs via Runes scripting.
//! Same-class pairs produce target +127, different-class pairs produce target -127.

use std::sync::Arc;
use atomic_neural_transistors::{AtomicNeuralTransistor, Signal, Value};
use atomic_neural_transistors::testdata::{Rng, generate_class_prototypes, add_noise};

fn main() {
    println!("=== CompareANT Mastery Runner ===\n");

    let mut rng = Rng::new(42);

    // Generate 4 class prototypes, 32-dim
    let prototypes = generate_class_prototypes(&mut rng, 4, 32);
    println!("Prototypes: {} classes, 32-dim", prototypes.len());

    // Generate pairs: 1000 total (200 test, 800 train)
    let n_total = 1000;
    let n_test = 200;
    let num_classes = prototypes.len();

    struct Pair {
        a: Vec<Signal>,
        b: Vec<Signal>,
        same: bool,
    }

    let mut pairs: Vec<Pair> = Vec::with_capacity(n_total);
    for i in 0..n_total {
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
        pairs.push(Pair { a, b, same });
    }

    let (test_set, train_set) = pairs.split_at(n_test);
    println!("Data: {} training, {} test\n", train_set.len(), test_set.len());

    // Load ANT from rune source
    let source = include_str!("../runes/compare_train.rune");
    let mut ant = AtomicNeuralTransistor::from_source(source)
        .expect("Failed to load compare_train.rune");

    let max_cycles = 50;
    let mut best_accuracy = 0.0f64;

    for cycle in 1..=max_cycles {
        // Train on each pair
        for pair in train_set.iter() {
            // Pack [a(32) ++ b(32) | target(1)] = 65 values
            let mut values: Vec<Value> = pair.a.iter()
                .chain(pair.b.iter())
                .map(|s| Value::Integer(s.current() as i64))
                .collect();
            let target = if pair.same { 127i64 } else { -127i64 };
            values.push(Value::Integer(target));

            ant.call_values("train", vec![Value::Array(Arc::new(values))])
                .expect("train call failed");
        }

        // Decay mastery pressure
        ant.call_values("decay", vec![])
            .expect("decay call failed");

        // Evaluate on test set
        let mut correct = 0usize;
        for pair in test_set.iter() {
            let input: Vec<Value> = pair.a.iter()
                .chain(pair.b.iter())
                .map(|s| Value::Integer(s.current() as i64))
                .collect();

            let result = ant.call_values("forward", vec![Value::Array(Arc::new(input))])
                .expect("forward call failed");

            // forward returns array with 1 element; check sign
            let output_val = match result {
                Value::Array(arr) => {
                    if let Some(Value::Integer(v)) = arr.first() { *v } else { 0 }
                }
                Value::Integer(v) => v,
                _ => 0,
            };

            let predicted_same = output_val > 0;
            if predicted_same == pair.same {
                correct += 1;
            }
        }

        let acc = correct as f64 / test_set.len() as f64 * 100.0;
        if acc > best_accuracy {
            best_accuracy = acc;
        }

        println!(
            "Cycle {:2}: test {:.1}% ({}/{})",
            cycle, acc, correct, test_set.len()
        );

        if acc >= 99.0 {
            println!("\nTarget accuracy 99%+ reached at cycle {}!", cycle);
            break;
        }
    }

    println!("\nBest test accuracy: {:.1}%", best_accuracy);
    println!("\n=== CompareANT Mastery Runner Complete ===");
}
