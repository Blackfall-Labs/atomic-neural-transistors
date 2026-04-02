//! ClassifierANT Mastery Runner
//!
//! Trains a 4-class classifier on 32-dim ternary signal patterns via Runes scripting.
//! Uses mastery learning through the AtomicNeuralTransistor rune engine.

use std::sync::Arc;
use atomic_neural_transistors::{AtomicNeuralTransistor, Value};
use atomic_neural_transistors::testdata::{Rng, generate_class_prototypes, generate_dataset};

fn main() {
    println!("=== ClassifierANT Mastery Runner ===\n");

    let mut rng = Rng::new(0xDEAD);

    // Generate 4-class Hadamard prototypes, 32-dim
    let prototypes = generate_class_prototypes(&mut rng, 4, 32);
    println!("Classes: {} with Hadamard polarity signatures", prototypes.len());

    // Generate 1000 samples: 200 test, 800 train
    let all_data = generate_dataset(&mut rng, 1000, &prototypes);
    let (test_set, train_set) = all_data.split_at(200);
    println!("Data: {} training, {} test\n", train_set.len(), test_set.len());

    // Load ANT from rune source
    let source = include_str!("../runes/classifier_train.rune");
    let mut ant = AtomicNeuralTransistor::from_source(source)
        .expect("Failed to load classifier_train.rune");

    // Training loop: 50 cycles
    let max_cycles = 50;
    let mut best_accuracy = 0.0f64;

    for cycle in 1..=max_cycles {
        let mut cycle_correct = 0usize;

        // Train on each sample
        for sample in train_set.iter() {
            // Pack [signal(32) | class_as_signal(1)] into flat array
            let mut values: Vec<Value> = sample.signal.iter()
                .map(|s| Value::Integer(s.current() as i64))
                .collect();
            values.push(Value::Integer(sample.class as i64));

            let result = ant.call_values("train", vec![Value::Array(Arc::new(values))])
                .expect("train call failed");

            // train() returns argmax (predicted class)
            if let Value::Integer(predicted) = result {
                if predicted as usize == sample.class {
                    cycle_correct += 1;
                }
            }
        }

        // Decay mastery pressure after each cycle
        ant.call_values("decay", vec![])
            .expect("decay call failed");

        // Evaluate on test set
        let mut test_correct = 0usize;
        for sample in test_set.iter() {
            let input: Vec<Value> = sample.signal.iter()
                .map(|s| Value::Integer(s.current() as i64))
                .collect();

            let result = ant.call_values("predict", vec![Value::Array(Arc::new(input))])
                .expect("predict call failed");

            if let Value::Integer(predicted) = result {
                if predicted as usize == sample.class {
                    test_correct += 1;
                }
            }
        }

        let train_acc = cycle_correct as f64 / train_set.len() as f64 * 100.0;
        let test_acc = test_correct as f64 / test_set.len() as f64 * 100.0;
        if test_acc > best_accuracy {
            best_accuracy = test_acc;
        }

        println!(
            "Cycle {:2}: train {:.1}% ({}/{}), test {:.1}% ({}/{})",
            cycle, train_acc, cycle_correct, train_set.len(),
            test_acc, test_correct, test_set.len()
        );

        if test_acc >= 99.0 {
            println!("\nTarget accuracy 99%+ reached at cycle {}!", cycle);
            break;
        }
    }

    println!("\nBest test accuracy: {:.1}%", best_accuracy);
    println!("\n=== ClassifierANT Mastery Runner Complete ===");
}
