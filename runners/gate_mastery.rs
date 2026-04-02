//! GateANT Mastery Runner
//!
//! Trains a gating network on 32-dim ternary signals via Runes scripting.
//! Input: signal(32) ++ control(32) = 64-dim
//! Target: per-dimension gate mask (255 = pass, 0 = block)
//! Evaluation: apply gate output to signal, check masked dims are near-zero.

use std::sync::Arc;
use atomic_neural_transistors::{AtomicNeuralTransistor, Signal, Value};
use atomic_neural_transistors::testdata::Rng;

fn main() {
    println!("=== GateANT Mastery Runner ===\n");

    let mut rng = Rng::new(0x6A7E);

    // Generate gating data
    // Each sample: random signal, random control, random mask of dims to pass/block
    let n_total = 1000;
    let n_test = 200;
    let dim = 32;

    struct GateSample {
        signal: Vec<Signal>,
        control: Vec<Signal>,
        target: Vec<i64>,   // per-dim: 255 = pass, 0 = block
        mask: Vec<bool>,    // which dims are passed
    }

    let mut samples: Vec<GateSample> = Vec::with_capacity(n_total);
    for _ in 0..n_total {
        // Random signal
        let signal: Vec<Signal> = (0..dim).map(|_| {
            let mag = 64 + (rng.next_u8() % 128);
            let pol: i8 = if rng.next() & 1 == 0 { 1 } else { -1 };
            Signal::new_raw(pol, mag, 1)
        }).collect();

        // Random mask: roughly half dims passed, half blocked
        let mask: Vec<bool> = (0..dim).map(|_| rng.next() & 1 == 0).collect();

        // Control signal encodes the mask pattern
        // Passed dims get positive control, blocked dims get negative
        let control: Vec<Signal> = mask.iter().map(|&pass| {
            if pass {
                Signal::new_raw(1, 128 + (rng.next_u8() % 64), 1)
            } else {
                Signal::new_raw(-1, 128 + (rng.next_u8() % 64), 1)
            }
        }).collect();

        // Target gate values: 255 for pass, 0 for block
        let target: Vec<i64> = mask.iter().map(|&pass| {
            if pass { 255 } else { 0 }
        }).collect();

        samples.push(GateSample { signal, control, target, mask });
    }

    let (test_set, train_set) = samples.split_at(n_test);
    println!("Data: {} training, {} test\n", train_set.len(), test_set.len());

    // Load ANT from rune source
    let source = include_str!("../runes/gate_train.rune");
    let mut ant = AtomicNeuralTransistor::from_source(source)
        .expect("Failed to load gate_train.rune");

    let max_cycles = 50;
    let mut best_accuracy = 0.0f64;

    for cycle in 1..=max_cycles {
        // Train on each sample
        for sample in train_set.iter() {
            // Pack [signal(32) ++ control(32) | target(32)] = 96 values
            let mut values: Vec<Value> = sample.signal.iter()
                .chain(sample.control.iter())
                .map(|s| Value::Integer(s.current() as i64))
                .collect();
            for &t in &sample.target {
                values.push(Value::Integer(t));
            }

            ant.call_values("train", vec![Value::Array(Arc::new(values))])
                .expect("train call failed");
        }

        // Decay mastery pressure
        ant.call_values("decay", vec![])
            .expect("decay call failed");

        // Evaluate on test set
        // Check: for each dim, if mask says block, the gate output should be < 128
        //         if mask says pass, the gate output should be >= 128
        let mut correct_dims = 0usize;
        let total_dims = test_set.len() * dim;

        for sample in test_set.iter() {
            let input: Vec<Value> = sample.signal.iter()
                .chain(sample.control.iter())
                .map(|s| Value::Integer(s.current() as i64))
                .collect();

            let result = ant.call_values("forward", vec![Value::Array(Arc::new(input))])
                .expect("forward call failed");

            let gate_vals: Vec<i64> = match result {
                Value::Array(arr) => {
                    arr.iter().map(|v| {
                        if let Value::Integer(n) = v { *n } else { 0 }
                    }).collect()
                }
                _ => vec![0; dim],
            };

            for (d, &pass) in sample.mask.iter().enumerate() {
                if d < gate_vals.len() {
                    let gate_val = gate_vals[d];
                    let correct = if pass { gate_val >= 128 } else { gate_val < 128 };
                    if correct {
                        correct_dims += 1;
                    }
                }
            }
        }

        let acc = correct_dims as f64 / total_dims as f64 * 100.0;
        if acc > best_accuracy {
            best_accuracy = acc;
        }

        println!(
            "Cycle {:2}: gate accuracy {:.1}% ({}/{} dims correct)",
            cycle, acc, correct_dims, total_dims
        );

        if acc >= 95.0 {
            println!("\nTarget gate accuracy 95%+ reached at cycle {}!", cycle);
            break;
        }
    }

    println!("\nBest gate accuracy: {:.1}%", best_accuracy);
    println!("\n=== GateANT Mastery Runner Complete ===");
}
