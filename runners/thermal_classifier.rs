//! Thermal Classifier Runner
//!
//! Compares frozen hidden (classifier_train.rune) vs all-thermal
//! (thermal_classifier.rune) on 4-class 32-dim classification.
//!
//! Phase 1: Train both on normal data (10 cycles)
//! Phase 2: Train both on shifted data (10 cycles) — domain adaptation
//! Phase 3: Verify thermal predictions are consistent (stability test)

use std::sync::Arc;
use atomic_neural_transistors::{AtomicNeuralTransistor, Value};
use atomic_neural_transistors::testdata::{Rng, Sample, generate_class_prototypes, add_noise, add_shifted_noise};

fn eval_accuracy(ant: &mut AtomicNeuralTransistor, func: &str, data: &[Sample]) -> f64 {
    let mut correct = 0usize;
    for sample in data {
        let input: Vec<Value> = sample.signal.iter()
            .map(|s| Value::Integer(s.current() as i64))
            .collect();
        let result = ant.call_values(func, vec![Value::Array(Arc::new(input))])
            .expect("predict call failed");
        if let Value::Integer(predicted) = result {
            if predicted as usize == sample.class {
                correct += 1;
            }
        }
    }
    correct as f64 / data.len() as f64 * 100.0
}

fn train_frozen_cycle(ant: &mut AtomicNeuralTransistor, data: &[Sample]) {
    for sample in data {
        let mut values: Vec<Value> = sample.signal.iter()
            .map(|s| Value::Integer(s.current() as i64))
            .collect();
        values.push(Value::Integer(sample.class as i64));
        ant.call_values("train", vec![Value::Array(Arc::new(values))])
            .expect("train call failed");
    }
    ant.call_values("decay", vec![]).expect("decay failed");
}

fn train_thermal_cycle(ant: &mut AtomicNeuralTransistor, data: &[Sample]) {
    for sample in data {
        let input: Vec<Value> = sample.signal.iter()
            .map(|s| Value::Integer(s.current() as i64))
            .collect();
        let input_arr = Value::Array(Arc::new(input));

        // Hidden forward
        let h = ant.call_values("hidden_forward", vec![input_arr.clone()])
            .expect("hidden_forward failed");

        // Output forward
        let out = ant.call_values("output_forward", vec![h.clone()])
            .expect("output_forward failed");

        // Compute target and error
        let out_vals: Vec<i64> = match &out {
            Value::Array(arr) => arr.iter().map(|v| {
                if let Value::Integer(n) = v { *n } else { 0 }
            }).collect(),
            _ => vec![0; 4],
        };

        let target_out: Vec<i64> = (0..4).map(|c| {
            if c == sample.class { 127i64 } else { -127i64 }
        }).collect();

        let errors: Vec<i64> = (0..4).map(|c| {
            target_out[c] - out_vals[c].clamp(-127, 127)
        }).collect();

        let predicted = out_vals.iter().enumerate()
            .max_by_key(|(_, v)| **v)
            .map(|(i, _)| i)
            .unwrap_or(0);
        let correct_val = if predicted == sample.class { 1i64 } else { 0i64 };

        let target_out_val = Value::Array(Arc::new(
            target_out.iter().map(|&t| Value::Integer(t)).collect()
        ));

        // Train output region
        ant.call_values("train_output", vec![
            h.clone(), out.clone(), target_out_val, Value::Integer(correct_val),
        ]).expect("train_output failed");

        // Derive hidden targets: nudge from error sum
        let h_vals: Vec<i64> = match &h {
            Value::Array(arr) => arr.iter().map(|v| {
                if let Value::Integer(n) = v { *n } else { 0 }
            }).collect(),
            _ => vec![0; 24],
        };

        let total_error: i64 = errors.iter().sum();
        let nudge = (total_error / 4).clamp(-32, 32);

        let target_hidden: Vec<Value> = h_vals.iter().map(|&hv| {
            Value::Integer((hv + nudge).clamp(-127, 127))
        }).collect();
        let target_hidden_val = Value::Array(Arc::new(target_hidden));

        ant.call_values("train_hidden", vec![
            input_arr, h, target_hidden_val, Value::Integer(correct_val),
        ]).expect("train_hidden failed");
    }
    ant.call_values("decay_all", vec![]).expect("decay_all failed");
}

fn main() {
    println!("=== Thermal Classifier: Frozen vs Thermal ===\n");

    let mut rng = Rng::new(0xDEAD);
    let prototypes = generate_class_prototypes(&mut rng, 4, 32);

    // Generate datasets
    let train_data: Vec<Sample> = (0..800).map(|i| {
        let class = i % prototypes.len();
        Sample { signal: add_noise(&mut rng, &prototypes[class]), class }
    }).collect();

    let test_normal: Vec<Sample> = (0..200).map(|i| {
        let class = i % prototypes.len();
        Sample { signal: add_noise(&mut rng, &prototypes[class]), class }
    }).collect();

    let test_shifted: Vec<Sample> = (0..200).map(|i| {
        let class = i % prototypes.len();
        Sample { signal: add_shifted_noise(&mut rng, &prototypes[class]), class }
    }).collect();

    println!("Data: {} train, {} test (normal), {} test (shifted)\n",
        train_data.len(), test_normal.len(), test_shifted.len());

    // Load frozen network (classifier_train.rune)
    let mut frozen = AtomicNeuralTransistor::from_source(
        include_str!("../runes/classifier_train.rune")
    ).expect("Failed to load classifier_train.rune");

    // Load thermal network (thermal_classifier.rune)
    let mut thermal = AtomicNeuralTransistor::from_source(
        include_str!("../runes/thermal_classifier.rune")
    ).expect("Failed to load thermal_classifier.rune");

    println!("{:>6}  {:>10}  {:>10}  {:>10}  {:>10}  {:>12}",
        "Cycle", "Frz-Norm", "Frz-Shift", "Thm-Norm", "Thm-Shift", "Thm-Temp");

    // Phase 1: Train on normal data
    println!("\n--- Phase 1: Training on normal noise ---");
    for cycle in 1..=10 {
        train_frozen_cycle(&mut frozen, &train_data);
        train_thermal_cycle(&mut thermal, &train_data);

        let fn_acc = eval_accuracy(&mut frozen, "predict", &test_normal);
        let fs_acc = eval_accuracy(&mut frozen, "predict", &test_shifted);
        let tn_acc = eval_accuracy(&mut thermal, "predict", &test_normal);
        let ts_acc = eval_accuracy(&mut thermal, "predict", &test_shifted);

        let summary = thermal.call_values("summary_hidden", vec![])
            .unwrap_or(Value::Array(Arc::new(vec![])));
        let temp_str = match &summary {
            Value::Array(arr) if arr.len() >= 4 => {
                let vals: Vec<i64> = arr.iter().map(|v| {
                    if let Value::Integer(n) = v { *n } else { 0 }
                }).collect();
                format!("H={} W={} C={} D={}", vals[0], vals[1], vals[2], vals[3])
            }
            _ => "N/A".to_string(),
        };

        println!("{:>6}  {:>9.1}%  {:>9.1}%  {:>9.1}%  {:>9.1}%  {}",
            cycle, fn_acc, fs_acc, tn_acc, ts_acc, temp_str);
    }

    // Phase 2: Train on shifted data (domain adaptation)
    println!("\n--- Phase 2: Adapting to shifted noise ---");
    let train_shifted: Vec<Sample> = (0..800).map(|i| {
        let class = i % prototypes.len();
        Sample { signal: add_shifted_noise(&mut rng, &prototypes[class]), class }
    }).collect();

    for cycle in 1..=10 {
        train_frozen_cycle(&mut frozen, &train_shifted);
        train_thermal_cycle(&mut thermal, &train_shifted);

        let fn_acc = eval_accuracy(&mut frozen, "predict", &test_normal);
        let fs_acc = eval_accuracy(&mut frozen, "predict", &test_shifted);
        let tn_acc = eval_accuracy(&mut thermal, "predict", &test_normal);
        let ts_acc = eval_accuracy(&mut thermal, "predict", &test_shifted);

        let summary = thermal.call_values("summary_hidden", vec![])
            .unwrap_or(Value::Array(Arc::new(vec![])));
        let temp_str = match &summary {
            Value::Array(arr) if arr.len() >= 4 => {
                let vals: Vec<i64> = arr.iter().map(|v| {
                    if let Value::Integer(n) = v { *n } else { 0 }
                }).collect();
                format!("H={} W={} C={} D={}", vals[0], vals[1], vals[2], vals[3])
            }
            _ => "N/A".to_string(),
        };

        println!("{:>6}  {:>9.1}%  {:>9.1}%  {:>9.1}%  {:>9.1}%  {}",
            cycle + 10, fn_acc, fs_acc, tn_acc, ts_acc, temp_str);
    }

    // Phase 3: Consistency check (deterministic predictions)
    println!("\n--- Phase 3: Prediction Consistency ---");
    let mut consistent = 0usize;
    let n_check = 50;
    for sample in test_shifted.iter().take(n_check) {
        let input: Vec<Value> = sample.signal.iter()
            .map(|s| Value::Integer(s.current() as i64))
            .collect();
        let r1 = thermal.call_values("predict", vec![Value::Array(Arc::new(input.clone()))])
            .expect("predict failed");
        let r2 = thermal.call_values("predict", vec![Value::Array(Arc::new(input))])
            .expect("predict failed");
        if r1 == r2 {
            consistent += 1;
        }
    }
    println!("Deterministic: {}/{} predictions consistent", consistent, n_check);

    // Summary
    println!("\n--- Summary ---");
    let final_summary = thermal.call_values("summary_hidden", vec![])
        .unwrap_or(Value::Array(Arc::new(vec![])));
    if let Value::Array(arr) = &final_summary {
        if arr.len() >= 4 {
            let vals: Vec<i64> = arr.iter().map(|v| {
                if let Value::Integer(n) = v { *n } else { 0 }
            }).collect();
            println!("Thermal hidden temp: HOT={} WARM={} COOL={} COLD={}",
                vals[0], vals[1], vals[2], vals[3]);
        }
    }

    println!("\n=== Thermal Classifier Runner Complete ===");
}
