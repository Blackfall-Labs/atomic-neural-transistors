//! Multiplex Classification Runner
//!
//! Uses multiplex_features.rune with 3 ANT feature extractors:
//! product_features, projection_features, identity_features.
//! Orchestrates salience routing, prediction engine, and neuromodulator gating.
//! Trains for 20 cycles, prints routing stats.

use std::sync::Arc;
use atomic_neural_transistors::{AtomicNeuralTransistor, Value};
use atomic_neural_transistors::testdata::{Rng, generate_class_prototypes, generate_dataset};

fn main() {
    println!("=== Multiplex Classification Runner ===\n");

    let mut rng = Rng::new(0xABC_DEF_123);

    let prototypes = generate_class_prototypes(&mut rng, 4, 32);
    let train_data = generate_dataset(&mut rng, 500, &prototypes);
    let test_data = generate_dataset(&mut rng, 200, &prototypes);

    // Load multiplex features rune
    let source = include_str!("../runes/multiplex_features.rune");
    let mut ant = AtomicNeuralTransistor::from_source(source)
        .expect("Failed to load multiplex_features.rune");

    // Create handles for salience, prediction, neuromod
    let nm_handle = ant.call_values("create_neuromod", vec![])
        .expect("create_neuromod failed");
    let pred_handle = ant.call_values("create_predictor", vec![
        Value::Integer(32), Value::Integer(2), Value::Integer(40),
    ]).expect("create_predictor failed");
    let sal_handle = ant.call_values("create_salience", vec![
        Value::Integer(3), Value::Integer(32),
    ]).expect("create_salience failed");

    // Class targets: Hadamard patterns
    let class_targets: Vec<Vec<Value>> = (0..4).map(|c| {
        (0..32).map(|d| {
            let bits = ((c ^ d) as u32).count_ones();
            let pol: i32 = if bits % 2 == 0 { 1 } else { -1 };
            Value::Integer((pol * 100) as i64)
        }).collect()
    }).collect();

    println!("Architecture:");
    println!("  ANT 0: product features vs prototype 0");
    println!("  ANT 1: projection features (frozen random)");
    println!("  ANT 2: identity features (normalized)");
    println!("  Salience, Prediction, Neuromod via Runes verbs");
    println!();

    let mut surprise_count = 0u32;
    let mut learning_count = 0u32;

    // Training loop: 20 cycles
    println!("--- Training ({} samples, 20 cycles) ---", train_data.len());
    for cycle in 1..=20 {
        let mut cycle_correct = 0usize;

        for sample in &train_data {
            let input: Vec<Value> = sample.signal.iter()
                .map(|s| Value::Integer(s.current() as i64))
                .collect();
            let input_arr = Value::Array(Arc::new(input.clone()));

            // ANT 0: product features (input ++ prototype[0])
            let proto0: Vec<Value> = prototypes[0].iter()
                .map(|s| Value::Integer(s.current() as i64))
                .collect();
            let mut data0 = input.clone();
            data0.extend(proto0);
            let out0 = ant.call_values("product_features", vec![Value::Array(Arc::new(data0))])
                .expect("product_features failed");

            // ANT 1: projection features
            let out1 = ant.call_values("projection_features", vec![input_arr.clone()])
                .expect("projection_features failed");

            // ANT 2: identity features
            let out2 = ant.call_values("identity_features", vec![input_arr.clone()])
                .expect("identity_features failed");

            // Concatenate all outputs for salience routing
            let cat01 = ant.call_values("join", vec![out0, out1])
                .expect("join failed");
            let all_outputs = ant.call_values("join", vec![cat01, out2])
                .expect("join failed");

            // Route through salience
            let route_result = ant.call_values("route", vec![sal_handle.clone(), all_outputs])
                .expect("route failed");

            // Extract routed output (first 32 elements)
            let routed = ant.call_values("extract", vec![route_result, Value::Integer(0), Value::Integer(32)])
                .expect("extract failed");

            // Prediction: observe with target
            let target = Value::Array(Arc::new(class_targets[sample.class].clone()));
            let surprise = ant.call_values("observe", vec![
                pred_handle.clone(), routed.clone(), target,
            ]).expect("observe failed");

            let is_surprising = match &surprise {
                Value::Array(arr) => {
                    if let Some(Value::Integer(v)) = arr.get(1) { *v != 0 } else { false }
                }
                _ => false,
            };

            if is_surprising {
                surprise_count += 1;
                learning_count += 1;
            }

            // Check classification: compare routed output to class targets
            let routed_vals: Vec<i64> = match &routed {
                Value::Array(arr) => arr.iter().map(|v| {
                    if let Value::Integer(n) = v { *n } else { 0 }
                }).collect(),
                _ => vec![0; 32],
            };

            let predicted = (0..4).min_by_key(|&c| {
                class_targets[c].iter().zip(routed_vals.iter()).map(|(t, o)| {
                    let tv = if let Value::Integer(n) = t { *n } else { 0 };
                    (tv - o).abs()
                }).sum::<i64>()
            }).unwrap_or(0);

            if predicted == sample.class {
                cycle_correct += 1;
            }

            // Tick neuromod
            ant.call_values("tick", vec![nm_handle.clone()]).ok();
        }

        let acc = cycle_correct as f64 / train_data.len() as f64 * 100.0;
        if cycle % 5 == 0 || cycle == 1 {
            println!("  Cycle {:2}: accuracy {:.1}%", cycle, acc);
        }
    }

    // Evaluation on test set
    println!("\n--- Evaluation ({} test samples) ---", test_data.len());
    let mut test_correct = 0usize;

    for sample in &test_data {
        let input: Vec<Value> = sample.signal.iter()
            .map(|s| Value::Integer(s.current() as i64))
            .collect();
        let input_arr = Value::Array(Arc::new(input.clone()));

        let proto0: Vec<Value> = prototypes[0].iter()
            .map(|s| Value::Integer(s.current() as i64))
            .collect();
        let mut data0 = input.clone();
        data0.extend(proto0);
        let out0 = ant.call_values("product_features", vec![Value::Array(Arc::new(data0))])
            .expect("product_features failed");
        let out1 = ant.call_values("projection_features", vec![input_arr.clone()])
            .expect("projection_features failed");
        let out2 = ant.call_values("identity_features", vec![input_arr])
            .expect("identity_features failed");

        let cat01 = ant.call_values("join", vec![out0, out1]).expect("join failed");
        let all_outputs = ant.call_values("join", vec![cat01, out2]).expect("join failed");

        let route_result = ant.call_values("route", vec![sal_handle.clone(), all_outputs])
            .expect("route failed");
        let routed = ant.call_values("extract", vec![route_result, Value::Integer(0), Value::Integer(32)])
            .expect("extract failed");

        let routed_vals: Vec<i64> = match &routed {
            Value::Array(arr) => arr.iter().map(|v| {
                if let Value::Integer(n) = v { *n } else { 0 }
            }).collect(),
            _ => vec![0; 32],
        };

        let predicted = (0..4).min_by_key(|&c| {
            class_targets[c].iter().zip(routed_vals.iter()).map(|(t, o)| {
                let tv = if let Value::Integer(n) = t { *n } else { 0 };
                (tv - o).abs()
            }).sum::<i64>()
        }).unwrap_or(0);

        if predicted == sample.class {
            test_correct += 1;
        }
    }

    let test_acc = test_correct as f64 / test_data.len() as f64 * 100.0;
    println!("  Test accuracy: {:.1}% ({}/{})", test_acc, test_correct, test_data.len());
    println!("  Surprises detected: {}", surprise_count);
    println!("  Learning moments: {}", learning_count);

    // Read neuromod state
    let da = ant.call_values("read_chem", vec![nm_handle.clone(), Value::String("da".into())])
        .unwrap_or(Value::Integer(128));
    println!("\n  DA final: {:?}", da);

    println!("\n=== Multiplex Classification Runner Complete ===");
}
