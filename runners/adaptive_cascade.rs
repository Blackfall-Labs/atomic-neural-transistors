//! Adaptive Cascade Runner
//!
//! Phase 1: Specialization — train product-feature ANTs via multiplex
//! Phase 2: Distribution shift — flip polarity on dims 0-15
//! Phase 3: Cascaded verification — multiplex output feeds cascade_features.rune verifier

use std::sync::Arc;
use atomic_neural_transistors::{AtomicNeuralTransistor, Signal, Value};
use atomic_neural_transistors::testdata::{Rng, generate_class_prototypes, generate_dataset};

fn main() {
    println!("=== Adaptive Cascade Runner ===");
    println!("  3 phases: Specialization -> Adaptation -> Cascade Verification\n");

    let mut rng = Rng::new(0xCAFE_F00D);

    let prototypes = generate_class_prototypes(&mut rng, 4, 32);

    // Class targets for distance-based classification
    let class_targets: Vec<Vec<i64>> = (0..4).map(|c| {
        (0..32).map(|d| {
            let bits = ((c ^ d) as u32).count_ones();
            let pol: i64 = if bits % 2 == 0 { 1 } else { -1 };
            pol * 100
        }).collect()
    }).collect();

    // Load multiplex features and cascade features runes
    let mux_source = include_str!("../runes/multiplex_features.rune");
    let mut mux = AtomicNeuralTransistor::from_source(mux_source)
        .expect("Failed to load multiplex_features.rune");

    let cascade_source = include_str!("../runes/cascade_features.rune");
    let mut cascade = AtomicNeuralTransistor::from_source(cascade_source)
        .expect("Failed to load cascade_features.rune");

    // Create handles
    let nm_handle = mux.call_values("neuromod_new", vec![])
        .expect("create_neuromod failed");
    let pred_handle = mux.call_values("predict_new", vec![
        Value::Integer(32), Value::Integer(2), Value::Integer(20),
    ]).expect("create_predictor failed");
    let sal_handle = mux.call_values("salience_new", vec![
        Value::Integer(3), Value::Integer(32),
    ]).expect("create_salience failed");

    // Helper: process one sample through multiplex pipeline
    fn process_mux(
        mux: &mut AtomicNeuralTransistor,
        signal: &[Signal],
        protos: &[Vec<Signal>],
        sal_handle: &Value,
        pred_handle: &Value,
        nm_handle: &Value,
        target: Option<&[i64]>,
    ) -> (Vec<i64>, bool) {
        let input: Vec<Value> = signal.iter()
            .map(|s| Value::Integer(s.current() as i64))
            .collect();
        let input_arr = Value::Array(Arc::new(input.clone()));

        // 3 ANTs: product vs proto[0], projection, identity
        let proto0: Vec<Value> = protos[0].iter()
            .map(|s| Value::Integer(s.current() as i64))
            .collect();
        let mut data0 = input.clone();
        data0.extend(proto0);
        let out0 = mux.call_values("product_features", vec![Value::Array(Arc::new(data0))])
            .expect("product_features failed");
        let out1 = mux.call_values("projection_features", vec![input_arr.clone()])
            .expect("projection_features failed");
        let out2 = mux.call_values("identity_features", vec![input_arr])
            .expect("identity_features failed");

        let cat01 = mux.call_values("concat", vec![out0, out1]).expect("join failed");
        let all = mux.call_values("concat", vec![cat01, out2]).expect("join failed");

        let routed = mux.call_values("salience_route", vec![sal_handle.clone(), all])
            .expect("route failed");
        let output = mux.call_values("slice", vec![routed, Value::Integer(0), Value::Integer(32)])
            .expect("extract failed");

        // Prediction
        let target_val = match target {
            Some(t) => Value::Array(Arc::new(t.iter().map(|&v| Value::Integer(v)).collect())),
            None => Value::Nil,
        };
        let surprise = mux.call_values("predict_observe", vec![
            pred_handle.clone(), output.clone(), target_val,
        ]).expect("observe failed");

        let is_surprising = match &surprise {
            Value::Array(arr) => {
                if let Some(Value::Integer(v)) = arr.get(1) { *v != 0 } else { false }
            }
            _ => false,
        };

        mux.call_values("neuromod_tick", vec![nm_handle.clone()]).ok();

        let vals: Vec<i64> = match &output {
            Value::Array(arr) => arr.iter().map(|v| {
                if let Value::Integer(n) = v { *n } else { 0 }
            }).collect(),
            _ => vec![0; 32],
        };

        (vals, is_surprising)
    }

    fn classify_by_distance(output: &[i64], targets: &[Vec<i64>]) -> usize {
        targets.iter().enumerate()
            .min_by_key(|(_, tgt)| {
                output.iter().zip(tgt.iter())
                    .map(|(o, t)| (o - t).abs())
                    .sum::<i64>()
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    // Phase 1: Specialization (160 samples)
    println!("--- Phase 1: Specialization (160 samples) ---");
    let phase1_data = generate_dataset(&mut rng, 160, &prototypes);
    let mut p1_surprises = 0u32;

    for sample in &phase1_data {
        let target = &class_targets[sample.class];
        let (_output, is_surprising) = process_mux(
            &mut mux, &sample.signal, &prototypes,
            &sal_handle, &pred_handle, &nm_handle, Some(target),
        );
        if is_surprising { p1_surprises += 1; }

        // Train verifier on multiplex output
        let proto0: Vec<Value> = prototypes[0].iter()
            .map(|s| Value::Integer(s.current() as i64))
            .collect();
        let mut data0: Vec<Value> = sample.signal.iter()
            .map(|s| Value::Integer(s.current() as i64)).collect();
        data0.extend(proto0);
        let out0 = mux.call_values("product_features", vec![Value::Array(Arc::new(data0))])
            .expect("product_features failed");
        // Use out0 as verifier input (simplified cascade)
        let out0_vals: Vec<Value> = match &out0 {
            Value::Array(arr) => arr.iter().cloned().collect(),
            _ => vec![],
        };

        // Train verifier: pack [input(32) | onehot_target(4)]
        let onehot: Vec<Value> = (0..4).map(|c| {
            if c == sample.class { Value::Integer(127) } else { Value::Integer(-127) }
        }).collect();
        let mut ver_data = out0_vals;
        ver_data.extend(onehot);
        cascade.call_values("verifier_train", vec![Value::Array(Arc::new(ver_data))])
            .expect("verifier_train failed");
    }

    println!("  Surprises: {}", p1_surprises);

    // Phase 2: Distribution shift (80 samples)
    println!("\n--- Phase 2: Distribution Shift (80 samples) ---");
    println!("  Polarity inversion on dims 0-15");

    let shifted_protos: Vec<Vec<Signal>> = prototypes.iter().map(|proto| {
        proto.iter().enumerate().map(|(d, s)| {
            if d < 16 {
                Signal::from_current(-s.current())
            } else {
                *s
            }
        }).collect()
    }).collect();

    let phase2_data = generate_dataset(&mut rng, 80, &shifted_protos);
    let mut p2_surprises = 0u32;

    for sample in &phase2_data {
        let target = &class_targets[sample.class];
        let (_output, is_surprising) = process_mux(
            &mut mux, &sample.signal, &prototypes,
            &sal_handle, &pred_handle, &nm_handle, Some(target),
        );
        if is_surprising { p2_surprises += 1; }
    }

    println!("  Surprises: {}", p2_surprises);

    // Phase 3: Cascaded verification (60 samples, no learning)
    println!("\n--- Phase 3: Cascaded Verification (60 samples) ---");
    let phase3_data = generate_dataset(&mut rng, 60, &shifted_protos);
    let mut mux_correct = 0u32;
    let mut ver_correct = 0u32;
    let mut agree = 0u32;

    for sample in &phase3_data {
        let (output, _) = process_mux(
            &mut mux, &sample.signal, &prototypes,
            &sal_handle, &pred_handle, &nm_handle, None,
        );

        let mux_class = classify_by_distance(&output, &class_targets);

        // Verifier prediction
        let ver_input: Vec<Value> = output.iter().map(|&v| Value::Integer(v)).collect();
        let ver_pred = cascade.call_values("verifier_predict", vec![Value::Array(Arc::new(ver_input))])
            .expect("verifier_predict failed");
        let ver_class = match ver_pred {
            Value::Integer(n) => n as usize,
            _ => 0,
        };

        if mux_class == sample.class { mux_correct += 1; }
        if ver_class == sample.class { ver_correct += 1; }
        if mux_class == ver_class { agree += 1; }
    }

    let total = phase3_data.len() as u32;
    println!("  Multiplex accuracy: {:.1}% ({}/{})",
        mux_correct as f64 / total as f64 * 100.0, mux_correct, total);
    println!("  Verifier accuracy:  {:.1}% ({}/{})",
        ver_correct as f64 / total as f64 * 100.0, ver_correct, total);
    println!("  Agreement rate:     {:.1}% ({}/{})",
        agree as f64 / total as f64 * 100.0, agree, total);

    let da = mux.call_values("neuromod_read", vec![nm_handle.clone(), Value::String("da".into())])
        .unwrap_or(Value::Integer(128));
    println!("  DA final: {:?}", da);

    println!("\n=== Adaptive Cascade Runner Complete ===");
}
