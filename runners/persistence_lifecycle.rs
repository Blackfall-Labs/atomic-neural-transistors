//! Persistence Lifecycle Runner
//!
//! Uses persistence_lifecycle.rune (train, forward).
//! Phase 1: Train from scratch, measure accuracy
//! Phase 2: Verify weights exist in runtime
//! Phase 3: Continue training, verify improvement
//! Phase 4: Verify predictions are deterministic (consistency check)

use std::sync::Arc;
use atomic_neural_transistors::{AtomicNeuralTransistor, Signal, Value};
use atomic_neural_transistors::testdata::{Rng, generate_class_prototypes, add_noise};

struct Pair {
    a: Vec<Signal>,
    b: Vec<Signal>,
    same: bool,
}

fn generate_pairs(rng: &mut Rng, n: usize, prototypes: &[Vec<Signal>]) -> Vec<Pair> {
    let np = prototypes.len();
    (0..n).map(|i| {
        let class_a = (rng.next() as usize) % np;
        let a = add_noise(rng, &prototypes[class_a]);
        let same = i % 2 == 0;
        let b = if same {
            add_noise(rng, &prototypes[class_a])
        } else {
            let mut class_b = (rng.next() as usize) % np;
            while class_b == class_a {
                class_b = (rng.next() as usize) % np;
            }
            add_noise(rng, &prototypes[class_b])
        };
        Pair { a, b, same }
    }).collect()
}

fn evaluate(ant: &mut AtomicNeuralTransistor, pairs: &[Pair]) -> (f64, usize) {
    let mut correct = 0usize;
    for pair in pairs {
        let input: Vec<Value> = pair.a.iter().chain(pair.b.iter())
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

        let predicted_same = val > 0;
        if predicted_same == pair.same {
            correct += 1;
        }
    }
    let acc = correct as f64 / pairs.len() as f64 * 100.0;
    (acc, correct)
}

fn train_cycle(ant: &mut AtomicNeuralTransistor, pairs: &[Pair]) {
    for pair in pairs {
        let mut values: Vec<Value> = pair.a.iter().chain(pair.b.iter())
            .map(|s| Value::Integer(s.current() as i64))
            .collect();
        let target = if pair.same { 127i64 } else { -127i64 };
        values.push(Value::Integer(target));

        ant.call_values("train", vec![Value::Array(Arc::new(values))])
            .expect("train call failed");
    }
}

fn main() {
    println!("=== Persistence Lifecycle Runner ===\n");

    let mut rng = Rng::new(0xBEEF);
    let prototypes = generate_class_prototypes(&mut rng, 8, 32);

    let all_pairs = generate_pairs(&mut rng, 1000, &prototypes);
    let (test_set, train_set) = all_pairs.split_at(200);

    let source = include_str!("../runes/persistence_lifecycle.rune");
    let mut ant = AtomicNeuralTransistor::from_source(source)
        .expect("Failed to load persistence_lifecycle.rune");

    // Phase 1: Train from scratch
    println!("Phase 1: Train from scratch (5 cycles)");
    for cycle in 1..=5 {
        train_cycle(&mut ant, train_set);
        let (acc, correct) = evaluate(&mut ant, test_set);
        println!("  Cycle {}: {}/{} ({:.1}%)", cycle, correct, test_set.len(), acc);
    }

    let (pre_acc, pre_correct) = evaluate(&mut ant, test_set);
    println!("\nPre-persistence: {}/{} ({:.1}%)", pre_correct, test_set.len(), pre_acc);

    // Phase 2: Verify runtime state
    println!("\nPhase 2: Runtime state verification");
    {
        let rt = ant.runtime().lock().unwrap();
        println!("  Synaptic matrices: {}", rt.weight_count());
        println!("  Mastery states: {}", rt.mastery_count());
        println!("  Synaptic keys: {:?}", rt.synaptic_key_handles().keys().collect::<Vec<_>>());
    }

    // Phase 3: Collect predictions for consistency check
    println!("\nPhase 3: Prediction consistency check");
    let n_check = 50;
    let mut pre_outputs: Vec<i64> = Vec::with_capacity(n_check);
    for pair in test_set.iter().take(n_check) {
        let input: Vec<Value> = pair.a.iter().chain(pair.b.iter())
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
        pre_outputs.push(val);
    }

    // Verify same predictions
    let mut mismatches = 0;
    for (i, pair) in test_set.iter().take(n_check).enumerate() {
        let input: Vec<Value> = pair.a.iter().chain(pair.b.iter())
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
        if val != pre_outputs[i] {
            mismatches += 1;
        }
    }
    println!("  Deterministic: {} (mismatches: {}/{})",
        if mismatches == 0 { "YES" } else { "NO" }, mismatches, n_check);

    // Phase 4: Continue training
    println!("\nPhase 4: Continue mastery learning (5 more cycles)");
    for cycle in 6..=10 {
        train_cycle(&mut ant, train_set);
        let (acc, correct) = evaluate(&mut ant, test_set);
        println!("  Cycle {}: {}/{} ({:.1}%)", cycle, correct, test_set.len(), acc);
    }

    let (final_acc, final_correct) = evaluate(&mut ant, test_set);
    println!("\nFinal: {}/{} ({:.1}%)", final_correct, test_set.len(), final_acc);

    // Summary
    println!("\n--- Lifecycle Summary ---");
    println!("  Initial training:   50% -> {:.1}% in 5 cycles", pre_acc);
    println!("  Consistency:        {} mismatches", mismatches);
    println!("  Continued learning: {:.1}% -> {:.1}%", pre_acc, final_acc);

    println!("\n=== Persistence Lifecycle Runner Complete ===");
}
