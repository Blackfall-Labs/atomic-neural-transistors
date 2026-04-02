//! Debug Mastery Runner
//!
//! Simple diagnostic runner for mastery learning mechanics.
//! Test 1: minimal config — call train_minimal 10 times, print inspect_minimal
//! Test 2: production config — call train_production 10 times, print inspect_production

use std::sync::Arc;
use atomic_neural_transistors::{AtomicNeuralTransistor, Value};

fn main() {
    println!("=== Debug Mastery Runner ===\n");

    let source = include_str!("../runes/debug_mastery.rune");
    let mut ant = AtomicNeuralTransistor::from_source(source)
        .expect("Failed to load debug_mastery.rune");

    // Generate simple test data: input = [64, 64], target = [127]
    let data: Vec<Value> = vec![
        Value::Integer(64),
        Value::Integer(64),
        Value::Integer(127),
    ];
    let data_arr = Value::Array(Arc::new(data));

    // Test 1: minimal config
    println!("--- Test 1: Minimal Config (threshold=1, decay=0, gate=0) ---");
    for step in 1..=10 {
        let result = ant.call_values("train_minimal", vec![data_arr.clone()])
            .expect("train_minimal failed");
        let output_str = match &result {
            Value::Array(arr) => {
                let vals: Vec<String> = arr.iter().map(|v| {
                    if let Value::Integer(n) = v { n.to_string() } else { "?".to_string() }
                }).collect();
                format!("[{}]", vals.join(", "))
            }
            Value::Integer(n) => format!("{}", n),
            _ => format!("{:?}", result),
        };
        println!("  Step {:2}: output = {}", step, output_str);
    }

    let state = ant.call_values("inspect_minimal", vec![])
        .expect("inspect_minimal failed");
    println!("  State: {:?}\n", state);

    // Test 2: production config
    println!("--- Test 2: Production Config (threshold=3, decay=1, gate=5) ---");
    for step in 1..=10 {
        let result = ant.call_values("train_production", vec![data_arr.clone()])
            .expect("train_production failed");
        let output_str = match &result {
            Value::Array(arr) => {
                let vals: Vec<String> = arr.iter().map(|v| {
                    if let Value::Integer(n) = v { n.to_string() } else { "?".to_string() }
                }).collect();
                format!("[{}]", vals.join(", "))
            }
            Value::Integer(n) => format!("{}", n),
            _ => format!("{:?}", result),
        };
        println!("  Step {:2}: output = {}", step, output_str);
    }

    let state = ant.call_values("inspect_production", vec![])
        .expect("inspect_production failed");
    println!("  State: {:?}", state);

    // Check runtime diagnostics
    let rt = ant.runtime().lock().unwrap();
    println!("\n--- Runtime Summary ---");
    println!("  Synaptic matrices: {}", rt.weight_count());
    println!("  Mastery states: {}", rt.mastery_count());

    println!("\n=== Debug Mastery Runner Complete ===");
}
