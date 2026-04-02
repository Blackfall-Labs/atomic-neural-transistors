//! Pipeline Planning Runner
//!
//! Two ANTs: ClassifierANT (classifier_train.rune) selects action,
//! CompareANT (compare_train.rune) verifies state change.
//! Train both independently, then compose into a planning loop.

use std::sync::Arc;
use atomic_neural_transistors::{AtomicNeuralTransistor, Signal, Value};
use atomic_neural_transistors::testdata::{Rng, generate_class_prototypes, add_noise};

fn main() {
    println!("=== Pipeline Planning Runner ===\n");

    let mut rng = Rng::new(0x91A4);
    let n_states = 4;
    let prototypes = generate_class_prototypes(&mut rng, n_states, 32);
    let num_classes = prototypes.len();

    // Load and train ClassifierANT
    let class_source = include_str!("../runes/classifier_train.rune");
    let mut classifier = AtomicNeuralTransistor::from_source(class_source)
        .expect("Failed to load classifier_train.rune");

    println!("Training ClassifierANT...");
    for _cycle in 0..5 {
        for i in 0..800 {
            let class = i % n_states;
            let signal = add_noise(&mut rng, &prototypes[class]);
            let mut values: Vec<Value> = signal.iter()
                .map(|s| Value::Integer(s.current() as i64))
                .collect();
            values.push(Value::Integer(class as i64));
            classifier.call_values("train", vec![Value::Array(Arc::new(values))])
                .expect("train call failed");
        }
        classifier.call_values("decay", vec![]).expect("decay failed");
    }

    // Load and train CompareANT
    let compare_source = include_str!("../runes/compare_train.rune");
    let mut comparer = AtomicNeuralTransistor::from_source(compare_source)
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

            comparer.call_values("train", vec![Value::Array(Arc::new(values))])
                .expect("train call failed");
        }
        comparer.call_values("decay", vec![]).expect("decay failed");
    }
    println!("Both ANTs trained.\n");

    // Verify classifier accuracy
    let mut class_correct = 0;
    for _ in 0..200 {
        let class = (rng.next() as usize) % n_states;
        let signal = add_noise(&mut rng, &prototypes[class]);
        let input: Vec<Value> = signal.iter()
            .map(|s| Value::Integer(s.current() as i64))
            .collect();
        let result = classifier.call_values("predict", vec![Value::Array(Arc::new(input))])
            .expect("predict failed");
        if let Value::Integer(predicted) = result {
            if predicted as usize == class {
                class_correct += 1;
            }
        }
    }
    println!("Classifier accuracy: {}/200 ({:.1}%)\n",
        class_correct, class_correct as f64 / 200.0 * 100.0);

    // Planning loop
    fn apply_action(current: usize, action: usize, n: usize) -> usize {
        (current + action) % n
    }

    fn compare_signals(
        comparer: &mut AtomicNeuralTransistor,
        a: &[Signal], b: &[Signal],
    ) -> bool {
        let input: Vec<Value> = a.iter().chain(b.iter())
            .map(|s| Value::Integer(s.current() as i64))
            .collect();
        let result = comparer.call_values("forward", vec![Value::Array(Arc::new(input))])
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

    let n_episodes = 100;
    let max_steps = 10;
    let mut successes = 0;
    let mut total_steps = 0;

    println!("Running {} planning episodes (max {} steps each)...\n", n_episodes, max_steps);

    for _ in 0..n_episodes {
        let start = (rng.next() as usize) % n_states;
        let mut goal = (rng.next() as usize) % n_states;
        while goal == start { goal = (rng.next() as usize) % n_states; }

        let mut state = start;
        let mut reached = false;

        for step in 0..max_steps {
            if state == goal {
                reached = true;
                total_steps += step;
                break;
            }

            // ClassifierANT identifies current state
            let state_signal = add_noise(&mut rng, &prototypes[state]);
            let input: Vec<Value> = state_signal.iter()
                .map(|s| Value::Integer(s.current() as i64))
                .collect();
            let result = classifier.call_values("predict", vec![Value::Array(Arc::new(input))])
                .expect("predict failed");
            let classified = if let Value::Integer(n) = result { n as usize } else { 0 };

            // Policy: action = (goal - classified) mod n_states
            let action = (goal + n_states - classified) % n_states;

            // Apply action
            let prev_state = state;
            state = apply_action(state, action, n_states);

            // CompareANT verifies state changed
            let prev_signal = add_noise(&mut rng, &prototypes[prev_state]);
            let new_signal = add_noise(&mut rng, &prototypes[state]);
            let _same = compare_signals(&mut comparer, &prev_signal, &new_signal);
        }

        if state == goal && !reached {
            reached = true;
            total_steps += max_steps;
        }
        if reached { successes += 1; }
    }

    let success_rate = successes as f64 / n_episodes as f64 * 100.0;
    let avg_steps = if successes > 0 { total_steps as f64 / successes as f64 } else { 0.0 };

    println!("Results:");
    println!("  Success rate: {successes}/{n_episodes} ({success_rate:.1}%)");
    println!("  Avg steps to goal: {avg_steps:.1}");
    println!("  Max steps allowed: {max_steps}");

    // Breakdown by distance
    println!("\nBy start-goal distance:");
    for dist in 1..n_states {
        let mut dist_success = 0;
        let mut dist_total = 0;
        for start in 0..n_states {
            let goal = (start + dist) % n_states;
            for _ in 0..10 {
                let mut state = start;
                let mut reached = false;
                for _step in 0..max_steps {
                    if state == goal { reached = true; break; }
                    let state_signal = add_noise(&mut rng, &prototypes[state]);
                    let input: Vec<Value> = state_signal.iter()
                        .map(|s| Value::Integer(s.current() as i64))
                        .collect();
                    let result = classifier.call_values("predict", vec![Value::Array(Arc::new(input))])
                        .expect("predict failed");
                    let classified = if let Value::Integer(n) = result { n as usize } else { 0 };
                    let action = (goal + n_states - classified) % n_states;
                    state = apply_action(state, action, n_states);
                }
                if state == goal { reached = true; }
                dist_total += 1;
                if reached { dist_success += 1; }
            }
        }
        println!("  Distance {dist}: {dist_success}/{dist_total} ({:.1}%)",
            dist_success as f64 / dist_total as f64 * 100.0);
    }

    println!("\n=== Pipeline Planning Runner Complete ===");
}
