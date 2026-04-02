//! Composition: Sudoku Constraint Validation Runner
//!
//! Trains a CompareANT using compare_train.rune, then validates 4x4 Sudoku grids:
//! check no duplicates in each row, column, and 2x2 box.

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

fn has_duplicate(ant: &mut AtomicNeuralTransistor, group: &[&Vec<Signal>]) -> bool {
    for i in 0..group.len() {
        for j in (i + 1)..group.len() {
            if compare(ant, group[i], group[j]) {
                return true;
            }
        }
    }
    false
}

fn valid_sudoku(ant: &mut AtomicNeuralTransistor, grid: &[Vec<Signal>; 16]) -> bool {
    // Check 4 rows
    for r in 0..4 {
        let row: Vec<&Vec<Signal>> = (0..4).map(|c| &grid[r * 4 + c]).collect();
        if has_duplicate(ant, &row) { return false; }
    }
    // Check 4 columns
    for c in 0..4 {
        let col: Vec<&Vec<Signal>> = (0..4).map(|r| &grid[r * 4 + c]).collect();
        if has_duplicate(ant, &col) { return false; }
    }
    // Check 4 boxes (2x2)
    for box_r in 0..2 {
        for box_c in 0..2 {
            let bx: Vec<&Vec<Signal>> = vec![
                &grid[(box_r * 2) * 4 + box_c * 2],
                &grid[(box_r * 2) * 4 + box_c * 2 + 1],
                &grid[(box_r * 2 + 1) * 4 + box_c * 2],
                &grid[(box_r * 2 + 1) * 4 + box_c * 2 + 1],
            ];
            if has_duplicate(ant, &bx) { return false; }
        }
    }
    true
}

fn generate_valid_grid(rng: &mut Rng) -> [usize; 16] {
    let base: [usize; 16] = [
        0, 1, 2, 3,
        2, 3, 0, 1,
        1, 0, 3, 2,
        3, 2, 1, 0,
    ];
    let mut perm = [0usize, 1, 2, 3];
    for i in (1..4).rev() {
        let j = (rng.next() as usize) % (i + 1);
        perm.swap(i, j);
    }
    let mut grid = [0usize; 16];
    for i in 0..16 {
        grid[i] = perm[base[i]];
    }
    if rng.next() & 1 == 1 {
        for c in 0..4 { grid.swap(c, 4 + c); }
    }
    if rng.next() & 1 == 1 {
        for c in 0..4 { grid.swap(8 + c, 12 + c); }
    }
    grid
}

fn make_invalid_grid(rng: &mut Rng, valid: &[usize; 16]) -> [usize; 16] {
    let mut grid = *valid;
    let row = (rng.next() as usize) % 4;
    let c1 = (rng.next() as usize) % 4;
    let mut c2 = (rng.next() as usize) % 4;
    while c2 == c1 || grid[row * 4 + c1] == grid[row * 4 + c2] {
        c2 = (rng.next() as usize) % 4;
    }
    grid[row * 4 + c2] = grid[row * 4 + c1];
    grid
}

fn main() {
    println!("=== Composition: Sudoku Constraint Validation Runner ===\n");

    let mut rng = Rng::new(0x50D0);

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

    // Generate Sudoku prototypes (4 values)
    let sudoku_protos = generate_class_prototypes(&mut rng, 4, 32);

    // Generate test grids
    let n_samples = 200;
    let mut correct = 0usize;
    let mut false_valid = 0usize;
    let mut false_invalid = 0usize;

    for i in 0..n_samples {
        let valid_grid = generate_valid_grid(&mut rng);
        let (values, is_valid) = if i % 2 == 0 {
            (valid_grid, true)
        } else {
            (make_invalid_grid(&mut rng, &valid_grid), false)
        };

        let grid: [Vec<Signal>; 16] = std::array::from_fn(|idx| {
            add_noise(&mut rng, &sudoku_protos[values[idx]])
        });

        let predicted_valid = valid_sudoku(&mut ant, &grid);
        if predicted_valid == is_valid {
            correct += 1;
        } else if predicted_valid {
            false_valid += 1;
        } else {
            false_invalid += 1;
        }
    }

    let accuracy = correct as f64 / n_samples as f64 * 100.0;
    println!("Results:");
    println!("  Accuracy:     {correct}/{n_samples} ({accuracy:.1}%)");
    println!("  False valid:  {false_valid} (accepted invalid grid)");
    println!("  False invalid: {false_invalid} (rejected valid grid)");

    let total_comparisons = n_samples * 72;
    println!("\n  Total comparisons: {total_comparisons}");
    println!("  Comparisons per grid: 72");

    println!("\n=== Composition: Sudoku Runner Complete ===");
}
