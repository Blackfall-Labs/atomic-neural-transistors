//! Composition algebra example
//!
//! Shows how complex operations compose from primitives without training.

use atomic_neural_transistors::composition::{
    all_unique, contains, count_occurrences, find_positions, has_duplicate,
    find_all_objects, grids_equal, PerfectEquality,
};

fn main() {
    println!("=== ANT Composition Algebra ===\n");

    // Use perfect equality for demo (real ANTs achieve 99.5%+)
    let checker = PerfectEquality;

    // Sequence operations
    println!("Sequence Operations:");
    println!("-------------------");

    let sequence = [1, 2, 3, 5, 7, 2, 9];
    println!("Sequence: {:?}\n", sequence);

    // contains = OR(AreEqual(query, seq[i]) for all i)
    println!("contains(5, seq) = {}", contains(&checker, 5, &sequence));
    println!("contains(4, seq) = {}", contains(&checker, 4, &sequence));

    // has_duplicate = OR(AreEqual(seq[i], seq[j]) for all i < j)
    println!("\nhas_duplicate(seq) = {}", has_duplicate(&checker, &sequence));
    println!("has_duplicate([1,2,3]) = {}", has_duplicate(&checker, &[1, 2, 3]));

    // all_unique = NOT(has_duplicate)
    println!("\nall_unique(seq) = {}", all_unique(&checker, &sequence));
    println!("all_unique([1,2,3]) = {}", all_unique(&checker, &[1, 2, 3]));

    // count_occurrences = SUM(AreEqual > 0.5)
    println!("\ncount_occurrences(2, seq) = {}", count_occurrences(&checker, 2, &sequence));

    // find_positions
    println!("find_positions(2, seq) = {:?}", find_positions(&checker, 2, &sequence));

    // Grid operations
    println!("\n\nGrid Operations:");
    println!("----------------");

    let grid_a = vec![
        vec![1, 1, 0, 2],
        vec![1, 0, 0, 2],
        vec![0, 0, 3, 0],
    ];

    let grid_b = grid_a.clone();
    let grid_c = vec![
        vec![1, 1, 0, 2],
        vec![1, 0, 0, 2],
        vec![0, 0, 4, 0],  // Different!
    ];

    println!("Grid A:");
    for row in &grid_a {
        println!("  {:?}", row);
    }

    println!("\ngrids_equal(A, B) = {}", grids_equal(&checker, &grid_a, &grid_b));
    println!("grids_equal(A, C) = {}", grids_equal(&checker, &grid_a, &grid_c));

    // Object detection
    println!("\nfind_all_objects(A, background=0):");
    let objects = find_all_objects(&checker, &grid_a, 0);
    for (i, obj) in objects.iter().enumerate() {
        println!("  Object {}: value={}, size={}, positions={:?}",
            i, obj.value, obj.size(), obj.positions);
    }

    println!("\n\nKey Insight:");
    println!("------------");
    println!("All these operations are COMPOSED from a single AreEqual ANT.");
    println!("No additional training required - just loop and aggregate!");
    println!("\ncontains = loop(AreEqual) + max");
    println!("has_duplicate = pairwise(AreEqual) + max");
    println!("count = loop(AreEqual > 0.5) + sum");
}
