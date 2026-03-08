//! Composition: Sudoku Constraint Validation — Example 8
//!
//! Composes a trained CompareANT into a Sudoku constraint validator
//! without any additional mastery learning. Checks that a 4×4 mini-Sudoku
//! grid has no duplicates in any row, column, or 2×2 box.
//!
//! Pipeline:
//!   valid_sudoku(grid) = AND(
//!     NOT has_duplicate(row[i]) for all rows,
//!     NOT has_duplicate(col[j]) for all cols,
//!     NOT has_duplicate(box[k]) for all boxes
//!   )
//!
//! where has_duplicate uses the CompareANT from Example 6.
//!
//! This demonstrates that algebraic composition scales to complex
//! constraint satisfaction: a single trained comparison primitive composes
//! into a full constraint checker.

use atomic_neural_transistors::core::weight_matrix::{packed_from_current, WeightMatrix};
use atomic_neural_transistors::learning::{MasteryConfig, MasteryState};
use atomic_neural_transistors::PackedSignal;

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Self(seed) }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn next_u8(&mut self) -> u8 { (self.next() & 0xFF) as u8 }
}

fn matmul(input: &[PackedSignal], w: &WeightMatrix) -> Vec<PackedSignal> {
    let mut output = Vec::with_capacity(w.rows);
    for i in 0..w.rows {
        let mut sum: i64 = 0;
        for j in 0..w.cols {
            sum += input[j].current() as i64 * w.data[i * w.cols + j].current() as i64;
        }
        output.push(packed_from_current(sum.clamp(i32::MIN as i64, i32::MAX as i64) as i32));
    }
    output
}

fn comparison_features(a: &[PackedSignal], b: &[PackedSignal]) -> Vec<PackedSignal> {
    a.iter().zip(b.iter()).map(|(x, y)| {
        let product = x.current() as i64 * y.current() as i64;
        packed_from_current((product / 256) as i32)
    }).collect()
}

struct CompareNetwork { w_out: WeightMatrix }

impl CompareNetwork {
    fn compare(&self, a: &[PackedSignal], b: &[PackedSignal]) -> bool {
        let features = comparison_features(a, b);
        let out = matmul(&features, &self.w_out);
        out[0].current() > 0
    }
}

fn generate_prototypes(rng: &mut Rng, count: usize) -> Vec<Vec<PackedSignal>> {
    (0..count).map(|p| {
        (0..32).map(|d| {
            let bits = ((p & d) as u32).count_ones();
            let pol: i8 = if bits % 2 == 0 { 1 } else { -1 };
            PackedSignal::pack(pol, 128 + rng.next_u8() % 64, 1)
        }).collect()
    }).collect()
}

fn add_noise(rng: &mut Rng, proto: &[PackedSignal]) -> Vec<PackedSignal> {
    proto.iter().map(|s| {
        packed_from_current(s.current().saturating_add((rng.next() % 81) as i32 - 40))
    }).collect()
}

fn train_compare_network(rng: &mut Rng) -> CompareNetwork {
    let prototypes = generate_prototypes(rng, 8);
    let config = MasteryConfig { pressure_threshold: 3, decay_rate: 1, participation_gate: 5 };
    let mut w_out = WeightMatrix::zeros(1, 32);
    let mut ms = MasteryState::new(32, config);

    for _ in 0..2 {
        for i in 0..800 {
            let proto_a = (rng.next() as usize) % prototypes.len();
            let a = add_noise(rng, &prototypes[proto_a]);
            let identical = i % 2 == 0;
            let b = if identical {
                add_noise(rng, &prototypes[proto_a])
            } else {
                let mut proto_b = (rng.next() as usize) % prototypes.len();
                while proto_b == proto_a { proto_b = (rng.next() as usize) % prototypes.len(); }
                add_noise(rng, &prototypes[proto_b])
            };
            let features = comparison_features(&a, &b);
            let raw_out = matmul(&features, &w_out);
            let clamped = packed_from_current(raw_out[0].current().clamp(-127, 127));
            let target = if identical {
                PackedSignal::pack(1, 127, 1)
            } else {
                PackedSignal::pack(-1, 127, 1)
            };
            ms.update(&mut w_out, &features, &[clamped], &[target]);
        }
        ms.decay();
    }
    println!("CompareANT trained: {} transitions", ms.transitions);
    CompareNetwork { w_out }
}

// ---------------------------------------------------------------------------
// Sudoku constraint checking via composition
// ---------------------------------------------------------------------------

/// Check if a group (row/col/box) has duplicates using CompareANT
fn has_duplicate(net: &CompareNetwork, group: &[&Vec<PackedSignal>]) -> bool {
    for i in 0..group.len() {
        for j in (i + 1)..group.len() {
            if net.compare(group[i], group[j]) {
                return true;
            }
        }
    }
    false
}

/// Validate a 4×4 Sudoku grid: no duplicates in rows, cols, or 2×2 boxes
fn valid_sudoku(net: &CompareNetwork, grid: &[Vec<PackedSignal>; 16]) -> bool {
    // Check 4 rows
    for r in 0..4 {
        let row: Vec<&Vec<PackedSignal>> = (0..4).map(|c| &grid[r * 4 + c]).collect();
        if has_duplicate(net, &row) { return false; }
    }
    // Check 4 columns
    for c in 0..4 {
        let col: Vec<&Vec<PackedSignal>> = (0..4).map(|r| &grid[r * 4 + c]).collect();
        if has_duplicate(net, &col) { return false; }
    }
    // Check 4 boxes (2×2)
    for box_r in 0..2 {
        for box_c in 0..2 {
            let bx: Vec<&Vec<PackedSignal>> = vec![
                &grid[(box_r * 2) * 4 + box_c * 2],
                &grid[(box_r * 2) * 4 + box_c * 2 + 1],
                &grid[(box_r * 2 + 1) * 4 + box_c * 2],
                &grid[(box_r * 2 + 1) * 4 + box_c * 2 + 1],
            ];
            if has_duplicate(net, &bx) { return false; }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Grid generation
// ---------------------------------------------------------------------------

/// All valid 4×4 Sudoku solutions (values 0-3 in each cell)
/// Uses simple backtracking to generate
fn generate_valid_grid(rng: &mut Rng) -> [usize; 16] {
    // Start with a known valid arrangement and permute
    let base: [usize; 16] = [
        0, 1, 2, 3,
        2, 3, 0, 1,
        1, 0, 3, 2,
        3, 2, 1, 0,
    ];

    // Permute the values (0-3) → random assignment
    let mut perm = [0usize, 1, 2, 3];
    for i in (1..4).rev() {
        let j = (rng.next() as usize) % (i + 1);
        perm.swap(i, j);
    }

    let mut grid = [0usize; 16];
    for i in 0..16 {
        grid[i] = perm[base[i]];
    }

    // Optionally swap rows within bands
    if rng.next() & 1 == 1 {
        // Swap rows 0,1
        for c in 0..4 { grid.swap(c, 4 + c); }
    }
    if rng.next() & 1 == 1 {
        // Swap rows 2,3
        for c in 0..4 { grid.swap(8 + c, 12 + c); }
    }

    grid
}

fn make_invalid_grid(rng: &mut Rng, valid: &[usize; 16]) -> [usize; 16] {
    let mut grid = *valid;
    // Swap two cells in the same row to create a duplicate
    let row = (rng.next() as usize) % 4;
    let c1 = (rng.next() as usize) % 4;
    let mut c2 = (rng.next() as usize) % 4;
    while c2 == c1 || grid[row * 4 + c1] == grid[row * 4 + c2] {
        c2 = (rng.next() as usize) % 4;
    }
    // Copy c1's value to c2 → duplicate in the row
    grid[row * 4 + c2] = grid[row * 4 + c1];
    grid
}

struct SudokuSample {
    grid: [Vec<PackedSignal>; 16],
    valid: bool,
}

fn grid_to_signals(
    rng: &mut Rng, values: &[usize; 16], prototypes: &[Vec<PackedSignal>],
) -> [Vec<PackedSignal>; 16] {
    std::array::from_fn(|i| add_noise(rng, &prototypes[values[i]]))
}

fn main() {
    println!("=== Composition: Sudoku Constraint Validation ===\n");

    let mut rng = Rng::new(0x50D0);

    // Train CompareANT
    println!("Training CompareANT...");
    let net = train_compare_network(&mut rng);

    // Generate prototypes for 4 Sudoku values
    let prototypes = generate_prototypes(&mut rng, 4);

    // Generate test samples
    let n_samples = 200;
    let mut samples = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let valid_grid = generate_valid_grid(&mut rng);
        if i % 2 == 0 {
            let grid = grid_to_signals(&mut rng, &valid_grid, &prototypes);
            samples.push(SudokuSample { grid, valid: true });
        } else {
            let invalid_grid = make_invalid_grid(&mut rng, &valid_grid);
            let grid = grid_to_signals(&mut rng, &invalid_grid, &prototypes);
            samples.push(SudokuSample { grid, valid: false });
        }
    }

    println!("Test grids: {} ({} valid, {} invalid)\n",
        samples.len(),
        samples.iter().filter(|s| s.valid).count(),
        samples.iter().filter(|s| !s.valid).count(),
    );

    // Evaluate
    let mut correct = 0;
    let mut false_valid = 0;  // said valid when invalid
    let mut false_invalid = 0; // said invalid when valid

    for sample in &samples {
        let predicted_valid = valid_sudoku(&net, &sample.grid);
        if predicted_valid == sample.valid {
            correct += 1;
        } else if predicted_valid {
            false_valid += 1;
        } else {
            false_invalid += 1;
        }
    }

    let accuracy = correct as f64 / samples.len() as f64 * 100.0;
    println!("Results:");
    println!("  Accuracy:     {correct}/{} ({accuracy:.1}%)", samples.len());
    println!("  False valid:  {false_valid} (accepted invalid grid)");
    println!("  False invalid: {false_invalid} (rejected valid grid)");

    // Stats
    // 4 rows × C(4,2) + 4 cols × C(4,2) + 4 boxes × C(4,2) = 72 per grid
    let total_comparisons = samples.len() * 72;
    println!("\n  Total comparisons: {total_comparisons}");
    println!("  Comparisons per grid: {}", 12 * 6);

    println!("\n=== Composition: Sudoku Constraint Validation Complete ===");
}
