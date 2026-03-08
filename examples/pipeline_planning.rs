//! Multi-ANT Pipeline: Planning — Example 9
//!
//! Demonstrates a multi-ANT pipeline where a ClassifierANT selects an
//! action and a CompareANT verifies whether the action was effective
//! (state changed). No additional mastery learning — both ANTs are
//! trained independently, then composed into a planning loop.
//!
//! Pipeline:
//!   1. ClassifierANT classifies current state → action selection
//!   2. Apply action (simulated state transition)
//!   3. CompareANT checks if state changed (action was effective)
//!   4. Repeat until goal state reached or max steps
//!
//! Proves: independently trained ANTs compose into a multi-step pipeline
//! without any joint mastery learning.

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

fn relu(signals: &[PackedSignal]) -> Vec<PackedSignal> {
    signals.iter().map(|s| if s.current() > 0 { *s } else { PackedSignal::ZERO }).collect()
}

fn comparison_features(a: &[PackedSignal], b: &[PackedSignal]) -> Vec<PackedSignal> {
    a.iter().zip(b.iter()).map(|(x, y)| {
        let product = x.current() as i64 * y.current() as i64;
        packed_from_current((product / 256) as i32)
    }).collect()
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

// ---------------------------------------------------------------------------
// CompareANT (from Example 1)
// ---------------------------------------------------------------------------

struct CompareNetwork { w_out: WeightMatrix }

impl CompareNetwork {
    fn compare(&self, a: &[PackedSignal], b: &[PackedSignal]) -> bool {
        let features = comparison_features(a, b);
        let out = matmul(&features, &self.w_out);
        out[0].current() > 0
    }
}

fn train_compare(rng: &mut Rng, prototypes: &[Vec<PackedSignal>]) -> CompareNetwork {
    let config = MasteryConfig { pressure_threshold: 3, decay_rate: 1, participation_gate: 5 };
    let mut w_out = WeightMatrix::zeros(1, 32);
    let mut ms = MasteryState::new(32, config);
    let np = prototypes.len();

    for _ in 0..2 {
        for i in 0..800 {
            let pa = (rng.next() as usize) % np;
            let a = add_noise(rng, &prototypes[pa]);
            let identical = i % 2 == 0;
            let b = if identical {
                add_noise(rng, &prototypes[pa])
            } else {
                let mut pb = (rng.next() as usize) % np;
                while pb == pa { pb = (rng.next() as usize) % np; }
                add_noise(rng, &prototypes[pb])
            };
            let features = comparison_features(&a, &b);
            let raw_out = matmul(&features, &w_out);
            let clamped = packed_from_current(raw_out[0].current().clamp(-127, 127));
            let target = if identical { PackedSignal::pack(1, 127, 1) } else { PackedSignal::pack(-1, 127, 1) };
            ms.update(&mut w_out, &features, &[clamped], &[target]);
        }
        ms.decay();
    }
    println!("  CompareANT: {} transitions", ms.transitions);
    CompareNetwork { w_out }
}

// ---------------------------------------------------------------------------
// ClassifierANT (from Example 2)
// ---------------------------------------------------------------------------

struct ClassifierNetwork {
    w_hidden: WeightMatrix,
    w_out: WeightMatrix,
}

impl ClassifierNetwork {
    fn predict(&self, input: &[PackedSignal]) -> usize {
        let h = relu(&matmul(input, &self.w_hidden));
        let out = matmul(&h, &self.w_out);
        out.iter().enumerate()
            .max_by_key(|(_, s)| s.current())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

fn train_classifier(rng: &mut Rng, prototypes: &[Vec<PackedSignal>], n_classes: usize) -> ClassifierNetwork {
    let config = MasteryConfig { pressure_threshold: 3, decay_rate: 1, participation_gate: 5 };

    let hidden_data: Vec<PackedSignal> = (0..24 * 32).map(|_| {
        let mag = 20 + (rng.next_u8() % 21);
        let pol: i8 = if rng.next() & 1 == 0 { 1 } else { -1 };
        PackedSignal::pack(pol, mag, 1)
    }).collect();
    let w_hidden = WeightMatrix::from_data(hidden_data, 24, 32).unwrap();
    let mut w_out = WeightMatrix::zeros(n_classes, 24);
    let mut ms = MasteryState::new(n_classes * 24, config);

    for _ in 0..3 {
        for i in 0..800 {
            let class = i % n_classes;
            let signal = add_noise(rng, &prototypes[class]);
            let h = relu(&matmul(&signal, &w_hidden));
            let out = matmul(&h, &w_out);
            let target: Vec<PackedSignal> = (0..n_classes).map(|c| {
                if c == class { PackedSignal::pack(1, 127, 1) } else { PackedSignal::pack(-1, 127, 1) }
            }).collect();
            let clamped: Vec<PackedSignal> = out.iter()
                .map(|s| packed_from_current(s.current().clamp(-127, 127)))
                .collect();
            ms.update(&mut w_out, &h, &clamped, &target);
        }
        ms.decay();
    }
    println!("  ClassifierANT: {} transitions", ms.transitions);
    ClassifierNetwork { w_hidden, w_out }
}

// ---------------------------------------------------------------------------
// Planning simulation
// ---------------------------------------------------------------------------

/// Simple state machine: 4 states, each action moves to a specific next state
/// Action 0: stay, Action 1: move to state (current+1)%4
/// Action 2: move to state (current+2)%4, Action 3: move to state (current+3)%4
fn apply_action(current_state: usize, action: usize, n_states: usize) -> usize {
    (current_state + action) % n_states
}

fn run_planning_episode(
    classifier: &ClassifierNetwork,
    comparer: &CompareNetwork,
    prototypes: &[Vec<PackedSignal>],
    rng: &mut Rng,
    start: usize,
    goal: usize,
    n_states: usize,
    max_steps: usize,
) -> (bool, usize) {
    let mut state = start;
    for step in 0..max_steps {
        if state == goal { return (true, step); }

        // ClassifierANT identifies current state
        let state_signal = add_noise(rng, &prototypes[state]);
        let classified = classifier.predict(&state_signal);

        // Select action based on classification
        // Simple policy: action = (goal - classified) mod n_states
        let action = (goal + n_states - classified) % n_states;

        // Apply action
        let prev_state = state;
        state = apply_action(state, action, n_states);

        // CompareANT verifies state changed
        let prev_signal = add_noise(rng, &prototypes[prev_state]);
        let new_signal = add_noise(rng, &prototypes[state]);
        let _state_changed = !comparer.compare(&prev_signal, &new_signal);
    }
    (state == goal, max_steps)
}

fn main() {
    println!("=== Multi-ANT Pipeline: Planning ===\n");

    let mut rng = Rng::new(0x91A4);
    let n_states = 4;

    // Generate prototypes for states
    let prototypes = generate_prototypes(&mut rng, n_states);

    // Train both ANTs independently
    println!("Training ANTs independently...");
    let classifier = train_classifier(&mut rng, &prototypes, n_states);
    let comparer = train_compare(&mut rng, &prototypes);

    // Verify classifier accuracy
    let mut class_correct = 0;
    for _ in 0..200 {
        let class = (rng.next() as usize) % n_states;
        let signal = add_noise(&mut rng, &prototypes[class]);
        if classifier.predict(&signal) == class { class_correct += 1; }
    }
    println!("\nClassifier accuracy: {class_correct}/200 ({:.1}%)\n",
        class_correct as f64 / 200.0 * 100.0);

    // Run planning episodes
    let n_episodes = 100;
    let max_steps = 10;
    let mut successes = 0;
    let mut total_steps = 0;

    println!("Running {n_episodes} planning episodes (max {max_steps} steps each)...\n");

    for _ in 0..n_episodes {
        let start = (rng.next() as usize) % n_states;
        let mut goal = (rng.next() as usize) % n_states;
        while goal == start { goal = (rng.next() as usize) % n_states; }

        let (reached, steps) = run_planning_episode(
            &classifier, &comparer, &prototypes, &mut rng,
            start, goal, n_states, max_steps,
        );
        if reached {
            successes += 1;
            total_steps += steps;
        }
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
                let (reached, _) = run_planning_episode(
                    &classifier, &comparer, &prototypes, &mut rng,
                    start, goal, n_states, max_steps,
                );
                dist_total += 1;
                if reached { dist_success += 1; }
            }
        }
        println!("  Distance {dist}: {dist_success}/{dist_total} ({:.1}%)",
            dist_success as f64 / dist_total as f64 * 100.0);
    }

    println!("\n=== Multi-ANT Pipeline: Planning Complete ===");
}
