//! Debug mastery learning to find why CompareANT isn't training.

use atomic_neural_transistors::core::weight_matrix::{WeightMatrix, packed_from_current};
use atomic_neural_transistors::learning::{MasteryConfig, MasteryState};
use ternary_signal::PackedSignal;

fn main() {
    println!("=== Mastery Learning Debug ===\n");

    // Test 1: Simple 1x1 matrix, no decay, no gate
    println!("--- Test 1: Minimal config (threshold=1, decay=0, gate=0) ---");
    let mut w = WeightMatrix::zeros(1, 2);
    let mut state = MasteryState::new(2, MasteryConfig {
        pressure_threshold: 1,
        decay_rate: 0,
        participation_gate: 0,
    });
    let input = vec![PackedSignal::pack(1, 64, 1), PackedSignal::pack(1, 32, 1)];
    let output = vec![PackedSignal::ZERO];
    let target = vec![PackedSignal::pack(1, 127, 1)];

    println!("Before: w[0]={}, w[1]={}", w.data[0].current(), w.data[1].current());
    state.update(&mut w, &input, &output, &target);
    println!("After 1 step: w[0]={}, w[1]={}", w.data[0].current(), w.data[1].current());
    println!("Transitions: {}, Steps: {}\n", state.transitions, state.steps);

    // Test 2: Production config [3, 1, 5]
    println!("--- Test 2: Production config (threshold=3, decay=1, gate=5) ---");
    let mut w = WeightMatrix::zeros(1, 2);
    let mut state = MasteryState::new(2, MasteryConfig {
        pressure_threshold: 3,
        decay_rate: 1,
        participation_gate: 5,
    });
    let input = vec![PackedSignal::pack(1, 64, 1), PackedSignal::pack(1, 32, 1)];
    let target = vec![PackedSignal::pack(1, 127, 1)];

    for i in 0..20 {
        let output = vec![PackedSignal::ZERO]; // output stays zero since we're not doing matmul
        state.update(&mut w, &input, &output, &target);
        if i < 10 || state.transitions > 0 {
            println!(
                "Step {}: w[0]={}, w[1]={}, pressure=[{}, {}], participation=[{}, {}], transitions={}",
                i + 1,
                w.data[0].current(),
                w.data[1].current(),
                state.pressure[0],
                state.pressure[1],
                state.participation[0],
                state.participation[1],
                state.transitions,
            );
        }
    }

    // Test 3: Runes-based training - check if weights change
    println!("\n--- Test 3: Runes-based training ---");
    use atomic_neural_transistors::AtomicNeuralTransistor;

    let source = r#"rune "debug" do
  version 1
end
use :ant_ml

def train(data) do
    input = slice(data, 0, 4)
    target = slice(data, 4, 2)

    w = load_synaptic("debug.w", 2, 4)
    out = matmul(input, w, 2, 4)
    out = relu(out)

    mastery_update(w, input, out, target, [1, 0, 0])

    out
end"#;

    let mut ant = AtomicNeuralTransistor::from_source_with_thermogram(
        source, None, "debug", None,
    ).unwrap();

    let input = vec![
        PackedSignal::pack(1, 64, 1),
        PackedSignal::pack(1, 32, 1),
        PackedSignal::pack(-1, 100, 1),
        PackedSignal::pack(1, 200, 1),
        // target (2 elements)
        PackedSignal::pack(1, 127, 1),
        PackedSignal::pack(1, 127, 1),
    ];

    for i in 0..10 {
        let out = ant.call("train", &input).unwrap();
        let out_vals: Vec<i32> = out.iter().map(|s| s.current()).collect();
        println!("Step {}: output = {:?}", i + 1, out_vals);
    }

    // Check if weights changed via runtime
    let rt = ant.runtime().lock().unwrap();
    println!("\nRuntime weight count: {}", rt.weight_count());
    println!("Runtime mastery state count: {}", rt.mastery_count());
}
