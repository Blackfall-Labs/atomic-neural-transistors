//! ANT Benchmarks — proving microsecond latency, determinism, and spec compliance.
//!
//! All benchmarks use Signal (not PackedSignal) and Runes-based execution.
//! Thermogram benchmarks removed — replaced with .ant v3 format benchmarks.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::sync::Arc;
use atomic_neural_transistors::{
    AtomicNeuralTransistor, Signal, Value,
    SalienceRouter, PredictionEngine, NeuromodState, Chemical,
    ThermalWeightMatrix,
};

// ---------------------------------------------------------------------------
// Input generators
// ---------------------------------------------------------------------------

fn make_input(dim: usize, seed: u8) -> Vec<Signal> {
    (0..dim).map(|i| Signal::new_raw(1, ((i as u8).wrapping_mul(seed)) % 255, 1)).collect()
}

fn signals_to_value(signals: &[Signal]) -> Value {
    Value::Array(Arc::new(
        signals.iter().map(|s| Value::Integer(s.current() as i64)).collect()
    ))
}

// ---------------------------------------------------------------------------
// Forward pass latency benchmarks (via Runes)
// ---------------------------------------------------------------------------

fn bench_classifier_forward(c: &mut Criterion) {
    let mut ant = AtomicNeuralTransistor::from_source(
        include_str!("../runes/classifier.rune")
    ).unwrap();
    let input = make_input(32, 7);
    let input_val = signals_to_value(&input);

    c.bench_function("classifier_forward", |b| {
        b.iter(|| ant.call_values("forward", vec![black_box(input_val.clone())]))
    });
}

fn bench_compare_forward(c: &mut Criterion) {
    let mut ant = AtomicNeuralTransistor::from_source(
        include_str!("../runes/compare.rune")
    ).unwrap();
    let a = make_input(32, 5);
    let b = make_input(32, 3);
    let mut input_signals = a;
    input_signals.extend_from_slice(&b);
    let input_val = signals_to_value(&input_signals);

    c.bench_function("compare_forward", |b_iter| {
        b_iter.iter(|| ant.call_values("forward", vec![black_box(input_val.clone())]))
    });
}

fn bench_gate_forward(c: &mut Criterion) {
    let mut ant = AtomicNeuralTransistor::from_source(
        include_str!("../runes/gate.rune")
    ).unwrap();
    let input = make_input(64, 11);
    let input_val = signals_to_value(&input);

    c.bench_function("gate_forward", |b| {
        b.iter(|| ant.call_values("forward", vec![black_box(input_val.clone())]))
    });
}

fn bench_diff_forward(c: &mut Criterion) {
    let mut ant = AtomicNeuralTransistor::from_source(
        include_str!("../runes/diff.rune")
    ).unwrap();
    let input = make_input(64, 13);
    let input_val = signals_to_value(&input);

    c.bench_function("diff_forward", |b| {
        b.iter(|| ant.call_values("forward", vec![black_box(input_val.clone())]))
    });
}

fn bench_merge_forward(c: &mut Criterion) {
    let mut ant = AtomicNeuralTransistor::from_source(
        include_str!("../runes/merge.rune")
    ).unwrap();
    let input = make_input(64, 17);
    let input_val = signals_to_value(&input);

    c.bench_function("merge_forward", |b| {
        b.iter(|| ant.call_values("forward", vec![black_box(input_val.clone())]))
    });
}

// ---------------------------------------------------------------------------
// Determinism proof
// ---------------------------------------------------------------------------

fn bench_determinism_classifier(c: &mut Criterion) {
    let mut ant = AtomicNeuralTransistor::from_source(
        include_str!("../runes/classifier.rune")
    ).unwrap();
    let input = make_input(32, 7);
    let input_val = signals_to_value(&input);

    // Get reference output
    let ref_output = ant.call_values("forward", vec![input_val.clone()]).unwrap();

    c.bench_function("determinism_classifier_1000x", |b| {
        b.iter(|| {
            for _ in 0..1000 {
                let out = ant.call_values("forward", vec![input_val.clone()]).unwrap();
                assert_eq!(format!("{:?}", out), format!("{:?}", ref_output));
            }
        })
    });
}

// ---------------------------------------------------------------------------
// Mastery update latency
// ---------------------------------------------------------------------------

fn bench_mastery_update(c: &mut Criterion) {
    let mut ant = AtomicNeuralTransistor::from_source(
        include_str!("../runes/classifier_train.rune")
    ).unwrap();

    // Pack [input(32) | class(1)]
    let mut data: Vec<Value> = make_input(32, 7).iter()
        .map(|s| Value::Integer(s.current() as i64))
        .collect();
    data.push(Value::Integer(2)); // class 2
    let data_val = Value::Array(Arc::new(data));

    c.bench_function("mastery_update_classifier", |b| {
        b.iter(|| ant.call_values("train", vec![black_box(data_val.clone())]))
    });
}

// ---------------------------------------------------------------------------
// Thermal benchmarks
// ---------------------------------------------------------------------------

fn bench_thermal_matmul(c: &mut Criterion) {
    let mut ant = AtomicNeuralTransistor::from_source(
        include_str!("../runes/thermal_classifier.rune")
    ).unwrap();
    let input = make_input(32, 7);
    let input_val = signals_to_value(&input);

    c.bench_function("thermal_forward", |b| {
        b.iter(|| ant.call_values("forward", vec![black_box(input_val.clone())]))
    });
}

// ---------------------------------------------------------------------------
// .ant v3 persistence latency
// ---------------------------------------------------------------------------

fn bench_ant_save_load(c: &mut Criterion) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bench.ant");

    let twm = ThermalWeightMatrix::random_hot(48, 80, 42);

    c.bench_function("ant_v3_save", |b| {
        b.iter(|| twm.save(black_box(&path)))
    });

    // Ensure file exists for load
    twm.save(&path).unwrap();

    c.bench_function("ant_v3_load", |b| {
        b.iter(|| ThermalWeightMatrix::load(black_box(&path)))
    });
}

// ---------------------------------------------------------------------------
// Salience routing latency
// ---------------------------------------------------------------------------

fn bench_salience_route(c: &mut Criterion) {
    let router = SalienceRouter::new(3, 32);
    let outputs: Vec<Signal> = make_input(96, 23); // 3 × 32

    c.bench_function("salience_route_3x32", |b| {
        b.iter(|| router.route(black_box(&outputs)))
    });
}

// ---------------------------------------------------------------------------
// Prediction engine latency
// ---------------------------------------------------------------------------

fn bench_predict_observe(c: &mut Criterion) {
    let mut pred = PredictionEngine::new(32, 3, 40);
    let signal = make_input(32, 29);

    c.bench_function("predict_observe_32dim", |b| {
        b.iter(|| pred.observe(black_box(&signal), None))
    });
}

// ---------------------------------------------------------------------------
// Neuromod cycle latency
// ---------------------------------------------------------------------------

fn bench_neuromod_cycle(c: &mut Criterion) {
    let mut nm = NeuromodState::new();

    c.bench_function("neuromod_full_cycle", |b| {
        b.iter(|| {
            nm.inject(Chemical::Dopamine, 20);
            nm.inject(Chemical::Norepinephrine, 10);
            nm.inject(Chemical::Serotonin, -5);
            let _ = nm.plasticity_open();
            let _ = nm.participation_divisor();
            let _ = nm.decay_multiplier();
            nm.tick();
        })
    });
}

// ---------------------------------------------------------------------------
// Groups
// ---------------------------------------------------------------------------

criterion_group!(
    forward_latency,
    bench_classifier_forward,
    bench_compare_forward,
    bench_gate_forward,
    bench_diff_forward,
    bench_merge_forward,
);

criterion_group!(
    determinism,
    bench_determinism_classifier,
);

criterion_group!(
    mastery,
    bench_mastery_update,
);

criterion_group!(
    thermal,
    bench_thermal_matmul,
);

criterion_group!(
    persistence,
    bench_ant_save_load,
);

criterion_group!(
    subsystems,
    bench_salience_route,
    bench_predict_observe,
    bench_neuromod_cycle,
);

criterion_main!(
    forward_latency,
    determinism,
    mastery,
    thermal,
    persistence,
    subsystems,
);
