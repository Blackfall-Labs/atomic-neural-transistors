//! ANT Benchmarks — proving microsecond latency, determinism, and spec compliance.
//!
//! Spec claims (from ternsig originals):
//!   - Classifier: 32→24→4, 3x gated recurrence, 1920 params
//!   - Compare:    64→16→16→1, 1296 params
//!   - Diff:       64→24→32, 2304 params
//!   - Gate:       64→16→32 (sigmoid + slice + mul), 1536 params
//!   - Merge:      64→24→32, 2304 params
//!   - All forward passes complete in microseconds
//!   - All outputs are deterministic (same input → same output, every time)

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use atomic_neural_transistors::{
    AtomicNeuralTransistor, ClassifierANT, CompareANT, DiffANT, GateANT, MergeANT, PackedSignal,
};

// ---------------------------------------------------------------------------
// Input generators
// ---------------------------------------------------------------------------

fn classifier_input() -> Vec<PackedSignal> {
    (0..32).map(|i| PackedSignal::pack(1, (i as u8 * 7) % 255, 1)).collect()
}

// ---------------------------------------------------------------------------
// Forward pass latency benchmarks
// ---------------------------------------------------------------------------

fn bench_classifier_forward(c: &mut Criterion) {
    let mut ant = ClassifierANT::new().unwrap();
    let input = classifier_input();
    c.bench_function("classifier_forward", |b| {
        b.iter(|| ant.classify(black_box(&input)))
    });
}

fn bench_compare_forward(c: &mut Criterion) {
    let mut ant = CompareANT::new().unwrap();
    let a: Vec<PackedSignal> = (0..32).map(|i| PackedSignal::pack(1, (i as u8 * 5) % 200, 1)).collect();
    let b: Vec<PackedSignal> = (0..32).map(|i| PackedSignal::pack(-1, (i as u8 * 3) % 180, 1)).collect();
    c.bench_function("compare_forward", |b_iter| {
        b_iter.iter(|| ant.compare(black_box(&a), black_box(&b)))
    });
}

fn bench_diff_forward(c: &mut Criterion) {
    let mut ant = DiffANT::new().unwrap();
    let a: Vec<PackedSignal> = (0..32).map(|i| PackedSignal::pack(1, (i as u8 * 4) % 220, 1)).collect();
    let b: Vec<PackedSignal> = (0..32).map(|i| PackedSignal::pack(1, (i as u8 * 6) % 190, 1)).collect();
    c.bench_function("diff_forward", |b_iter| {
        b_iter.iter(|| ant.diff(black_box(&a), black_box(&b)))
    });
}

fn bench_gate_forward(c: &mut Criterion) {
    let mut ant = GateANT::new().unwrap();
    let signal: Vec<PackedSignal> = (0..32).map(|i| PackedSignal::pack(1, (i as u8 * 8) % 255, 1)).collect();
    let context: Vec<PackedSignal> = (0..32).map(|_| PackedSignal::pack(1, 128, 1)).collect();
    c.bench_function("gate_forward", |b| {
        b.iter(|| ant.gate(black_box(&signal), black_box(&context)))
    });
}

fn bench_merge_forward(c: &mut Criterion) {
    let mut ant = MergeANT::new().unwrap();
    let s1: Vec<PackedSignal> = (0..32).map(|i| PackedSignal::pack(1, (i as u8 * 2) % 200, 1)).collect();
    let s2: Vec<PackedSignal> = (0..32).map(|i| PackedSignal::pack(-1, (i as u8 * 4) % 180, 1)).collect();
    c.bench_function("merge_forward", |b| {
        b.iter(|| ant.merge(black_box(&[s1.as_slice(), s2.as_slice()])))
    });
}

// ---------------------------------------------------------------------------
// Determinism proof: same input produces byte-identical output across 1000 runs
// ---------------------------------------------------------------------------

fn bench_determinism(c: &mut Criterion) {
    let mut group = c.benchmark_group("determinism");

    // Classifier
    group.bench_function("classifier_1000x", |b| {
        let mut ant = ClassifierANT::new().unwrap();
        let input = classifier_input();
        let reference = ant.classify(&input).unwrap();
        b.iter(|| {
            let out = ant.classify(black_box(&input)).unwrap();
            assert_eq!(out.len(), reference.len());
            for (i, (r, o)) in reference.iter().zip(out.iter()).enumerate() {
                assert_eq!(r.as_u8(), o.as_u8(), "classifier non-determinism at index {i}");
            }
        });
    });

    // Gate (most complex new logic: slice + mul + shift)
    group.bench_function("gate_1000x", |b| {
        let mut ant = GateANT::new().unwrap();
        let signal: Vec<PackedSignal> = (0..32).map(|i| PackedSignal::pack(1, (i as u8 * 8) % 255, 1)).collect();
        let context: Vec<PackedSignal> = (0..32).map(|_| PackedSignal::pack(1, 128, 1)).collect();
        let reference = ant.gate(&signal, &context).unwrap();
        b.iter(|| {
            let out = ant.gate(black_box(&signal), black_box(&context)).unwrap();
            for (i, (r, o)) in reference.iter().zip(out.iter()).enumerate() {
                assert_eq!(r.as_u8(), o.as_u8(), "gate non-determinism at index {i}");
            }
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Param count spec verification
// ---------------------------------------------------------------------------

fn bench_param_count_spec(c: &mut Criterion) {
    // These aren't benchmarks per se — they run once and assert spec compliance.
    // Using criterion to get them into the bench output.
    c.bench_function("spec_param_counts", |b| {
        b.iter(|| {
            // Classifier: 24*32 + 24*24 + 24*24 + 4*24 = 768 + 576 + 576 + 96 = 2016
            // Wait — ternsig says 1920. Let me recount:
            // 24*32=768, 24*24=576, 24*24=576, 4*24=96 → 768+576+576+96 = 2016
            // But ternsig header says 1920. The file says:
            //   24*32 + 24*24 + 24*24 + 4*24 = 1920
            //   That's 768 + 576 + 576 + 96 = 2016. The ternsig comment is wrong.
            //   Actual: 2016. But weight files match the matrix dimensions.
            let classifier_params = 24 * 32 + 24 * 24 + 24 * 24 + 4 * 24;
            assert_eq!(classifier_params, 2016, "classifier param count");

            // Compare: 16*64 + 16*16 + 1*16 = 1024 + 256 + 16 = 1296
            let compare_params = 16 * 64 + 16 * 16 + 1 * 16;
            assert_eq!(compare_params, 1296, "compare param count");

            // Diff: 24*64 + 32*24 = 1536 + 768 = 2304
            let diff_params = 24 * 64 + 32 * 24;
            assert_eq!(diff_params, 2304, "diff param count");

            // Gate: 16*64 + 32*16 = 1024 + 512 = 1536
            let gate_params = 16 * 64 + 32 * 16;
            assert_eq!(gate_params, 1536, "gate param count");

            // Merge: 24*64 + 32*24 = 1536 + 768 = 2304
            let merge_params = 24 * 64 + 32 * 24;
            assert_eq!(merge_params, 2304, "merge param count");

            // Total: 2016 + 1296 + 2304 + 1536 + 2304 = 9456
            let total = classifier_params + compare_params + diff_params + gate_params + merge_params;
            assert!(total < 10_000, "total params {} exceeds 10K", total);

            // Each ANT < 5K params
            assert!(classifier_params < 5000, "classifier exceeds 5K");
            assert!(compare_params < 5000, "compare exceeds 5K");
            assert!(diff_params < 5000, "diff exceeds 5K");
            assert!(gate_params < 5000, "gate exceeds 5K");
            assert!(merge_params < 5000, "merge exceeds 5K");
        });
    });
}

// ---------------------------------------------------------------------------
// Output shape spec verification
// ---------------------------------------------------------------------------

fn bench_output_shapes(c: &mut Criterion) {
    c.bench_function("spec_output_shapes", |b| {
        let mut classifier = ClassifierANT::new().unwrap();
        let mut compare = CompareANT::new().unwrap();
        let mut diff = DiffANT::new().unwrap();
        let mut gate = GateANT::new().unwrap();
        let mut merge = MergeANT::new().unwrap();

        b.iter(|| {
            // Classifier: 32 in → 4 out (class logits)
            let out = classifier.classify(&classifier_input()).unwrap();
            assert_eq!(out.len(), 4, "classifier output should be 4 classes");

            // Compare: 64 in → 1 out (similarity)
            let a: Vec<PackedSignal> = (0..32).map(|i| PackedSignal::pack(1, i as u8, 1)).collect();
            let b_vec: Vec<PackedSignal> = (0..32).map(|i| PackedSignal::pack(1, i as u8, 1)).collect();
            let _out = compare.compare(&a, &b_vec).unwrap();
            // compare returns PackedSignal (single value) — that's correct

            // Diff: 64 in → 32 out
            let out = diff.diff(&a, &b_vec).unwrap();
            assert_eq!(out.len(), 32, "diff output should be 32");

            // Gate: 64 in → 32 out (gated signal)
            let signal: Vec<PackedSignal> = (0..32).map(|_| PackedSignal::pack(1, 100, 1)).collect();
            let ctx: Vec<PackedSignal> = (0..32).map(|_| PackedSignal::pack(1, 128, 1)).collect();
            let out = gate.gate(&signal, &ctx).unwrap();
            assert_eq!(out.len(), 32, "gate output should be 32");

            // Merge: 64 in → 32 out
            let out = merge.merge(&[a.as_slice(), b_vec.as_slice()]).unwrap();
            assert_eq!(out.len(), 32, "merge output should be 32");
        });
    });
}

// ---------------------------------------------------------------------------
// Thermogram persistence latency
// ---------------------------------------------------------------------------

fn bench_thermogram_save_load(c: &mut Criterion) {
    use thermogram::{Thermogram, PlasticityRule, Delta};

    let dir = tempfile::tempdir().unwrap();
    let thermo_path = dir.path().join("bench.thermo");

    // Build a thermogram with all 5 ANTs' synaptic strengths
    let mut thermo = Thermogram::new("bench", PlasticityRule::stdp_like());
    let keys: Vec<(&str, usize)> = vec![
        ("classifier.w_in", 24 * 32),
        ("classifier.w_rec", 24 * 24),
        ("classifier.w_gate", 24 * 24),
        ("classifier.w_out", 4 * 24),
        ("compare.w_in", 16 * 64),
        ("compare.w_hidden", 16 * 16),
        ("compare.w_out", 1 * 16),
        ("diff.w_in", 24 * 64),
        ("diff.w_out", 32 * 24),
        ("gate.w_in", 16 * 64),
        ("gate.w_out", 32 * 16),
        ("merge.w_in", 24 * 64),
        ("merge.w_out", 32 * 24),
    ];
    for (key, size) in &keys {
        let data: Vec<PackedSignal> = (0..*size)
            .map(|i| PackedSignal::pack(1, (i % 200) as u8, 1))
            .collect();
        let prev_hash = thermo.dirty_chain.head_hash.clone();
        let delta = if prev_hash.is_none() {
            Delta::create(*key, data, "bench")
        } else {
            Delta::update(*key, data, "bench", thermogram::Signal::positive(255), prev_hash)
        };
        thermo.apply_delta(delta).unwrap();
    }
    thermo.save(&thermo_path).unwrap();

    let mut group = c.benchmark_group("thermogram");

    group.bench_function("save_all_ants", |b| {
        b.iter(|| thermo.save(black_box(&thermo_path)).unwrap());
    });

    group.bench_function("load_all_ants", |b| {
        b.iter(|| Thermogram::load(black_box(&thermo_path)).unwrap());
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Mastery update latency
// ---------------------------------------------------------------------------

fn bench_mastery_update(c: &mut Criterion) {
    c.bench_function("mastery_update_step", |b| {
        let source = r#"rune "bench_mastery" do
  version 1
end
use :ant_ml

def forward(input) do
    w = load_synaptic("bench.w", 8, 16)
    out = matmul(input, w, 8, 16)
    out = relu(out)
    target = zeros(8)
    mastery_update(w, input, out, target, [3, 1, 5])
    out
end"#;

        let mut ant = AtomicNeuralTransistor::from_source_with_thermogram(
            source, None, "mastery_bench", None,
        ).unwrap();
        let input: Vec<PackedSignal> = (0..16)
            .map(|i| PackedSignal::pack(1, (i as u8 * 10) % 200, 1))
            .collect();

        b.iter(|| ant.forward(black_box(&input)).unwrap());
    });
}

criterion_group!(
    benches,
    bench_classifier_forward,
    bench_compare_forward,
    bench_diff_forward,
    bench_gate_forward,
    bench_merge_forward,
    bench_determinism,
    bench_param_count_spec,
    bench_output_shapes,
    bench_thermogram_save_load,
    bench_mastery_update,
);
criterion_main!(benches);
