//! Equivalence tests: Rust reference implementation vs Rune script output.
//! Proves that the Rune scripts produce byte-identical computation to the ternsig originals.

use atomic_neural_transistors::core::weight_matrix::{packed_from_current, WeightMatrix};
use atomic_neural_transistors::{AtomicNeuralTransistor, PackedSignal};

// ---------------------------------------------------------------------------
// Rust reference implementations matching ternsig assembly
// ---------------------------------------------------------------------------

fn ternary_sigmoid(current: i32) -> u8 {
    if current <= -512 {
        0
    } else if current >= 512 {
        255
    } else {
        ((current as i64 + 512) * 255 / 1024) as u8
    }
}

fn ref_relu(signals: &[PackedSignal]) -> Vec<PackedSignal> {
    signals
        .iter()
        .map(|s| {
            if s.is_negative() {
                PackedSignal::ZERO
            } else {
                *s
            }
        })
        .collect()
}

fn ref_sigmoid(signals: &[PackedSignal]) -> Vec<PackedSignal> {
    signals
        .iter()
        .map(|s| {
            let val = ternary_sigmoid(s.current());
            PackedSignal::pack(1, val, 1)
        })
        .collect()
}

fn ref_mul(a: &[PackedSignal], b: &[PackedSignal]) -> Vec<PackedSignal> {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let product = x.current() as i64 * y.current() as i64;
            packed_from_current(product as i32)
        })
        .collect()
}

fn ref_add(a: &[PackedSignal], b: &[PackedSignal]) -> Vec<PackedSignal> {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let sum = x.current() as i64 + y.current() as i64;
            packed_from_current(sum as i32)
        })
        .collect()
}

fn ref_shift(signals: &[PackedSignal], n: u32) -> Vec<PackedSignal> {
    signals
        .iter()
        .map(|s| packed_from_current(s.current() >> n))
        .collect()
}

fn ref_softmax(signals: &[PackedSignal]) -> Vec<PackedSignal> {
    if signals.is_empty() {
        return vec![];
    }
    let currents: Vec<i32> = signals.iter().map(|s| s.current()).collect();
    let min_c = *currents.iter().min().unwrap();
    let shifted: Vec<u64> = currents
        .iter()
        .map(|&c| (c as i64 - min_c as i64) as u64)
        .collect();
    let total: u64 = shifted.iter().sum();
    if total == 0 {
        let uniform = (255 / signals.len() as u8).max(1);
        signals
            .iter()
            .map(|_| PackedSignal::pack(1, uniform, 1))
            .collect()
    } else {
        shifted
            .iter()
            .map(|&s| {
                let mag = ((s * 255) / total) as u8;
                PackedSignal::pack(1, mag, 1)
            })
            .collect()
    }
}

/// Reference classifier: 32→24→4 with 3x gated recurrence.
/// Matches classifier.ternsig exactly.
fn ref_classifier(
    input: &[PackedSignal],
    w_in: &WeightMatrix,
    w_rec: &WeightMatrix,
    w_gate: &WeightMatrix,
    w_out: &WeightMatrix,
) -> Vec<PackedSignal> {
    // Input projection
    let mut h = ref_relu(&w_in.matmul(input));

    // 3x gated recurrent iterations
    for _ in 0..3 {
        let update = ref_relu(&w_rec.matmul(&h));
        let gate = ref_sigmoid(&w_gate.matmul(&h));
        let gated = ref_mul(&update, &gate);
        h = ref_add(&h, &gated);
        h = ref_shift(&h, 1);
    }

    // Output projection
    let out = w_out.matmul(&h);
    ref_softmax(&out)
}

/// Reference gate: [signal(32), context(32)] → 16 → 32 (sigmoid) → gated signal.
/// Matches gate.ternsig exactly.
fn ref_gate(
    input: &[PackedSignal],
    w_in: &WeightMatrix,
    w_out: &WeightMatrix,
) -> Vec<PackedSignal> {
    // Compute gate values
    let h = ref_relu(&w_in.matmul(input));
    let gate = ref_sigmoid(&w_out.matmul(&h));

    // Extract signal (first 32 elements)
    let signal = &input[..32];

    // Apply gating
    let gated = ref_mul(signal, &gate);
    ref_shift(&gated, 8)
}

/// Reference compare: [a(32), b(32)] → 16 → 16 → 1.
fn ref_compare(
    input: &[PackedSignal],
    w_in: &WeightMatrix,
    w_hidden: &WeightMatrix,
    w_out: &WeightMatrix,
) -> Vec<PackedSignal> {
    let h = ref_relu(&w_in.matmul(input));
    let h = ref_relu(&w_hidden.matmul(&h));
    w_out.matmul(&h)
}

/// Reference diff: [a(32), b(32)] → 24 → 32.
fn ref_diff(
    input: &[PackedSignal],
    w_in: &WeightMatrix,
    w_out: &WeightMatrix,
) -> Vec<PackedSignal> {
    let h = ref_relu(&w_in.matmul(input));
    w_out.matmul(&h)
}

/// Reference merge: [sig1(32), sig2(32)] → 24 → 32.
fn ref_merge(
    input: &[PackedSignal],
    w_in: &WeightMatrix,
    w_out: &WeightMatrix,
) -> Vec<PackedSignal> {
    let h = ref_relu(&w_in.matmul(input));
    w_out.matmul(&h)
}

// ---------------------------------------------------------------------------
// Helper: build an ANT with Thermogram (self-init zeros) for Rune execution
// ---------------------------------------------------------------------------

fn make_ant(source: &str) -> AtomicNeuralTransistor {
    AtomicNeuralTransistor::from_source_with_thermogram(source, None, "test", None).unwrap()
}

/// Create deterministic non-zero synaptic strengths for testing.
fn seeded_matrix(rows: usize, cols: usize, seed: u8) -> WeightMatrix {
    let mut data = Vec::with_capacity(rows * cols);
    for i in 0..(rows * cols) {
        let val = ((i as u16).wrapping_mul(seed as u16).wrapping_add(37)) % 256;
        let pol: i8 = if (i + seed as usize) % 3 == 0 {
            -1
        } else {
            1
        };
        let mag = (val as u8).wrapping_add(i as u8) % 8;
        let mul = (val as u8).wrapping_add(seed) % 8;
        data.push(PackedSignal::pack(pol, mag, mul));
    }
    WeightMatrix::from_data(data, rows, cols).unwrap()
}

/// Apply multiple keyed matrices to a thermogram, chaining hashes correctly.
fn load_matrices_into_thermogram(
    thermo: &mut thermogram::Thermogram,
    entries: &[(&str, &WeightMatrix)],
) {
    for (key, wm) in entries {
        let data: Vec<PackedSignal> = wm.data.clone();
        let prev_hash = thermo.dirty_chain.head_hash.clone();
        let delta = thermogram::Delta::update(
            *key,
            data,
            "test",
            thermogram::Signal::positive(255),
            prev_hash,
        );
        thermo.apply_delta(delta).unwrap();
    }
}

fn assert_byte_identical(label: &str, a: &[PackedSignal], b: &[PackedSignal]) {
    assert_eq!(
        a.len(),
        b.len(),
        "{label}: length mismatch ({} vs {})",
        a.len(),
        b.len()
    );
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.as_u8(),
            y.as_u8(),
            "{label}[{i}]: reference=0x{:02x} vs rune=0x{:02x}",
            x.as_u8(),
            y.as_u8()
        );
    }
}

// ---------------------------------------------------------------------------
// Equivalence tests
// ---------------------------------------------------------------------------

#[test]
fn test_classifier_equivalence_zeros() {
    // With zero synaptic strengths: both should produce uniform softmax
    let w_in = WeightMatrix::zeros(24, 32);
    let w_rec = WeightMatrix::zeros(24, 24);
    let w_gate = WeightMatrix::zeros(24, 24);
    let w_out = WeightMatrix::zeros(4, 24);

    let input: Vec<PackedSignal> = (0..32)
        .map(|i| PackedSignal::pack(1, i as u8 * 8, 1))
        .collect();

    let ref_out = ref_classifier(&input, &w_in, &w_rec, &w_gate, &w_out);

    // Rune script with load_synaptic self-inits zeros (no Thermogram data)
    let mut ant = make_ant(include_str!("../runes/classifier.rune"));
    let rune_out = ant.forward(&input).unwrap();

    assert_byte_identical("classifier_zeros", &ref_out, &rune_out);
}

#[test]
fn test_classifier_equivalence_seeded() {
    let w_in = seeded_matrix(24, 32, 7);
    let w_rec = seeded_matrix(24, 24, 13);
    let w_gate = seeded_matrix(24, 24, 19);
    let w_out = seeded_matrix(4, 24, 31);

    let input: Vec<PackedSignal> = (0..32)
        .map(|i| PackedSignal::pack(1, (i as u8 * 5) % 255, 1))
        .collect();

    let ref_out = ref_classifier(&input, &w_in, &w_rec, &w_gate, &w_out);

    // Build ANT with pre-loaded synaptic strengths via Thermogram
    let mut thermo =
        thermogram::Thermogram::new("classifier_test", thermogram::PlasticityRule::stdp_like());
    load_matrices_into_thermogram(&mut thermo, &[
        ("classifier.w_in", &w_in),
        ("classifier.w_rec", &w_rec),
        ("classifier.w_gate", &w_gate),
        ("classifier.w_out", &w_out),
    ]);

    let mut ant = AtomicNeuralTransistor::from_source_with_thermogram(
        include_str!("../runes/classifier.rune"),
        None,
        "classifier_test",
        None,
    )
    .unwrap();
    {
        let rt = ant.runtime();
        let mut guard = rt.lock().unwrap();
        guard.set_thermogram(thermo);
    }

    let rune_out = ant.forward(&input).unwrap();
    assert_byte_identical("classifier_seeded", &ref_out, &rune_out);
}

#[test]
fn test_gate_equivalence_zeros() {
    let w_in = WeightMatrix::zeros(16, 64);
    let w_out = WeightMatrix::zeros(32, 16);

    let signal: Vec<PackedSignal> = (0..32)
        .map(|i| PackedSignal::pack(1, i as u8 * 8, 1))
        .collect();
    let context: Vec<PackedSignal> = (0..32)
        .map(|_| PackedSignal::pack(1, 128, 1))
        .collect();
    let mut input = signal.clone();
    input.extend_from_slice(&context);

    let ref_out = ref_gate(&input, &w_in, &w_out);

    let mut ant = make_ant(include_str!("../runes/gate.rune"));
    let rune_out = ant.forward(&input).unwrap();

    assert_byte_identical("gate_zeros", &ref_out, &rune_out);
}

#[test]
fn test_gate_equivalence_seeded() {
    let w_in = seeded_matrix(16, 64, 41);
    let w_out = seeded_matrix(32, 16, 53);

    let signal: Vec<PackedSignal> = (0..32)
        .map(|i| PackedSignal::pack(1, (i as u8 * 7) % 255, 1))
        .collect();
    let context: Vec<PackedSignal> = (0..32)
        .map(|i| PackedSignal::pack(-1, (i as u8 * 3) % 200, 1))
        .collect();
    let mut input = signal.clone();
    input.extend_from_slice(&context);

    let ref_out = ref_gate(&input, &w_in, &w_out);

    let mut thermo =
        thermogram::Thermogram::new("gate_test", thermogram::PlasticityRule::stdp_like());
    load_matrices_into_thermogram(&mut thermo, &[
        ("gate.w_in", &w_in),
        ("gate.w_out", &w_out),
    ]);

    let mut ant = AtomicNeuralTransistor::from_source_with_thermogram(
        include_str!("../runes/gate.rune"),
        None,
        "gate_test",
        None,
    )
    .unwrap();
    {
        let rt = ant.runtime();
        let mut guard = rt.lock().unwrap();
        guard.set_thermogram(thermo);
    }

    let rune_out = ant.forward(&input).unwrap();
    assert_byte_identical("gate_seeded", &ref_out, &rune_out);
}

#[test]
fn test_compare_equivalence_seeded() {
    let w_in = seeded_matrix(16, 64, 59);
    let w_hidden = seeded_matrix(16, 16, 67);
    let w_out = seeded_matrix(1, 16, 71);

    let a: Vec<PackedSignal> = (0..32)
        .map(|i| PackedSignal::pack(1, i as u8 * 4, 1))
        .collect();
    let b: Vec<PackedSignal> = (0..32)
        .map(|i| PackedSignal::pack(-1, i as u8 * 6, 1))
        .collect();
    let mut input = a.clone();
    input.extend_from_slice(&b);

    let ref_out = ref_compare(&input, &w_in, &w_hidden, &w_out);

    let mut thermo =
        thermogram::Thermogram::new("compare_test", thermogram::PlasticityRule::stdp_like());
    load_matrices_into_thermogram(&mut thermo, &[
        ("compare.w_in", &w_in),
        ("compare.w_hidden", &w_hidden),
        ("compare.w_out", &w_out),
    ]);

    let mut ant = AtomicNeuralTransistor::from_source_with_thermogram(
        include_str!("../runes/compare.rune"),
        None,
        "compare_test",
        None,
    )
    .unwrap();
    {
        let rt = ant.runtime();
        let mut guard = rt.lock().unwrap();
        guard.set_thermogram(thermo);
    }

    let rune_out = ant.forward(&input).unwrap();
    assert_byte_identical("compare_seeded", &ref_out, &rune_out);
}

#[test]
fn test_diff_equivalence_seeded() {
    let w_in = seeded_matrix(24, 64, 73);
    let w_out = seeded_matrix(32, 24, 79);

    let a: Vec<PackedSignal> = (0..32)
        .map(|i| PackedSignal::pack(1, i as u8 * 3, 1))
        .collect();
    let b: Vec<PackedSignal> = (0..32)
        .map(|i| PackedSignal::pack(1, 255 - i as u8 * 3, 1))
        .collect();
    let mut input = a.clone();
    input.extend_from_slice(&b);

    let ref_out = ref_diff(&input, &w_in, &w_out);

    let mut thermo =
        thermogram::Thermogram::new("diff_test", thermogram::PlasticityRule::stdp_like());
    load_matrices_into_thermogram(&mut thermo, &[
        ("diff.w_in", &w_in),
        ("diff.w_out", &w_out),
    ]);

    let mut ant = AtomicNeuralTransistor::from_source_with_thermogram(
        include_str!("../runes/diff.rune"),
        None,
        "diff_test",
        None,
    )
    .unwrap();
    {
        let rt = ant.runtime();
        let mut guard = rt.lock().unwrap();
        guard.set_thermogram(thermo);
    }

    let rune_out = ant.forward(&input).unwrap();
    assert_byte_identical("diff_seeded", &ref_out, &rune_out);
}

#[test]
fn test_merge_equivalence_seeded() {
    let w_in = seeded_matrix(24, 64, 83);
    let w_out = seeded_matrix(32, 24, 89);

    let sig1: Vec<PackedSignal> = (0..32)
        .map(|i| PackedSignal::pack(1, i as u8 * 2, 1))
        .collect();
    let sig2: Vec<PackedSignal> = (0..32)
        .map(|i| PackedSignal::pack(-1, i as u8 * 4, 1))
        .collect();
    let mut input = sig1.clone();
    input.extend_from_slice(&sig2);

    let ref_out = ref_merge(&input, &w_in, &w_out);

    let mut thermo =
        thermogram::Thermogram::new("merge_test", thermogram::PlasticityRule::stdp_like());
    load_matrices_into_thermogram(&mut thermo, &[
        ("merge.w_in", &w_in),
        ("merge.w_out", &w_out),
    ]);

    let mut ant = AtomicNeuralTransistor::from_source_with_thermogram(
        include_str!("../runes/merge.rune"),
        None,
        "merge_test",
        None,
    )
    .unwrap();
    {
        let rt = ant.runtime();
        let mut guard = rt.lock().unwrap();
        guard.set_thermogram(thermo);
    }

    let rune_out = ant.forward(&input).unwrap();
    assert_byte_identical("merge_seeded", &ref_out, &rune_out);
}
