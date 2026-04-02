//! Equivalence tests: Rust reference implementation vs Rune script output.
//! Proves that the Rune scripts produce identical computation to the Rust originals.

use atomic_neural_transistors::core::weight_matrix::WeightMatrix;
use atomic_neural_transistors::{AtomicNeuralTransistor, Signal};

// ---------------------------------------------------------------------------
// Rust reference implementations
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

fn ref_relu(signals: &[Signal]) -> Vec<Signal> {
    signals.iter().map(|s| {
        if s.current() < 0 { Signal::ZERO } else { *s }
    }).collect()
}

fn ref_sigmoid(signals: &[Signal]) -> Vec<Signal> {
    signals.iter().map(|s| {
        let val = ternary_sigmoid(s.current());
        Signal::new_raw(1, val, 1)
    }).collect()
}

fn ref_mul(a: &[Signal], b: &[Signal]) -> Vec<Signal> {
    a.iter().zip(b.iter())
        .map(|(x, y)| {
            let product = x.current() as i64 * y.current() as i64;
            Signal::from_current(product as i32)
        })
        .collect()
}

fn ref_add(a: &[Signal], b: &[Signal]) -> Vec<Signal> {
    a.iter().zip(b.iter())
        .map(|(x, y)| {
            let sum = x.current() as i64 + y.current() as i64;
            Signal::from_current(sum as i32)
        })
        .collect()
}

fn ref_shift(signals: &[Signal], n: u32) -> Vec<Signal> {
    signals.iter()
        .map(|s| Signal::from_current(s.current() >> n))
        .collect()
}

fn ref_softmax(signals: &[Signal]) -> Vec<Signal> {
    if signals.is_empty() { return vec![]; }
    let currents: Vec<i32> = signals.iter().map(|s| s.current()).collect();
    let min_c = *currents.iter().min().unwrap();
    let shifted: Vec<u64> = currents.iter().map(|&c| (c as i64 - min_c as i64) as u64).collect();
    let total: u64 = shifted.iter().sum();
    if total == 0 {
        let uniform = (255 / signals.len() as u8).max(1);
        signals.iter().map(|_| Signal::new_raw(1, uniform, 1)).collect()
    } else {
        shifted.iter().map(|&s| {
            let mag = ((s * 255) / total) as u8;
            Signal::new_raw(1, mag, 1)
        }).collect()
    }
}

/// Reference classifier: 32→24→4 with 3x gated recurrence.
fn ref_classifier(
    input: &[Signal],
    w_in: &WeightMatrix,
    w_rec: &WeightMatrix,
    w_gate: &WeightMatrix,
    w_out: &WeightMatrix,
) -> Vec<Signal> {
    let mut h = ref_relu(&w_in.matmul(input));
    for _ in 0..3 {
        let update = ref_relu(&w_rec.matmul(&h));
        let gate = ref_sigmoid(&w_gate.matmul(&h));
        let gated = ref_mul(&update, &gate);
        h = ref_add(&h, &gated);
        h = ref_shift(&h, 1);
    }
    let out = w_out.matmul(&h);
    ref_softmax(&out)
}

/// Reference gate: [signal(32), context(32)] → 16 → 32 (sigmoid) → gated signal.
fn ref_gate(
    input: &[Signal],
    w_in: &WeightMatrix,
    w_out: &WeightMatrix,
) -> Vec<Signal> {
    let h = ref_relu(&w_in.matmul(input));
    let gate = ref_sigmoid(&w_out.matmul(&h));
    let signal = &input[..32];
    let gated = ref_mul(signal, &gate);
    ref_shift(&gated, 8)
}

/// Reference compare: [a(32), b(32)] → 16 → 16 → 1.
fn ref_compare(
    input: &[Signal],
    w_in: &WeightMatrix,
    w_hidden: &WeightMatrix,
    w_out: &WeightMatrix,
) -> Vec<Signal> {
    let h = ref_relu(&w_in.matmul(input));
    let h = ref_relu(&w_hidden.matmul(&h));
    w_out.matmul(&h)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_ant(source: &str) -> AtomicNeuralTransistor {
    AtomicNeuralTransistor::from_source(source).unwrap()
}

fn assert_equivalent(label: &str, a: &[Signal], b: &[Signal]) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch ({} vs {})", a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.current(), y.current(),
            "{label}[{i}]: reference={} vs rune={}",
            x.current(), y.current()
        );
    }
}

// ---------------------------------------------------------------------------
// Equivalence tests (zeros — no pre-loaded weights needed)
// ---------------------------------------------------------------------------

#[test]
fn test_classifier_equivalence_zeros() {
    let w_in = WeightMatrix::zeros(24, 32);
    let w_rec = WeightMatrix::zeros(24, 24);
    let w_gate = WeightMatrix::zeros(24, 24);
    let w_out = WeightMatrix::zeros(4, 24);

    let input: Vec<Signal> = (0..32)
        .map(|i| Signal::new_raw(1, i as u8 * 8, 1))
        .collect();

    let ref_out = ref_classifier(&input, &w_in, &w_rec, &w_gate, &w_out);

    let mut ant = make_ant(include_str!("../runes/classifier.rune"));
    let rune_out = ant.forward(&input).unwrap();

    assert_equivalent("classifier_zeros", &ref_out, &rune_out);
}

#[test]
fn test_gate_equivalence_zeros() {
    let w_in = WeightMatrix::zeros(16, 64);
    let w_out = WeightMatrix::zeros(32, 16);

    let signal: Vec<Signal> = (0..32)
        .map(|i| Signal::new_raw(1, i as u8 * 8, 1))
        .collect();
    let context: Vec<Signal> = (0..32)
        .map(|_| Signal::new_raw(1, 128, 1))
        .collect();
    let mut input = signal;
    input.extend_from_slice(&context);

    let ref_out = ref_gate(&input, &w_in, &w_out);

    let mut ant = make_ant(include_str!("../runes/gate.rune"));
    let rune_out = ant.forward(&input).unwrap();

    assert_equivalent("gate_zeros", &ref_out, &rune_out);
}

#[test]
fn test_load_synaptic_deduplicates() {
    let source = r#"rune "test_dedup" do
  version 1
end
use :ant_ml

def forward(input) do
    w1 = load_synaptic("dedup.w", 2, 4)
    w2 = load_synaptic("dedup.w", 2, 4)
    a = matmul(input, w1)
    b = matmul(input, w2)
    add(a, b)
end"#;

    let mut ant = AtomicNeuralTransistor::from_source(source).unwrap();
    let input: Vec<Signal> = vec![Signal::new_raw(1, 10, 1); 4];
    let output = ant.forward(&input).unwrap();
    assert_eq!(output.len(), 2);
}
