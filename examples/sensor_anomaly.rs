//! Sensor Anomaly Detection — Example 13
//!
//! Real-world use case: an industrial sensor array produces 32-dim signal
//! vectors. The system learns normal baseline, detects anomalies via
//! surprise, classifies anomaly types, and adapts to sensor recalibration.
//!
//! ANT computation is defined in `runes/sensor_anomaly.rune`.
//! This runner orchestrates signal generation, multiplex routing,
//! prediction, and neuromod gating.
//!
//! **Phase 1 — Baseline Learning** (200 ticks, normal signals only)
//!   PredictionEngine locks onto normal pattern. Salience router trains.
//!
//! **Phase 2 — Anomaly Injection** (300 ticks, 60% normal + 4 anomaly types)
//!   Surprise detects anomalies. Classification via per-dim error patterns.
//!
//! **Phase 3 — Sensor Recalibration** (100 ticks, shifted baseline)
//!   Global offset simulates recalibration. System adapts as EMA catches up.

use atomic_neural_transistors::core::weight_matrix::packed_from_current;
use atomic_neural_transistors::{AtomicNeuralTransistor, PackedSignal, Value};

// ---------------------------------------------------------------------------
// PRNG (xorshift64)
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn next_i32(&mut self, lo: i32, hi: i32) -> i32 {
        let range = (hi - lo + 1) as u64;
        lo + (self.next() % range) as i32
    }
}

// ---------------------------------------------------------------------------
// Signal Generator — stationary baseline + noise (real sensor behavior)
// ---------------------------------------------------------------------------

struct SignalGenerator {
    baselines: [i32; 32],
    rng: Rng,
}

impl SignalGenerator {
    fn new(rng: &mut Rng) -> Self {
        let mut baselines = [0i32; 32];
        for b in baselines.iter_mut() {
            *b = 40 + (rng.next() % 81) as i32;
        }
        Self {
            baselines,
            rng: Rng::new(rng.next()),
        }
    }

    fn normal(&mut self) -> Vec<PackedSignal> {
        (0..32)
            .map(|d| {
                let noise = self.rng.next_i32(-10, 10);
                packed_from_current(self.baselines[d] + noise)
            })
            .collect()
    }

    fn spike(&mut self) -> Vec<PackedSignal> {
        let mut signal = self.normal();
        let n_dims = 4 + (self.rng.next() % 5) as usize;
        for _ in 0..n_dims {
            let d = (self.rng.next() % 32) as usize;
            let impulse = if self.rng.next() % 2 == 0 { 200 } else { -200 };
            let current = signal[d].current();
            signal[d] = packed_from_current(current.saturating_add(impulse));
        }
        signal
    }

    fn drift(&mut self, offset: i32) -> Vec<PackedSignal> {
        let mut signal = self.normal();
        for d in 0..32 {
            let current = signal[d].current();
            signal[d] = packed_from_current(current.saturating_add(offset));
        }
        signal
    }

    fn dropout(&mut self) -> Vec<PackedSignal> {
        let mut signal = self.normal();
        let n_dims = 8 + (self.rng.next() % 5) as usize;
        for _ in 0..n_dims {
            let d = (self.rng.next() % 32) as usize;
            signal[d] = PackedSignal::ZERO;
        }
        signal
    }

    fn oscillation(&mut self, tick: u32) -> Vec<PackedSignal> {
        let mut signal = self.normal();
        let n_dims = 8 + (self.rng.next() % 7) as usize;
        let polarity: i32 = if tick % 2 == 0 { 80 } else { -80 };
        for _ in 0..n_dims {
            let d = (self.rng.next() % 32) as usize;
            let current = signal[d].current();
            signal[d] = packed_from_current(current.saturating_add(polarity));
        }
        signal
    }

    fn normal_with_offset(&mut self, offset: i32) -> Vec<PackedSignal> {
        (0..32)
            .map(|d| {
                let noise = self.rng.next_i32(-10, 10);
                packed_from_current(self.baselines[d] + noise + offset)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Anomaly types and classification
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum AnomalyType {
    Normal,
    Spike,
    Drift,
    Dropout,
    Oscillation,
}

impl AnomalyType {
    fn label(&self) -> &'static str {
        match self {
            AnomalyType::Normal => "normal",
            AnomalyType::Spike => "spike",
            AnomalyType::Drift => "drift",
            AnomalyType::Dropout => "dropout",
            AnomalyType::Oscillation => "oscillation",
        }
    }
}

/// Classify anomaly type from raw input deviation vs baseline.
fn classify_from_deviation(signal: &[PackedSignal], baseline: &[i32]) -> usize {
    let n = signal.len().min(baseline.len());
    if n == 0 {
        return 0;
    }

    let deviations: Vec<i32> = (0..n)
        .map(|d| signal[d].current() - baseline[d])
        .collect();

    let extreme_count = deviations.iter().filter(|d| d.abs() > 170).count();
    let large_count = deviations.iter().filter(|d| d.abs() > 50).count();
    let near_zero = deviations.iter().filter(|d| d.abs() < 15).count();
    let sum: i64 = deviations.iter().map(|d| *d as i64).sum();
    let positive_shift = deviations.iter().filter(|d| **d > 20).count();

    if extreme_count >= 2 && extreme_count <= 10 && near_zero > 16 {
        return 0; // spike
    }
    if sum.abs() > 400 && positive_shift > 18 {
        return 1; // drift
    }
    let near_zero_signal = signal.iter().filter(|s| s.current().abs() < 10).count();
    if near_zero_signal > 5 {
        return 2; // dropout
    }
    if large_count > 3 {
        return 3; // oscillation
    }
    if extreme_count > 0 {
        return 0; // spike fallback
    }
    0
}

// ---------------------------------------------------------------------------
// Runes helpers
// ---------------------------------------------------------------------------

fn signals_to_value(signals: &[PackedSignal]) -> Value {
    Value::Array(signals.iter().map(|s| Value::Integer(s.as_u8() as i64)).collect())
}

fn value_to_signals(val: &Value) -> Vec<PackedSignal> {
    match val {
        Value::Array(arr) => arr.iter().map(|v| match v {
            Value::Integer(n) => PackedSignal::from_raw(*n as u8),
            _ => PackedSignal::ZERO,
        }).collect(),
        _ => vec![],
    }
}

/// Extract integer from Value::Array at index, or from Value::Integer directly.
fn value_array_int(val: &Value, idx: usize) -> i64 {
    match val {
        Value::Array(arr) => match arr.get(idx) {
            Some(Value::Integer(n)) => *n,
            _ => 0,
        },
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("=== Sensor Anomaly Detection — Example 13 ===");
    println!();
    println!("Signal: 32-dim, 600 ticks total (stationary baseline + noise)");
    println!("ANTs: magnitude_profiler, delta_detector, frequency_analyzer (Runes)");

    let mut rng = Rng::new(0xDEAD_BEEF_CAFE);
    let output_dim = 32;

    let mut gen = SignalGenerator::new(&mut rng);

    // Load .rune script
    let rune_path = std::path::Path::new("runes/sensor_anomaly.rune");
    let mut ant = AtomicNeuralTransistor::from_file(rune_path)
        .expect("Failed to load sensor_anomaly.rune");

    // Create handles via Runes verbs
    let nm_handle = ant
        .call_values("neuromod_new", vec![])
        .expect("neuromod_new failed");
    let pred_handle = ant
        .call_values("predict_new", vec![Value::Integer(output_dim as i64), Value::Integer(2), Value::Integer(25)])
        .expect("predict_new failed");
    let sal_handle = ant
        .call_values("salience_new", vec![Value::Integer(3), Value::Integer(output_dim as i64)])
        .expect("salience_new failed");

    let mut previous = vec![PackedSignal::ZERO; output_dim];

    // =========================================================================
    // Phase 1 — Baseline Learning (200 ticks)
    // =========================================================================
    println!();
    println!("--- Phase 1: Baseline Learning (200 ticks) ---");

    let mut warmup_surprises = 0u32;
    let mut warmup_surprises_after_20 = 0u32;
    let mut baseline_sum = vec![0i64; output_dim];
    let baseline_count = 200i64;

    for tick in 0..200u32 {
        let signal = gen.normal();
        for d in 0..output_dim {
            baseline_sum[d] += signal[d].current() as i64;
        }

        let (routed, surprise_mag, is_surprising, _direction, _da) =
            process_tick(&mut ant, &signal, &previous, &nm_handle, &pred_handle, &sal_handle, Some(&signal));

        if is_surprising {
            warmup_surprises += 1;
            if tick >= 20 {
                warmup_surprises_after_20 += 1;
            }
        }

        previous = routed;
        let _ = (surprise_mag, _direction, _da);
    }

    let baseline: Vec<i32> = baseline_sum
        .iter()
        .map(|s| (*s / baseline_count) as i32)
        .collect();

    let da_val = read_da(&mut ant, &nm_handle);
    let gate_open = read_gate(&mut ant, &nm_handle);
    println!(
        "  Total surprises: {} (after warmup: {})",
        warmup_surprises, warmup_surprises_after_20
    );
    println!("  DA final: {}", da_val);
    println!(
        "  Plasticity gate: {} (DA {} > gate 77)",
        if gate_open { "OPEN" } else { "CLOSED" },
        da_val
    );

    // =========================================================================
    // Phase 2 — Anomaly Injection (300 ticks)
    // =========================================================================
    println!();
    println!("--- Phase 2: Anomaly Detection (300 ticks) ---");

    let mut total_anomalies = 0u32;
    let mut total_normals = 0u32;
    let mut detected_anomalies = 0u32;
    let mut false_positives = 0u32;
    let mut per_type_injected = [0u32; 4];
    let mut per_type_detected = [0u32; 4];
    let mut per_type_classified = [0u32; 4];
    let mut da_min: i64 = 255;
    let mut da_max: i64 = 0;
    let mut drift_offset = 0i32;

    for tick in 0..300u32 {
        let anomaly_roll = rng.next() % 100;
        let (signal, anomaly_type) = if anomaly_roll < 60 {
            (gen.normal(), AnomalyType::Normal)
        } else if anomaly_roll < 70 {
            (gen.spike(), AnomalyType::Spike)
        } else if anomaly_roll < 80 {
            drift_offset += 8;
            (gen.drift(drift_offset), AnomalyType::Drift)
        } else if anomaly_roll < 90 {
            (gen.dropout(), AnomalyType::Dropout)
        } else {
            (gen.oscillation(tick), AnomalyType::Oscillation)
        };

        let is_anomaly = anomaly_type != AnomalyType::Normal;

        // No target during detection phase — pure observation
        let (routed, surprise_mag, is_surprising, _direction, da) =
            process_tick(&mut ant, &signal, &previous, &nm_handle, &pred_handle, &sal_handle, None);

        da_min = da_min.min(da);
        da_max = da_max.max(da);

        if is_anomaly {
            total_anomalies += 1;
            let type_idx = match anomaly_type {
                AnomalyType::Spike => 0,
                AnomalyType::Drift => 1,
                AnomalyType::Dropout => 2,
                AnomalyType::Oscillation => 3,
                _ => unreachable!(),
            };
            per_type_injected[type_idx] += 1;

            if is_surprising {
                detected_anomalies += 1;
                per_type_detected[type_idx] += 1;

                let classified = classify_from_deviation(&signal, &baseline);
                if classified == type_idx {
                    per_type_classified[type_idx] += 1;
                }

                // Reward detection with DA
                ant.call_values(
                    "neuromod_inject",
                    vec![nm_handle.clone(), Value::String("da".into()), Value::Integer(15)],
                )
                .ok();
            }
        } else {
            total_normals += 1;
            if is_surprising {
                false_positives += 1;
            }
        }

        if tick < 50 && is_surprising && is_anomaly {
            println!(
                "  [tick {:>3}] DETECTED {} (mag={})",
                tick,
                anomaly_type.label(),
                surprise_mag
            );
        }

        previous = routed;
    }

    let detection_rate = if total_anomalies > 0 {
        detected_anomalies as f64 / total_anomalies as f64 * 100.0
    } else {
        0.0
    };
    let fp_rate = if total_normals > 0 {
        false_positives as f64 / total_normals as f64 * 100.0
    } else {
        0.0
    };

    println!();
    println!(
        "  Anomalies injected: {} / {} ticks",
        total_anomalies, 300
    );
    println!(
        "  Detected (surprise): {}/{} ({:.1}%)",
        detected_anomalies, total_anomalies, detection_rate
    );
    println!(
        "  False positives: {}/{} ({:.1}%)",
        false_positives, total_normals, fp_rate
    );
    println!();
    println!("  Per-type detection:");
    for (i, label) in ["Spike", "Drift", "Dropout", "Oscillation"]
        .iter()
        .enumerate()
    {
        if per_type_injected[i] > 0 {
            let det_pct = per_type_detected[i] as f64 / per_type_injected[i] as f64 * 100.0;
            let cls_pct = if per_type_detected[i] > 0 {
                per_type_classified[i] as f64 / per_type_detected[i] as f64 * 100.0
            } else {
                0.0
            };
            println!(
                "    {:>11}: detected {}/{} ({:.0}%), classified {}/{} ({:.0}%)",
                label,
                per_type_detected[i],
                per_type_injected[i],
                det_pct,
                per_type_classified[i],
                per_type_detected[i],
                cls_pct,
            );
        }
    }
    println!();
    let da_final = read_da(&mut ant, &nm_handle);
    println!("  DA range: {} - {}", da_min, da_max);
    println!("  DA final: {}", da_final);

    // =========================================================================
    // Phase 3 — Sensor Recalibration (100 ticks)
    // =========================================================================
    println!();
    println!("--- Phase 3: Sensor Recalibration (100 ticks) ---");

    // Reset predictor: create fresh handles
    let pred_handle = ant
        .call_values("predict_new", vec![Value::Integer(output_dim as i64), Value::Integer(2), Value::Integer(25)])
        .expect("predict_new failed");
    let nm_handle = ant
        .call_values("neuromod_new", vec![])
        .expect("neuromod_new failed");

    // Re-establish baseline (20 ticks)
    for _ in 0..20u32 {
        let signal = gen.normal();
        let (routed, _, _, _, _) =
            process_tick(&mut ant, &signal, &previous, &nm_handle, &pred_handle, &sal_handle, None);
        previous = routed;
    }
    println!("  Baseline re-established (20 ticks), now applying offset +80...");

    let recal_offset = 80i32;
    let mut surprise_count_window = [false; 100];
    let mut adapted_at: Option<usize> = None;
    let mut initial_surprise_count = 0u32;
    let mut final_surprise_count = 0u32;

    for tick in 0..100usize {
        let signal = gen.normal_with_offset(recal_offset);

        let (routed, _, is_surprising, _, _) =
            process_tick(&mut ant, &signal, &previous, &nm_handle, &pred_handle, &sal_handle, None);
        surprise_count_window[tick] = is_surprising;

        if tick < 20 && is_surprising {
            initial_surprise_count += 1;
        }
        if tick >= 80 && is_surprising {
            final_surprise_count += 1;
        }

        if tick >= 10 && adapted_at.is_none() {
            let window_surprises: u32 = (tick - 9..=tick)
                .map(|t| if surprise_count_window[t] { 1u32 } else { 0 })
                .sum();
            if window_surprises < 2 {
                adapted_at = Some(tick);
            }
        }

        previous = routed;
    }

    let initial_rate = initial_surprise_count as f64 / 20.0 * 100.0;
    let final_rate = final_surprise_count as f64 / 20.0 * 100.0;

    let da_final = read_da(&mut ant, &nm_handle);
    let ne_val = read_chemical(&mut ant, &nm_handle, "ne");
    let sht_val = read_chemical(&mut ant, &nm_handle, "5ht");

    println!(
        "  Initial surprise rate (first 20): {:.0}% ({}/20)",
        initial_rate, initial_surprise_count
    );
    match adapted_at {
        Some(t) => println!("  Adapted at tick: {} (surprise window < 20%)", t),
        None => println!("  Adaptation: not fully converged in 100 ticks"),
    }
    println!(
        "  Final surprise rate (last 20): {:.0}% ({}/20)",
        final_rate, final_surprise_count
    );
    println!("  DA final: {}", da_final);
    println!(
        "  Neuromod state: DA={} NE={} 5HT={}",
        da_final, ne_val, sht_val
    );

    println!();
    println!("=== Sensor anomaly detection complete ===");
}

// ---------------------------------------------------------------------------
// Process one tick through the Runes-defined ANT pipeline
// ---------------------------------------------------------------------------

/// Process one tick: call 3 ANT forwards (Runes), route, predict, tick neuromod.
/// Returns (routed_output, surprise_magnitude, is_surprising, direction, da).
fn process_tick(
    ant: &mut AtomicNeuralTransistor,
    signal: &[PackedSignal],
    previous: &[PackedSignal],
    nm_handle: &Value,
    pred_handle: &Value,
    sal_handle: &Value,
    target: Option<&[PackedSignal]>,
) -> (Vec<PackedSignal>, i64, bool, i64, i64) {
    let input_val = signals_to_value(signal);
    let prev_val = signals_to_value(previous);

    // Call ANT forwards defined in .rune
    let out_mag = ant.call_values("magnitude_forward", vec![input_val.clone()])
        .expect("magnitude_forward failed");
    let out_delta = ant.call_values("delta_forward", vec![input_val.clone(), prev_val])
        .expect("delta_forward failed");
    let out_freq = ant.call_values("frequency_forward", vec![input_val])
        .expect("frequency_forward failed");

    // Concatenate all ANT outputs for salience routing
    let mag_delta = ant.call_values("concat", vec![out_mag, out_delta])
        .expect("concat failed");
    let all_outputs = ant.call_values("concat", vec![mag_delta, out_freq])
        .expect("concat failed");

    // Route through salience
    let route_result = ant.call_values("salience_route", vec![sal_handle.clone(), all_outputs])
        .expect("salience_route failed");

    // Extract routed output (first 32 elements)
    let routed = ant.call_values("slice", vec![route_result, Value::Integer(0), Value::Integer(32)])
        .expect("slice failed");
    let routed_signals = value_to_signals(&routed);

    // Prediction: observe routed output with optional target
    let target_val = match target {
        Some(t) => signals_to_value(t),
        None => Value::Nil,
    };
    let surprise = ant.call_values("predict_observe", vec![pred_handle.clone(), routed.clone(), target_val])
        .expect("predict_observe failed");

    let surprise_mag = value_array_int(&surprise, 0);
    let is_surprising = value_array_int(&surprise, 1) != 0;
    let direction = value_array_int(&surprise, 2);

    // Tick neuromod
    ant.call_values("neuromod_tick", vec![nm_handle.clone()]).ok();

    // Read DA
    let da = read_da_from_value(ant, nm_handle);

    (routed_signals, surprise_mag, is_surprising, direction, da)
}

fn read_da(ant: &mut AtomicNeuralTransistor, nm_handle: &Value) -> i64 {
    read_da_from_value(ant, nm_handle)
}

fn read_da_from_value(ant: &mut AtomicNeuralTransistor, nm_handle: &Value) -> i64 {
    match ant.call_values("neuromod_read", vec![nm_handle.clone(), Value::String("da".into())]) {
        Ok(Value::Integer(n)) => n,
        _ => 128,
    }
}

fn read_gate(ant: &mut AtomicNeuralTransistor, nm_handle: &Value) -> bool {
    match ant.call_values("neuromod_gate", vec![nm_handle.clone()]) {
        Ok(Value::Bool(b)) => b,
        _ => false,
    }
}

fn read_chemical(ant: &mut AtomicNeuralTransistor, nm_handle: &Value, chem: &str) -> i64 {
    match ant.call_values("neuromod_read", vec![nm_handle.clone(), Value::String(chem.into())]) {
        Ok(Value::Integer(n)) => n,
        _ => 128,
    }
}
