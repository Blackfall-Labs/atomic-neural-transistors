//! Sensor Anomaly Detection Runner
//!
//! Uses sensor_anomaly.rune (magnitude_forward, delta_forward, frequency_forward).
//! 3 phases: baseline learning, anomaly injection, recalibration.
//! Uses salience/prediction/neuromod verbs for orchestration.

use std::sync::Arc;
use atomic_neural_transistors::{AtomicNeuralTransistor, Signal, Value};
use atomic_neural_transistors::testdata::Rng;

fn signals_to_value(signals: &[Signal]) -> Value {
    Value::Array(Arc::new(
        signals.iter().map(|s| Value::Integer(s.current() as i64)).collect()
    ))
}

fn value_array_int(val: &Value, idx: usize) -> i64 {
    match val {
        Value::Array(arr) => match arr.get(idx) {
            Some(Value::Integer(n)) => *n,
            _ => 0,
        },
        _ => 0,
    }
}

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
        Self { baselines, rng: Rng::new(rng.next()) }
    }

    fn normal(&mut self) -> Vec<Signal> {
        (0..32).map(|d| {
            let noise = self.rng.next_i32(21) - 10;
            Signal::from_current(self.baselines[d] + noise)
        }).collect()
    }

    fn spike(&mut self) -> Vec<Signal> {
        let mut signal = self.normal();
        let n_dims = 4 + (self.rng.next() % 5) as usize;
        for _ in 0..n_dims {
            let d = (self.rng.next() % 32) as usize;
            let impulse = if self.rng.next() % 2 == 0 { 200 } else { -200 };
            let c = signal[d].current();
            signal[d] = Signal::from_current(c.saturating_add(impulse));
        }
        signal
    }

    fn drift(&mut self, offset: i32) -> Vec<Signal> {
        let mut signal = self.normal();
        for d in 0..32 {
            let c = signal[d].current();
            signal[d] = Signal::from_current(c.saturating_add(offset));
        }
        signal
    }

    fn dropout(&mut self) -> Vec<Signal> {
        let mut signal = self.normal();
        let n_dims = 8 + (self.rng.next() % 5) as usize;
        for _ in 0..n_dims {
            let d = (self.rng.next() % 32) as usize;
            signal[d] = Signal::from_current(0);
        }
        signal
    }

    fn oscillation(&mut self, tick: u32) -> Vec<Signal> {
        let mut signal = self.normal();
        let n_dims = 8 + (self.rng.next() % 7) as usize;
        let polarity: i32 = if tick % 2 == 0 { 80 } else { -80 };
        for _ in 0..n_dims {
            let d = (self.rng.next() % 32) as usize;
            let c = signal[d].current();
            signal[d] = Signal::from_current(c.saturating_add(polarity));
        }
        signal
    }

    fn normal_with_offset(&mut self, offset: i32) -> Vec<Signal> {
        (0..32).map(|d| {
            let noise = self.rng.next_i32(21) - 10;
            Signal::from_current(self.baselines[d] + noise + offset)
        }).collect()
    }
}

fn process_tick(
    ant: &mut AtomicNeuralTransistor,
    signal: &[Signal],
    previous: &[Signal],
    nm_handle: &Value,
    pred_handle: &Value,
    sal_handle: &Value,
    target: Option<&[Signal]>,
) -> (Vec<Signal>, i64, bool, i64) {
    let input_val = signals_to_value(signal);
    let prev_val = signals_to_value(previous);

    let out_mag = ant.call_values("magnitude_forward", vec![input_val.clone()])
        .expect("magnitude_forward failed");
    let out_delta = ant.call_values("delta_forward", vec![input_val.clone(), prev_val])
        .expect("delta_forward failed");
    let out_freq = ant.call_values("frequency_forward", vec![input_val])
        .expect("frequency_forward failed");

    let mag_delta = ant.call_values("join", vec![out_mag, out_delta])
        .expect("join failed");
    let all_outputs = ant.call_values("join", vec![mag_delta, out_freq])
        .expect("join failed");

    let route_result = ant.call_values("route", vec![sal_handle.clone(), all_outputs])
        .expect("route failed");
    let routed = ant.call_values("extract", vec![route_result, Value::Integer(0), Value::Integer(32)])
        .expect("extract failed");

    let routed_signals: Vec<Signal> = match &routed {
        Value::Array(arr) => arr.iter().map(|v| {
            if let Value::Integer(n) = v { Signal::from_current(*n as i32) } else { Signal::from_current(0) }
        }).collect(),
        _ => vec![Signal::from_current(0); 32],
    };

    let target_val = match target {
        Some(t) => signals_to_value(t),
        None => Value::Nil,
    };
    let surprise = ant.call_values("observe", vec![
        pred_handle.clone(), routed, target_val,
    ]).expect("observe failed");

    let surprise_mag = value_array_int(&surprise, 0);
    let is_surprising = value_array_int(&surprise, 1) != 0;
    let direction = value_array_int(&surprise, 2);

    ant.call_values("tick", vec![nm_handle.clone()]).ok();

    (routed_signals, surprise_mag, is_surprising, direction)
}

fn read_da(ant: &mut AtomicNeuralTransistor, nm_handle: &Value) -> i64 {
    match ant.call_values("read_chem", vec![nm_handle.clone(), Value::String("da".into())]) {
        Ok(Value::Integer(n)) => n,
        _ => 128,
    }
}

fn main() {
    println!("=== Sensor Anomaly Detection Runner ===\n");

    let mut rng = Rng::new(0xDEAD_BEEF);
    let output_dim = 32;
    let mut gen = SignalGenerator::new(&mut rng);

    let source = include_str!("../runes/sensor_anomaly.rune");
    let mut ant = AtomicNeuralTransistor::from_source(source)
        .expect("Failed to load sensor_anomaly.rune");

    let nm_handle = ant.call_values("create_neuromod", vec![]).expect("create_neuromod failed");
    let pred_handle = ant.call_values("create_predictor", vec![
        Value::Integer(output_dim as i64), Value::Integer(2), Value::Integer(25),
    ]).expect("create_predictor failed");
    let sal_handle = ant.call_values("create_salience", vec![
        Value::Integer(3), Value::Integer(output_dim as i64),
    ]).expect("create_salience failed");

    let mut previous = vec![Signal::from_current(0); output_dim];

    // Phase 1: Baseline Learning (200 ticks)
    println!("--- Phase 1: Baseline Learning (200 ticks) ---");
    let mut warmup_surprises = 0u32;

    for _tick in 0..200u32 {
        let signal = gen.normal();
        let (routed, _, is_surprising, _) = process_tick(
            &mut ant, &signal, &previous, &nm_handle, &pred_handle, &sal_handle, Some(&signal),
        );
        if is_surprising { warmup_surprises += 1; }
        previous = routed;
    }

    let da_val = read_da(&mut ant, &nm_handle);
    println!("  Surprises: {}", warmup_surprises);
    println!("  DA final: {}", da_val);

    // Phase 2: Anomaly Injection (300 ticks)
    println!("\n--- Phase 2: Anomaly Detection (300 ticks) ---");
    let mut total_anomalies = 0u32;
    let mut total_normals = 0u32;
    let mut detected_anomalies = 0u32;
    let mut false_positives = 0u32;
    let mut drift_offset = 0i32;

    for tick in 0..300u32 {
        let anomaly_roll = rng.next() % 100;
        let (signal, is_anomaly) = if anomaly_roll < 60 {
            (gen.normal(), false)
        } else if anomaly_roll < 70 {
            (gen.spike(), true)
        } else if anomaly_roll < 80 {
            drift_offset += 8;
            (gen.drift(drift_offset), true)
        } else if anomaly_roll < 90 {
            (gen.dropout(), true)
        } else {
            (gen.oscillation(tick), true)
        };

        let (routed, _surprise_mag, is_surprising, _) = process_tick(
            &mut ant, &signal, &previous, &nm_handle, &pred_handle, &sal_handle, None,
        );

        if is_anomaly {
            total_anomalies += 1;
            if is_surprising {
                detected_anomalies += 1;
                ant.call_values("inject", vec![
                    nm_handle.clone(), Value::String("da".into()), Value::Integer(15),
                ]).ok();
            }
        } else {
            total_normals += 1;
            if is_surprising { false_positives += 1; }
        }

        previous = routed;
    }

    let detection_rate = if total_anomalies > 0 {
        detected_anomalies as f64 / total_anomalies as f64 * 100.0
    } else { 0.0 };
    let fp_rate = if total_normals > 0 {
        false_positives as f64 / total_normals as f64 * 100.0
    } else { 0.0 };

    println!("  Anomalies injected: {}", total_anomalies);
    println!("  Detected: {}/{} ({:.1}%)", detected_anomalies, total_anomalies, detection_rate);
    println!("  False positives: {}/{} ({:.1}%)", false_positives, total_normals, fp_rate);

    // Phase 3: Sensor Recalibration (100 ticks)
    println!("\n--- Phase 3: Sensor Recalibration (100 ticks) ---");

    // Reset predictor
    let pred_handle = ant.call_values("create_predictor", vec![
        Value::Integer(output_dim as i64), Value::Integer(2), Value::Integer(25),
    ]).expect("create_predictor failed");
    let nm_handle = ant.call_values("create_neuromod", vec![]).expect("create_neuromod failed");

    // Re-establish baseline
    for _ in 0..20u32 {
        let signal = gen.normal();
        let (routed, _, _, _) = process_tick(
            &mut ant, &signal, &previous, &nm_handle, &pred_handle, &sal_handle, None,
        );
        previous = routed;
    }
    println!("  Baseline re-established, applying offset +80...");

    let recal_offset = 80i32;
    let mut initial_surprise_count = 0u32;
    let mut final_surprise_count = 0u32;

    for tick in 0..100usize {
        let signal = gen.normal_with_offset(recal_offset);
        let (routed, _, is_surprising, _) = process_tick(
            &mut ant, &signal, &previous, &nm_handle, &pred_handle, &sal_handle, None,
        );
        if tick < 20 && is_surprising { initial_surprise_count += 1; }
        if tick >= 80 && is_surprising { final_surprise_count += 1; }
        previous = routed;
    }

    println!("  Initial surprise rate (first 20): {:.0}% ({}/20)",
        initial_surprise_count as f64 / 20.0 * 100.0, initial_surprise_count);
    println!("  Final surprise rate (last 20): {:.0}% ({}/20)",
        final_surprise_count as f64 / 20.0 * 100.0, final_surprise_count);

    let da_final = read_da(&mut ant, &nm_handle);
    println!("  DA final: {}", da_final);

    println!("\n=== Sensor Anomaly Detection Runner Complete ===");
}
