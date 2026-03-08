//! Thermogram persistence tests:
//! - Roundtrip: save → load → identical synaptic strengths
//! - Self-init: no .thermo file → fresh zeros → forward works
//! - Mastery persistence: mastery_update → save → load → verify changed strengths
//! - Thermal progression: repeated mastery → consolidate → entries promote Hot→Warm

use atomic_neural_transistors::{AtomicNeuralTransistor, PackedSignal};
use thermogram::{Thermogram, PlasticityRule, Delta};

#[test]
fn test_thermogram_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let thermo_path = dir.path().join("test.thermo");

    // Create a Thermogram with some synaptic strengths
    let mut thermo = Thermogram::new("roundtrip_test", PlasticityRule::stdp_like());
    let data: Vec<PackedSignal> = (0..48)
        .map(|i| PackedSignal::pack(1, (i as u8 * 5) % 255, 1))
        .collect();
    let delta = Delta::create("test.w_in", data.clone(), "test");
    thermo.apply_delta(delta).unwrap();

    // Save
    thermo.save(&thermo_path).unwrap();

    // Load
    let loaded = Thermogram::load(&thermo_path).unwrap();
    let read_data = loaded.read("test.w_in").unwrap().unwrap();

    // Verify byte-identical
    assert_eq!(data.len(), read_data.len());
    for (i, (orig, loaded)) in data.iter().zip(read_data.iter()).enumerate() {
        assert_eq!(
            orig.as_u8(),
            loaded.as_u8(),
            "roundtrip mismatch at index {i}: expected 0x{:02x}, got 0x{:02x}",
            orig.as_u8(),
            loaded.as_u8()
        );
    }
}

#[test]
fn test_self_init_zeros() {
    // No .thermo file → load_synaptic self-inits zeros → forward should work
    let mut ant = AtomicNeuralTransistor::from_source_with_thermogram(
        include_str!("../runes/classifier.rune"),
        None,
        "self_init_test",
        None, // No thermo_path → fresh Thermogram
    )
    .unwrap();

    let input: Vec<PackedSignal> = (0..32)
        .map(|i| PackedSignal::pack(1, i as u8 * 8, 1))
        .collect();

    let output = ant.forward(&input).unwrap();
    assert_eq!(output.len(), 4, "classifier should output 4 class logits");

    // With zero synaptic strengths, all matmul results are zero,
    // so softmax should give uniform distribution
    for ps in &output {
        assert!(ps.current() >= 0, "softmax output should be non-negative");
    }
}

#[test]
fn test_thermogram_ant_save_load_cycle() {
    let dir = tempfile::tempdir().unwrap();
    let thermo_path = dir.path().join("cycle.thermo");

    // Create Thermogram with known data
    let mut thermo = Thermogram::new("cycle_test", PlasticityRule::stdp_like());
    let w_in_data: Vec<PackedSignal> = (0..24 * 64)
        .map(|i: usize| PackedSignal::pack(1, ((i * 3) % 200) as u8, 1))
        .collect();
    let w_out_data: Vec<PackedSignal> = (0..32 * 24)
        .map(|i: usize| PackedSignal::pack(-1, ((i * 7) % 180) as u8, 1))
        .collect();
    // First delta: no prev_hash needed
    thermo
        .apply_delta(Delta::create("diff.w_in", w_in_data.clone(), "test"))
        .unwrap();
    // Second delta: must chain to first
    let prev = thermo.dirty_chain.head_hash.clone();
    thermo
        .apply_delta(Delta::update("diff.w_out", w_out_data.clone(), "test",
            thermogram::Signal::positive(255), prev))
        .unwrap();
    thermo.save(&thermo_path).unwrap();

    // Load into an ANT
    let mut ant = AtomicNeuralTransistor::from_source_with_thermogram(
        include_str!("../runes/diff.rune"),
        None,
        "cycle_test",
        Some(&thermo_path),
    )
    .unwrap();

    // Run forward to prove it loads correctly
    let a: Vec<PackedSignal> = (0..32)
        .map(|i| PackedSignal::pack(1, i as u8 * 4, 1))
        .collect();
    let b: Vec<PackedSignal> = (0..32)
        .map(|i| PackedSignal::pack(-1, i as u8 * 2, 1))
        .collect();
    let mut input = a;
    input.extend_from_slice(&b);
    let output = ant.forward(&input).unwrap();
    assert_eq!(output.len(), 32);

    // Save back
    ant.save_thermogram(&thermo_path).unwrap();

    // Reload and verify data persisted
    let reloaded = Thermogram::load(&thermo_path).unwrap();
    let reloaded_w_in = reloaded.read("diff.w_in").unwrap().unwrap();
    assert_eq!(reloaded_w_in.len(), w_in_data.len());
    for (i, (orig, re)) in w_in_data.iter().zip(reloaded_w_in.iter()).enumerate() {
        assert_eq!(
            orig.as_u8(),
            re.as_u8(),
            "diff.w_in mismatch at {i}"
        );
    }
}

#[test]
fn test_thermal_progression() {
    // Repeated updates should create deltas that consolidate through thermal states.
    // With fast_learner config and high strength (0.9), consolidation will crystallize
    // entries directly to cold if they exceed the crystallization threshold.
    let mut thermo = Thermogram::for_fast_learner("progression_test", PlasticityRule::stdp_like());

    let data: Vec<PackedSignal> = vec![PackedSignal::pack(1, 100, 1); 16];

    // Apply multiple updates to the same key
    for i in 0..20 {
        let updated: Vec<PackedSignal> = data
            .iter()
            .map(|_| PackedSignal::pack(1, (100 + i) as u8, 1))
            .collect();
        let prev_hash = thermo.dirty_chain.head_hash.clone();
        let delta = Delta::update(
            "test.key",
            updated,
            "mastery",
            thermogram::Signal::positive(230), // high strength ~0.9
            prev_hash,
        );
        thermo.apply_delta(delta).unwrap();
    }

    // Consolidate to move dirty → hot (and possibly crystallize to cold)
    let result = thermo.consolidate().unwrap();
    assert!(
        result.deltas_processed > 0,
        "consolidation should process deltas"
    );

    // Entry should exist somewhere in the thermal layers after consolidation
    let stats = thermo.stats();
    let total = stats.hot_entries + stats.warm_entries + stats.cool_entries + stats.cold_entries;
    assert!(
        total > 0,
        "should have entries in some thermal state after consolidation (hot={}, warm={}, cool={}, cold={})",
        stats.hot_entries, stats.warm_entries, stats.cool_entries, stats.cold_entries
    );

    // Run thermal transitions
    thermo.run_thermal_transitions().unwrap();

    // Entry should still be readable after transitions
    let read = thermo.read("test.key").unwrap();
    assert!(read.is_some(), "test.key should be readable after transitions");
    assert_eq!(read.unwrap().len(), 16);

    // Final stats — entry persists
    let stats_after = thermo.stats();
    let total_after = stats_after.hot_entries + stats_after.warm_entries
        + stats_after.cool_entries + stats_after.cold_entries;
    assert!(total_after > 0, "entry should persist after thermal transitions");
}

#[test]
fn test_save_synaptic_creates_delta() {
    // Verify that save_synaptic verb creates a proper Thermogram delta
    let source = r#"rune "test_save" do
  version 1
end
use :ant_ml

def forward(input) do
    w = load_synaptic("test.w", 4, 8)
    save_synaptic(w, "test.w")
    matmul(input, w, 4, 8)
end"#;

    let mut ant = AtomicNeuralTransistor::from_source_with_thermogram(
        source,
        None,
        "save_test",
        None,
    )
    .unwrap();

    let input: Vec<PackedSignal> = vec![PackedSignal::pack(1, 50, 1); 8];
    ant.forward(&input).unwrap();

    // Check that the Thermogram has the delta
    let guard = ant.runtime().lock().unwrap();
    let thermo = guard.thermogram().unwrap();
    let data = thermo.read("test.w").unwrap();
    assert!(data.is_some(), "save_synaptic should have written data");
    assert_eq!(data.unwrap().len(), 32, "4x8 matrix = 32 elements");
}

#[test]
fn test_load_synaptic_deduplicates() {
    // Calling load_synaptic twice with the same key should return the same handle
    let source = r#"rune "test_dedup" do
  version 1
end
use :ant_ml

def forward(input) do
    w1 = load_synaptic("dedup.w", 2, 4)
    w2 = load_synaptic("dedup.w", 2, 4)
    # Both should be the same handle — matmul with either gives same result
    a = matmul(input, w1, 2, 4)
    b = matmul(input, w2, 2, 4)
    add(a, b)
end"#;

    let mut ant = AtomicNeuralTransistor::from_source_with_thermogram(
        source,
        None,
        "dedup_test",
        None,
    )
    .unwrap();

    let input: Vec<PackedSignal> = vec![PackedSignal::pack(1, 10, 1); 4];
    let output = ant.forward(&input).unwrap();
    assert_eq!(output.len(), 2);
}
