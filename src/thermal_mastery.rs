//! Thermal mastery learning — pressure-based plasticity with per-weight temperature gating.
//!
//! Same mastery algorithm as learning.rs, but each weight's temperature controls
//! how much pressure is needed for a transition. HOT weights change easily.
//! COLD weights are frozen. Hits cool weights down. Errors warm them back up.

use crate::core::thermal::{ThermalWeight, ThermalWeightMatrix, ThermalMasteryConfig};
use crate::core::weight_matrix::packed_from_current;
use ternary_signal::PackedSignal;

/// Representable positive magnitudes in PackedSignal (sorted).
const REPR_LEVELS: &[i32] = &[
    0, 1, 4, 16, 32, 64, 128, 255, 256, 512, 1020, 1024, 2048, 4080, 4096,
    8160, 8192, 16320, 16384, 32640, 32768, 65025,
];

fn step_up(current: i32) -> i32 {
    let abs = current.unsigned_abs() as i32;
    for &level in REPR_LEVELS {
        if level > abs {
            return if current >= 0 { level } else { -level };
        }
    }
    current
}

fn step_down(current: i32) -> i32 {
    let abs = current.unsigned_abs() as i32;
    let mut prev = 0i32;
    for &level in REPR_LEVELS {
        if level >= abs { break; }
        prev = level;
    }
    if current > 0 { prev } else if current < 0 { -prev } else { 0 }
}

/// Per-matrix learning state (tracks total steps/transitions).
#[derive(Clone, Debug)]
pub struct ThermalMasteryState {
    pub config: ThermalMasteryConfig,
    pub steps: u64,
    pub transitions: u64,
}

impl ThermalMasteryState {
    pub fn new(config: ThermalMasteryConfig) -> Self {
        Self { config, steps: 0, transitions: 0 }
    }

    /// Run one mastery learning step with thermal gating.
    ///
    /// Same algorithm as learning.rs MasteryState::update, but:
    /// - Pressure is accumulated per-weight (stored IN the ThermalWeight)
    /// - Pressure threshold is scaled by weight temperature
    /// - COLD weights are skipped entirely
    /// - Correct participations → hit() on contributing weights
    pub fn update(
        &mut self,
        weights: &mut ThermalWeightMatrix,
        input: &[PackedSignal],
        output: &[PackedSignal],
        target: &[PackedSignal],
        correct: bool, // was this a correct detection overall?
    ) {
        assert_eq!(input.len(), weights.cols);
        assert_eq!(output.len(), weights.rows);
        assert_eq!(target.len(), weights.rows);

        self.steps += 1;

        let max_input = input.iter()
            .map(|s| s.current().unsigned_abs())
            .max()
            .unwrap_or(1)
            .max(1);
        let activity_threshold = max_input / 4; // top 25%

        // Pass 1: Hit counting on correct detections (BEFORE error loop).
        // Weights that participate in correct detections get cooled.
        // This happens even when error == 0 (output matches target perfectly).
        if correct {
            for j in 0..weights.cols {
                let input_abs = input[j].current().unsigned_abs();
                if input_abs > activity_threshold {
                    // Hit ALL rows' weights for this active input column
                    for i in 0..weights.rows {
                        let w_idx = i * weights.cols + j;
                        weights.data[w_idx].hit(self.config.cooling_rate);
                    }
                }
            }
        }

        // Pass 2: Pressure accumulation and transitions (only when there IS error).
        for i in 0..weights.rows {
            let error = target[i].current() as i64 - output[i].current() as i64;
            if error == 0 { continue; }
            let direction = error.signum() as i32;

            for j in 0..weights.cols {
                let w_idx = i * weights.cols + j;
                let tw = &weights.data[w_idx];

                // COLD weights: skip entirely
                if tw.pressure_multiplier() == 0 { continue; }

                let input_abs = input[j].current().unsigned_abs();
                let input_sign = (input[j].current() as i64).signum() as i32;

                // Activity gate
                if input_abs <= activity_threshold { continue; }

                // Activity strength
                let activity_strength = ((input_abs - activity_threshold) as i64 * 15
                    / max_input as i64).max(1) as i32;

                // Error magnitude
                let error_mag = ((error.abs().min(127) as i32) + 31) / 32;

                // Accumulate pressure
                let pressure_delta = direction * input_sign * activity_strength * error_mag;
                let tw = &mut weights.data[w_idx];
                tw.pressure = tw.pressure.saturating_add(pressure_delta as i16);

                // Threshold gate: scaled by temperature band
                let effective_threshold = self.config.pressure_threshold as i16
                    * tw.pressure_multiplier() as i16;

                if tw.pressure.abs() >= effective_threshold {
                    let needed = tw.pressure.signum() as i32;
                    apply_transition(&mut tw.signal, needed, &mut self.transitions);
                    tw.pressure = 0;
                }
            }
        }
    }

    /// Decay pressure on all weights. Call once per epoch.
    pub fn decay(&self, weights: &mut ThermalWeightMatrix) {
        for tw in &mut weights.data {
            if tw.pressure > 0 {
                tw.pressure = (tw.pressure - self.config.decay_rate as i16).max(0);
            } else if tw.pressure < 0 {
                tw.pressure = (tw.pressure + self.config.decay_rate as i16).min(0);
            }

            // Warming: if a weight has been under sustained pressure despite being cool,
            // warm it back up so it can adapt
            if tw.pressure.abs() as i32 >= self.config.warming_threshold
                && tw.temperature < 128
            {
                tw.warm(self.config.warming_step);
                tw.pressure = 0; // reset after warming
            }
        }
    }
}

/// Apply weaken-before-flip transition.
fn apply_transition(weight: &mut PackedSignal, needed_direction: i32, transitions: &mut u64) {
    let current = weight.current();
    let current_sign = if current > 0 { 1 } else if current < 0 { -1 } else { 0 };

    if current_sign == needed_direction {
        let stepped = step_up(current);
        if stepped != current {
            *weight = packed_from_current(stepped);
            *transitions += 1;
        }
    } else if current_sign == -needed_direction {
        let stepped = step_down(current);
        *weight = packed_from_current(stepped);
        *transitions += 1;
    } else {
        let initial = if needed_direction > 0 { 1 } else { -1 };
        *weight = packed_from_current(initial);
        *transitions += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::thermal::ThermalWeightMatrix;

    #[test]
    fn basic_thermal_mastery() {
        let config = ThermalMasteryConfig {
            pressure_threshold: 1, // low threshold for quick transitions
            participation_gate: 0,
            ..Default::default()
        };
        let mut state = ThermalMasteryState::new(config);
        let mut weights = ThermalWeightMatrix::zeros(1, 2);

        let input = vec![PackedSignal::pack(1, 64, 1), PackedSignal::pack(1, 64, 1)];
        let output = vec![PackedSignal::ZERO];
        let target = vec![PackedSignal::pack(1, 127, 1)];

        state.update(&mut weights, &input, &output, &target, true);
        assert!(state.transitions > 0, "should have transitions with low threshold");
    }

    #[test]
    fn cold_weights_dont_change() {
        let config = ThermalMasteryConfig {
            pressure_threshold: 1,
            participation_gate: 0,
            ..Default::default()
        };
        let mut state = ThermalMasteryState::new(config);
        let mut weights = ThermalWeightMatrix::zeros(1, 2);

        // Make all weights COLD
        for tw in &mut weights.data {
            tw.temperature = 0;
        }

        let input = vec![PackedSignal::pack(1, 128, 1), PackedSignal::pack(1, 128, 1)];
        let output = vec![PackedSignal::ZERO];
        let target = vec![PackedSignal::pack(1, 127, 1)];

        state.update(&mut weights, &input, &output, &target, true);
        assert_eq!(state.transitions, 0, "COLD weights should not transition");
        assert!(weights.data.iter().all(|tw| tw.signal.current() == 0),
            "COLD weights should remain zero");
    }

    #[test]
    fn hot_weights_change_faster_than_warm() {
        let config = ThermalMasteryConfig {
            pressure_threshold: 3,
            participation_gate: 0,
            ..Default::default()
        };

        let input = vec![PackedSignal::pack(1, 128, 1)];
        let output = vec![PackedSignal::ZERO];
        let target = vec![PackedSignal::pack(1, 127, 1)];

        // HOT weights
        let mut state_hot = ThermalMasteryState::new(config.clone());
        let mut weights_hot = ThermalWeightMatrix::zeros(1, 1);
        weights_hot.data[0].temperature = 255; // HOT
        for _ in 0..5 {
            state_hot.update(&mut weights_hot, &input, &output, &target, true);
        }

        // WARM weights
        let mut state_warm = ThermalMasteryState::new(config);
        let mut weights_warm = ThermalWeightMatrix::zeros(1, 1);
        weights_warm.data[0].temperature = 150; // WARM
        for _ in 0..5 {
            state_warm.update(&mut weights_warm, &input, &output, &target, true);
        }

        assert!(state_hot.transitions >= state_warm.transitions,
            "HOT should have more transitions than WARM: hot={} warm={}",
            state_hot.transitions, state_warm.transitions);
    }

    #[test]
    fn hits_cool_weights() {
        let config = ThermalMasteryConfig {
            pressure_threshold: 1,
            participation_gate: 0,
            cooling_rate: 1, // cool by 1 on every hit for testing
            ..Default::default()
        };
        let mut state = ThermalMasteryState::new(config);
        let mut weights = ThermalWeightMatrix::zeros(1, 1);

        let input = vec![PackedSignal::pack(1, 128, 1)];
        let output = vec![PackedSignal::ZERO];
        let target = vec![PackedSignal::pack(1, 127, 1)];

        let initial_temp = weights.data[0].temperature;
        for _ in 0..10 {
            state.update(&mut weights, &input, &output, &target, true);
        }
        assert!(weights.data[0].temperature < initial_temp,
            "temperature should decrease with hits: {} vs {}",
            weights.data[0].temperature, initial_temp);
    }
}
