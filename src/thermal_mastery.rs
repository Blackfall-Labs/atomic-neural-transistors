//! Thermal mastery learning — pressure-based plasticity with per-strength temperature gating.
//!
//! Same mastery algorithm as learning.rs, but each strength's temperature controls
//! how much pressure is needed for a transition. HOT strengths change easily.
//! COLD strengths are frozen. Hits cool strengths down. Errors warm them back up.

use crate::core::thermal::{ThermalWeight, ThermalWeightMatrix, ThermalMasteryConfig};
use ternary_signal::{Polarity, Signal};

/// Step the magnitude UP to the next level.
fn step_up_magnitude(magnitude: u8) -> u8 {
    magnitude.saturating_add(1)
}

/// Step the magnitude DOWN toward zero.
fn step_down_magnitude(magnitude: u8) -> u8 {
    magnitude.saturating_sub(1)
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
    /// - Pressure is accumulated per-strength (stored IN the ThermalWeight)
    /// - Pressure threshold is scaled by strength temperature
    /// - COLD strengths are skipped entirely
    /// - Correct participations → hit() on contributing strengths
    pub fn update(
        &mut self,
        weights: &mut ThermalWeightMatrix,
        input: &[Signal],
        output: &[Signal],
        target: &[Signal],
        correct: bool,
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

        // Compute prediction margin: difference between correct class output
        // and highest incorrect class output. Higher margin = more confident.
        let mut max_correct: i64 = 0;
        let mut max_incorrect: i64 = 0;
        let mut has_correct = false;
        let mut has_incorrect = false;
        for i in 0..weights.rows {
            let out_val = output[i].current() as i64;
            let tgt_val = target[i].current() as i64;
            if tgt_val > 0 {
                if !has_correct || out_val > max_correct { max_correct = out_val; }
                has_correct = true;
            } else {
                if !has_incorrect || out_val > max_incorrect { max_incorrect = out_val; }
                has_incorrect = true;
            }
        }
        let margin = if has_correct && has_incorrect {
            (max_correct - max_incorrect).max(0) as u16
        } else {
            0
        };

        // Pass 1: Hit/miss counting — performance-conditioned cooling.
        for j in 0..weights.cols {
            let input_abs = input[j].current().unsigned_abs();
            if input_abs > activity_threshold {
                for i in 0..weights.rows {
                    let w_idx = i * weights.cols + j;
                    if correct {
                        weights.data[w_idx].hit(self.config.cooling_rate, margin);
                    } else {
                        weights.data[w_idx].miss();
                    }
                }
            }
        }

        // Pass 2: Pressure accumulation and transitions.
        for i in 0..weights.rows {
            let error = target[i].current() as i64 - output[i].current() as i64;
            if error == 0 { continue; }
            let direction = error.signum() as i32;

            for j in 0..weights.cols {
                let w_idx = i * weights.cols + j;
                let tw = &weights.data[w_idx];

                // COLD strengths: skip entirely
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
                    apply_transition(tw, needed, &mut self.transitions);
                    tw.pressure = 0;
                }
            }
        }
    }

    /// Decay pressure on all strengths. Call once per cycle.
    pub fn decay(&self, weights: &mut ThermalWeightMatrix) {
        for tw in &mut weights.data {
            if tw.pressure > 0 {
                tw.pressure = (tw.pressure - self.config.decay_rate as i16).max(0);
            } else if tw.pressure < 0 {
                tw.pressure = (tw.pressure + self.config.decay_rate as i16).min(0);
            }

            // Warming: if a strength has been under sustained pressure despite being cool,
            // warm it back up so it can adapt
            if tw.pressure.abs() as i32 >= self.config.warming_threshold
                && tw.temperature < 128
            {
                tw.warm(self.config.warming_step);
                tw.pressure = 0;
            }
        }
    }
}

/// Apply weaken-before-flip transition on a ThermalWeight.
fn apply_transition(tw: &mut ThermalWeight, needed_direction: i32, transitions: &mut u64) {
    let current = tw.current();
    let current_sign = if current > 0 { 1 } else if current < 0 { -1 } else { 0 };

    if current_sign == needed_direction {
        // Polarity matches: strengthen (step up magnitude)
        let new_mag = step_up_magnitude(tw.magnitude);
        if new_mag != tw.magnitude {
            tw.magnitude = new_mag;
            if tw.multiplier == 0 { tw.multiplier = 1; }
            *transitions += 1;
        }
    } else if current_sign == -needed_direction {
        // Polarity opposes: weaken first (step down toward zero)
        let new_mag = step_down_magnitude(tw.magnitude);
        tw.magnitude = new_mag;
        if new_mag == 0 {
            tw.polarity = Polarity::Zero;
            tw.multiplier = 0;
        }
        *transitions += 1;
    } else {
        // Current is zero: set initial polarity in needed direction
        tw.polarity = if needed_direction > 0 { Polarity::Positive } else { Polarity::Negative };
        tw.magnitude = 1;
        tw.multiplier = 1;
        *transitions += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_thermal_mastery() {
        let config = ThermalMasteryConfig {
            pressure_threshold: 1,
            participation_gate: 0,
            ..Default::default()
        };
        let mut state = ThermalMasteryState::new(config);
        let mut weights = ThermalWeightMatrix::zeros(1, 2);

        let input = vec![Signal::new_raw(1, 64, 1), Signal::new_raw(1, 64, 1)];
        let output = vec![Signal::ZERO];
        let target = vec![Signal::new_raw(1, 127, 1)];

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

        for tw in &mut weights.data {
            tw.temperature = 0;
        }

        let input = vec![Signal::new_raw(1, 128, 1), Signal::new_raw(1, 128, 1)];
        let output = vec![Signal::ZERO];
        let target = vec![Signal::new_raw(1, 127, 1)];

        state.update(&mut weights, &input, &output, &target, true);
        assert_eq!(state.transitions, 0, "COLD strengths should not transition");
        assert!(weights.data.iter().all(|tw| tw.current() == 0),
            "COLD strengths should remain zero");
    }

    #[test]
    fn hot_weights_change_faster_than_warm() {
        let config = ThermalMasteryConfig {
            pressure_threshold: 3,
            participation_gate: 0,
            ..Default::default()
        };

        let input = vec![Signal::new_raw(1, 128, 1)];
        let output = vec![Signal::ZERO];
        let target = vec![Signal::new_raw(1, 127, 1)];

        let mut state_hot = ThermalMasteryState::new(config.clone());
        let mut weights_hot = ThermalWeightMatrix::zeros(1, 1);
        weights_hot.data[0].temperature = 255;
        for _ in 0..5 {
            state_hot.update(&mut weights_hot, &input, &output, &target, true);
        }

        let mut state_warm = ThermalMasteryState::new(config);
        let mut weights_warm = ThermalWeightMatrix::zeros(1, 1);
        weights_warm.data[0].temperature = 150;
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
            cooling_rate: 1,
            ..Default::default()
        };
        let mut state = ThermalMasteryState::new(config);
        let mut weights = ThermalWeightMatrix::zeros(1, 1);

        let input = vec![Signal::new_raw(1, 128, 1)];
        let output = vec![Signal::ZERO];
        let target = vec![Signal::new_raw(1, 127, 1)];

        let initial_temp = weights.data[0].temperature;
        for _ in 0..10 {
            state.update(&mut weights, &input, &output, &target, true);
        }
        assert!(weights.data[0].temperature < initial_temp,
            "temperature should decrease with hits: {} vs {}",
            weights.data[0].temperature, initial_temp);
    }
}
