//! Mastery learning for Signal synaptic strength matrices.
//!
//! Integer-only, pressure-based learning. No gradients.
//! Ternary pressure transitions gated by participation and neuromodulation.

use crate::core::weight_matrix::WeightMatrix;
use crate::neuromod::NeuromodState;
use ternary_signal::Signal;

/// Configuration for mastery learning.
#[derive(Clone, Debug)]
pub struct MasteryConfig {
    /// Pressure magnitude needed for a transition (default: 3).
    pub pressure_threshold: i32,
    /// Pressure decay per learning step (default: 1).
    pub decay_rate: i32,
    /// Minimum uses before learning applies (default: 5).
    pub participation_gate: u32,
}

impl Default for MasteryConfig {
    fn default() -> Self {
        Self {
            pressure_threshold: 3,
            decay_rate: 1,
            participation_gate: 5,
        }
    }
}

/// Per-matrix learning state.
#[derive(Clone, Debug)]
pub struct MasteryState {
    /// Per-strength pressure accumulator.
    pub pressure: Vec<i32>,
    /// Per-strength usage counter.
    pub participation: Vec<u32>,
    /// Learning configuration.
    pub config: MasteryConfig,
    /// Total learning steps applied.
    pub steps: u64,
    /// Total transitions applied.
    pub transitions: u64,
}

impl MasteryState {
    /// Create new mastery state for a matrix of given size.
    pub fn new(weight_count: usize, config: MasteryConfig) -> Self {
        Self {
            pressure: vec![0; weight_count],
            participation: vec![0; weight_count],
            config,
            steps: 0,
            transitions: 0,
        }
    }

    /// Run one mastery learning step.
    pub fn update(
        &mut self,
        weights: &mut WeightMatrix,
        input: &[Signal],
        output: &[Signal],
        target: &[Signal],
    ) {
        self.update_gated(weights, input, output, target, None);
    }

    /// Run one mastery learning step with optional neuromodulator gating.
    pub fn update_gated(
        &mut self,
        weights: &mut WeightMatrix,
        input: &[Signal],
        output: &[Signal],
        target: &[Signal],
        neuromod: Option<&NeuromodState>,
    ) {
        assert_eq!(input.len(), weights.cols);
        assert_eq!(output.len(), weights.rows);
        assert_eq!(target.len(), weights.rows);

        // DA gate: no learning if dopamine is below threshold
        if let Some(nm) = neuromod {
            if !nm.plasticity_open() {
                return;
            }
        }

        self.steps += 1;

        let max_input = input.iter()
            .map(|s| s.current().unsigned_abs())
            .max()
            .unwrap_or(1)
            .max(1);

        // NE controls participation divisor (default 4 = top 25%)
        let divisor = neuromod
            .map(|nm| nm.participation_divisor())
            .unwrap_or(4);
        let activity_threshold = max_input / divisor;

        for i in 0..weights.rows {
            let error = target[i].current() as i64 - output[i].current() as i64;
            if error == 0 {
                continue;
            }

            let direction = error.signum() as i32;

            for j in 0..weights.cols {
                let w_idx = i * weights.cols + j;
                let input_abs = input[j].current().unsigned_abs();

                // Record participation for any non-zero input
                if input_abs > 0 {
                    self.participation[w_idx] = self.participation[w_idx].saturating_add(1);
                }

                // Only learn if participation gate is met
                if self.participation[w_idx] < self.config.participation_gate {
                    continue;
                }

                // Activity-weighted pressure: only active inputs contribute
                if input_abs <= activity_threshold {
                    continue;
                }

                // Scale pressure by activity strength
                let activity_strength = ((input_abs - activity_threshold) as i64 * 15
                    / max_input as i64)
                    .max(1) as i32;

                // Error magnitude scaling: larger errors push harder (1-4)
                let error_mag = ((error.abs().min(127) as i32) + 31) / 32;

                // Pressure = direction × input_sign × activity_strength × error_mag
                let input_sign = (input[j].current() as i64).signum() as i32;
                self.pressure[w_idx] += direction * input_sign * activity_strength * error_mag;

                // Threshold gate: transition when |pressure| >= threshold
                if self.pressure[w_idx].abs() >= self.config.pressure_threshold {
                    let needed_direction = self.pressure[w_idx].signum();
                    apply_transition(&mut weights.data[w_idx], needed_direction, &mut self.transitions);
                    self.pressure[w_idx] = 0;
                }
            }
        }
    }

    /// Apply pressure decay to all strengths.
    /// Call once per cycle, not per sample.
    pub fn decay(&mut self) {
        self.decay_gated(None);
    }

    /// Apply pressure decay with optional neuromodulator gating.
    pub fn decay_gated(&mut self, neuromod: Option<&NeuromodState>) {
        let multiplier = neuromod
            .map(|nm| nm.decay_multiplier())
            .unwrap_or(1);
        let effective_decay = self.config.decay_rate * multiplier;

        for p in self.pressure.iter_mut() {
            if *p > 0 {
                *p = (*p - effective_decay).max(0);
            } else if *p < 0 {
                *p = (*p + effective_decay).min(0);
            }
        }
    }
}

/// Apply a transition using the weaken-before-flip pattern.
///
/// Works directly on Signal: step magnitude up/down, flip polarity only when depleted.
fn apply_transition(signal: &mut Signal, needed_direction: i32, transitions: &mut u64) {
    let current = signal.current();
    let current_sign = if current > 0 { 1 } else if current < 0 { -1 } else { 0 };

    if current_sign == needed_direction {
        // Polarity matches: strengthen
        let new_mag = signal.magnitude.saturating_add(1);
        if new_mag != signal.magnitude {
            signal.magnitude = new_mag;
            if signal.multiplier == 0 { signal.multiplier = 1; }
            *transitions += 1;
        }
    } else if current_sign == -needed_direction {
        // Polarity opposes: weaken first
        let new_mag = signal.magnitude.saturating_sub(1);
        signal.magnitude = new_mag;
        if new_mag == 0 {
            signal.polarity = 0;
            signal.multiplier = 0;
        }
        *transitions += 1;
    } else {
        // Current is zero: set initial polarity
        signal.polarity = if needed_direction > 0 { 1 } else { -1 };
        signal.magnitude = 1;
        signal.multiplier = 1;
        *transitions += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mastery_config_default() {
        let cfg = MasteryConfig::default();
        assert_eq!(cfg.pressure_threshold, 3);
        assert_eq!(cfg.decay_rate, 1);
        assert_eq!(cfg.participation_gate, 5);
    }

    #[test]
    fn test_mastery_state_creation() {
        let state = MasteryState::new(100, MasteryConfig::default());
        assert_eq!(state.pressure.len(), 100);
        assert_eq!(state.participation.len(), 100);
        assert_eq!(state.steps, 0);
        assert_eq!(state.transitions, 0);
    }

    #[test]
    fn test_pressure_accumulates_or_transitions() {
        let mut weights = WeightMatrix::zeros(1, 2);
        let mut state = MasteryState::new(2, MasteryConfig {
            pressure_threshold: 3,
            decay_rate: 0,
            participation_gate: 0,
        });

        let input = vec![Signal::new_raw(1, 32, 1), Signal::new_raw(1, 32, 1)];
        let output = vec![Signal::ZERO];
        let target = vec![Signal::new_raw(1, 64, 1)];

        state.update(&mut weights, &input, &output, &target);
        assert_eq!(state.steps, 1);
        let has_pressure = state.pressure[0] > 0 || state.pressure[1] > 0;
        let has_transition = state.transitions > 0;
        assert!(has_pressure || has_transition,
            "Expected pressure accumulation or transition");
    }

    #[test]
    fn test_weight_transitions() {
        let mut weights = WeightMatrix::zeros(1, 1);
        let mut state = MasteryState::new(1, MasteryConfig {
            pressure_threshold: 1,
            decay_rate: 0,
            participation_gate: 0,
        });

        let input = vec![Signal::new_raw(1, 64, 1)];
        let output = vec![Signal::ZERO];
        let target = vec![Signal::new_raw(1, 128, 1)];

        state.update(&mut weights, &input, &output, &target);
        assert!(state.transitions > 0);
    }

    #[test]
    fn test_participation_gate() {
        let mut weights = WeightMatrix::zeros(1, 1);
        let mut state = MasteryState::new(1, MasteryConfig {
            pressure_threshold: 1,
            decay_rate: 0,
            participation_gate: 10,
        });

        let input = vec![Signal::new_raw(1, 64, 1)];
        let output = vec![Signal::ZERO];
        let target = vec![Signal::new_raw(1, 128, 1)];

        for _ in 0..9 {
            state.update(&mut weights, &input, &output, &target);
        }
        assert_eq!(state.transitions, 0);

        state.update(&mut weights, &input, &output, &target);
        assert!(state.transitions > 0);
    }
}
