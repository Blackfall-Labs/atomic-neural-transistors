//! Mastery learning for PackedSignal weight matrices.
//!
//! Integer-only, pressure-based learning. No gradients.
//! Ternary pressure transitions gated by participation and neuromodulation.

use crate::core::weight_matrix::{packed_from_current, WeightMatrix};
use crate::neuromod::NeuromodState;
use ternary_signal::PackedSignal;

/// Configuration for mastery learning.
#[derive(Clone, Debug)]
pub struct MasteryConfig {
    /// Pressure magnitude needed for a weight transition (default: 3).
    pub pressure_threshold: i32,
    /// Pressure decay per learning step (default: 1).
    pub decay_rate: i32,
    /// Minimum uses before learning applies to a weight (default: 5).
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

/// Per-weight-matrix learning state.
#[derive(Clone, Debug)]
pub struct MasteryState {
    /// Per-weight pressure accumulator.
    pub pressure: Vec<i32>,
    /// Per-weight usage counter.
    pub participation: Vec<u32>,
    /// Learning configuration.
    pub config: MasteryConfig,
    /// Total learning steps applied.
    pub steps: u64,
    /// Total transitions applied.
    pub transitions: u64,
}

impl MasteryState {
    /// Create new mastery state for a weight matrix of given size.
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
    ///
    /// Production mastery algorithm from astromind-archive:
    /// 1. Compute error signal: target - output (as current values)
    /// 2. Activity-weighted participation: only top 25% of active inputs contribute
    /// 3. Pressure accumulates from direction × activity_strength × error_magnitude
    /// 4. Threshold gate: transition only when |pressure| >= threshold
    /// 5. Weaken-before-flip: deplete magnitude to zero before polarity flip
    pub fn update(
        &mut self,
        weights: &mut WeightMatrix,
        input: &[PackedSignal],
        output: &[PackedSignal],
        target: &[PackedSignal],
    ) {
        self.update_gated(weights, input, output, target, None);
    }

    /// Run one mastery learning step with optional neuromodulator gating.
    ///
    /// When `neuromod` is provided:
    /// - Plasticity gate: no learning if DA is below gate threshold
    /// - Participation breadth: NE controls activity threshold divisor
    ///   (high NE = broader participation, low NE = narrower)
    pub fn update_gated(
        &mut self,
        weights: &mut WeightMatrix,
        input: &[PackedSignal],
        output: &[PackedSignal],
        target: &[PackedSignal],
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

        // Compute activity threshold based on neuromodulator state
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

            let direction = error.signum() as i32; // +1 or -1

            for j in 0..weights.cols {
                let w_idx = i * weights.cols + j;
                let input_abs = input[j].current().unsigned_abs();
                let input_sign = (input[j].current() as i64).signum() as i32;

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

                // Scale pressure by activity strength (production used ×15 scale)
                let activity_strength = ((input_abs - activity_threshold) as i64 * 15
                    / max_input as i64)
                    .max(1) as i32;

                // Error magnitude scaling: larger errors push harder (1-4)
                let error_mag = ((error.abs().min(127) as i32) + 31) / 32;

                // Pressure = direction × input_sign × activity_strength × error_mag
                self.pressure[w_idx] += direction * input_sign * activity_strength * error_mag;

                // Threshold gate: transition when |pressure| >= threshold
                if self.pressure[w_idx].abs() >= self.config.pressure_threshold {
                    let needed_direction = self.pressure[w_idx].signum(); // +1 or -1
                    apply_transition(&mut weights.data[w_idx], needed_direction, &mut self.transitions);
                    self.pressure[w_idx] = 0;
                }
            }
        }
    }

    /// Apply pressure decay to all weights.
    /// Call once per training cycle (epoch), not per sample.
    pub fn decay(&mut self) {
        self.decay_gated(None);
    }

    /// Apply pressure decay with optional neuromodulator gating.
    ///
    /// When `neuromod` is provided, 5HT controls decay rate:
    /// - High 5HT (255) → 2× decay (harder to accumulate pressure)
    /// - Neutral 5HT (128) → 1× decay (default behavior)
    /// - Low 5HT (0) → 0× decay (pressure accumulates freely)
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

/// Apply a weight transition using the weaken-before-flip pattern.
///
/// Production rule from astromind-archive:
/// 1. If polarity matches needed direction: strengthen (step magnitude up)
/// 2. If polarity opposes needed direction: weaken first (step magnitude down)
/// 3. Only flip polarity when magnitude is depleted to zero
fn apply_transition(weight: &mut PackedSignal, needed_direction: i32, transitions: &mut u64) {
    let current = weight.current();
    let current_sign = if current > 0 { 1 } else if current < 0 { -1 } else { 0 };

    if current_sign == needed_direction {
        // Polarity matches: strengthen (step up in magnitude)
        let stepped = step_up(current);
        if stepped != current {
            *weight = packed_from_current(stepped);
            *transitions += 1;
        }
    } else if current_sign == -needed_direction {
        // Polarity opposes: weaken first (step down toward zero)
        let stepped = step_down(current);
        *weight = packed_from_current(stepped);
        *transitions += 1;
        // Note: if stepped == 0, next transition will flip polarity
    } else {
        // Current is zero: set initial polarity in needed direction
        let initial = if needed_direction > 0 { 1 } else { -1 };
        *weight = packed_from_current(initial);
        *transitions += 1;
    }
}

/// Representable positive magnitudes in PackedSignal (sorted).
/// These are all products of LOG_LUT[mc] * LOG_LUT[uc] where
/// LOG_LUT = [0, 1, 4, 16, 32, 64, 128, 255].
const REPR_LEVELS: &[i32] = &[
    0, 1, 4, 16, 32, 64, 128, 255, 256, 512, 1020, 1024, 2048, 4080, 4096,
    8160, 8192, 16320, 16384, 32640, 32768, 65025,
];

/// Step the current value UP to the next representable PackedSignal magnitude.
fn step_up(current: i32) -> i32 {
    let abs = current.unsigned_abs() as i32;
    for &level in REPR_LEVELS {
        if level > abs {
            return if current >= 0 { level } else { -level };
        }
    }
    current // already at max
}

/// Step the current value DOWN to the previous representable PackedSignal magnitude.
fn step_down(current: i32) -> i32 {
    let abs = current.unsigned_abs() as i32;
    let mut prev = 0i32;
    for &level in REPR_LEVELS {
        if level >= abs {
            break;
        }
        prev = level;
    }
    if current > 0 {
        prev
    } else if current < 0 {
        -prev
    } else {
        0
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
            decay_rate: 0, // no decay for testing
            participation_gate: 0, // no gate for testing
        });

        let input = vec![PackedSignal::pack(1, 32, 1), PackedSignal::pack(1, 32, 1)];
        let output = vec![PackedSignal::ZERO];
        let target = vec![PackedSignal::pack(1, 64, 1)];

        // First step: activity-weighted pressure fires immediately with strong input,
        // causing transition. Either pressure accumulated or transition occurred.
        state.update(&mut weights, &input, &output, &target);
        assert_eq!(state.steps, 1);
        let has_pressure = state.pressure[0] > 0 || state.pressure[1] > 0;
        let has_transition = state.transitions > 0;
        assert!(has_pressure || has_transition,
            "Expected pressure accumulation or weight transition");
    }

    #[test]
    fn test_weight_transitions() {
        let mut weights = WeightMatrix::zeros(1, 1);
        let mut state = MasteryState::new(1, MasteryConfig {
            pressure_threshold: 1, // immediate transition
            decay_rate: 0,
            participation_gate: 0,
        });

        let input = vec![PackedSignal::pack(1, 64, 1)];
        let output = vec![PackedSignal::ZERO];
        let target = vec![PackedSignal::pack(1, 128, 1)];

        state.update(&mut weights, &input, &output, &target);
        // Weight should have transitioned from zero
        assert!(state.transitions > 0);
    }

    #[test]
    fn test_participation_gate() {
        let mut weights = WeightMatrix::zeros(1, 1);
        let mut state = MasteryState::new(1, MasteryConfig {
            pressure_threshold: 1,
            decay_rate: 0,
            participation_gate: 10, // high gate
        });

        let input = vec![PackedSignal::pack(1, 64, 1)];
        let output = vec![PackedSignal::ZERO];
        let target = vec![PackedSignal::pack(1, 128, 1)];

        // Won't transition until participation >= 10
        for _ in 0..9 {
            state.update(&mut weights, &input, &output, &target);
        }
        assert_eq!(state.transitions, 0);

        // 10th step should allow learning
        state.update(&mut weights, &input, &output, &target);
        assert!(state.transitions > 0);
    }
}
