//! Salience Router — gate-based multi-ANT fusion with learned routing.
//!
//! Gates outputs from parallel ANTs based on learned salience and output
//! confidence. Untrained fallback: winner-take-all by output magnitude.
//!
//! From CROSS_ANT_ROUTING.md Option B: Gate-Based Fusion.

use crate::core::weight_matrix::WeightMatrix;
use crate::learning::{MasteryConfig, MasteryState};
use ternary_signal::Signal;

/// Routes and combines outputs from multiple parallel ANTs.
#[derive(Debug, Clone)]
pub struct SalienceRouter {
    /// Number of ANT sources.
    n_sources: usize,
    /// Output dimension per source.
    source_dim: usize,
    /// Gate synaptic strengths: n_sources × (n_sources * source_dim).
    /// Maps concatenated outputs → per-source gate value.
    gate_weights: WeightMatrix,
    /// Mastery state for gate synaptic strengths.
    gate_mastery: MasteryState,
}

/// Result of routing through the salience router.
#[derive(Debug, Clone)]
pub struct RouteResult {
    /// Combined output after gating.
    pub output: Vec<Signal>,
    /// Per-source confidence (output magnitude).
    pub confidences: Vec<i64>,
    /// Per-source gate values (0-255, from learned sigmoid gates).
    pub gate_values: Vec<u8>,
    /// Index of the highest-confidence source.
    pub winner: usize,
}

impl SalienceRouter {
    /// Create a new salience router.
    ///
    /// - `n_sources`: number of parallel ANTs
    /// - `source_dim`: output dimension per ANT
    pub fn new(n_sources: usize, source_dim: usize) -> Self {
        let concat_dim = n_sources * source_dim;
        Self {
            n_sources,
            source_dim,
            gate_weights: WeightMatrix::zeros(n_sources, concat_dim),
            gate_mastery: MasteryState::new(
                n_sources * concat_dim,
                MasteryConfig {
                    pressure_threshold: 3,
                    decay_rate: 1,
                    participation_gate: 3,
                },
            ),
        }
    }

    /// Number of sources this router handles.
    pub fn n_sources(&self) -> usize {
        self.n_sources
    }

    /// Output dimension per source.
    pub fn source_dim(&self) -> usize {
        self.source_dim
    }

    /// Route combined ANT outputs to produce a gated result.
    ///
    /// `outputs` is a flat slice of length `n_sources * source_dim`, containing
    /// all ANT outputs concatenated in order.
    pub fn route(&self, outputs: &[Signal]) -> RouteResult {
        assert_eq!(
            outputs.len(),
            self.n_sources * self.source_dim,
            "outputs length must equal n_sources × source_dim"
        );

        // Compute per-source confidence (output magnitude)
        let mut confidences = Vec::with_capacity(self.n_sources);
        for s in 0..self.n_sources {
            let start = s * self.source_dim;
            let end = start + self.source_dim;
            let mag: i64 = outputs[start..end]
                .iter()
                .map(|p| p.current().abs() as i64)
                .sum();
            confidences.push(mag);
        }

        // Winner = highest confidence
        let winner = confidences
            .iter()
            .enumerate()
            .max_by_key(|(_, &m)| m)
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Compute gate values via matmul through gate synaptic strengths
        let raw_gates = self.gate_weights.matmul(outputs);
        let gate_values: Vec<u8> = raw_gates
            .iter()
            .map(|g| integer_sigmoid(g.current()))
            .collect();

        // Check if gate synaptic strengths are all zero (untrained)
        let gates_trained = self
            .gate_weights
            .data
            .iter()
            .any(|w| w.current() != 0);

        // Produce gated output
        let mut output = vec![Signal::ZERO; self.source_dim];

        if gates_trained {
            // Gate-based fusion: scale each source by its gate value, sum
            for s in 0..self.n_sources {
                let gate = gate_values[s] as i64;
                let start = s * self.source_dim;
                for d in 0..self.source_dim {
                    let src = outputs[start + d].current() as i64;
                    // Scale by gate (0-255) then normalize by 255
                    let gated = src * gate / 255;
                    let existing = output[d].current() as i64;
                    output[d] = Signal::from_current((existing + gated).clamp(i32::MIN as i64, i32::MAX as i64) as i32);
                }
            }
        } else {
            // Untrained fallback: winner-take-all
            let start = winner * self.source_dim;
            output = outputs[start..start + self.source_dim].to_vec();
        }

        RouteResult {
            output,
            confidences,
            gate_values,
            winner,
        }
    }

    /// Train gate synaptic strengths via mastery learning.
    ///
    /// `outputs` is the concatenated ANT outputs (flat).
    /// `routed_output` is what the router produced (unused, reserved for future).
    /// `target` is the desired output (length = source_dim).
    pub fn train_route(
        &mut self,
        outputs: &[Signal],
        _routed_output: &[Signal],
        target: &[Signal],
    ) {
        assert_eq!(outputs.len(), self.n_sources * self.source_dim);

        // Compute per-source error relative to target:
        // Source closest to target gets high gate, others get low.
        let mut source_errors: Vec<i64> = Vec::with_capacity(self.n_sources);
        for s in 0..self.n_sources {
            let start = s * self.source_dim;
            let err: i64 = (0..self.source_dim.min(target.len()))
                .map(|d| {
                    let t = target[d].current() as i64;
                    let o = outputs[start + d].current() as i64;
                    (t - o).abs()
                })
                .sum();
            source_errors.push(err);
        }

        // Build gate targets: low error = high gate (127), high error = low gate (-127)
        let max_err = source_errors.iter().copied().max().unwrap_or(1).max(1);
        let gate_targets: Vec<Signal> = source_errors
            .iter()
            .map(|&err| {
                // Invert: low error = positive gate, high error = negative
                let normalized = 127 - (err * 254 / max_err) as i32;
                Signal::from_current(normalized)
            })
            .collect();

        // Current gate output (raw, before sigmoid)
        let raw_gates = self.gate_weights.matmul(outputs);

        self.gate_mastery.update(
            &mut self.gate_weights,
            outputs,
            &raw_gates,
            &gate_targets,
        );
    }

    /// Apply mastery decay to gate synaptic strengths.
    pub fn decay(&mut self) {
        self.gate_mastery.decay();
    }

    /// Access gate mastery state (for stats).
    pub fn gate_mastery(&self) -> &MasteryState {
        &self.gate_mastery
    }
}

/// Integer sigmoid: maps i32 → 0-255. Piecewise linear, centered at 0.
fn integer_sigmoid(current: i32) -> u8 {
    if current <= -512 {
        0
    } else if current >= 512 {
        255
    } else {
        ((current as i64 + 512) * 255 / 1024) as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ternary_signal::Signal;

    fn make_signals(values: &[i32]) -> Vec<Signal> {
        values.iter().map(|v| Signal::from_current(*v)).collect()
    }

    #[test]
    fn test_untrained_winner_take_all() {
        let router = SalienceRouter::new(3, 4);
        // Source 0: weak, Source 1: strong, Source 2: medium
        let outputs = make_signals(&[
            1, 1, 1, 1, // source 0
            100, 100, 100, 100, // source 1 (strongest)
            50, 50, 50, 50, // source 2
        ]);
        let result = router.route(&outputs);
        assert_eq!(result.winner, 1, "strongest source should win");
        // Untrained → winner-take-all output
        for d in 0..4 {
            assert_eq!(
                result.output[d].current(),
                outputs[4 + d].current(),
                "output should match winner's output"
            );
        }
    }

    #[test]
    fn test_confidence_computation() {
        let router = SalienceRouter::new(2, 3);
        let outputs = make_signals(&[
            10, 20, 30, // source 0: mag = 60ish
            -100, -100, -100, // source 1: mag = 300ish
        ]);
        let result = router.route(&outputs);
        assert!(
            result.confidences[1] > result.confidences[0],
            "source 1 should have higher confidence"
        );
    }

    #[test]
    fn test_integer_sigmoid() {
        assert_eq!(integer_sigmoid(-1000), 0);
        assert_eq!(integer_sigmoid(1000), 255);
        let mid = integer_sigmoid(0);
        assert!(mid > 120 && mid < 135, "midpoint should be ~127, got {}", mid);
    }

    #[test]
    fn test_router_dimensions() {
        let router = SalienceRouter::new(4, 8);
        assert_eq!(router.n_sources(), 4);
        assert_eq!(router.source_dim(), 8);
    }
}
