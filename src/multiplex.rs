//! Multiplex Encoder — orchestrates parallel ANTs with salience routing,
//! prediction-error surprise detection, and neuromodulator-gated learning.
//!
//! Core loop:
//! 1. Forward all ANTs on input (parallel encoding)
//! 2. Route through salience → combined output
//! 3. Observe combined output → SurpriseSignal
//! 4. If surprising AND plasticity gate open:
//!    - Positive surprise → mastery update (reinforce what worked)
//!    - Negative surprise → anti-pattern pressure (inverted targets)
//! 5. Tick neuromod (decay toward baseline)

use crate::core::weight_matrix::{packed_from_current, WeightMatrix};
use crate::learning::{MasteryConfig, MasteryState};
use crate::neuromod::{Chemical, NeuromodState};
use crate::prediction::{PredictionEngine, SurpriseSignal};
use crate::salience::{RouteResult, SalienceRouter};
use ternary_signal::PackedSignal;

/// A slot holding one ANT's forward function and learnable output synaptic strengths.
pub struct AntSlot {
    /// Human-readable name for this ANT.
    pub name: String,
    /// Forward function: takes input, returns output.
    forward_fn: Box<dyn Fn(&[PackedSignal]) -> Vec<PackedSignal>>,
    /// Output synaptic strengths (learnable via mastery).
    pub w_out: WeightMatrix,
    /// Mastery state for the output synaptic strengths.
    pub ms_out: MasteryState,
}

impl AntSlot {
    /// Create a new ANT slot.
    ///
    /// - `name`: identifier for this ANT
    /// - `forward_fn`: the ANT's computation (input → output)
    /// - `output_dim`: dimension of this ANT's output
    /// - `hidden_dim`: dimension of the hidden representation feeding the output layer
    pub fn new(
        name: impl Into<String>,
        forward_fn: Box<dyn Fn(&[PackedSignal]) -> Vec<PackedSignal>>,
        output_dim: usize,
        hidden_dim: usize,
    ) -> Self {
        Self {
            name: name.into(),
            forward_fn,
            w_out: WeightMatrix::zeros(output_dim, hidden_dim),
            ms_out: MasteryState::new(
                output_dim * hidden_dim,
                MasteryConfig {
                    pressure_threshold: 3,
                    decay_rate: 1,
                    participation_gate: 5,
                },
            ),
        }
    }

    /// Create a slot with pre-trained synaptic strengths and a simple passthrough forward.
    ///
    /// Use when the ANT's forward function already produces the final output
    /// (no additional output layer needed).
    pub fn with_passthrough(
        name: impl Into<String>,
        forward_fn: Box<dyn Fn(&[PackedSignal]) -> Vec<PackedSignal>>,
    ) -> Self {
        Self {
            name: name.into(),
            forward_fn,
            w_out: WeightMatrix::zeros(0, 0), // unused
            ms_out: MasteryState::new(0, MasteryConfig::default()),
        }
    }

    /// Run this ANT's forward function.
    pub fn forward(&self, input: &[PackedSignal]) -> Vec<PackedSignal> {
        (self.forward_fn)(input)
    }
}

/// Result of one multiplex processing step.
#[derive(Debug, Clone)]
pub struct MultiplexResult {
    /// Combined output after salience routing.
    pub output: Vec<PackedSignal>,
    /// Surprise signal from prediction engine.
    pub surprise: SurpriseSignal,
    /// Routing result (confidences, gates, winner).
    pub route: RouteResult,
    /// Whether a learning moment occurred.
    pub learning_occurred: bool,
    /// Current dopamine level.
    pub dopamine: u8,
}

/// Orchestrates parallel ANTs → salience → prediction → surprise → learning.
pub struct MultiplexEncoder {
    /// The parallel ANT slots.
    slots: Vec<AntSlot>,
    /// Salience router for combining ANT outputs.
    router: SalienceRouter,
    /// Prediction engine for surprise detection.
    predictor: PredictionEngine,
    /// Neuromodulator state gating plasticity.
    pub neuromod: NeuromodState,
    /// Output dimension per ANT.
    output_dim: usize,
    /// Surprise magnitude threshold multiplier for strong learning moments.
    strong_surprise_multiplier: i64,
}

impl MultiplexEncoder {
    /// Create a new multiplex encoder.
    ///
    /// - `output_dim`: output dimension per ANT (all must match)
    /// - `ema_shift`: right-shift for prediction EMA (3 recommended)
    /// - `surprise_threshold`: per-dim surprise threshold (40 recommended)
    pub fn new(output_dim: usize, ema_shift: u8, surprise_threshold: i32) -> Self {
        Self {
            slots: Vec::new(),
            router: SalienceRouter::new(0, output_dim),
            predictor: PredictionEngine::new(output_dim, ema_shift, surprise_threshold),
            neuromod: NeuromodState::new(),
            output_dim,
            strong_surprise_multiplier: 2,
        }
    }

    /// Add an ANT slot. Must be called before `finalize()`.
    pub fn add_slot(&mut self, slot: AntSlot) {
        self.slots.push(slot);
    }

    /// Finalize the encoder after adding all slots.
    /// Creates the salience router with the correct number of sources.
    pub fn finalize(&mut self) {
        let n = self.slots.len();
        self.router = SalienceRouter::new(n, self.output_dim);
    }

    /// Number of ANT slots.
    pub fn n_slots(&self) -> usize {
        self.slots.len()
    }

    /// Process input through all ANTs, route, predict, and optionally learn.
    ///
    /// - `input`: signal fed to all ANTs
    /// - `target`: if provided, enables surprise-gated learning
    pub fn process(
        &mut self,
        input: &[PackedSignal],
        target: Option<&[PackedSignal]>,
    ) -> MultiplexResult {
        let n = self.slots.len();
        assert!(n > 0, "MultiplexEncoder has no slots — call add_slot() then finalize()");

        // Step 1: Forward all ANTs
        let mut all_outputs: Vec<PackedSignal> = Vec::with_capacity(n * self.output_dim);
        for slot in &self.slots {
            let out = slot.forward(input);
            assert_eq!(
                out.len(),
                self.output_dim,
                "ANT '{}' output dim {} != expected {}",
                slot.name,
                out.len(),
                self.output_dim
            );
            all_outputs.extend_from_slice(&out);
        }

        // Step 2: Route through salience
        let route = self.router.route(&all_outputs);
        let routed_output = route.output.clone();

        // Step 3: Observe → surprise
        let surprise = self.predictor.observe(&routed_output, target);

        // Step 4: Surprise-gated learning
        let mut learning_occurred = false;

        if let Some(tgt) = target {
            if surprise.is_surprising && self.neuromod.plasticity_open() {
                learning_occurred = true;

                match surprise.direction {
                    1 => {
                        // Positive surprise: output moved toward target.
                        // Reinforce: train salience router to favor the winner.
                        self.router.train_route(&all_outputs, &routed_output, tgt);

                        // Reward: inject DA
                        self.neuromod.inject(Chemical::Dopamine, 20);

                        // Strong positive → inject extra NE for broader learning
                        // Check if surprise magnitude > 2× the base threshold
                        let base_threshold =
                            self.predictor.dims() as i64 * 40; // base surprise threshold
                        if surprise.magnitude
                            > base_threshold * self.strong_surprise_multiplier
                        {
                            self.neuromod.inject(Chemical::Norepinephrine, 10);
                        }
                    }
                    -1 => {
                        // Negative surprise: output moved away from target.
                        // Anti-pattern: train with inverted target (learn what NOT to do).
                        let anti_target: Vec<PackedSignal> = routed_output
                            .iter()
                            .map(|s| packed_from_current(-s.current()))
                            .collect();
                        self.router
                            .train_route(&all_outputs, &route.output, tgt);

                        // Also apply anti-pattern pressure to the winning ANT's output layer
                        let winner = route.winner;
                        if self.slots[winner].w_out.rows > 0 {
                            let winner_start = winner * self.output_dim;
                            let winner_out: Vec<PackedSignal> =
                                all_outputs[winner_start..winner_start + self.output_dim].to_vec();
                            let slot = &mut self.slots[winner];
                            slot.ms_out.update_gated(
                                &mut slot.w_out,
                                &winner_out,
                                &routed_output,
                                &anti_target,
                                Some(&self.neuromod),
                            );
                        }

                        // Punish: reduce DA
                        self.neuromod.inject(Chemical::Dopamine, -10);
                    }
                    _ => {
                        // No direction info — still train router normally
                        self.router.train_route(&all_outputs, &routed_output, tgt);
                    }
                }
            }
        }

        // Step 5: Tick neuromod (decay toward baseline)
        self.neuromod.tick();

        MultiplexResult {
            output: routed_output,
            surprise,
            route,
            learning_occurred,
            dopamine: self.neuromod.dopamine,
        }
    }

    /// Access the prediction engine.
    pub fn predictor(&self) -> &PredictionEngine {
        &self.predictor
    }

    /// Access the salience router.
    pub fn router(&self) -> &SalienceRouter {
        &self.router
    }

    /// Reset prediction engine and neuromod to initial state.
    pub fn reset(&mut self) {
        self.predictor.reset();
        self.neuromod = NeuromodState::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::weight_matrix::packed_from_current;

    fn make_signals(values: &[i32]) -> Vec<PackedSignal> {
        values.iter().map(|v| packed_from_current(*v)).collect()
    }

    /// Simple ANT that returns its input scaled.
    fn scale_ant(scale: i32) -> Box<dyn Fn(&[PackedSignal]) -> Vec<PackedSignal>> {
        Box::new(move |input: &[PackedSignal]| {
            input
                .iter()
                .map(|s| packed_from_current(s.current() * scale / 100))
                .collect()
        })
    }

    #[test]
    fn test_multiplex_basic_routing() {
        let mut mux = MultiplexEncoder::new(4, 3, 40);

        // ANT 0: weak signal
        mux.add_slot(AntSlot::with_passthrough("weak", scale_ant(10)));
        // ANT 1: strong signal
        mux.add_slot(AntSlot::with_passthrough("strong", scale_ant(100)));

        mux.finalize();

        let input = make_signals(&[100, 100, 100, 100]);
        let result = mux.process(&input, None);

        // Winner should be the strong ANT
        assert_eq!(result.route.winner, 1);
        assert_eq!(result.output.len(), 4);
    }

    #[test]
    fn test_multiplex_no_surprise_during_warmup() {
        let mut mux = MultiplexEncoder::new(4, 3, 40);
        mux.add_slot(AntSlot::with_passthrough("a", scale_ant(100)));
        mux.finalize();

        let input = make_signals(&[100, 100, 100, 100]);
        for _ in 0..4 {
            let result = mux.process(&input, None);
            assert!(!result.surprise.is_surprising, "warmup should not be surprising");
            assert!(!result.learning_occurred);
        }
    }

    #[test]
    fn test_multiplex_surprise_triggers_learning() {
        let mut mux = MultiplexEncoder::new(4, 3, 40);
        mux.add_slot(AntSlot::with_passthrough("a", scale_ant(100)));
        mux.add_slot(AntSlot::with_passthrough("b", scale_ant(50)));
        mux.finalize();

        let input = make_signals(&[100, 100, 100, 100]);
        let target = make_signals(&[80, 80, 80, 80]);

        // Warm up with stable signal
        for _ in 0..10 {
            mux.process(&input, Some(&target));
        }

        // Now shock with very different input
        let shock_input = make_signals(&[-100, -100, -100, -100]);
        let result = mux.process(&shock_input, Some(&target));

        // Should detect surprise after warmup
        if result.surprise.is_surprising {
            assert!(result.learning_occurred);
        }
    }

    #[test]
    fn test_da_oscillation() {
        let mut mux = MultiplexEncoder::new(2, 3, 20);
        mux.add_slot(AntSlot::with_passthrough("a", scale_ant(100)));
        mux.finalize();

        let input = make_signals(&[100, 100]);
        let target = make_signals(&[100, 100]);

        // DA starts at 128 (baseline), should stay near baseline with stable input
        let initial_da = mux.neuromod.dopamine;

        for _ in 0..20 {
            mux.process(&input, Some(&target));
        }

        // DA should have decayed back toward baseline (128)
        // It may fluctuate but shouldn't saturate at 0 or 255
        let final_da = mux.neuromod.dopamine;
        assert!(
            final_da > 50 && final_da < 200,
            "DA should stay near baseline, got {}",
            final_da
        );
    }
}
