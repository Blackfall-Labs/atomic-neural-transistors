//! Prediction Engine — integer-only EMA predictor with surprise detection.
//!
//! Tracks expected output via exponential moving average (using right-shift
//! instead of float multiplication). Computes surprise signal when actual
//! output deviates significantly from prediction.
//!
//! Surprise triggers learning moments in the multiplex encoder:
//! - Positive surprise → mastery update (reinforce what worked)
//! - Negative surprise → anti-pattern pressure (learn what NOT to do)
//!
//! From astromind-archive: prediction error drives ignition, not confidence.

use crate::PackedSignal;

/// Result of observing an actual output against prediction.
#[derive(Debug, Clone)]
pub struct SurpriseSignal {
    /// Total |actual - predicted| across all dimensions.
    pub magnitude: i64,
    /// Whether magnitude exceeds the surprise threshold.
    pub is_surprising: bool,
    /// +1 = positive surprise (moved toward target), -1 = negative (moved away).
    /// 0 = no target provided or during warmup.
    pub direction: i32,
    /// Per-dimension prediction error (actual - predicted).
    pub per_dim_error: Vec<i32>,
}

/// Integer-only EMA predictor with surprise detection.
#[derive(Debug, Clone)]
pub struct PredictionEngine {
    /// Running mean per dimension (i32 for headroom).
    ema: Vec<i32>,
    /// Right-shift for EMA smoothing. 3 = α≈0.125, 4 = α≈0.0625.
    ema_shift: u8,
    /// Observations before predictions are trusted.
    warmup: usize,
    /// Total observations so far.
    observations: usize,
    /// Per-dimension magnitude threshold for surprise.
    surprise_threshold: i32,
}

impl PredictionEngine {
    /// Create a new predictor.
    ///
    /// - `dims`: number of output dimensions to track
    /// - `ema_shift`: right-shift for smoothing (3 recommended)
    /// - `surprise_threshold`: per-dim threshold (40 recommended for ±128 signals)
    pub fn new(dims: usize, ema_shift: u8, surprise_threshold: i32) -> Self {
        Self {
            ema: vec![0; dims],
            ema_shift,
            warmup: 5,
            observations: 0,
            surprise_threshold,
        }
    }

    /// Current prediction (the EMA).
    pub fn predict(&self) -> &[i32] {
        &self.ema
    }

    /// Number of dimensions tracked.
    pub fn dims(&self) -> usize {
        self.ema.len()
    }

    /// Whether the predictor has seen enough data to trust predictions.
    pub fn is_warm(&self) -> bool {
        self.observations >= self.warmup
    }

    /// Observe actual output. Updates EMA and returns surprise signal.
    ///
    /// If `target` is provided, direction indicates whether output moved
    /// toward (+1) or away from (-1) the target relative to prediction.
    pub fn observe(
        &mut self,
        actual: &[PackedSignal],
        target: Option<&[PackedSignal]>,
    ) -> SurpriseSignal {
        let dims = self.ema.len().min(actual.len());
        let mut per_dim_error = Vec::with_capacity(dims);
        let mut magnitude: i64 = 0;

        for j in 0..dims {
            let actual_val = actual[j].current();
            let error = actual_val - self.ema[j];
            per_dim_error.push(error);
            magnitude += error.abs() as i64;

            // EMA update: ema += (actual - ema) >> shift
            self.ema[j] += error >> self.ema_shift;
        }

        self.observations += 1;

        let is_surprising = self.is_warm()
            && magnitude > (self.surprise_threshold as i64 * dims as i64);

        // Compute direction if target is available
        let direction = if let Some(tgt) = target {
            if !self.is_warm() {
                0
            } else {
                // Positive = output is closer to target than prediction was
                // Negative = output is farther from target than prediction was
                let mut target_distance_from_pred: i64 = 0;
                let mut target_distance_from_actual: i64 = 0;
                for j in 0..dims.min(tgt.len()) {
                    let t = tgt[j].current();
                    target_distance_from_pred += (t - self.ema[j]).abs() as i64;
                    target_distance_from_actual += (t - actual[j].current()).abs() as i64;
                }
                if target_distance_from_actual < target_distance_from_pred {
                    1 // moved toward target — positive surprise
                } else if target_distance_from_actual > target_distance_from_pred {
                    -1 // moved away from target — negative surprise
                } else {
                    0
                }
            }
        } else {
            0
        };

        SurpriseSignal {
            magnitude,
            is_surprising,
            direction,
            per_dim_error,
        }
    }

    /// Reset the predictor state.
    pub fn reset(&mut self) {
        self.ema.fill(0);
        self.observations = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::weight_matrix::packed_from_current;

    fn make_signals(values: &[i32]) -> Vec<PackedSignal> {
        values.iter().map(|v| packed_from_current(*v)).collect()
    }

    #[test]
    fn test_warmup_not_surprising() {
        let mut pred = PredictionEngine::new(4, 3, 40);
        let sig = make_signals(&[100, 100, 100, 100]);
        for _ in 0..4 {
            let surprise = pred.observe(&sig, None);
            assert!(!surprise.is_surprising, "should not be surprising during warmup");
        }
        assert!(!pred.is_warm());
    }

    #[test]
    fn test_surprise_after_warmup() {
        let mut pred = PredictionEngine::new(4, 3, 40);
        let stable = make_signals(&[100, 100, 100, 100]);
        // Warm up with stable signal
        for _ in 0..10 {
            pred.observe(&stable, None);
        }
        assert!(pred.is_warm());

        // Now send a very different signal
        let shock = make_signals(&[-100, -100, -100, -100]);
        let surprise = pred.observe(&shock, None);
        assert!(surprise.is_surprising);
        assert!(surprise.magnitude > 0);
    }

    #[test]
    fn test_stable_not_surprising() {
        let mut pred = PredictionEngine::new(4, 3, 40);
        let stable = make_signals(&[100, 100, 100, 100]);
        for _ in 0..20 {
            let surprise = pred.observe(&stable, None);
            if pred.observations > 10 {
                assert!(!surprise.is_surprising, "stable signal should not be surprising");
            }
        }
    }

    #[test]
    fn test_positive_direction() {
        let mut pred = PredictionEngine::new(4, 3, 40);
        let bad = make_signals(&[-100, -100, -100, -100]);
        let target = make_signals(&[100, 100, 100, 100]);
        // Train on bad signal
        for _ in 0..10 {
            pred.observe(&bad, Some(&target));
        }
        // Now output closer to target
        let good = make_signals(&[80, 80, 80, 80]);
        let surprise = pred.observe(&good, Some(&target));
        assert_eq!(surprise.direction, 1, "moving toward target = positive");
    }

    #[test]
    fn test_negative_direction() {
        let mut pred = PredictionEngine::new(4, 3, 40);
        let ok = make_signals(&[50, 50, 50, 50]);
        let target = make_signals(&[100, 100, 100, 100]);
        for _ in 0..10 {
            pred.observe(&ok, Some(&target));
        }
        // Now output farther from target
        let worse = make_signals(&[-50, -50, -50, -50]);
        let surprise = pred.observe(&worse, Some(&target));
        assert_eq!(surprise.direction, -1, "moving away from target = negative");
    }

    #[test]
    fn test_ema_converges() {
        let mut pred = PredictionEngine::new(1, 3, 40);
        let sig = make_signals(&[100]);
        for _ in 0..50 {
            pred.observe(&sig, None);
        }
        // EMA should be close to 100 after many observations
        // PackedSignal quantization means 100 encodes as a nearby representable value
        assert!((pred.predict()[0] - 100).abs() < 30, "EMA should converge near 100, got {}", pred.predict()[0]);
    }
}
