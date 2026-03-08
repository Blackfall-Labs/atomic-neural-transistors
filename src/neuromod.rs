//! Neuromodulator State — integer-only chemical gating for plasticity.
//!
//! Three chemicals gate mastery learning:
//! - **Dopamine**: Reward signal. Plasticity only fires when DA > gate threshold.
//! - **Norepinephrine**: Arousal. Controls participation breadth (how many synapses learn).
//! - **Serotonin**: Stability. Controls pressure decay rate (higher = harder to learn).
//!
//! All values are u8 (0-255) with baseline at 128. Chemicals decay toward
//! baseline each tick. Signed injection allows both excitation and inhibition.
//!
//! From astromind-archive production: DA ↔ 5HT antagonism prevents saturation.

/// Which chemical to inject.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Chemical {
    Dopamine,
    Norepinephrine,
    Serotonin,
}

/// Integer-only neuromodulator state that gates plasticity.
#[derive(Debug, Clone)]
pub struct NeuromodState {
    pub dopamine: u8,
    pub norepinephrine: u8,
    pub serotonin: u8,
    /// DA must exceed this for plasticity to fire (~0.3 = 77/255).
    pub dopamine_gate: u8,
    /// Baseline all chemicals decay toward.
    pub baseline: u8,
}

impl NeuromodState {
    /// Create neutral state — all chemicals at baseline.
    pub fn new() -> Self {
        Self {
            dopamine: 128,
            norepinephrine: 128,
            serotonin: 128,
            dopamine_gate: 77,
            baseline: 128,
        }
    }

    /// Create with custom DA gate threshold.
    pub fn with_gate(dopamine_gate: u8) -> Self {
        Self { dopamine_gate, ..Self::new() }
    }

    /// Is the dopamine gate open? Plasticity only fires when true.
    pub fn plasticity_open(&self) -> bool {
        self.dopamine > self.dopamine_gate
    }

    /// Participation breadth divisor based on norepinephrine.
    ///
    /// Default mastery uses top-25% (divisor = 4).
    /// High NE (255) → divisor 2 (top-50%, broader participation).
    /// Low NE (0) → divisor 8 (top-12%, narrow participation).
    /// Neutral NE (128) → divisor 4 (default).
    pub fn participation_divisor(&self) -> u32 {
        // Map NE 0-255 → divisor 8-2 (inverted: more NE = smaller divisor = broader)
        // divisor = 8 - (NE * 6 / 255) = 8 - NE/42
        let shift = (self.norepinephrine as u32 * 6) / 255;
        (8 - shift).max(2)
    }

    /// Pressure decay multiplier based on serotonin.
    ///
    /// Default decay_rate is 1 per cycle.
    /// High 5HT (255) → multiply by 2 (faster decay, harder to learn).
    /// Low 5HT (0) → multiply by 0 (no decay, pressure accumulates freely).
    /// Neutral 5HT (128) → multiply by 1 (default behavior).
    pub fn decay_multiplier(&self) -> i32 {
        // Map 5HT 0-255 → multiplier 0-2
        (self.serotonin as i32 * 2) / 255
    }

    /// Inject a signed chemical delta. Positive = excite, negative = inhibit.
    /// Result clamped to 0-255.
    pub fn inject(&mut self, chemical: Chemical, amount: i8) {
        let target = match chemical {
            Chemical::Dopamine => &mut self.dopamine,
            Chemical::Norepinephrine => &mut self.norepinephrine,
            Chemical::Serotonin => &mut self.serotonin,
        };
        let new_val = (*target as i16) + (amount as i16);
        *target = new_val.clamp(0, 255) as u8;

        // Antagonism: DA ↔ 5HT (from astromind Trial 8)
        // High DA suppresses 5HT, high 5HT suppresses DA
        match chemical {
            Chemical::Dopamine if amount > 0 => {
                let suppress = (amount as i16 / 3).min(self.serotonin as i16) as u8;
                self.serotonin = self.serotonin.saturating_sub(suppress);
            }
            Chemical::Serotonin if amount > 0 => {
                let suppress = (amount as i16 / 3).min(self.dopamine as i16) as u8;
                self.dopamine = self.dopamine.saturating_sub(suppress);
            }
            _ => {}
        }
    }

    /// Decay all chemicals toward baseline by 1 per tick.
    pub fn tick(&mut self) {
        self.dopamine = decay_toward(self.dopamine, self.baseline);
        self.norepinephrine = decay_toward(self.norepinephrine, self.baseline);
        self.serotonin = decay_toward(self.serotonin, self.baseline);
    }
}

impl Default for NeuromodState {
    fn default() -> Self {
        Self::new()
    }
}

fn decay_toward(current: u8, baseline: u8) -> u8 {
    if current > baseline {
        current - 1
    } else if current < baseline {
        current + 1
    } else {
        current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neutral_state() {
        let nm = NeuromodState::new();
        assert!(nm.plasticity_open()); // 128 > 77
        assert_eq!(nm.participation_divisor(), 5); // 8 - 128*6/255 ≈ 8 - 3 = 5
        assert_eq!(nm.decay_multiplier(), 1); // 128*2/255 = 1
    }

    #[test]
    fn test_da_gate() {
        let mut nm = NeuromodState::new();
        assert!(nm.plasticity_open());
        nm.dopamine = 50; // below gate
        assert!(!nm.plasticity_open());
        nm.dopamine = 78; // above gate
        assert!(nm.plasticity_open());
    }

    #[test]
    fn test_injection_clamps() {
        let mut nm = NeuromodState::new();
        nm.dopamine = 250;
        nm.inject(Chemical::Dopamine, 100); // would overflow
        assert_eq!(nm.dopamine, 255); // clamped
        nm.dopamine = 5;
        nm.inject(Chemical::Dopamine, -100); // would underflow
        assert_eq!(nm.dopamine, 0); // clamped
    }

    #[test]
    fn test_antagonism() {
        let mut nm = NeuromodState::new();
        let initial_5ht = nm.serotonin;
        nm.inject(Chemical::Dopamine, 30);
        assert!(nm.serotonin < initial_5ht, "DA injection should suppress 5HT");
    }

    #[test]
    fn test_tick_decay() {
        let mut nm = NeuromodState::new();
        nm.dopamine = 200;
        nm.serotonin = 50;
        nm.tick();
        assert_eq!(nm.dopamine, 199); // decay toward 128
        assert_eq!(nm.serotonin, 51); // decay toward 128
        nm.norepinephrine = 128;
        nm.tick();
        assert_eq!(nm.norepinephrine, 128); // already at baseline
    }

    #[test]
    fn test_ne_participation() {
        let mut nm = NeuromodState::new();
        nm.norepinephrine = 255; // max arousal
        assert_eq!(nm.participation_divisor(), 2); // broadest
        nm.norepinephrine = 0; // min arousal
        assert_eq!(nm.participation_divisor(), 8); // narrowest
    }

    #[test]
    fn test_5ht_decay_modifier() {
        let mut nm = NeuromodState::new();
        nm.serotonin = 255; // max stability
        assert_eq!(nm.decay_multiplier(), 2); // fastest decay
        nm.serotonin = 0; // min stability
        assert_eq!(nm.decay_multiplier(), 0); // no decay
    }
}
