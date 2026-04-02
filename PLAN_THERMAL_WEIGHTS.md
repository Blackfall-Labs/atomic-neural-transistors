# Thermal Weights — Per-Weight Plasticity Gating for ANTs

## Problem

Current ANT synaptic matrices have two modes: frozen or fully plastic. There is no middle ground.

- **Frozen hidden + learned output**: Converges fast (79% train in 1 cycle) but random projections don't generalize. Cross-speaker phoneme detection hits 3-11% test accuracy. The frozen hidden region can't discover input-invariant features because it never changes.

- **Both regions learning**: Hidden region thrashes — millions of transitions per cycle destroy train accuracy. Alternating freeze/unfreeze helps (test improves from 2% to 7%) but the unfreeze cycles are too violent. The output region has to completely relearn against the new hidden representation each time.

The root cause: every synaptic strength in a matrix has the same plasticity. There is no way for individual strengths that have proven useful across diverse inputs to resist change while strengths that are still searching remain plastic.

## Solution: Thermal Weights

Each synaptic strength carries its own thermal state. Temperature controls how much a strength changes per mastery step.

```
Per synaptic strength:
  polarity:    Polarity  — Excitatory (+1), Inhibitory (-1), Silent (0)
  magnitude:   u8        — base intensity
  multiplier:  u8        — contextual scaling
  temperature: u8        — HOT (255) → COLD (0)
  hits:        u16       — total reinforcement count
  pressure:    i16       — accumulated mastery pressure
```

8 bytes per synaptic strength. A 48×80 hidden region = 3,840 strengths = 30 KB.

### Temperature Controls Plasticity

| Temperature Range | State | Plasticity | Behavior |
|-------------------|-------|------------|----------|
| 192-255 | HOT | Full | Strength changes freely on every mastery step |
| 128-191 | WARM | Reduced | Pressure threshold doubled — needs stronger evidence |
| 64-127 | COOL | Low | Pressure threshold 4× — only strong consistent error moves it |
| 0-63 | COLD | Frozen | Strength locked. No mastery updates. |

### Hits Drive Cooling

Every time a synaptic strength participates in a correct detection, its hit count increments. Temperature drops as hits accumulate:

```
new_temperature = max(0, temperature - (hits / cooling_rate))
```

- `cooling_rate = 100`: a strength needs 100 hits to drop 1 temperature unit
- A strength hit consistently across 32 speakers × 50 exposures = 1,600 hits → temperature drops by 16
- After 25,500 hits (across many sessions) → fully COLD

Hits only count when the ANT's detection was CORRECT. Wrong detections don't cool strengths — they add pressure instead. A strength that participates in both correct and incorrect detections stays warm because hits and pressure pull in opposite directions.

### Warming (Demotion)

If a COOL or COLD strength consistently participates in wrong detections (pressure accumulates despite high threshold), it warms back up:

```
if accumulated_pressure > warming_threshold:
    temperature = min(255, temperature + warming_step)
    pressure = 0
```

This handles concept drift: a strength that was correct for mastery data but fails on new speakers heats back up and becomes plastic again.

## Three Initialization Strategies

### A) Grow From Nothing

All strengths start as zero/silent with temperature = HOT (255).

```
polarity = Zero, magnitude = 0, multiplier = 0, temperature = 255, hits = 0
```

Structure emerges entirely from input. Only strengths that receive consistent mastery pressure ever become non-zero. The ANT literally grows its own wiring from experience.

**Pros**: No bias from initialization. Maximally sparse — only useful strengths exist.
**Cons**: Slow to start. May never discover certain projections if pressure doesn't reach them.

### B) Random Pool

All strengths start with random polarity and low magnitude, temperature = HOT (255).

```
polarity = ±1 (random), magnitude = low, multiplier = 1, temperature = 255, hits = 0
```

Everything is plastic. Strengths that align with real signal patterns get reinforced and cool down. Strengths that don't stay noisy, fail to accumulate hits, and eventually decay or get overwritten by pressure.

**Pros**: Many starting directions — more likely to find useful projections early.
**Cons**: Initial noise means early detections are unreliable. More total transitions needed.

### C) Seeded With Structural Hints

Strengths initialized with domain-specific priors, but all at HOT temperature.

For audio/phoneme detection: hidden region strengths seeded with filterbank band groupings — adjacent frequency bands get correlated initial strengths, formant regions (300-3000Hz) get higher initial magnitude. The structure matches known spectral properties of speech.

```
polarity = domain_hint, magnitude = medium, multiplier = 1, temperature = 255, hits = 0
```

**Pros**: Fastest convergence — starts near useful projections, mastery fine-tunes.
**Cons**: Domain-specific. The hints might be wrong and need to be unlearned.
**Key**: Everything is still HOT. The hints are suggestions, not constraints. If the data disagrees with the seeding, the strengths will change. They just have a shorter path to the right answer if the hints are good.

## Integration With Existing ANT Architecture

### ThermalWeight

```rust
pub struct ThermalWeight {
    pub polarity: Polarity,     // ternary_signal::Polarity — enforced {-1, 0, +1}
    pub magnitude: u8,          // base intensity
    pub multiplier: u8,         // contextual scaling
    pub temperature: u8,        // 255=HOT → 0=COLD
    pub hits: u16,              // reinforcement count
    pub pressure: i16,          // accumulated mastery pressure
}
```

Uses `ternary_signal::Polarity` enum — not raw `i8`. Invalid states like `polarity = 2` are unrepresentable.

### Mastery Update Changes

`ThermalMasteryState::update()` operates on `Signal` inputs (not PackedSignal):

1. Compute pressure as before (error × input_sign × activity_strength)
2. Before applying transition: check strength temperature
3. Scale pressure threshold by temperature band:
   - HOT: threshold × 1 (normal)
   - WARM: threshold × 2
   - COOL: threshold × 4
   - COLD: skip entirely
4. After correct detection: increment hits on participating strengths
5. Temperature update: `temp = max(0, temp - hits/cooling_rate)`

### Runes Integration

The `load_synaptic` verb loads thermal synaptic matrices. Thermal state is invisible to the forward cascade — matmul uses signal values only. Mastery update respects thermal state automatically.

```rune
# Loads ThermalWeightMatrix from .ant file
w = load_synaptic("detector.w_hidden", 48, 80)

# Thermal state is invisible to forward cascade
# Mastery update respects thermal state automatically
```

No changes to .rune scripts. The thermal behavior is in the synaptic storage, not the computation.

## Test Plan

### Unit Tests
1. ThermalWeight creation, cooling, warming
2. ThermalWeightMatrix matmul produces same results regardless of thermal state (temperature doesn't affect cascade)
3. Mastery update respects temperature — COLD strengths don't change
4. Hit counting on correct detection
5. .ant file save/load round-trip

### Strategy Comparison (Synthetic Data)
1. 4-class classification (Hadamard patterns, same as existing classifier_mastery example)
2. Run all three init strategies (A, B, C) for 10 cycles
3. Compare: convergence speed, final accuracy, total transitions, sparsity

### Cross-Speaker Phoneme Detection (Audio)
1. Load LibriSpeech word spool (40 speakers, 54K words)
2. 39 phoneme detector ANTs with ThermalWeightMatrix
3. Curriculum: 10 → 25 → 50 → 100 words
4. Gate: 80% test accuracy on held-out speakers to advance
5. Track per-strength temperature distribution over cycles
6. Compare A/B/C strategies

### Persistence Lifecycle
1. Mastery session 1, save .ant files
2. Load session 2, verify strengths + temperatures + hits restored
3. Continue mastery, verify COLD strengths don't change
4. Verify accuracy maintained or improved across sessions

## Common Failure Modes

- Random hidden projections don't generalize across domains (speaker variation, etc.)
- Unfreezing both regions simultaneously: hidden thrashes, output can't keep up
- Too-low participation gate: learns from noise
- Too-high pressure threshold: never transitions
- Not decaying pressure per cycle: pressure accumulates across cycles, causing delayed spurious transitions

## Testing Best Practices

- Always track mastery AND evaluation accuracy per cycle — the gap tells you everything
- Flat mastery accuracy = converged. Flat evaluation accuracy = not generalizing.
- Evaluation accuracy climbing = the right features are being learned
- Mastery high / evaluation low = overfitting to mastery distribution
- Use curriculum: start small, prove it works, scale up
- Gate advancement on EVALUATION accuracy, not mastery
- Multiple sessions with persistence: verify accuracy carries over
