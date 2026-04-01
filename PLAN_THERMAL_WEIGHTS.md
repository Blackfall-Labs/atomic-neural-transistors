# Thermal Weights — Per-Weight Plasticity Gating for ANTs

## Problem

Current ANT weight matrices have two modes: frozen or fully plastic. There is no middle ground.

- **Frozen hidden + learned output**: Converges fast (79% train in 1 epoch) but random projections don't generalize. Cross-speaker phoneme detection hits 3-11% test accuracy. The frozen hidden layer can't discover input-invariant features because it never changes.

- **Both layers learning**: Hidden layer thrashes — millions of transitions per epoch destroy train accuracy. Alternating freeze/unfreeze helps (test improves from 2% to 7%) but the unfreeze epochs are too violent. The output layer has to completely relearn against the new hidden representation each time.

The root cause: every weight in a matrix has the same plasticity. There is no way for individual weights that have proven useful across diverse inputs to resist change while weights that are still searching remain plastic.

## Solution: Thermal Weights

Each weight carries its own thermal state. Temperature controls how much a weight changes per mastery step.

```
Per weight:
  polarity:    i8    — excitatory (+1), inhibitory (-1), silent (0)
  magnitude:   u8    — base intensity (LOG_LUT indexed)
  multiplier:  u8    — contextual scaling (LOG_LUT indexed)
  temperature: u8    — HOT (255) → COLD (0)
  hits:        u16   — total reinforcement count
```

6 bytes per weight. A 48×80 hidden layer = 3,840 weights = 23 KB.

### Temperature Controls Plasticity

| Temperature Range | State | Plasticity | Behavior |
|-------------------|-------|------------|----------|
| 192-255 | HOT | Full | Weight changes freely on every mastery step |
| 128-191 | WARM | Reduced | Pressure threshold doubled — needs stronger evidence |
| 64-127 | COOL | Low | Pressure threshold 4× — only strong consistent error moves it |
| 0-63 | COLD | Frozen | Weight locked. No mastery updates. |

### Hits Drive Cooling

Every time a weight participates in a correct detection, its hit count increments. Temperature drops as hits accumulate:

```
new_temperature = max(0, temperature - (hits / cooling_rate))
```

- `cooling_rate = 100`: a weight needs 100 hits to drop 1 temperature unit
- A weight hit consistently across 32 speakers × 50 exposures = 1,600 hits → temperature drops by 16
- After 25,500 hits (across many sessions) → fully COLD

Hits only count when the ANT's detection was CORRECT. Wrong detections don't cool weights — they add pressure instead. A weight that participates in both correct and incorrect detections stays warm because hits and pressure pull in opposite directions.

### Warming (Demotion)

If a COOL or COLD weight consistently participates in wrong detections (pressure accumulates despite high threshold), it warms back up:

```
if accumulated_pressure > warming_threshold:
    temperature = min(255, temperature + warming_step)
    pressure = 0
```

This handles concept drift: a weight that was correct for training data but fails on new speakers heats back up and becomes plastic again.

## .ant File Format

Binary format. No JSON. No serde overhead.

```
Header (16 bytes):
  magic:      [u8; 4]    — b"ANT\x01" (version 1)
  n_layers:   u16        — number of weight matrices
  n_weights:  u32        — total weight count across all layers
  flags:      u16        — reserved
  checksum:   u32        — CRC32 of weight data

Per layer header (8 bytes):
  rows:       u16
  cols:       u16
  reserved:   u32

Per weight (6 bytes):
  polarity:    i8
  magnitude:   u8
  multiplier:  u8
  temperature: u8
  hits:        u16 (little-endian)
```

A 2-layer detector (48×80 hidden + 1×48 output) = 3,888 weights = 23,328 bytes + 32 bytes headers = **~23 KB per ANT**.

39 phoneme ANTs = ~900 KB total. Loads in microseconds.

### Save/Load

```rust
// Save
ant.save("phoneme_h.ant")?;

// Load — weights, temperatures, hit counts all restored
let ant = PhonemeDetector::load("phoneme_h.ant")?;

// Inspect
for (i, w) in ant.weights().enumerate() {
    println!("{}: pol={} mag={} mul={} temp={} hits={}",
        i, w.polarity, w.magnitude, w.multiplier, w.temperature, w.hits);
}
```

## Three Initialization Strategies

### A) Grow From Nothing

All weights start as zero/silent with temperature = HOT (255).

```
polarity = 0, magnitude = 0, multiplier = 0, temperature = 255, hits = 0
```

Structure emerges entirely from input. Only weights that receive consistent mastery pressure ever become non-zero. The ANT literally grows its own wiring from experience.

**Pros**: No bias from initialization. Maximally sparse — only useful weights exist.
**Cons**: Slow to start. May never discover certain projections if pressure doesn't reach them.

### B) Random Pool

All weights start with random polarity and low magnitude, temperature = HOT (255).

```
polarity = ±1 (random), magnitude = LUT[1..3], multiplier = 1, temperature = 255, hits = 0
```

Everything is plastic. Weights that align with real signal patterns get reinforced and cool down. Weights that don't stay noisy, fail to accumulate hits, and eventually decay or get overwritten by pressure.

**Pros**: Many starting directions — more likely to find useful projections early.
**Cons**: Initial noise means early detections are unreliable. More total transitions needed.

### C) Seeded With Structural Hints

Weights initialized with domain-specific priors, but all at HOT temperature.

For audio/phoneme detection: hidden layer weights seeded with filterbank band groupings — adjacent frequency bands get correlated initial weights, formant regions (300-3000Hz) get higher initial magnitude. The structure matches known spectral properties of speech.

```
polarity = domain_hint, magnitude = LUT[2..4], multiplier = 1, temperature = 255, hits = 0
```

**Pros**: Fastest convergence — starts near useful projections, mastery fine-tunes.
**Cons**: Domain-specific. The hints might be wrong and need to be unlearned.
**Key**: Everything is still HOT. The hints are suggestions, not constraints. If the data disagrees with the seeding, the weights will change. They just have a shorter path to the right answer if the hints are good.

## Integration With Existing ANT Architecture

### WeightMatrix Upgrade

Current `WeightMatrix` stores `Vec<PackedSignal>` — 1 byte per weight, no temperature, no hits.

New `ThermalWeightMatrix` stores `Vec<ThermalWeight>` — 6 bytes per weight, full thermal state.

```rust
pub struct ThermalWeight {
    pub signal: PackedSignal,   // polarity × magnitude × multiplier (1 byte)
    pub temperature: u8,        // 255=HOT → 0=COLD
    pub hits: u16,              // reinforcement count
}

pub struct ThermalWeightMatrix {
    pub data: Vec<ThermalWeight>,
    pub rows: usize,
    pub cols: usize,
}
```

### Mastery Update Changes

`MasteryState::update()` currently treats all weights equally. With thermal weights:

1. Compute pressure as before (error × input_sign × activity_strength)
2. Before applying transition: check weight temperature
3. Scale pressure threshold by temperature band:
   - HOT: threshold × 1 (normal)
   - WARM: threshold × 2
   - COOL: threshold × 4
   - COLD: skip entirely
4. After correct detection: increment hits on participating weights
5. Temperature update: `temp = max(0, temp - hits/cooling_rate)`

### Runes Integration

The `load_synaptic` verb already loads weight matrices. Extend to support `.ant` files:

```rune
# Loads ThermalWeightMatrix from .ant file
w = load_synaptic("detector.w_hidden", 48, 80)

# Thermal state is invisible to forward pass — matmul uses signal values only
# Mastery update respects thermal state automatically
```

No changes to .rune scripts. The thermal behavior is in the weight storage, not the computation.

## Test Plan

### Unit Tests
1. ThermalWeight creation, cooling, warming
2. ThermalWeightMatrix matmul produces same results as WeightMatrix (thermal state doesn't affect forward pass)
3. Mastery update respects temperature — COLD weights don't change
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
5. Track per-weight temperature distribution over epochs
6. Compare A/B/C strategies

### Persistence Lifecycle
1. Train session 1, save .ant files
2. Load session 2, verify weights + temperatures + hits restored
3. Continue training, verify COLD weights don't change
4. Verify accuracy maintained or improved across sessions
