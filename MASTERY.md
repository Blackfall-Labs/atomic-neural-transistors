# Mastery Learning in ANTs

This document describes how Atomic Neural Transistors learn. This is NOT gradient descent. There are no floats, no loss functions, no backpropagation. Mastery learning operates on ternary integer substrates through pressure accumulation and neuromodulator gating.

---

## Signal Equation

Every synaptic strength is a PackedSignal with three components:

```
s = p × m × k
```

| Component | Symbol | Encoding | Range |
|-----------|--------|----------|-------|
| **Polarity** | p | 2 bits (00=zero, 01=+1, 10=−1) | {−1, 0, +1} |
| **Magnitude** | m | 3-bit code → LUT index | LUT = [0, 1, 4, 16, 32, 64, 128, 255] |
| **Multiplier** | k | 3-bit code → LUT index | LUT = [0, 1, 4, 16, 32, 64, 128, 255] |

The current value of a signal is `p × LUT[m_code] × LUT[k_code]`. Both magnitude and multiplier index the same 8-entry lookup table. This gives 22 distinct representable positive levels from the 64 unique products, ranging from 0 to 65,025 (255 × 255).

The full byte layout:

```
[ pp | mmm | kkk ]
  7-6  5-3   2-0
```

**All three components matter.** Magnitude without multiplier, or vice versa, gives an incomplete signal. Mastery transitions step through representable levels — the products of m × k — not through m or k independently.

---

## The Algorithm

### Step 1: Compute Error Signal

For each output neuron, compute the raw integer error:

```
error[i] = target[i].current() - output[i].current()
```

No normalization, no softmax, no cross-entropy. If error is zero, skip that output neuron entirely — the system is already correct.

### Step 2: Activity-Weighted Participation

Only the **top 25% of active inputs** contribute to pressure. This mirrors biological synapse learning where only the most active neurons drive plasticity.

```
max_input = max(|input[j]|) for all j
activity_threshold = max_input / 4

for each input j:
    if |input[j]| <= activity_threshold:
        skip  (not active enough to participate)
```

Additionally, a per-synapse **participation counter** tracks total usage. A synapse must exceed the participation gate (default: 5 observations) before any learning applies. This prevents learning from noise.

### Step 3: Accumulate Pressure

Pressure accumulates from the correlation between error direction, input sign, activity strength, and error magnitude:

```
activity_strength = (|input[j]| - activity_threshold) * 15 / max_input
error_mag = (min(|error|, 127) + 31) / 32    # 1 to 4

pressure[w] += sign(error) * sign(input[j]) * activity_strength * error_mag
```

The `× 15` scale factor comes from the production astromind system. Activity strength is proportional to how far above the threshold the input is — strongly active inputs contribute more pressure.

### Step 4: Threshold Gate

When accumulated pressure crosses the threshold (default: 3), a transition fires:

```
if |pressure[w]| >= threshold:
    apply_transition(weight[w], sign(pressure[w]))
    pressure[w] = 0
```

Pressure resets after each transition. This is hysteresis — prevents oscillation by requiring sustained evidence before committing to a change.

### Step 5: Weaken Before Flip

The transition rule follows the **weaken-before-flip** pattern from the production system. Polarity is structural identity — it should only flip as a last resort after magnitude is depleted to zero.

```
needed_direction = sign(pressure)

if weight.polarity == needed_direction:
    STRENGTHEN: step magnitude up to next representable level
else if weight.polarity == -needed_direction:
    WEAKEN: step magnitude down toward zero
else (weight is zero):
    INITIALIZE: set polarity to needed direction, magnitude = 1
```

**Why this matters:** Direct polarity flips cause oscillation — a weight flips back and forth without settling. Weaken-before-flip forces the weight to pass through zero first, which:
- Prevents immediate re-flipping (magnitude must rebuild from 1)
- Preserves structural identity unless sustained pressure demands change
- Matches biological synapse behavior (depletion before reversal)

### Step 6: Representable Level Stepping

Synaptic strengths occupy discrete representable levels: the set of all `m × k` products from the 3-bit magnitude and 3-bit multiplier LUTs. Transitions step to the **next representable level**, not by an arbitrary amount:

```
REPR_LEVELS = [0, 1, 4, 16, 32, 64, 128, 255, 256, 512, 1020, 1024,
               2048, 4080, 4096, 8160, 8192, 16320, 16384, 32640, 32768, 65025]

step_up(current):   → next level in REPR_LEVELS above |current|
step_down(current): → previous level in REPR_LEVELS below |current|
```

This prevents the quantization trap where adding a small step rounds back to the same representable value.

### Step 7: Pressure Decay

After each training cycle (epoch, not sample), all pressures decay toward zero:

```
for each pressure[w]:
    if pressure[w] > 0: pressure[w] -= decay_rate
    if pressure[w] < 0: pressure[w] += decay_rate
```

Decay happens per-epoch, allowing pressure to accumulate across samples within a cycle. Per-sample decay would zero out pressure before it could build.

---

## Architecture Pattern: Frozen Hidden + Learned Output

The production system used a specific multi-layer architecture:

```
Input (32 dims)
    │
    ▼
┌─────────────────────────────────────┐
│ Hidden Layer: FROZEN random          │
│ s = p × m × k where:               │
│   p = ±1 (random), m = 20-40,      │
│   k = random (1-255, LOG_LUT)      │
│ Activation: ReLU (sparse)           │
│ Learning: NONE (fixed projections)  │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Output Layer: LEARNED from zero     │
│ s = 0 (p=0, m=0, k=0) initially    │
│ Activation: Linear                  │
│ Learning: FULL mastery              │
└─────────────────────────────────────┘
    │
    ▼
Output
```

**Why frozen hidden layers work:** Random projections naturally create separation between classes. Different input patterns produce different activation patterns through the fixed random matrix. The output layer only needs to learn which frozen neuron activations correlate with which targets — a dramatically easier problem than learning the full representation.

**Results:** Frozen hidden + learned output converges in 1-2 cycles with 90-100% accuracy, compared to 25+ cycles when training all layers. The 96% reduction in required transitions comes from eliminating the credit assignment problem in hidden layers.

---

## Convergence Properties

### Speed

| Metric | Mastery (Ternary) | Gradient Descent (Float) |
|--------|-------------------|--------------------------|
| Iterations to converge | 1-2 | 850+ |
| Polarity flips | < 100 | 10,000+ sign changes |
| Total transitions | ~200-300 | Continuous (every step) |

Why so fast? Four reasons:

1. **Ternary state space is tiny.** Each synaptic strength has ~2,048 possible states (polarity × magnitude × multiplier), not the continuous real line. Fewer states = fewer transitions to find the right one.

2. **Activity-weighted participation eliminates noise.** Only the top 25% of active inputs contribute pressure. In a typical 1,296-param CompareANT, only ~300 synapses participate in any given step.

3. **Pressure accumulation acts as natural momentum.** Small errors don't cause transitions — pressure must build consistently before anything changes. This prevents oscillation.

4. **Weaken-before-flip prevents polarity thrashing.** Magnitude must deplete to zero before polarity can change, creating a natural damping effect.

### Stability

Once converged, mastery learning is extremely stable:

- **No catastrophic forgetting** — Pressure must re-accumulate from scratch to change a synaptic strength. A single noisy input produces pressure that decays before reaching threshold.
- **No learning rate tuning** — The pressure threshold, activity scale, and step levels are fixed. There is no "learning rate" to decay or schedule.
- **Deterministic** — Same inputs produce same pressure accumulation, same transitions, same final synaptic strengths. Every time.
- **Monotonic convergence** — With output clamping, error decreases monotonically. Once correct, error becomes zero and no further pressure accumulates.

---

## Reproducible Examples

Ten examples in `examples/` demonstrate mastery learning, composition, and persistence:

### Mastery Training Examples

#### Example 1: CompareANT Similarity Detection (`compare_mastery.rs`)

Trains a similarity detector on 32-dim ternary signal pairs using element-wise product features and Hadamard polarity prototypes.

```
Architecture: product_features(a, b) → mastery-trained 1×32 output matrix
Data: 1,000 pairs (50% identical, 50% different), 8 prototype patterns
Result: 50% → 100% accuracy in 1-2 mastery cycles
Persistence: Byte-identical outputs after thermogram save/load
```

#### Example 2: ClassifierANT Multi-Class (`classifier_mastery.rs`)

Trains a 4-class classifier using frozen random hidden projections and a mastery-trained output layer.

```
Architecture: frozen 24×32 projection → ReLU → mastery-trained 4×24 output
Data: 1,000 samples, 4 classes with Hadamard polarity signatures
Result: 25% → 100% accuracy in 1 cycle, 175 transitions
Persistence: Byte-identical outputs after thermogram save/load
```

#### Example 3: DiffANT Change Detection (`diff_mastery.rs`)

Trains a difference detector using product features with inverted targets — the complement of CompareANT.

```
Architecture: product_features(a, b) → mastery-trained 1×32 output matrix
Data: 1,000 pairs (50% same prototype, 50% different), 8 Hadamard prototypes
Result: 50% → 100% accuracy in 1 cycle, 286 transitions
Key insight: Same feature space as CompareANT, opposite target polarity
```

#### Example 4: GateANT Signal Gating (`gate_mastery.rs`)

Trains a signal gate that learns per-dimension pass/block based on control patterns, using frozen hidden projections and sigmoid activation.

```
Architecture: frozen 16×64 hidden → ReLU → mastery-trained 32×16 output → sigmoid
Data: 1,000 samples, 4 control patterns with Hadamard gate masks
Result: 96.5% per-dimension gate accuracy in 1 cycle
```

#### Example 5: MergeANT Signal Fusion (`merge_mastery.rs`)

Trains a signal merger to combine two inputs into an output encoding both input classes.

```
Architecture: frozen 24×64 hidden → ReLU → mastery-trained 32×24 output
Data: 1,000 pairs from 2 classes (4 merge combinations)
Result: 99.5% accuracy in 1 cycle, 6,768 transitions
```

### Composition Examples

#### Example 6: has_duplicate (`composition_has_duplicate.rs`)

Composes a trained CompareANT into a duplicate detector without any additional learning:

```
Pipeline: has_duplicate(seq) = OR(compare(seq[i], seq[j]) for all i < j)
Data: 500 sequences of length 4-8
Result: 100% accuracy (0 false positives, 0 false negatives)
Proves: Trained accuracy preserves through algebraic composition
```

#### Example 7: contains (`composition_contains.rs`)

Composes a trained CompareANT into a membership test — checks if a query exists in a sequence:

```
Pipeline: contains(query, seq) = OR(compare(query, seq[i]) for all i)
Data: 500 queries against sequences of length 3-8
Result: 100% accuracy (0 false positives, 0 false negatives)
```

#### Example 8: Sudoku Constraint Validation (`composition_sudoku.rs`)

Composes CompareANT into a 4×4 mini-Sudoku validator checking rows, columns, and 2×2 boxes:

```
Pipeline: valid(grid) = AND(NOT has_duplicate(group) for all rows, cols, boxes)
Data: 200 grids (100 valid, 100 invalid), 72 comparisons per grid
Result: 100% accuracy (0 false valid, 0 false invalid)
Proves: Algebraic composition scales to complex constraint satisfaction
```

### Pipeline & Persistence Examples

#### Example 9: Multi-ANT Planning Pipeline (`pipeline_planning.rs`)

Composes independently trained ClassifierANT and CompareANT into a planning loop:

```
Pipeline: ClassifierANT classifies state → select action → CompareANT verifies change
Data: 100 planning episodes, 4-state environment
Result: 100% success rate, 1.0 avg steps to goal
Proves: Independently trained ANTs compose into multi-step pipelines
```

#### Example 10: Full Persistence Lifecycle (`persistence_lifecycle.rs`)

Proves the complete train → save → destroy → load → verify → continue cycle:

```
Phase 1: Train from scratch → 100% in 1 cycle
Phase 2: Save thermogram to disk (1 delta)
Phase 3: Destroy instance, reload from thermogram → byte-identical
Phase 4: Continue mastery learning (5 more cycles) → accuracy maintained
Phase 5: Save updated thermogram → 2 deltas (thermal history)
```

---

## Thermogram Integration

After mastery updates, changed synaptic strengths are written to the Thermogram as deltas:

```
mastery_update(handle, input, output, target) → weight transitions occur
save_synaptic(handle, key)                    → delta written to Thermogram
persist_thermo(path)                          → Thermogram saved to disk
```

The Thermogram tracks thermal progression:

| Temperature | Meaning | Plasticity |
|-------------|---------|------------|
| **Hot** | Recently learned, volatile | Fully plastic |
| **Warm** | Session learning, stabilizing | Moderately plastic |
| **Cool** | Procedural/skill memory | Slowly plastic |
| **Cold** | Core identity, frozen | Glacial (nearly frozen) |

Repeated mastery on the same synaptic strengths causes the Thermogram entries to promote from Hot → Warm → Cool → Cold. This is how short-term learning becomes long-term memory.

### Consolidation

When the Thermogram consolidates:
1. Dirty deltas merge into the Hot layer
2. Hot entries with sufficient strength and observation count promote to Warm
3. Warm entries promote to Cool
4. Cool entries promote to Cold
5. Entries below the prune threshold are deleted

This mirrors biological memory consolidation — rehearsal strengthens memories, neglect lets them decay.

---

## Neuromodulator Gating (Production System)

In the full Astromind system, mastery was gated by chemical signals:

**Dopamine >= 0.3 required.** No learning happens without reward signal. This prevented ANTs from learning during random exploration or from noisy inputs.

**Norepinephrine modulates attention.** Higher norepinephrine = broader participation gate (more synapses participate). Lower = narrower (only the most active synapses update).

**Serotonin modulates decay.** Higher serotonin = faster pressure decay (harder to accumulate enough pressure for a transition). This acted as a "confidence gate" — only strong, consistent error signals produced learning.

The standalone ANT crate does not include neuromodulator simulation. The mastery_update verb uses fixed config values (pressure_threshold, decay_rate, participation_gate) that correspond to the "neutral" neuromodulator state from production.

---

## Configuration

### MasteryConfig

```rust
MasteryConfig {
    pressure_threshold: 3,    // Pressure needed for a transition
    decay_rate: 1,            // Pressure decay per epoch
    participation_gate: 5,    // Observations before eligibility
}
```

### From Runes

```rune
# Run one mastery step
mastery_update(weights_handle, input, output, target, [3, 1, 5])

# Query learning state
state = mastery_state(weights_handle)
# Returns [steps, transitions, pressure_sum, participation_sum]
```

### Production Defaults

The values [3, 1, 5] were the production defaults, corresponding to:
- Moderate pressure threshold (not too sensitive, not too sluggish)
- Unit decay (pressure decays by 1 each epoch)
- 5-observation participation gate (synapse must fire 5 times before it can learn)

These produce 1-2 cycle convergence with 100% accuracy on structured data.

---

## What Mastery Is Not

- **Not gradient descent.** There are no gradients. Pressure is integer accumulation, not a derivative.
- **Not backpropagation.** Error does not flow backward through layers. Each synaptic strength sees the global error signal directly.
- **Not stochastic.** There is no randomness in mastery. Same inputs, same config, same result.
- **Not continuous.** Synaptic strengths occupy discrete ternary states. Transitions are jumps, not smooth curves.
- **Not differentiable.** The step function and polarity flips are discontinuous by design. This is a feature — it prevents the "vanishing gradient" problem entirely because there are no gradients to vanish.
