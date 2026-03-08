# Atomic Neural Transistors (ANTs)

**Ultra-small (<5K param) composable ternary neural primitives for CPU-only real-time AI**

[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org/)

---

## What are ANTs?

ANTs are the **transistors of neural computing** — atomic units that perform a single operation with high precision and compose into larger systems without retraining.

Just as silicon transistors:

- Do one thing (switch on/off)
- Compose into gates, then circuits, then CPUs
- Are fast and predictable

ANTs:

- Do one thing (compare, classify, diff, gate, merge)
- Compose into pipelines, constraint checkers, and cognitive systems
- Run in **microseconds** with **deterministic**, **integer-only** computation

**No floats. No gradients. No GPUs.** ANTs operate on ternary signals (polarity x magnitude) using integer arithmetic throughout. They learn via [mastery learning](MASTERY.md) — a pressure-based discrete plasticity system, not gradient descent.

---

## Specialized ANTs

| ANT | Purpose | Params | Forward Latency |
|-----|---------|--------|-----------------|
| `CompareANT` | Binary similarity detection | 1,296 | 64 µs |
| `DiffANT` | Change detection | 2,304 | 49 µs |
| `MergeANT` | Signal fusion | 2,304 | 49 µs |
| `GateANT` | Signal gating (sigmoid) | 1,536 | 80 µs |
| `ClassifierANT` | Multi-class classification | 2,016 | 194 µs |

Total across all 5 ANTs: **9,456 parameters**. Each parameter is a `PackedSignal` (1 byte).

---

## Multiplex Encoding

Multiple ANTs process the same input in parallel, with learned salience routing, prediction-error surprise detection, and neuromodulator-gated learning.

```
input → [ANT_α, ANT_β, ANT_γ] → Salience Router → output
                                        ↓
                                 Prediction Engine
                                   (EMA tracker)
                                        ↓
                                surprise > threshold?
                                   ↓           ↓
                                  YES          NO
                              Learning       (skip)
                              Moment
                           ↙         ↘
                     positive      negative
                     (reinforce)   (anti-pattern)
                     DA +20        DA -10
```

| Component | Latency | Purpose |
|-----------|---------|---------|
| `MultiplexEncoder` (3 ANTs) | 5.0 µs | Full process cycle |
| `SalienceRouter` (3×32) | 622 ns | Gate-based fusion |
| `PredictionEngine` (32-dim) | 85 ns | EMA + surprise detection |
| `NeuromodState` (full cycle) | 10 ns | DA/NE/5HT gating |

### Neuromodulator Gating

Three chemicals gate plasticity (integer-only, 0-255 each):

| Chemical | Role | Effect on Learning |
|----------|------|--------------------|
| **Dopamine** | Reward signal | DA > gate (77) required for any learning |
| **Norepinephrine** | Arousal | Controls participation breadth (narrow ↔ broad) |
| **Serotonin** | Stability | Controls pressure decay rate (faster ↔ slower) |

DA ↔ 5HT antagonism prevents saturation. All chemicals decay toward baseline (128) each tick.

---

## Mastery Learning

ANTs learn via **mastery learning** — an integer-only, pressure-based plasticity system. No gradients, no backpropagation, no loss functions.

| Metric | Mastery (Ternary) | Gradient Descent (Float) |
|--------|-------------------|--------------------------|
| Cycles to converge | 1-2 | 850+ |
| Polarity flips | < 100 | 10,000+ sign changes |
| Total transitions | ~200-300 | Continuous (every step) |
| Substrate | Integer | Float |
| Deterministic | Yes | No (float rounding) |

Key mechanisms:
- **Activity-weighted pressure**: Only the top 25% of active inputs contribute
- **Weaken-before-flip**: Magnitude depletes to zero before polarity can change
- **Representable level stepping**: Transitions jump to the next discrete level, not by arbitrary amounts
- **Pressure threshold gating**: Sustained evidence required before any synaptic change
- **Neuromodulator gating**: Optional DA/NE/5HT modulation of plasticity

See [MASTERY.md](MASTERY.md) for the full algorithm.

---

## Quick Start

### Using Runes scripts (recommended)

ANTs are defined as Runes scripts that compose built-in verbs:

```rune
rune "compare" do
  version 1
end
use :ant_ml

def forward(input) do
    w_in = load_synaptic("compare.w_in", 16, 64)
    w_hidden = load_synaptic("compare.w_hidden", 16, 16)
    w_out = load_synaptic("compare.w_out", 1, 16)

    h = matmul(input, w_in, 16, 64)
    h = relu(h)
    h = matmul(h, w_hidden, 16, 16)
    h = relu(h)
    matmul(h, w_out, 1, 16)
end
```

### Using Rust directly

```rust
use atomic_neural_transistors::{CompareANT, PackedSignal};

let mut compare = CompareANT::new()?;
let a: Vec<PackedSignal> = (0..32).map(|i| PackedSignal::pack(1, i * 8, 1)).collect();
let b: Vec<PackedSignal> = (0..32).map(|i| PackedSignal::pack(1, i * 8, 1)).collect();
let similar = compare.compare(&a, &b)?; // PackedSignal: positive = similar
```

### Multiplex encoding with surprise-gated learning

```rust
use atomic_neural_transistors::{MultiplexEncoder, AntSlot, PackedSignal};

let mut mux = MultiplexEncoder::new(32, 3, 40); // 32-dim output, EMA shift 3, threshold 40
mux.add_slot(AntSlot::with_passthrough("compare", compare_fn));
mux.add_slot(AntSlot::with_passthrough("classify", classify_fn));
mux.finalize();

let result = mux.process(&input, Some(&target));
// result.surprise.is_surprising  — prediction error exceeded threshold
// result.learning_occurred        — mastery update was triggered
// result.dopamine                 — current DA level (gates future learning)
```

---

## Composition Algebra

Complex operations compose from trained primitives **without additional learning**:

```rust
use atomic_neural_transistors::composition::sequence::{contains, has_duplicate};

// contains = OR(compare(query, seq[i]) for all i)
// has_duplicate = OR(compare(seq[i], seq[j]) for all i < j)
```

Composition results from the benchmark suite:

| Composition | Built From | Accuracy | Comparisons |
|-------------|-----------|----------|-------------|
| `has_duplicate` (seq len 4-8) | CompareANT | 100% | O(n^2) |
| `contains` (seq len 3-8) | CompareANT | 100% | O(n) |
| Sudoku 4x4 validation | CompareANT | 100% | 72 per grid |
| Planning pipeline | ClassifierANT + CompareANT | 100% | per step |

---

## Persistence

ANTs persist via **Thermogram** — a delta-chained format that tracks thermal progression of synaptic strengths (Hot → Warm → Cool → Cold).

```rust
use thermogram::{Thermogram, Delta, PlasticityRule};

// Save
let mut thermo = Thermogram::new("compare", PlasticityRule::stdp_like());
thermo.apply_delta(Delta::create("compare.w_out", weights.data.clone(), "mastery"))?;
thermo.save(&path)?;

// Load → byte-identical outputs
let loaded = Thermogram::load(&path)?;
```

Thermogram I/O benchmarks:
- **Save** (all 5 ANTs, 9,456 params): 4.9 ms
- **Load** (all 5 ANTs): 105 µs
- File sizes: 454 bytes – 2.9 KB per trained ANT

---

## Examples

13 reproducible examples demonstrate mastery learning, composition, multiplex encoding, and real-world anomaly detection:

```bash
# Mastery training (all converge in 1-2 cycles)
cargo run --example compare_mastery      # 100% — similarity detection
cargo run --example classifier_mastery   # 100% — 4-class classification
cargo run --example diff_mastery         # 100% — change detection
cargo run --example gate_mastery         # 96.5% — signal gating
cargo run --example merge_mastery        # 99.5% — signal fusion

# Composition (no additional learning)
cargo run --example composition_has_duplicate   # 100% on 500 sequences
cargo run --example composition_contains        # 100% on 500 queries
cargo run --example composition_sudoku          # 100% on 200 4x4 grids

# Multi-ANT pipelines
cargo run --example pipeline_planning    # 100% success — classifier + comparer
cargo run --example persistence_lifecycle  # train → save → destroy → load → verify

# Multiplex encoding with surprise-gated learning
cargo run --example multiplex_classify   # 3 ANTs, salience routing, DA dynamics
cargo run --example adaptive_cascade     # specialization + shift adaptation + cascade verification

# Real-world use case
cargo run --example sensor_anomaly       # 87% detection, 12% FP, 4-type classification, adaptation
```

---

## Benchmarks

Run with `cargo bench`:

```
ANT Forward Passes:
  classifier_forward         194 µs
  compare_forward             64 µs
  diff_forward                49 µs
  gate_forward                80 µs
  merge_forward               49 µs

Multiplex System:
  multiplex/3_ant_process    5.0 µs    (3 ANTs + routing + prediction + learning)
  salience_route_3x32        622 ns    (gate-based fusion, 3 sources × 32 dim)
  prediction_observe_32dim    85 ns    (EMA update + surprise detection)
  neuromod/full_cycle         10 ns    (inject DA/NE/5HT + gate check + tick)

Learning:
  mastery_update_step         51 µs    (one mastery step via Runes)

Persistence:
  thermogram/save           4.9 ms    (all 5 ANTs, 9,456 params)
  thermogram/load           105 µs    (all 5 ANTs)
```

Example wall times (release, including data generation + training + evaluation):

| Example | Wall Time |
|---------|-----------|
| CompareANT mastery | 544 ms |
| ClassifierANT mastery | 371 ms |
| DiffANT mastery | 413 ms |
| GateANT mastery | 424 ms |
| MergeANT mastery | 453 ms |
| Sudoku (14,400 comparisons) | 352 ms |
| Planning pipeline | 363 ms |
| Persistence lifecycle | 363 ms |
| Multiplex classify | 36 ms |
| Adaptive cascade | 33 ms |
| Sensor anomaly (600 ticks) | 31 ms |

Binary sizes: 177–523 KB per example (release build).

---

## Architecture

```
atomic-neural-transistors/
├── src/
│   ├── core/               # AtomicNeuralTransistor runtime (Runes VM bridge)
│   │   ├── atomic_neural_transistor.rs
│   │   └── weight_matrix.rs  # WeightMatrix, PackedSignal matmul
│   ├── ants/               # Specialized ANT wrappers
│   │   ├── compare.rs      # CompareANT — binary similarity
│   │   ├── diff.rs         # DiffANT — change detection
│   │   ├── merge.rs        # MergeANT — signal fusion
│   │   ├── gate.rs         # GateANT — sigmoid gating
│   │   └── classifier.rs   # ClassifierANT — multi-class
│   ├── learning.rs         # MasteryState, MasteryConfig, pressure engine
│   ├── neuromod.rs         # NeuromodState — DA/NE/5HT chemical gating
│   ├── prediction.rs       # PredictionEngine — EMA + surprise detection
│   ├── salience.rs         # SalienceRouter — gate-based multi-ANT fusion
│   ├── multiplex.rs        # MultiplexEncoder — parallel ANTs + routing + learning
│   ├── modules/
│   │   └── ant_ml.rs       # Runes verb implementations (matmul, relu, neuromod, etc.)
│   ├── composition/        # Composition algebra (contains, has_duplicate, grid)
│   └── weights_init.rs     # Self-initialization from Thermogram
├── runes/                  # Runes scripts defining ANT computation
│   ├── compare.rune
│   ├── classifier.rune
│   ├── diff.rune
│   ├── gate.rune
│   └── merge.rune
├── examples/               # 12 reproducible examples
├── benches/                # Criterion benchmarks
├── trained/                # Thermogram files (persisted synaptic strengths)
├── MASTERY.md              # Full mastery learning algorithm documentation
├── CROSS_ANT_ROUTING.md    # Research notes on inter-ANT routing patterns
└── REAL_WORLD_USES.md      # Production deployment history
```

---

## Dependencies

- **ternary-signal** — `PackedSignal`, `Signal`, `Polarity` types (the atom of all signaling)
- **runes** — Behavioral scripting engine (parser, evaluator, core)
- **thermogram-rs** — Delta-chained persistence with thermal progression

All computation is integer-only. No floating point dependencies. No BLAS. No GPU.

---

## License

MIT OR Apache-2.0
