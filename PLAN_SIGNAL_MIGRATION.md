# PLAN: Gut PackedSignal from ANTs

**PackedSignal is REMOVED from this repo entirely. It stays in ternary-signal for audio (lossy storage). For ANTs, it's destructive — all computation AND storage uses raw Signal (Polarity × Magnitude × Multitude).**

**No public Rust API. Runes is the only public interface.**

---

## File Audit — What Changes

| File | Lines | Action |
|------|-------|--------|
| `src/lib.rs` | 55 | Remove `PackedSignal` re-export. Export `Signal` only. |
| `src/error.rs` | 31 | No change |
| `src/encoding.rs` | 438 | Static table → `Signal`. All functions → `Signal`. Remove all `PackedSignal` imports. |
| `src/learning.rs` | 348 | All params → `Signal`. `packed_from_current()` → Signal construction. Remove import. |
| `src/memory.rs` | 420 | All API → `Signal`. Databank storage serializes Signal directly (3 bytes, not 1). Remove import. |
| `src/neuromod.rs` | 201 | No change (never used PackedSignal) |
| `src/prediction.rs` | 234 | All params → `Signal`. Remove import. |
| `src/salience.rs` | 277 | All I/O → `Signal`. Remove import. |
| `src/multiplex.rs` | 376 | All throughput → `Signal`. Remove import. |
| `src/weights_init.rs` | 84 | Init with `Signal::ZERO`. Remove import. |
| `src/thermal_mastery.rs` | ~200 | All params → `Signal`. Remove import. |
| `src/core/mod.rs` | 10 | No change |
| `src/core/weight_matrix.rs` | 382 | **`data: Vec<Signal>`**. matmul/relu/softmax all Signal. Kill `packed_from_current()`. Kill `relu_packed()`. Kill `softmax_packed()`. `.ant` format writes 3 bytes per signal (pol, mag, mul) not 1. |
| `src/core/thermal.rs` | ~200 | **`ThermalWeight.signal: Signal`**. matmul → Signal. `random_hot()` → Signal. `.ant` format → 3 bytes per signal. |
| `src/core/atomic_neural_transistor.rs` | ~100 | No PackedSignal conversion. Signal throughout. |
| `src/modules/ant_ml.rs` | ~500 | **All verbs operate on Signal.** Value representation carries Signal. No PackedSignal anywhere. |
| `src/ants/classifier.rs` | ~80 | Internal only (called via Runes). Signal throughout. |
| `src/ants/compare.rs` | ~80 | Same |
| `src/ants/diff.rs` | ~60 | Same |
| `src/ants/gate.rs` | ~60 | Same |
| `src/ants/merge.rs` | ~60 | Same |

**After this: `grep -r PackedSignal src/` returns zero results.**

---

## What Gets Killed

- `pub use ternary_signal::PackedSignal` — gone from lib.rs
- `packed_from_current()` — gone. Construct `Signal` directly.
- `relu_packed()` — renamed to `relu()`, operates on `&[Signal]`
- `softmax_packed()` — renamed to `softmax()`, operates on `&[Signal]`
- `PackedSignal::pack()` calls — replaced with `Signal::new()` or direct struct construction
- `PackedSignal::current()` calls — replaced with `Signal::current()` (already exists, returns `polarity * magnitude * multiplier` as i32)
- `PackedSignal::ZERO` — replaced with `Signal::ZERO`
- `PackedSignal::from_signal()` / `.to_signal()` — gone, no conversion needed
- `.as_u8()` serialization — replaced with 3-byte serialization (pol, mag, mul)

---

## Storage Format Change

### .ant file format → v3

See `ANT_FORMAT.md` for full specification.

**v2** (dead): 1 byte per synaptic strength (PackedSignal = u8). Lossy. 6 bytes per ThermalWeight.
**v3** (current): 3 bytes per signal (Polarity i8 + magnitude u8 + multiplier u8). Lossless. 8 bytes per ThermalWeight.

Breaking change. Old .ant files won't load. They were trained with broken precision anyway.

### ThermalWeight struct

```rust
pub struct ThermalWeight {
    pub polarity: Polarity,     // ternary_signal::Polarity enum — enforced {-1, 0, +1}
    pub magnitude: u8,
    pub multiplier: u8,
    pub temperature: u8,        // 255=HOT → 0=COLD
    pub hits: u16,
    pub pressure: i16,
}
```

Uses `ternary_signal::Polarity` — not raw i8. Invalid polarities are unrepresentable.

---

## Examples → Runes

All 17 Rust examples become Runes scripts. The .rs files stay as internal test/reference but are NOT the public interface.

| Current Rust Example | Runes Equivalent |
|---------------------|-----------------|
| `classifier_mastery.rs` | `classifier.rune` (exists, extend with mastery) |
| `thermal_classifier.rs` | `thermal_classifier.rune` (new) |
| `compare_mastery.rs` | `compare_train.rune` (exists) |
| `diff_mastery.rs` | `diff.rune` (exists) |
| `gate_mastery.rs` | `gate.rune` (exists) |
| `merge_mastery.rs` | `merge.rune` (exists) |
| `sensor_anomaly.rs` | `sensor_anomaly.rune` (exists) |
| `debug_mastery.rs` | `debug_mastery.rune` (new) |
| `colonist_threat.rs` | `colonist_threat.rune` (new) |
| `multiplex_classify.rs` | `multiplex_classify.rune` (new) |
| `adaptive_cascade.rs` | `adaptive_cascade.rune` (new) |
| `composition_contains.rs` | `composition_contains.rune` (new) |
| `composition_has_duplicate.rs` | `composition_has_duplicate.rune` (new) |
| `composition_sudoku.rs` | `composition_sudoku.rune` (new) |
| `persistence_lifecycle.rs` | `persistence.rune` (new) |
| `pipeline_planning.rs` | `pipeline.rune` (new) |

---

## Missing Runes Verbs

Verbs needed for full Runes-only interface:

- `thermal_mastery_update` — ThermalMasteryState::update()
- `thermal_load_synaptic` — load ThermalWeightMatrix
- `thermal_save` — persist ThermalWeightMatrix
- `thermal_decay` — ThermalMasteryState::decay()
- `thermal_summary` — temperature distribution query
- `one_hot` — generate one-hot target signal vector
- `error_signal` — compute target - output
- `hidden_target` — derive hidden region targets from output error

---

## Execution Order

### Phase 1: Core types
1. `weight_matrix.rs` — `data: Vec<Signal>`, all methods Signal-native, kill packed helpers
2. `thermal.rs` — `ThermalWeight.signal: Signal`, matmul Signal-native
3. `.ant` format → 3 bytes per signal
4. `cargo check`

### Phase 2: Learning
1. `learning.rs` — MasteryState::update() all Signal params
2. `thermal_mastery.rs` — ThermalMasteryState::update() all Signal params
3. `cargo check`

### Phase 3: Runes module
1. `ant_ml.rs` — all verbs Signal-native, Value carries Signal
2. `cargo check`

### Phase 4: ANT types & supporting modules
1. `encoding.rs`, `memory.rs`, `prediction.rs`, `salience.rs`, `multiplex.rs`
2. All ANT variants (classifier, compare, diff, gate, merge)
3. `weights_init.rs`
4. `cargo check`

### Phase 5: lib.rs + cleanup
1. Remove `pub use PackedSignal` from lib.rs
2. Grep for any remaining PackedSignal references — kill them all
3. Update tests, benches
4. `cargo check` → `cargo test`

### Phase 6: Examples → Runes
1. Write .rune equivalents
2. Add missing verbs as discovered
3. Keep .rs files as internal reference

### Phase 7: ant-mnist
1. Rewrite network as .rune script
2. Rust host loads and runs the rune
3. Hot-reload without recompilation

---

## Thermogram Dependency — REMOVED

thermogram-rs is no longer relevant. ANTs has its own persistence format (.ant v3). Drop the thermogram dependency from Cargo.toml entirely. Kill all thermogram imports, Delta usage, and Thermogram save/load code. 41 occurrences across 8 files.
