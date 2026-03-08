# Plan: Reproduce ANT Accuracy Benchmarks

Reproduce every accuracy claim from the Astromind production system as runnable examples in this repo. Each example trains from scratch on synthetic data, measures accuracy, persists learned thermograms, and verifies accuracy holds after reload.

---

## Example 1: CompareANT Equality (`examples/compare_mastery.rs`)

**Target:** 99.5% accuracy on random embedding pair equality detection.

**Training data:**
- Generate 1,000 random 32-dim ternary signal pairs
- 50% identical pairs (label: positive similarity)
- 50% different pairs (label: zero/negative similarity)
- Hold out 200 pairs for evaluation

**Training loop:**
1. Create CompareANT with Thermogram
2. For each training pair:
   - Forward pass: `compare(a, b)` -> similarity signal
   - Compute target: PackedSignal::MAX_POSITIVE if identical, PackedSignal::ZERO if different
   - `mastery_update(w_in, input, output, target)`
   - `mastery_update(w_hidden, input_hidden, output_hidden, target_hidden)`
   - `mastery_update(w_out, input_out, output_out, target_out)`
3. Repeat for N epochs until convergence

**Evaluation:**
- Run all 200 held-out pairs
- Threshold: similarity.current() > 0 = "same", else "different"
- Report accuracy, false positive rate, false negative rate

**Persistence test:**
- Save thermogram after training
- Create new CompareANT from saved thermogram
- Re-evaluate on held-out set
- Assert accuracy matches pre-save accuracy exactly (byte-identical outputs)

**Convergence measurement:**
- Track accuracy after each epoch
- Report epoch at which 99% accuracy first reached
- Report total mastery transitions and polarity flips

---

## Example 2: ClassifierANT Multi-Class (`examples/classifier_mastery.rs`)

**Target:** 98%+ accuracy on 4-class synthetic pattern classification.

**Training data:**
- Generate 1,000 random 32-dim signals with 4 class labels
- Class 0: High magnitude in dims 0-7, low elsewhere
- Class 1: High magnitude in dims 8-15, low elsewhere
- Class 2: High magnitude in dims 16-23, low elsewhere
- Class 3: High magnitude in dims 24-31, low elsewhere
- Add noise: random perturbation of +/- 20% magnitude to all dims
- Hold out 200 for evaluation

**Training loop:**
1. Create ClassifierANT with Thermogram
2. For each training sample:
   - Forward pass: `classify(input)` -> 4 class logits
   - Compute target: one-hot encoded as PackedSignal (max magnitude for correct class, zero for others)
   - Mastery update on all 4 weight matrices
3. Repeat for N epochs

**Evaluation:**
- argmax of output = predicted class
- Report per-class accuracy and confusion matrix
- Report overall accuracy

**Persistence + convergence:** Same as Example 1.

---

## Example 3: DiffANT Change Detection (`examples/diff_mastery.rs`)

**Target:** 99%+ accuracy on detecting whether two vectors differ.

**Training data:**
- Generate 1,000 pairs of 32-dim signals
- 50%: identical (target diff = all zeros)
- 50%: one random dimension changed by significant amount (target diff = one-hot delta)
- Hold out 200

**Training loop:**
1. Create DiffANT with Thermogram
2. For each pair:
   - Forward: `diff(a, b)` -> 32-dim difference embedding
   - Target: known difference vector
   - Mastery update
3. Repeat

**Evaluation:**
- Magnitude of diff output: > threshold = "changed", else "unchanged"
- Report detection accuracy
- Report per-dimension sensitivity (which dims does DiffANT track best?)

---

## Example 4: GateANT Signal Filtering (`examples/gate_mastery.rs`)

**Target:** Selective signal pass-through with learned context gating.

**Training data:**
- Generate 1,000 signal+context pairs
- Context encodes "which half to pass": context dims 0-15 high = pass signal dims 0-15, suppress 16-31
- Target: gated signal (signal * expected_gate)
- Hold out 200

**Training loop:**
1. Create GateANT with Thermogram
2. For each pair:
   - Forward: `gate(signal, context)` -> gated output
   - Target: expected gated signal
   - Mastery update
3. Repeat

**Evaluation:**
- Correlation between output and target signal
- Report suppression ratio (how well does gate suppress irrelevant dims?)
- Report pass-through fidelity (how well does gate preserve relevant dims?)

---

## Example 5: MergeANT Signal Fusion (`examples/merge_mastery.rs`)

**Target:** Learned optimal combination of two signal sources.

**Training data:**
- Generate 1,000 signal pairs
- Target: hand-crafted "ideal merge" (e.g., max of each dimension, or weighted combination)
- Hold out 200

**Training loop:**
1. Create MergeANT with Thermogram
2. For each pair:
   - Forward: `merge([sig1, sig2])` -> merged output
   - Target: ideal merge
   - Mastery update
3. Repeat

**Evaluation:**
- Correlation between output and target
- Report dimension-wise accuracy

---

## Example 6: Composition — has_duplicate (`examples/composition_has_duplicate.rs`)

**Target:** 100% accuracy on duplicate detection using trained CompareANT.

**Prerequisite:** Trained CompareANT thermogram from Example 1.

**Test data:**
- Generate 500 sequences of length 4-9
- 50% contain at least one duplicate element
- 50% all unique

**Pipeline:**
1. Load trained CompareANT from thermogram
2. For each sequence:
   - For all pairs (i, j) where i < j:
     - `compare(seq[i], seq[j])`
     - If similarity > threshold: duplicate found
   - OR all pair results
3. Report accuracy

**This proves composition preserves trained accuracy without retraining.**

---

## Example 7: Composition — contains (`examples/composition_contains.rs`)

**Target:** ~97% accuracy on query-in-sequence search.

**Prerequisite:** Trained CompareANT thermogram from Example 1.

**Test data:**
- Generate 500 sequences of length 8
- Generate query for each
- 50% query is present in sequence
- 50% query is absent

**Pipeline:**
1. Load trained CompareANT from thermogram
2. For each (query, sequence):
   - For each element in sequence:
     - `compare(query, element)`
   - OR all results
3. Report accuracy, false positive rate, false negative rate

---

## Example 8: Composition — Sudoku Constraint Validation (`examples/composition_sudoku.rs`)

**Target:** 100% constraint violation detection.

**Prerequisite:** Trained CompareANT thermogram from Example 1.

**Test data:**
- Generate 100 valid 9x9 Sudoku grids
- Corrupt 50 grids by swapping two cells in the same row/column/box
- Encode each cell value as a 32-dim ternary signal

**Pipeline:**
1. Load trained CompareANT from thermogram
2. For each grid:
   - For each row: `has_duplicate(row)` -> violation?
   - For each column: `has_duplicate(col)` -> violation?
   - For each 3x3 box: `has_duplicate(box)` -> violation?
   - Grid valid = no violations detected
3. Report: detection accuracy on corrupt grids, false positive rate on valid grids

---

## Example 9: Multi-ANT Pipeline — Planning (`examples/pipeline_planning.rs`)

**Target:** Demonstrate ClassifierANT + DiffANT working together for action planning.

**Prerequisite:** Trained ClassifierANT and DiffANT thermograms.

**Scenario:**
1. Define 4 action types (fill, backtrack, validate, complete)
2. Generate plan states as 32-dim signals
3. ClassifierANT selects action type from current state
4. Execute action (modify state deterministically)
5. DiffANT measures progress (diff between previous and current state)
6. Loop until ClassifierANT outputs "complete" or max steps reached

**Evaluation:**
- Does the pipeline reach "complete" state?
- How many steps vs random action selection?
- Report progress curve (DiffANT magnitude over steps)

---

## Example 10: Full Persistence Lifecycle (`examples/persistence_lifecycle.rs`)

**Target:** Prove the complete train -> save -> load -> verify cycle.

**Pipeline:**
1. Train CompareANT from scratch (25 iterations)
2. Measure accuracy: should be >= 99%
3. Save thermogram to `trained/compare.thermo`
4. Destroy the ANT instance
5. Create new CompareANT, load from `trained/compare.thermo`
6. Measure accuracy again: must be identical to step 2
7. Run 5 more mastery iterations (continued learning)
8. Measure accuracy: should be >= 99.5%
9. Save updated thermogram
10. Verify thermogram has entries in Warm or Cool layer (thermal promotion from repeated mastery)

---

## Implementation Order

1. **Example 1 (CompareANT mastery)** — Foundation. All compositions depend on this.
2. **Example 2 (ClassifierANT mastery)** — Second most used ANT.
3. **Example 6 (has_duplicate)** — Simplest composition, proves the algebra works.
4. **Example 7 (contains)** — Second composition.
5. **Example 8 (Sudoku)** — The flagship demo from production.
6. **Example 10 (Persistence lifecycle)** — Proves thermogram persistence.
7. **Example 3 (DiffANT)** — Needed for pipeline example.
8. **Example 4 (GateANT)** — Standalone training.
9. **Example 5 (MergeANT)** — Standalone training.
10. **Example 9 (Planning pipeline)** — Multi-ANT integration, last because it needs trained models.

## Technical Notes

### Mastery Loop Structure

The mastery_update verb operates per-weight-matrix. For a CompareANT with 3 matrices (w_in, w_hidden, w_out), each forward pass requires 3 separate mastery_update calls — one per matrix. The Runes script needs to expose intermediate activations for this.

This means the training scripts cannot just use the existing `forward()` function. They need a `forward_with_mastery()` variant or a training-specific Runes script that calls mastery_update between layers.

### Training Runes Scripts

Each ANT needs a `train.rune` variant alongside its `forward.rune`:

```rune
def train(input, target) do
    w_in = load_synaptic("compare.w_in", 16, 64)
    w_hidden = load_synaptic("compare.w_hidden", 16, 16)
    w_out = load_synaptic("compare.w_out", 1, 16)

    # Layer 1
    h1 = matmul(input, w_in, 16, 64)
    h1 = relu(h1)
    mastery_update(w_in, input, h1, ???)  # need layer-local targets

    # Layer 2
    h2 = matmul(h1, w_hidden, 16, 16)
    h2 = relu(h2)

    # Output
    out = matmul(h2, w_out, 1, 16)
    mastery_update(w_out, h2, out, target)

    out
end
```

**Open question:** What are the layer-local targets for intermediate layers? In production, the global error signal was broadcast to all layers (each synapse sees `error * input_correlation`). The mastery_update verb already implements this — it takes the global input/output/target and internally computes per-synapse pressure from the correlation. So actually we can just call mastery_update with the full input/output/target for the output layer, and the internal correlation computation handles the rest.

Need to verify this against the learning.rs implementation before coding.
