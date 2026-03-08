# Cross-ANT Routing: From Attention to Salience

Research notes for inter-ANT communication and composition beyond algebraic operators.

---

## Current State

ANTs compose via two patterns today:

1. **Algebraic composition**: `OR(compare(a, b))`, `AND(NOT(has_dup))` — logical operators over scalar ANT outputs. Proven in examples 6-9 (has_duplicate, contains, Sudoku, planning pipeline).

2. **Concatenation**: Two signals get concatenated as `[sig_a, sig_b]` and fed to a single ANT's input. This is how DiffANT, MergeANT, and GateANT already work — they take 64-dim input formed by concatenating two 32-dim signals.

What's missing: **one ANT's output feeding as context into another ANT's input**, where the downstream ANT learns to use the upstream output as part of its computation.

---

## The Astromind Dual Encoder Pattern

From `astromind-archive/crates/astromind/src/models/time_model.rs`:

```
text_emb = embed_tokens(text_ids)           # Encoder A
service_emb = service_encoder(features)     # Encoder B
fused = concat([text_emb, service_emb])     # Concatenate along feature dim
output = fusion_proj(fused)                 # Project back to hidden_size
```

Two encoders process different modalities. Their outputs get concatenated and projected through a learned fusion layer. The fusion layer learns which combinations of the two encodings matter.

**Mapped to ANTs**: ANT A produces a 32-dim output. That output is concatenated with a new 32-dim input signal. ANT B takes the 64-dim `[ant_a_output, new_input]` through its frozen hidden projection and learned output layer. ANT B's mastery learning discovers which dimensions of ANT A's output correlate with which dimensions of the new input.

This is structurally identical to what MergeANT already does — the only difference is that one of its inputs would be the output of another ANT rather than raw signal.

---

## The Cross-Mesh Attention Pattern

From `astromind-archive/engineering/WHITEPAPER_ANT.md`, Section 5.4:

8 specialized meshes execute in parallel. After execution, a cross-mesh attention mechanism coordinates them:

```
Q_i = mesh_i.output @ W_q        # Query: what does mesh i need?
K   = all_outputs @ W_k          # Keys: what each mesh advertises
V   = all_outputs @ W_v          # Values: actual data to share

scores = Q_i · K^T / sqrt(d)     # Relevance matching
weights = softmax(scores)         # Normalize to distribution
output_i = weights · V + mesh_i.output   # Weighted blend + residual
```

Each mesh asks "who has what I need?" by comparing its Query against every mesh's Key. The dot product measures relevance. Softmax normalizes. The mesh receives a weighted blend of everyone's Values.

### Why This Doesn't Port Directly to ANTs

| Requirement | Attention | ANT Substrate |
|-------------|-----------|---------------|
| Score computation | Float dot product | Integer multiply-accumulate (OK) |
| Normalization | `softmax(exp(x) / sum(exp))` | No `exp()`, no float division |
| Blending | Weighted sum with continuous weights | No continuous weights |
| Output | Smooth mixture of all sources | Discrete selection or gating |

The softmax is the hard blocker. It requires `exp()` and float division to produce a probability distribution. There is no ternary integer equivalent that preserves the smooth blending property.

---

## The Ternary Alternative: Salience Routing

Instead of blending all sources smoothly, **select which source to use** via hard routing.

### Option A: Classifier-Based Routing

A ClassifierANT examines the current context and selects which downstream ANT to invoke:

```
context = [current_signal, task_descriptor]
route = ClassifierANT.predict(context)     # Returns class 0-N

match route:
    0 => CompareANT.forward(signal_a, signal_b)
    1 => DiffANT.forward(signal_a, signal_b)
    2 => MergeANT.forward(signal_a, signal_b)
    3 => GateANT.forward(signal, control)
```

Already proven: Example 9 (pipeline_planning) does exactly this — ClassifierANT classifies state, then the result determines the action.

**Pros**: Pure integer, deterministic, already works.
**Cons**: Hard routing — only one ANT fires per step. No blending.

### Option B: Gate-Based Fusion

Multiple ANTs execute in parallel. A GateANT learns which dimensions of each ANT's output to pass through:

```
out_compare = CompareANT.forward(a, b)     # 32-dim
out_diff = DiffANT.forward(a, b)           # 32-dim
out_merge = MergeANT.forward(a, b)         # 32-dim

combined = concat([out_compare, out_diff, out_merge])  # 96-dim
gate_mask = GateANT.forward(combined, context)         # 96-dim gate values
gated = element_wise_mul(combined, gate_mask)           # 96-dim gated output
```

The GateANT's sigmoid output (0-255 per dimension) acts as a soft-ish gate — not truly continuous like softmax attention, but not binary either. The 256 discrete levels provide enough granularity for selective routing.

**Pros**: Multiple ANTs contribute. Per-dimension control. Sigmoid gives 256 levels of gating.
**Cons**: All ANTs must execute (no compute savings). Gate must be trained on the combined output space.

### Option C: Winner-Take-All Salience

Each ANT produces output + a confidence signal (output magnitude). The highest-confidence ANT wins:

```
out_a = CompareANT.forward(a, b)
out_b = DiffANT.forward(a, b)
out_c = MergeANT.forward(a, b)

mag_a = sum(|out_a[j]|)    # Total output magnitude = confidence
mag_b = sum(|out_b[j]|)
mag_c = sum(|out_c[j]|)

winner = argmax(mag_a, mag_b, mag_c)
output = [out_a, out_b, out_c][winner]
```

No learned routing at all — the ANT that produces the strongest signal wins. This mirrors biological winner-take-all circuits in cortical columns.

**Pros**: Zero additional parameters. No training needed. Pure integer.
**Cons**: Only one winner. No partial contribution. May be unstable if magnitudes are close.

---

## Experiment Plan

### Experiment 1: Dual Encoder Chaining

Prove that ANT A's output can feed ANT B as context:

```
# Train CompareANT on similarity (already done)
# Train a new "ContextMergeANT" that takes [compare_output, new_signal]
# Target: classify the new_signal using compare_output as context

compare_out = CompareANT.forward(reference, candidate)
context = concat([compare_out_expanded, new_signal])  # 64-dim
result = ContextMergeANT.forward(context)
```

Key question: Does the downstream ANT learn to use the upstream output meaningfully, or does it ignore it?

### Experiment 2: Gate-Based Multi-ANT Fusion

Run 3 ANTs in parallel, gate their outputs:

```
# Train Compare, Diff, Merge independently (already done)
# Train a GateANT on the concatenated 96-dim output
# Target: produce correct classification from the gated combination
```

Key question: Does gating outperform any single ANT alone?

### Experiment 3: Winner-Take-All

Run multiple ANTs, pick highest-magnitude output:

```
# No training needed — just magnitude comparison
# Test across multiple task types
```

Key question: Does output magnitude correlate with correctness?

### Experiment 4: Ternary Approximate Attention

Try to approximate softmax attention using integer operations:

```
# Replace exp(x) with max(0, x)^2 (ReLU-squared)
# Replace division with right-shift
# Replace weighted sum with top-K selection + equal weighting

scores = [Q · K_j for j in meshes]
scores = [max(0, s)^2 for s in scores]    # ReLU-squared approximation
top_2 = indices of 2 largest scores
output = (V[top_2[0]] + V[top_2[1]]) >> 1  # Average top-2 values
```

Key question: Is approximate attention better than hard routing for ANT-scale systems?

---

## Open Questions

1. **Is smooth blending necessary?** The production astromind used float attention across meshes, but each individual mesh (containing ANTs) used integer computation internally. The attention was a coordination layer *between* integer subsystems, not within them. Maybe the coordination layer is allowed to be float, while ANTs stay integer.

2. **Scale matters.** Attention's advantage is proportional to the number of sources. With 2-3 ANTs, hard routing probably suffices. With 8+ meshes of ANTs, the combinatorial space makes learned routing more valuable.

3. **Neuromodulator gating from production.** The original astromind gated mastery learning with dopamine, norepinephrine, and serotonin levels. These same neuromodulators could gate inter-ANT routing: dopamine selects which ANTs are "rewarded" (their outputs get forwarded), norepinephrine controls how many ANTs participate (narrow vs broad routing), serotonin modulates confidence thresholds.
