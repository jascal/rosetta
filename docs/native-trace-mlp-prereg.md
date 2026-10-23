# Run D — where does the copy computation live? (whole-MLP ablation sweep)

Status: **registered before any run-D model request.** Runs A, B and C are closed
and published ([#62](https://github.com/jascal/rosetta/pull/62),
[#63](https://github.com/jascal/rosetta/pull/63),
[#64](https://github.com/jascal/rosetta/pull/64)); their evidence is immutable and
is not touched here.

## The question this run exists to answer

Run C held run B's nine trace coordinates fixed and ablated them as a group, as a
cumulative prefix, and by writing other instances' values over them. Across
**2,394 interventions on 114 fresh contexts, no copy-row answer ever changed** —
while those coordinates moved the model's logits about **nineteen times** more
than layer-matched random neurons, with almost all of that effect concentrated
in a single coordinate, layer 23 neuron 1792.

So the nine coordinates are **on the model's path and not the mechanism**. Run C
ruled nine coordinates out. It pointed at nothing.

Run D asks the next question: **if the copy answer is not computed by the nine
coordinates, is it computed by the MLP at the answer position at all?** The
instrument is a whole-layer ablation: zero every neuron's post-SwiGLU product in
one layer, at the answer position, and see whether the copy answer survives.

Three outcomes are possible, and they are different claims:

- **H-B (layer-localized).** Some layer's MLP at the answer position is
  necessary: zeroing it flips copy answers.
- **H-C (position-localized).** No layer's answer-position MLP is necessary, but
  some layer's MLP **is** necessary when it is removed along the whole sequence —
  the copy source is represented by that layer's MLP at an earlier position.
- **H-A (attention-only).** No layer's MLP is necessary at any position in the
  sweep's scope: the copy answer is computed by the attention path and the
  residual stream alone.

Run C cannot distinguish these, because it ablated nine coordinates rather than
whole layers, and only at the answer position.

## Inputs: run C's frozen contexts, reused unchanged

Run D reuses run C's **frozen** inputs and unpatched references, digest-pinned,
rather than generating new ones:

| Artifact | Source | Role |
|---|---|---|
| `dataset.json` | run C's frozen manifest | the 114 contexts (96 copy, 18 controls) |
| `references.json` | run C's frozen manifest | each row's own unpatched greedy answer |
| `base.json` | run C's frozen manifest | each row's unpatched reference/target logits |

The same contexts make run D **directly comparable to run C**: the nine treated
coordinates sit in layers 7, 15 and 23, and the sweep covers those layers among
the other 21. Nothing is fitted, selected or re-ranked; the treatment is the
layer grid, which is fixed in advance and covers all 24 layers with no
cherry-picking.

## The instrument, and what it cannot do

A probe patch sets one neuron's post-SwiGLU product (`hidden[position, neuron]`)
to a value before the layer's down-projection. Zeroing all 4,864 neurons of a
layer at the last position removes that layer's MLP contribution to the answer
position entirely. The pinned binary supports multiple simultaneous patches at
distinct coordinates of the same layer, and the same mechanism run C used and
validated exactly (its identity arm reproduced every answer bit-for-bit).

**Scope limit, registered in advance: this instrument can only ablate MLP
products.** Attention is not patchable through this API, so run D cannot rule
out an attention-side implementation of the copy; a null result here is "not the
MLP", not "not the model".

## Arms

All patches are at the **last position** (the answer position) unless an arm
says otherwise. "Zero layer *l*" means: every one of the 4,864 neurons of that
layer's MLP, at that position, set to 0.0.

| Arm | Count | Definition |
|---|---|---|
| **S** — single-layer sweep | 24 | zero layer *l* alone, for each *l* = 0..23 |
| **C** — cumulative | 7 | zero layers 0..*k*−1 together, for *k* = 1, 2, 4, 8, 12, 16, 24 |
| **I** — identity write-back | 3 | layers 7, 15, 23: write the row's **own** MLP products back |
| **W** — donor write-back | 3 | layers 7, 15, 23: write the **donor row's** MLP products (run C's registered matched-group opposite-class donor) |
| **F** — full-sequence | ≤6 | stage 2 (below): zero one layer's MLP at **every** position |

Stage 1 is S + C + I + W: **37 arms × 114 contexts = 4,218 requests**.

**Stage 2 is a registered conditional, not a post-hoc choice.** It runs on the
layer set

```
L2 = {layers whose S-arm flipped at least one copy answer}
     ∪ {the single layer with the largest mean |Δlogit| over copy rows in S}
     ∪ {7, 15, 23}
```

and it asks whether a layer's MLP matters at *earlier* positions (the copy
source). If stage 1 flips nothing anywhere, stage 2 is the decisive probe; if
stage 1 flips something, stage 2 asks whether that layer matters along the
sequence rather than only at the answer.

The cumulative arm's `k=1` is the same intervention as the sweep's layer-0 arm.
That duplication is a registered internal consistency check: the two arms must
produce identical answers and identical logits.

## Readouts

All in Datalog (`dl/mlp_ablation_summary.dl`), over the intervention results and
the unpatched references:

- `flip(i, arm)` — the patched answer differs from the row's own unpatched
  answer;
- `answer_correct(i, arm)` — for copy rows, the patched answer equals the
  generated copy target;
- `logit_delta(i, arm, d)` — the change in the logit of the row's own reference
  token;
- `summary(arm, n, nflip, nflipcopy, ncorrect, max_abs_logit_delta, mean_abs_logit_delta)`
  — per arm;
- `layer_table(arm, layer, n, nflip, nflipcopy, mean_abs_logit_delta)` — the sweep;
- `identity_ok(i)` / `identity_broken(i)` — the instrument control.

A flip is relative to the model's own unpatched answer, and copy-row flips are
counted separately from control-row flips, because a control row has no correct
answer to lose.

## Decision rules, fixed now

- **H-B is supported** iff some layer's S-arm flips ≥ 1 copy-row answer. The
  layer set is then exactly the flipping layers.
- **H-B is supported in aggregate** iff no single S-arm flips a copy-row answer
  but the joint arm (C, k=24) does: no layer is individually necessary, but the
  MLPs are collectively necessary.
- **H-C is supported** iff no S-arm flips a copy-row answer and some stage-2
  F-arm does.
- **H-A is supported** iff no S-arm and no F-arm flips a copy-row answer. That is
  a claim about the MLP path only, in the scope stated above.
- If the identity arm does not reproduce the unpatched answers exactly, **the
  run stops** and reports the instrument failure; no other arm is interpreted.

Brittleness is the obvious rival explanation for any flip, so every arm reports
its logit effect alongside its flips: a layer whose ablation flips answers while
moving the reference logit less than a layer that moves it more and flips
nothing is evidence of a local computation, not of a hair-thin margin.

## Integrity checks

Fail closed, before any result is interpreted:

- every zero-arm patch list covers **exactly** the neuron universe of each
  patched layer (no gaps, no duplicates, no out-of-range coordinate);
- every arm has the registered number of layers;
- the identity arm reproduces the row's own answer exactly (max |Δlogit| = 0);
- the cumulative `k=1` arm and the sweep's layer-0 arm agree exactly;
- the write-back arms are built from a complete observation file: every
  (row, layer) covers the whole neuron universe, with no missing or extra
  neuron.

## Claim discipline

Everything is **`empirical`**, over one pinned checkpoint, one pinned executable,
one task family, one set of 114 contexts, greedy decoding, and MLP products
only. Run D produces **no circuit and no equivalence certificate**, so it claims
no `proved` result; the program's `proved` results remain run B's 31
finite-domain certificates.

Not claimed, at any point:

- that the copy mechanism is located by a null result — a null here means "not
  the MLP path", never "found elsewhere";
- anything about attention, which this instrument cannot ablate;
- anything about other checkpoints, temperatures, positions or task classes;
- that an S-arm flip identifies a *mechanism* rather than a load-bearing
  contributor: necessity is not explanation.

## Cost

Measured on the pinned binary before registration, on a context outside the
study set: a bare request costs 208 ms, a 4,864-patch request 215 ms, the
116,736-patch joint arm 361 ms. Stage 1 is therefore ≈ 20 minutes of model
time, the identity collection 114 requests of ~1.6 MB, and stage 2 ≈ 7 minutes.
Total under 30 minutes, about 4,900 requests.

## Completion

Run D is complete when the outcome — a localized layer, an aggregate dependence,
a position dependence, or a null — is committed and published with its
counterexamples. No protocol element may be altered after any intervention
observation to change the verdict.
