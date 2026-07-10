# J-Lens bridge — can a Jacobian read-out propose rules our black-box miner never enumerates, and are they earlier in the layers? (`empirical` / `open`)

rosetta's stated #1 bottleneck is candidate generation, not certification: "hand-coding idiom
detectors … then *guessing* agreement / delimiter / coreference … is **a priori enumeration of a
space we cannot enumerate** … the models have circuits we'd never think to name — they're exactly
the ~60% holdout-generalization gap" (`IDIOM_LEARNER.md`). J-Lens (landed in `fieldrun` +
`pil`) is a new, cheap, **model-internal** attribution signal. This note asks whether it can seed
that generator, and whether it implies **earlier-in-the-layers** rules to find — and pre-registers
the disciplined call before any plumbing gets built.

**One-line verdict up front:** J-Lens is a well-matched *untrusted proposer* for rosetta's
generation bottleneck and hands rosetta an axis it has never had (a block/layer attribution to sit
beside its token-position axis). But its "structure resolves earlier" win is **scale-gated to
≥24 layers / ~400M params**, and it can **never** promote a tag. **Measured outcome (2026-07-10 —
see Step-0 below): do NOT build the proposer hook yet.** The full chain now runs on real artifacts,
but on every local model the derived prior fails to discriminate: `llama32_1b` (16L) is a clean null
(`dresolve ≥ 0` at all λ), and `qwen25coder15b` (28L) is marginal/inconclusive (`dresolve = −0.008`
only at λ=0.10, positive at λ≈0.25–0.5) and still non-discriminating. J-Lens stays analysis-only; the
hook is `open` on a larger ≥24L fit.

## What J-Lens is (the `fieldrun → pil` seam)

An **empirical mid-stack read-out probe**. The logit-lens reads an intermediate residual by
unembedding it directly, assuming the downstream network is the identity (`J_l = I`). J-Lens first
routes the residual through the layer's **averaged causal Jacobian**, then unembeds:

```
J_l = E_{t, t'≥t, prompt}[ ∂h_final,t' / ∂h_l,t ]     read(h_l) = softmax( W_U · norm( J_l · h_l ) )
```

so a layer-`l` activation is scored by *what the network is disposed to make it emit*, not by the
identity-path guess. fieldrun owns the forward pass, so it estimates `J_l` by a finite-difference
Hutchinson JVP (no autodiff), central-differenced, off the hot path (`fieldrun/JLENS.md`,
`src/jlens.rs`). **pil** consumes it to correct direct-logit-attribution: `c_b^v = ⟨J[l(b)] d̃_b,
U_v⟩` credits a block with its **total (direct + downstream)** linearised effect, not just its
direct logit contribution — and since PIL's `sources` *are* its circuits, this is literally a
**per-circuit attribution correction** (`pil/pil/fieldrun_io.py:jcorrect_sources`).

The framing is emphatically **not a certificate** (`fieldrun/JLENS.md`): "`J_l` is a first-order,
context-averaged approximation. The J-lens **never touches the forward path or the faithfulness
gate** — it only re-reads captured residuals. Treat its output as a probe, not a certificate."

Artifacts fieldrun now emits (numpy side-channel — **nothing lands in `whole.dl` or any `.dl`**):

| CLI | member | dtype / shape | meaning |
|---|---|---|---|
| `--jlens-export out.npz` | `J` | `f32 [n_layer,d,d]` | averaged causal Jacobian per layer; `J[n_layer−1]=I` |
| | `fitted` | `i4 [n_layer]` | `1` = fit; `0` = identity (reads as plain logit-lens) |
| `--tensors-export out.npz` | `U` | `f32 [vocab,d]` | unembedding rows (needed to turn a corrected read into logits) |
| | `gamma` | `f32 [d]` | final-norm gain (the γ-conjugation is load-bearing — see below) |

`.meta.json` carries the **apply convention** and **capture point**: `J[l]` maps *from* `h_l` = the
POST-block residual of layer `l` (after the attn+MLP residual add, PRE final-norm). Eval knob:
shrinkage `J' = (1−λ)I + λ·J`; `λ=0` reproduces the exact logit-lens, empirical sweet spot
`λ≈0.25–0.5`. Arch coverage: **rope** (RMSNorm; llama/qwen/stories) and **neox** (LayerNorm;
pythia) — i.e. *all* of rosetta's targets are fittable in principle.

**pil's headline result** (`experiments/jlens_correction_sweep.RESULTS.md`): the corrected read
makes the decode **resolve ~7–8% of depth (≈2 layers) earlier** than the logit-lens — **but only at
≥24 layers**. WIN at pythia-410m (24L) and Qwen2.5-0.5B (24L); **null at pythia-14m/70m/160m
(≤12L)**; Qwen2.5-1.5B (28L) inconclusive (a ~42h compute wall). The final-norm γ-conjugation
(`diag(γ) J diag(1/γ)`) is *causally necessary* for the win; the effect tracks **scale**
(threshold ~24L / ~400M params, depth confounded with width in the Pythia ladder), not architecture.

## The load-bearing fact: rosetta and J-Lens live in different worlds

Everything below follows from this table. Naming it is the point.

| | rosetta (live path) | J-Lens |
|---|---|---|
| What it reads | only `token → decide/logit` from `whole.dl` — a **token-level black box** | per-layer residual streams |
| Layer axis | **none exists** on the live path | layer-indexed by construction |
| Discovery signal | combinatorial mining + causal *input-token* perturbation flip-counts (`py/discover.py`) | first-order Jacobian attribution |
| Governing tag | `proved` **iff** `dl/equiv.dl` returns `nmiss=0 ∧ nuncov=0` | `empirical`, "a probe never a certificate" |

Consequence: J-Lens **cannot enter rosetta's certified path** — that would violate program-wide tag
discipline (a `proved` claim needs the artifact that backs it; a Jacobian probe is not that
artifact). The only legal integration is the one rosetta's architecture already runs on:
**untrusted generator + sound checker.** J-Lens *proposes*; `dl/equiv.dl` remains the sole arbiter
of `proved`. Anything else is off the table.

What it buys us that we lack today: rosetta localizes *which token positions* are load-bearing
(`discover.py` flip-counts) but has **no notion of which sub-computation** produces an answer.
J-Lens supplies exactly that missing **block/layer attribution axis** — a "where in the network,"
to complement rosetta's existing "where in the context."

## How J-Lens could improve rule identification (proposer side only)

All four keep `dl/` untouched — the J-Lens output only reorders/seeds the Python miner
(`py/idiom_learn.py`), never the certificate. This mirrors fieldrun's own discipline: off the
forward path, off the gate.

1. **Block-attribution as a "where to look" prior.** For a context+answer, rank blocks by the
   corrected incidence `⟨J[l] d̃_b, U_v⟩` (already computed in `pil`'s `jcorrect_sources`). The top
   blocks tell the miner which sub-computation to try to name and which operand slots plausibly
   matter — narrowing the blind combinatorial search over frames/operands in `learn_gates` /
   `learn_compose`. Every proposal still passes detect + causal-confirm + `equiv.dl`.
2. **Resolve-depth triage of the corpus.** J-Lens eval emits a per-position `resolve_layer` (the
   depth at which the read first locks to the decode). Windows that commit *early* are priors for
   simple deterministic rules (n-gram / select-gate); late-resolving windows flag genuine
   compose / induction depth. A cheap family-prior *before* mining, replacing uniform enumeration.
3. **Seed the parked SAE Phase 4.** `SAE_BRIDGE.md` shelved feature→feature transition circuits
   because "the bottleneck is the model, not the bridge." J-Lens supplies the attribution to decide
   *which* feature-transition edges are worth positing — and its "resolves ~2 layers earlier"
   finding says the informative transitions sit **mid-stack in deep models**. That is the targeting
   signal Phase 4 never had.
4. **Spuriousness flag for the ~40% / ~60% gaps.** If the block J-Lens credits for a decode is not
   the one a named rule's operands correspond to, the rule likely agrees *coincidentally* on the
   finite domain (cf. induction firing "promiscuously," ~40% match on natural text). Useful input to
   the parked confidence-gating line — a cross-check the token-only miner cannot produce.

## The central question: are there *earlier-in-the-layers* rules to find?

Two honest halves.

- **As an existence hint — yes, for deep models.** "Decode resolves ~2 layers earlier" is precisely
  the claim that decode-relevant structure is *present and legible mid-stack, earlier than the
  logit-lens shows*. Read as evidence, it says: in ≥24-layer models there is earlier-layer structure
  to find. If rosetta ever grows a layer axis (Phase 4), J-Lens argues that axis is populated
  earlier than a late-layer prior would assume.

- **Three caveats kill the naïve version:**
  1. **The scale gate lands squarely on rosetta's own models.** pil tested pythia-70m and
     pythia-160m — *the exact models rosetta certifies* — and found J-Lens **null** (≤12 layers).
  2. **Legibility ≠ certifiable rule.** "Resolves earlier" is a direction becoming readable, not a
     clean rule. J-Lens is first-order + context-averaged + `empirical`; every pointer must still
     pass `equiv.dl`, and the survival base-rate for behavioral candidates is low.
  3. **It never promotes a tag.** J-Lens can point; it can never make anything `proved`.

Where rosetta's targets sit relative to the ≥24L / ~400M gate (layer counts from
`models/*/bundle.fieldrun.json`):

| model | arch | layers | vs gate | J-Lens prediction |
|---|---|---|---|---|
| stories260K / 15M / 110M | rope (RMSNorm) | 5 / … | far below | identity ≈ logit-lens — nothing new |
| pythia70m | neox (LayerNorm) | 6 | below | **null** (pil confirmed on this model) |
| pythia160m | neox (LayerNorm) | 12 | below | **null** (pil confirmed on this model) |
| llama32_1b | rope (RMSNorm) | 16 | **untested band** (>400M params, <24L) | unknown — the cheap probe |
| qwen25coder15b | rope (RMSNorm) | 28 | **above** | plausibly a win; pil compute-walled at 1.5B/28L |

So on the toy and small models where most certified work lives today, the J-Lens result predicts
**no earlier-layer win to chase.** The only informative targets are the two deep rope models —
and both are RMSNorm, where the load-bearing γ-conjugation is **exact**.

## The disciplined call (measure-before-build, paid *before* any plumbing)

Same discipline as `SAE_BRIDGE.md`: run the narrow experiment that can kill the idea cheaply,
before touching `py/idiom_learn.py` or the certificate.

**Experiment (read-only, no `.dl` change, no fieldrun change):**
1. Fit + export J-Lens on **`llama32_1b`** first (16L — cheaper than the 28L Qwen, and it sits in
   the untested band between pil's null (≤12L) and win (≥24L)): `fieldrun --jlens-fit … --jlens-export
   llama32_1b.jlens.npz` + `--tensors-export llama32_1b.tensors.npz`. If it shows an earlier resolve,
   escalate to **`qwen25coder15b`** (28L, above the gate; budget for the ~42h fit pil hit).
2. Run `pil`'s `jcorrect_sources` on a rosetta corpus window set to get, per context+answer, the
   **block-attribution ranking** at `λ≈0.25–0.5`.
3. Feed that ranking to a *shim* proposer that orders `idiom_learn.py`'s candidate frames/operands
   by J-Lens block incidence instead of blind enumeration. Certify with the **unchanged** `equiv.dl`.
4. Measure: **how many J-Lens-seeded candidates survive `equiv.dl`, vs the current blind miner, per
   unit of search** — and whether any surviving rule keys on a *mid-stack* block the token-only
   miner would not have reached.

## Success / kill criteria

- **Success (justifies a proposer shim):** on `llama32_1b` or `qwen25coder15b`, J-Lens-seeded
  candidates clear `equiv.dl` at a **higher hit-rate per unit of search** than blind enumeration,
  and/or surface ≥1 certified family the token-only miner did not find — with the block J-Lens
  credited being genuinely earlier/mid-stack. Deliverable: a Python-side proposer ordering, `dl/`
  and the certificate untouched.
- **Kill signal:** J-Lens block-attribution does not improve candidate survival over blind
  enumeration on any target above the gate (or the "earlier resolve" simply doesn't reproduce on
  rosetta's corpora) → keep J-Lens as an *analysis-only* read-out; do not wire it into the miner.
  A useful negative, exactly like SAE_BRIDGE's "don't build the feature-cover plumbing."
- **Null-by-scale (expected on small models):** on stories\*/pythia70m/pythia160m, J-Lens reads as
  identity — do **not** interpret that as "no earlier rules exist," only as "this lens can't see them
  below ~24L." Report it as such; it is not a kill.

## Caveats

- **Empirical, first-order, context-averaged.** `J_l = E[·]` is one linearisation of a nonlinear
  map; a rule it points at can still be spurious. The certificate, not the probe, decides.
- **Arch/scale coverage.** J-Lens is fit only for rope + neox in fieldrun; the win is scale-gated
  and, on the Pythia ladder, depth is confounded with width — so "≥24L" is a *scale* threshold, not
  a depth-isolated one. `llama32_1b` (16L / 1.2B params) is the interesting off-ladder point.
- **γ-conjugation is load-bearing and exact only for RMSNorm.** For LayerNorm (pythia/neox) it omits
  the mean-centering rank-1 term and the `ln_f` bias (approximate). rosetta's deep targets are both
  RMSNorm, so this is not a blocker here — but it caps any neox use.
- **Tag discipline is absolute.** No J-Lens output may ever be tagged better than `empirical`, and
  nothing derived from it becomes `proved` without a clean `equiv.dl` certificate over a stated
  domain. J-Lens changes *which candidates we try*, never *what we can claim*.

## Relation to SAE_BRIDGE and the ecosystem

J-Lens and the SAE bridge are complementary probes for the same missing axis (what happens *inside*
the layers): the SAE bridge gives a **feature basis** (which directions), J-Lens gives **attribution
through depth** (which blocks, how much of the total effect). The natural composition is Phase 4 of
`SAE_BRIDGE.md` — `feature(L)→feature(L+k)` transition circuits — with J-Lens choosing which
transitions to posit. Both remain firmly on the untrusted-proposer side of rosetta's propose-and-check
line. Upstream: `pil` already owns the correction math (`jcorrect_sources`) and the sweep harness;
rosetta consumes its ranking, it does not re-implement the Jacobian. Emitter changes, if ever needed,
go in a `fieldrun` branch + PR — rosetta does not modify `fieldrun`.

## Proposer shim sketch (the `llama32_1b` experiment)

The shim is a **depth/family prior** that reorders `idiom_learn`'s enumeration and splits its
causal-confirmation budget — it never removes a candidate and never touches `dl/`. Three parts joined
by a plain JSONL file (the same decoupling as `fieldrun → rosetta` via `whole.dl`): the Jacobian math
stays in `pil`, rosetta consumes a file.

```
fieldrun (llama32_1b)                 pil (owns the Jacobian math)            rosetta (untrusted proposer)
  --jlens-fit/--jlens-export  ┐
  --tensors-export (U, γ)     ├─►  jcorrect_sources(D,blocks,J,fitted,λ,γ)  ─►  jlens_prior.jsonl
  --source-dump (D, cands)    ┘        per-block incidence toward decode           │
                                                                                   ▼
                                              py/jlens_propose.py ──► family order + budget weights
                                                                                   │
                                              py/idiom_learn.select_cover(..., prior=…)  ◄── consults; blind if absent
                                                                                   │
                                              dl/equiv.dl  ◄── UNCHANGED; sole arbiter of `proved`
```

**Step 0 — the kill gate (no rosetta code yet).** Fit and sweep on `llama32_1b`; only proceed if the
J-corrected read actually resolves earlier at λ≈0.25–0.5:

```bash
fieldrun --bundle models/llama32_1b --recursion-explain --jlens-fit --jlens-export ll.jlens.npz
fieldrun --bundle models/llama32_1b --tensors-export ll.tensors.npz          # U + γ (RMSNorm ⇒ γ exact)
fieldrun --bundle models/llama32_1b --source-dump ll.source.jsonl <corpus>   # per-position D, cands
python pil/experiments/jlens_correction_sweep.py ll.source.jsonl --jlens ll.jlens.npz --tensors ll.tensors.npz
#   → need a row with dresolve < 0 at lam 0.25/0.5. If not: STOP — null-by-scale (16L may be below the gate).
```
*(Confirm the exact `--source-dump` flag name in `fieldrun --help`; it is the dump `pil`'s sweep already
consumes. All three exports are read-only — no fieldrun source change.)*

**Step 1 — producer (pil side, ~25 lines).** Turn the export + dump into the prior JSONL rosetta reads.
Uses only pil's public `fieldrun_io`:

```python
# pil/experiments/jlens_rosetta_prior.py — emit rosetta's J-Lens prior JSONL
import json, numpy as np
from pil.fieldrun_io import load_source_dump, load_jlens, jcorrect_sources, _block_layer

def emit_prior(source_dump, jlens_npz, U, gamma, out, lam=0.5, layernorm=False):
    sb = load_source_dump(source_dump)                       # sb.D (N,nb,dim), sb.blocks, sb.cands
    J, fitted, _ = load_jlens(jlens_npz)
    n_layer = J.shape[0]
    Dc = jcorrect_sources(sb.D, sb.blocks, J, fitted, lam=lam, gamma=gamma, layernorm=layernorm)
    depths = np.array([_block_layer(b, n_layer) if _block_layer(b, n_layer) is not None else -1
                       for b in sb.blocks])                  # per-block inclusion depth for `resolve`
    pos = [r.get("pos") for r in raw_records(source_dump)]   # --source-dump carries pos → robust alignment
    with open(out, "w") as fh:
        for i in range(Dc.shape[0]):
            dec = int(sb.cands[i, 0])                        # the model decode (cands[:,0])
            inc = [[sb.blocks[b], float(Dc[i, b] @ U[dec])]  # J-corrected incidence toward the decode
                   for b in range(len(sb.blocks))]
            inc.sort(key=lambda t: -abs(t[1]))
            uc = U[sb.cands[i]]                              # candidate-restricted, as in the sweep
            resolve = next((l / max(n_layer - 1, 1) for l in range(n_layer)
                            if int(np.argmax(Dc[i, depths <= l].sum(0) @ uc.T)) == 0), 1.0)
            fh.write(json.dumps({"pos": pos[i], "decode": dec, "resolve": resolve,
                                 "block_inc": inc, "n_layer": n_layer}) + "\n")
```

**Step 2 — consumer (rosetta side).** `py/jlens_propose.py` (written) loads the JSONL, aligns each record
to its length-`w` corpus window by position (**aborting loudly** if the prior's `decode` disagrees with
`oracle.decide` on a sample — rosetta's no-vacuous-success rule), and exposes:
- `family_prior(rec)` → ranked families for one context. Validated shape mapping: shallow + early-resolve +
  one block → `ngram`/`select`; attn-dominated block → `induction` (copy); deep multi-mlp block + late
  resolve → `compose`. Ties fall back to today's blind priority, so nothing is dropped.
- `family_order(ctx_prior, insts, idxs)` → a corpus-level Borda order over families + normalized budget
  weights; returns `(None, None)` when no prior covers the set → blind fallback.

**Step 3 — integration hook (minimal, reversible).** One optional arg on `select_cover`, consulted in two
places; absent ⇒ byte-identical to today:

```python
def select_cover(insts, refs, idxs, w, decide_fn, fill=None, hold=0.3, s=str, prior=None):
    order, weight = jlens_propose.family_order(prior, insts, idxs) if prior else (None, None)
    # (a) budget: pass a per-family confirmation cap ∝ weight[fam] into learn_gates/learn_compose/…
    #     so the oracle-heavy causal-confirm loop spends first on the families J-Lens rates likely.
    # (b) admission: iterate `pend` in `order` (not set()) so MDL tries the prior-favored family first.
    #     `order`/`weight` = None → today's blind order + uniform budget. equiv.dl is untouched.
```

The leverage is entirely inside the existing `--confirm=<cap>` budget: J-Lens decides *which* frames and
operand-shapes burn that cap first. Positions still come from `discover.py` flip-counts — J-Lens supplies
the family/depth axis, `discover` the operand-position axis; the two multiply.

**Safety invariant (why this can't corrupt a certificate).** The prior only (i) reorders enumeration and
(ii) allocates a *pre-existing* budget. With an unbounded `--confirm`, output is identical to blind — every
candidate is still learned and MDL-admitted. With a bounded budget, the shim spends it on the highest-prior
candidates *first*, so at worst it matches blind and at best certifies more per unit search. `dl/equiv.dl`,
`circuit.dl`, and every `proved` tag are untouched. A wrong prior costs search, never soundness.

**Measurement (the doc's success criterion).** Run `idiom_learn --certify` on `llama32_1b` twice at a fixed
`--confirm` budget — `prior=None` vs `prior=load_prior("ll.jlens_prior.jsonl")` — and compare: (1) certified
families found, (2) `equiv.dl`-passing candidates per oracle call, (3) whether any newly-certified rule keys
on a context whose J-Lens attribution is genuinely mid-stack. Win ⇒ keep the shim; flat ⇒ J-Lens stays
analysis-only (the kill signal), exactly as `SAE_BRIDGE.md` parked the feature cover.

**Position extension (deferred, needs a fieldrun probe).** To get an operand *position* from J-Lens (not
just a family), an attention block's corrected incidence would be distributed back over source positions by
its attention weights — a direct operand-position prior to sharpen `discover.py`. That needs an attention
export the source dump doesn't carry (a `fieldrun` branch + PR), so it is out of scope for this first
experiment; the family/depth prior above stands alone.

## Step-0 dry run + status (2026-07-10)

The full chain was exercised end-to-end on **real fieldrun+pil artifacts**, first as a plumbing validation on
a small model, then — after the fieldrun unblock landed — as the **real kill gate on the deep models**.

**Blocker (RESOLVED, fieldrun PR #127 `955f3cc`).** The two above-gate bundles are int8-quantized;
`--tensors-export` used to fail on them (`upcast: quantised weight needs its scale; go through mm()`). The fix
dequantizes the unembed in `export_unembed` (rope + neox), so `--tensors-export` now emits `U`/`gamma` f32 on
quantized bundles (int8 dequant cosine 1.0000 vs the fp16 reference); `--source-dump` (recon 1.00) and
`--jlens-fit` already worked on quant. Off the forward path → faithfulness green. This confirms the workspace
rule in practice: the emitter fix went in a **fieldrun branch + PR**, and rosetta/pil consumed the released
binary — no local cross-repo edit.

**Deep-model kill gate — `llama32_1b` (16L, rope/RMSNorm, γ EXACT): NULL.** Real fit (10-prompt corpus, 24
probes, all 15 fittable layers, ~29 min; `‖J_l−I‖_F` decays L0≈251→L14≈93) → source-dump (160 records) →
sweep:

| λ | recon | resolve | **dresolve** |
|---|---|---|---|
| 0.00 | 1.000 | 0.680 | +0.000 (baseline) |
| 0.10 | 0.925 | 0.741 | +0.061 |
| 0.25 | 0.838 | 0.689 | **+0.009** |
| 0.50 | 0.625 | 0.737 | +0.056 |
| 1.00 | 0.175 | 0.783 | +0.102 |

`recon(λ=0)=1.000` ✓ (exact γ-fold). **`dresolve ≥ 0` at every λ** — no earlier-resolve win; the best point
(λ=0.25) is +0.009, statistically flat, never negative. This is consistent with pil's ~24L gate: 16L (the
untested band) lands on the **null side** — the effect needs more depth. On the real llama prior the consumer's
no-signal guard fired (top-family spread 159 compose / 1 induction → `family_order → None`, blind fallback),
validating graceful degradation on a real deep model. *(Caveat: a 10-prompt fit is modest; but `dresolve`
monotonically ≥0 across λ is a fairly robust null.)*

**Deep-model kill gate — `qwen25coder15b` (28L, *above* the gate): MARGINAL / INCONCLUSIVE — criterion not
met.** Real fit (matched: 10 prompts, 24 probes, 27 fitted layers, ~62 min; `‖J_l−I‖_F` shows the predicted
mid-stack plateau L6–L20 ≈ 88–95 between input and output decays) → source-dump (160 records) → sweep:

| λ | recon | resolve | **dresolve** |
|---|---|---|---|
| 0.00 | 1.000 | 0.562 | +0.000 (baseline) |
| 0.10 | 0.950 | 0.554 | **−0.008** |
| 0.25 | 0.925 | 0.632 | +0.070 |
| 0.50 | 0.856 | 0.741 | +0.178 |
| 1.00 | 0.500 | 0.817 | +0.254 |

A hair of earlier-resolve appears at **λ=0.10 (−0.008 ≈ 0.2 layers, within noise)**, but the pre-registered
criterion — `dresolve<0` at **λ≈0.25–0.5** — is **not met** (there it is clearly positive, +0.07/+0.18). This
mirrors pil's own result exactly: a clean win at Qwen2.5-**0.5B**/24L (−0.030 @λ0.5) but **inconclusive at
Qwen2.5-1.5B/28L** (compute-walled ~42h). And decisively for the shim: the derived per-context prior does
**not discriminate** — top-family spread ≈148 compose / ≈12 induction at both λ=0.10 and λ=0.25 → the no-signal
guard fires → **blind fallback**. Even where the sweep hints at an effect, the prior can't order candidates.

**Verdict (pre-committed): do NOT build the `select_cover` hook.** On current evidence no local model produces
a discriminating J-Lens prior: `llama32_1b` (16L) is a clean null, `qwen25coder15b` (28L) is
marginal/inconclusive and still non-discriminating. This is the same disciplined call as `SAE_BRIDGE.md`'s
"don't build the feature-cover plumbing." Tag: **`open`, not a hard null** — the fits are modest (10 prompts;
pil needed ~42h for a clean 1.5B/28L read), so a much larger ≥24L fit *might* surface the λ≈0.25–0.5 win and a
discriminating prior. The consumer + producer stay as validated, dormant machinery, re-runnable the instant a
higher-quality fit justifies it.

**What ran first (pythia70m, 6L f32, neox/LayerNorm — plumbing validation below the gate):** `--jlens-fit` (all 6 layers, tiny corpus)
→ `--jlens-export` (`J[6,512,512]`) → `--source-dump` (120 records, **recon argmax 0.99**) → pil's correction
sweep → producer → `py/jlens_propose`. The sweep gave `recon(λ=0)=0.992` ✓ and **`dresolve = +0.14/+0.20/+0.24`
at λ=0.25/0.5/1.0** — resolves *later*, i.e. the expected **null/negative below the gate** (also confounded by
a throwaway 2-prompt fit). This is not a result about the gate; it validates (a) the chain runs on real
artifacts and (b) graceful degradation.

**What the real (no-signal) prior taught the shim.** On that below-gate prior every context mapped to one
family (block-attribution is meaningless under an under-fit / sub-gate `J`). A prior that can't *discriminate*
between contexts adds no ordering value, so `family_order` now returns `(None, None)` — blind fallback — when
one family is top-choice for ≥90% of contexts (the `min_discrim` guard). Verified: the real pythia70m prior →
blind fallback; a discriminating synthetic prior (ngram/induction/compose) → a real order. Fit knob learned:
the fit's cost is dominated by the corpus size — pass a small `--jlens-corpus` (the default corpus is large
and times out); a few probes/src suffice for a directional read.

**Status (final for this round).** `py/jlens_propose.py` = the consumer, written and validated against real
fieldrun/pil artifacts across **three models** (pythia70m 6L, llama32_1b 16L, qwen25coder15b 28L) — incl.
`pos`-keyed alignment, the loud misalignment abort, and the no-signal guard (which correctly fired on all
three real priors). The producer ran as shown (staged for `pil/experiments/jlens_rosetta_prior.py`). The
fieldrun unblock is **landed** (PR #127). The `select_cover` hook is **not built**, per the pre-committed
verdict above — no local model yields a discriminating prior (16L null, 28L marginal + non-discriminating).
J-Lens stays an analysis-only read-out. **Re-open condition:** a ≥24L, large-corpus (f32-quality) fit that
shows `dresolve<0` at λ≈0.25–0.5 *and* a prior that clears the `min_discrim` guard — then the producer +
hook land unchanged. Everything needed to re-run is in place; only the fit budget was the limit.
