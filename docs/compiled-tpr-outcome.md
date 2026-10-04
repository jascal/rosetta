# Compiled TPR behind n-gram and idiom layers — outcome

**Pre-registration:** [`compiled-tpr-prereg.md`](compiled-tpr-prereg.md), pushed before any code (`8edc2ca`, 22:18).
- Code was frozen before the real run: rosetta `e22f017` (`py/benchmark_compiled_tpr.py`, sha256 `33b938ae…`, as
  pinned in `protocol.json`) and the pil twin `e2babef` (`experiments/compile_tpr.py`: HF GPT-2 decisions on train,
  the `cert`-objective TPR fit, weighted-fact export) (22:36). `--smoke` runs used non-study data only (600
  contexts, seeds + 1000).
- The driver was edited **after** the run, in review (see *Post-run audit*). The edits change neither selection nor
  verdicts. The run used the `e22f017` driver.
- Artifacts are in [`reference/benchmarks/gpt2_compiled_tpr/`](../reference/benchmarks/gpt2_compiled_tpr/). COPY's
  certificate evidence is omitted for size (see that folder's `EVIDENCE.md`).
- References are from the fieldrun GPT-2 bundle (`4edbcffe…`). Verdicts are `dl/equiv.dl` query results.

## Verdict

**Q1 (a compiled TPR certifies GPT-2 decisions that n-grams and idioms cannot): NO, on both tasks** (overall:
`no`). This matches the pre-registered rule. It is **not** a test of the compiled TPR on both tasks:

- **SVO** is a real test. The TPR layer was consulted and failed zero-error on one counterexample.
- **COPY** never consults the TPR. The frozen idiom fires on every test context, so the TPR domain is empty. COPY
  says nothing about whether a compiled TPR can cover the COPY residual.

### SVO

- Test contexts: 3600 (dropped for an unseen pair: 0). Candidates: 112. Datalog/Python parse parity: True. TPR θ = 175647 (fixed point, 2¹⁶).
- n-gram: 7200 rules learned on train, 0 kept by the dev filter. Idiom: verb whitelist [].
- Layer firing on test: {'ngram': 0, 'idiom': 0, 'tpr': 469}; abstain 3131.
- Certificates (`ndomain`, `nmiss`):
  - ngram: 0, 0, certified = None
  - idiom: 0, 0, certified = None
  - tpr: 469, 1, certified = False
  - composite: 469, 1, certified = False
- Residual (the deployed n-gram + idiom do not decide GPT-2's answer): 3600. Split (`R_in_sentence` extends the
  pre-registered §3 split, which names only `R_obj` and `R_other`): R_in_sentence 2948 (TPR decides 468, agrees 468); R_obj 424 (TPR decides 1, agrees 0); R_other 228 (TPR decides 0, agrees 0).
- **Additional certified coverage: 0** (Q1 fails).

### COPY

- Test contexts: 3600 (dropped for an unseen pair: 0). Candidates: 123. Datalog/Python parse parity: True. TPR θ = 376729 (fixed point, 2¹⁶).
- n-gram: 7185 rules learned on train, 5 kept by the dev filter. Idiom: frozen copy guard (unchanged).
- Layer firing on test: {'ngram': 0, 'idiom': 3600, 'tpr': 0}; abstain 0.
- Certificates (`ndomain`, `nmiss`):
  - ngram: 0, 0, certified = None
  - idiom: 3600, 122, certified = False
  - tpr: 0, 0, certified = None
  - composite: 3600, 122, certified = False
- Residual (the deployed n-gram + idiom do not decide GPT-2's answer): 122. Split: R_ctx 121 (TPR decides 0, agrees 0); R_out 1 (TPR decides 0, agrees 0).
- **Additional certified coverage: 0** (Q1 fails).

### Why each task fails

- **SVO.**
  - The idiom's verb whitelist came out **empty**: every verb has dev contexts where GPT-2 does not output the
    subject. So (a) + (b) decide nothing, and the residual is the whole test set.
  - The compiled TPR fires on 469 contexts and matches GPT-2 on 468. Its certificate fails on one counterexample:
    GPT-2 says ` captain`, the TPR says ` butcher`.
  - The 468 agreements are all `R_in_sentence` (subject) decisions: the ones a copy-subject idiom would make. The TPR
    fires on just 1 object-choice context, which is its one error, and on no non-sentence-token context.
- **COPY.**
  - The frozen copy guard fires on all 3,600 test contexts and is wrong on 122, so its certificate fails.
  - Those 122 are almost all `near_match8` cases (120), where GPT-2 follows the decoy.
  - Because the idiom fires everywhere, the TPR layer is never consulted. Q1 fails structurally, not on the TPR's
    merits.

## Post-hoc diagnostic (not pre-registered; `empirical`)

`py/diagnose_compiled_tpr.py` asks, with no re-selection: on the pre-registered residual, what does the compiled
TPR decide **alone**, with its dev-selected θ (guarded) and without θ (unguarded)?

| task | residual category | n | guarded fires | guarded agrees | unguarded agrees |
|---|---|---:|---:|---:|---:|
| SVO | R_in_sentence | 2948 | 468 | 468 | 2853 |
| SVO | R_obj | 424 | 1 | 0 | 228 |
| SVO | R_other | 228 | 0 | 0 | 52 |
| COPY | R_ctx/near_match8 | 120 | 0 | 0 | 10 |
| COPY | R_ctx/plain | 1 | 0 | 0 | 0 |
| COPY | R_out/plain | 1 | 0 | 0 | 0 |

## Post-run audit (review follow-up; re-selects nothing)

`py/audit_compiled_tpr.py` → [`audit.json`](../reference/benchmarks/gpt2_compiled_tpr/audit.json). It reads back the
stored datasets, circuits, frozen domains, θ and references, and recomputes what the frozen run left out. Each check
fails the script if it disagrees with `report.json`.

- **Full mismatch lists.** `report.json` kept only the first 50 per certificate. `audit.json` lists all of them,
  re-derived by `dl/equiv.dl` on the frozen domains, with identical `ndomain`/`nmiss`/`certified`.
  - COPY idiom: 122 (120 `near_match8`, 2 `plain`).
  - SVO TPR: 1.
- **HF/fieldrun disagreements (pre-reg §1, unmet in the run).** The run did not report these. Greedy next token,
  HF GPT-2 vs the fieldrun bundle `4edbcffe…` references:

  | task | train | dev | test |
  |---|---:|---:|---:|
  | SVO | 5 / 7200 | 0 / 1200 | 2 / 3600 |
  | COPY | 0 / 7200 | 0 / 1200 | 1 / 3600 |

  pil trained on HF decisions, so 5 SVO training targets differ from fieldrun. Certificates use fieldrun only.
- **Parse parity on the full train split.** The run checked `train[:500]` and recorded the result without gating.
  The audit checks all train contexts: SVO 7200 contexts and 21,600 pairs, COPY 7200 contexts and 220,992 pairs.
  Datalog and Python agree exactly on both.
- **Unguarded θ literal.** The unguarded program used `tp_theta(-2**60)`. That fits the 64-bit Soufflé used here,
  but it would wrap on a 32-bit `number`. θ selection reads `tp_margin`, not `tp_decide`, so the literal had no
  effect. Re-selecting with `tp_theta(0)` gives the same θ on both tasks (175647, 376729). The post-hoc diagnostic
  re-run with `tp_theta(0)` is byte-identical.

Driver edits after the run (for re-runs, not this result):

- `--tokenizer` is now an argument (it was a machine-local path). The diagnostic takes it as `argv[2]`.
- The unguarded program uses `tp_theta(0)`.
- Every mismatch is stored.
- Parse parity covers full train and aborts the run on mismatch.
- `report.json` gets a `tag_note`: the aggregates are `empirical`, but each `certified`/`nmiss` is an `equiv.dl`
  query result.

`tests/test_compiled_tpr.py` locks the protocol pieces on synthetic data:

- the role parse, Python vs Datalog, including rejection of off-template contexts;
- COPY role numbering;
- the whitelist rule;
- router precedence (n-gram, then idiom, then TPR).

## Reading (interpretation)

1. **Where the compiled TPR is confident, it reproduces the idiom, not the residual.**
   - Guarded, it certifies nothing GPT-2-specific. Its confident decisions are subject copies.
   - On the decisions no crisp rule predicts, the unguarded TPR is right sometimes: object choice 228/424, other
     tokens 52/228, decoy-following 10/120. But its margins there never clear a zero-error θ.
   - The structure it learned carries some signal about the residual, but not with margins that survive the guard.
2. **On SVO, the answer is no at this scale.** A learned, compiled TPR adds **no** certified coverage beyond
   n-grams and idioms. On COPY the question was not asked (see *Verdict*). The only COPY evidence on the TPR is the
   post-hoc table: unguarded 10/120 on decoy-following contexts, guarded 0.
   - What it certifies empirically is what an idiom already expresses.
   - The idiom-defying decisions are where it is least confident.
   - This matches pil #136: certification needs margins, and the residual is exactly the low-margin part of GPT-2's
     behaviour.
3. **Two design weaknesses are visible, and both are reported rather than repaired.**
   - (i) A verb-level, all-correct whitelist is too strict for an idiom that is right about 82% of the time (GPT-2
     outputs the subject on 981/1200 dev and 2948/3600 test contexts; `audit.json` `subject_copy_rate`), so no idiom
     coverage survived.
   - (ii) The frozen copy guard pre-empts every COPY context, so a layer-order design that lets the TPR veto low-margin
     idiom answers was not tested.
   - Either change would be a new pre-registration.
4. **What did work:**
   - a learned TPR compiled to a **Soufflé-only** program: weighted input facts, Datalog role parsing (parse parity
     with the training twin), and no weights or unembedding at runtime;
   - dev-selected guards and test domains frozen before references;
   - `equiv.dl` verdicts on every layer, including a replayable SVO counterexample.

**Scope:** GPT-2 small; two templated tasks; one TPR fit (d_F = 32, seed 0); certificates are proofs over these
finite test domains only.

