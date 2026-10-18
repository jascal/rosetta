# Handoff: native-trace run B is built and tested; collection has not started

Read `AGENTS.md`, [`native-trace-outcome.md`](native-trace-outcome.md),
[`native-trace-diagnosis.md`](native-trace-diagnosis.md) and
[`native-trace-prereg2.md`](native-trace-prereg2.md) first. The experiment this
branch designs has **never touched the model**. Collection is the only
unfinished step, and running it is a deliberate decision the user has not yet
made.

Branch `research/native-trace-prereg2`, pushed, head `cfce4d8`, eight commits
ahead of `master` (`4e38fe9`). Working tree clean except an unrelated untracked
`docs/native-trace-handoff.md` (run A's operational note; not evidence, not
committed).

## What run B is

Run A (`master`, `4e38fe9`) reported zero incremental certified coverage for a
native-neuron prior against behavior-only and shuffled-trace controls. The
post-hoc diagnosis in `native-trace-diagnosis.md` shows the failure had a
specific, fixable cause rather than saying anything about native neurons:
`discovery.json` held 288 rows and **zero** negative controls, so the
feature-selection negative class was exactly the 7 instances where the strict
suffix-3 rule errs on a positive copy, while every evaluation split carries 18
real negative controls. The threshold was a midpoint between two classes drawn
only from positive copies, so on a negative control it extrapolated off-manifold
and labelled **18/18** train negatives trace-good. A uniform signature made the
trace term collapse to a firing count, which made trace and shuffled scores
elementwise identical.

Run B changes exactly two things, both registered before any collection:

1. **Select on train, sign on a disjoint split**, with the selection negative
   class containing real negative controls (≥8 per class, or no vote and the
   shortfall reported).
2. **A signed-margin vote weighted by the SwiGLU gate** (`|gate| / mean|gate|`
   on the select split), replacing run A's binary majority — mirroring
   fieldrun PR #145's `w_router · act`. All of it in Datalog.

Plus a **non-degeneracy gate**: before the trace term ranks anything, the
signature must be non-uniform and must not cover every negative control. Failure
stops the run and reports it as degenerate — no fallback to behavior-only, no
threshold adjustment. Run A lacked this precondition and only noticed the
degeneracy afterwards.

## What is done and verified

| | |
|---|---|
| Prereg, sealed in `protocol.json` | `8135116` |
| `dl/native_trace_features2.dl`, `dl/native_trace_select2.dl` | `c50bae0` |
| `py/run_native_trace2.py` — inputs, features, select | `53cfa1f`, `883a2bf` |
| Driver completed: collect, freeze, evaluate | `cd13a67` |
| `py/audit_native_trace2.py` — model-free replay audit | `c0344cb` |
| Oracle attestation + refusal paths | `a15f00c` |
| `collect` plumbing against a stub oracle | `cfce4d8` |

Suite: **224 passing** (`.venv/bin/pytest`; the repo has no CI, so run it
locally). Run A's `frozen.json` still hashes to `451c1650…e327e` and its
retained evidence replays model-free with all 25 verdicts preserved.

Committed run-B inputs at `reference/benchmarks/qwen25_native_trace_b/`:
seeds 201/202/203 → select/sign/test, 114 rows each with 18 negative controls,
6,418 excluded tokens (all six historical Qwen datasets **plus run A's own
frozen inputs**, so no run-B input shares an embedding with the completed run).
`dataset.json` `80bce2e3…0634`, `hypotheses.dl` `6013d029…bbaaa` (1,344
candidates), prereg seal `e4458f90…3be5b`.

## The decision that is still open

**The gate weighting is the one judgement call in the design.** It was reasoned
from fieldrun PR #145 and implemented with tests, but it was never reviewed
against evidence. If it should be dropped in favour of a plain signed-margin
sum, amend `native-trace-prereg2.md` **first**, then re-seal:

```bash
.venv/bin/python py/run_native_trace2.py inputs --write reference/benchmarks/qwen25_native_trace_b
```

That regenerates `protocol.json` with the new prereg hash (it rewrites
`dataset.json` and `hypotheses.dl` identically — verify `git status` shows only
`protocol.json` changed). `input_gate` refuses to proceed if the seal does not
match, which is the point.

## Running it

Requires the pinned oracle, present at `/tmp/rosetta-native-qwen-f32/`:
binary `fieldrun-0948cbe` (`54de8fc8…67ea62`), bundle stem `model`
(`3a2fb8e5…f77f8d` / `5c013a3a…aaa12`). **Run B introduces no new oracle** — it
is the same validated binary and bundle as run A, so run A's preflight
(`reference/benchmarks/native_trace_preflight/`) *is* run B's gate, attested by
hash rather than re-run. `model_gate` refuses on any mismatch.

```bash
B=/tmp/rosetta-native-qwen-f32
S=reference/benchmarks/qwen25_native_trace_b   # or a fresh dir for a clean run
.venv/bin/python py/run_native_trace2.py collect $S --binary $B/fieldrun-0948cbe --bundle $B/model
.venv/bin/python py/run_native_trace2.py features $S
.venv/bin/python py/run_native_trace2.py collect $S --binary $B/fieldrun-0948cbe --bundle $B/model   # sign
.venv/bin/python py/run_native_trace2.py select  $S
.venv/bin/python py/run_native_trace2.py freeze  $S
.venv/bin/python py/run_native_trace2.py final   $S --binary $B/fieldrun-0948cbe --bundle $B/model
.venv/bin/python py/run_native_trace2.py evaluate $S
.venv/bin/python py/audit_native_trace2.py $S --write
```

`select` prints the non-degeneracy result and, if the gate fires, writes
`selection.json` with `proposed` empty and **does not rank**. That is a
completed, publishable null — do not fall back or substitute a rule.

To collect in the committed directory, copy it first (`cp -r`), since the
existing one holds the sealed inputs that collection will join.

## Things that will bite you

- **Run A is frozen and immutable.** Its inputs, refs, domains, rules and
  `frozen.json` (`451c1650…e327e`) must not change. Only `docs/` is outside the
  manifest; `protocol.json` seals `dataset.json`, `hypotheses.dl` and the prereg.
- **This is a plausible-positive.** The mechanism and the failure were diagnosed
  before run B existed, so the design was not fixed blind with respect to the
  outcome. Keep the shuffled control: it is what catches a prior that is uniform
  for a different reason. A fix that only works because it memorised seeds
  201–203 is the failure mode this experiment exists to detect.
- **Reporting a second null is success.** Do not alter the protocol after test
  answers to change the verdict.
- **The stub oracle is not the model.** `tests/test_native_trace2_collect.py`
  validates plumbing only; real response shapes have not been seen by this code
  path yet. One real request would confirm them cheaply.
- **Claim tags** are `proved` / `empirical` / `open` with domain limits.
  Domain-exhaustive finite checks are proved *over that domain*, never
  unconditionally. Do not promote a tag without the artifact behind it.
- No modifications to the sibling fieldrun checkout. No subagents.

## Bugs the tests caught that are worth not reintroducing

- `n_negative_good` unified its total and its count in one variable, so it
  emitted a row only when `good == total`. The run-A-shaped fixture hid it.
- `features2`'s completeness check accepted a sample missing one unit's
  observation, silently dropping that row from a class mean. Run A has the same
  weakness; run B requires every `(sample, unit)` pair.
- `collect` concatenated **every** returned position into one feature vector
  instead of the last. Against a 3-token context that yielded a 9-long vector
  where 3 was correct — position-mixed data that would have scored as a
  plausible number rather than an error. Fixed in run B; run A's driver still
  has the latent shape, harmless only because real responses carry one position.
- Souffle 2.5 has no `abs()`, refuses implicit `number`×`float` arithmetic, and
  rejects an anonymous `_` inside a negated atom within an aggregate.

## Not done

The experiment itself: collection, equivalence checks on real data, the replay
audit over run-B evidence, and the outcome write-up. No run-B PR is open — the
branch is pushed but unpublished, so the preregistration is branch-timestamped
rather than publicly registered.