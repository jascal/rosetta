# Copy/induction benchmark and isolated holdout evaluation

The benchmark compares a train-only lexical n-gram cover with a standalone copy circuit, then checks both against
an untouched test split. **Empirical:** on the local quantized Qwen2.5-0.5B-Instruct bundle, the copy hypothesis matches
22/24 test argmax predictions. It failed admission on train/validation, so the selected artifact retains the baseline
and abstains on all 24 test cases. **Open:** a certified Qwen copy circuit covering this entire test domain.

## Recorded results

All three runs use seed 0 and the same generated contexts. Each has 24 train cases (4 sequence groups), 12 validation
cases (2 groups), 24 test cases (4 groups), and 4 no-repeat negative contexts. Paired cases are correlated; these are
24 cases, not 24 independent sequence groups.

| Oracle | Copy train | Copy validation | Copy test | Copy admitted? | Selected test behavior |
|---|---:|---:|---:|---|---|
| [Authored copy control](../reference/benchmarks/copy_control/REPORT.md) | 24/24 | 12/12 | 24/24 | yes, suffix length 1 | 24 correct, exact certificate passes |
| [Constant-output control](../reference/benchmarks/constant_control/REPORT.md) | 0/24 | 0/12 | 0/24 | no | 24 abstentions |
| [Qwen2.5-0.5B-Instruct](../reference/benchmarks/qwen25_05b/REPORT.md) | 21/24 | 11/12 | 22/24 | no | 24 abstentions |

The lexical baseline answers zero test cases in each run: sequence groups deliberately have disjoint token
vocabularies. This tests transfer to unseen token identities; it is not a representative natural-language comparison.
The later [guarded-copy experiment](guarded-copy-benchmark.md) found and fixed a lexical-emitter bug affecting
shorter contexts in mixed-length batches. Each original run now has a separate `baseline-correction/` report:
the corrected baseline certifies 24/24 train cases and still abstains on all test cases. Original frozen artifacts
and copy certificates are preserved; these corrections are explicitly post-test re-evaluations of cached references.
All artifacts abstain on all four no-repeat contexts. This small negative set does not establish broad off-domain
reliability. Undefined precision (zero answers) is recorded as `null`, never 100%.

**Proved, finite domain:** the admitted copy-control circuit agrees exactly with the authored task references on all
24 stated test contexts (`nmiss=0`, `nuncov=0`, complete references). This proves the control task on those contexts,
not an LLM capability or a theorem over arbitrary sequences. Qwen references came from the resident fieldrun server;
the report retains hashes of the local bundle. This is Qwen2.5, not the recently discussed Qwen n-gram architecture.
Neither the raw Qwen copy hypothesis nor its abstaining selected baseline passes a full-domain equivalence certificate.

## Experimental separation

1. Generate length-12 nonce sequences from token IDs in `[200, 40000)`, with disjoint vocabulary per sequence group.
   For each sequence `S`, use `S + S[:p]` at fixed positions `p = 3, 6, 9`. Generate an intervention that replaces only
   the earlier copied-from token `S[p]`, preserving the final suffix. Keep each sequence and all its interventions
   in one split. The positive control's targets come from this recipe, independently of the detector.
2. Query the oracle only on train and validation. `dl/copy_select.dl` admits the shortest of suffix lengths 1–3 only
   with zero prediction failures and zero causal-pair failures, including successful pairs in both splits.
   This benchmark admits a whole copy hypothesis or rejects it; it does not discover a guard for a faithful subset.
3. `dl/ngram_train.dl` wraps `dl/ngram.dl` to learn the baseline exclusively from train facts. Python serializes the
   Datalog-chosen suffix facts. The copy rule matches the current suffix to its most recent earlier occurrence and
   returns that occurrence's successor. Length 1 is the predeclared diagnostic hypothesis if admission rejects all.
4. Write the candidates and `frozen.json` with dataset and artifact hashes **before requesting any test reference**.
   The selected artifact is the admitted copy rule, or the baseline on rejection. No test result changes that choice.
5. `dl/holdout.dl` checks group/context leakage, reference completeness, prediction ambiguity, and counts all outcomes.
   `dl/equiv.dl` independently checks the **entire** test domain. No failed cases are removed to obtain a certificate.

Admission and evaluation audits also emit a relation named `certified`; those verdicts mean input integrity for their
specific query. Only the report's `test_certificate` is an equivalence verdict. All verifier sources, exact staged
facts, output relations, tool version and hashes are retained under each run's `certificate-evidence/`.

The control's emitted copy program is 972 bytes versus 3,185 bytes for its 24-fact lexical baseline (Datalog
scaffolding included). Those are artifact sizes, not a model compression ratio. Runtime needs only Soufflé,
`circuits.dl`, `run.dl`, and `tok.facts`; no weights, Python, fieldrun, or `whole.dl` are needed.

## Reproduce and replay

Use fresh output directories; the runner refuses to overwrite a previous experiment.

```bash
python3 py/benchmark_induction.py /tmp/copy-control --oracle copy-control
python3 py/benchmark_induction.py /tmp/constant-control --oracle constant-control

# In one terminal, start the build-time oracle for the chosen local bundle:
fieldrun --bundle /path/to/bundle --serve 8187
# In another:
python3 py/benchmark_induction.py /tmp/qwen-copy --oracle fieldrun \
  --port 8187 --bundle /path/to/bundle
```

On this Mac, Soufflé preprocessing required `DEVELOPER_DIR=/Library/Developer/CommandLineTools` in the environment.
The caller must point `--bundle` at the bundle actually served on `--port`; its hash records provenance, not server
attestation. The synthetic controls need neither a server nor model files.

The exact evidence path is in `report.json` → `results` → `selected` → `test_certificate` → `evidence`.
Replay needs only Python and Soufflé:

```bash
python3 py/certificate.py reference/benchmarks/copy_control/certificate-evidence/<check-directory>
```

An unsuccessful hypothesis is a valid experimental result: the benchmark command finishes normally, while certificate
replay exits nonzero for a failed equivalence verdict. Moving the full evidence directory preserves replayability.

## Package holdouts

For gated `pack.build` cover builds, `py/pack/holdout.py` now partitions the corpus **before** extraction, retaining
normalized-line groups, seed and file hashes. The cover receives only `evaluation/train.txt`; evaluation tokenizes
held-out lines independently and rejects missing or changed inputs. The gate fails when required evidence is absent.
This evaluates the cover tier against corpus continuations, not model argmax. Retrieval and curated answers need
their own evaluator before making a whole-package claim. The legacy `holdout_score.py` and `idiom_learn --select`
use their split for family selection: their scores are exploratory validation, not an untouched final test.

## Next experiment

Implemented as the [guarded-copy experiment](guarded-copy-benchmark.md), with fresh sequence groups and a firing
domain frozen before test references: Qwen's selected guard certifies 96/96 covered test cases and abstains on 48.

Treat the Qwen mismatches as discovery data for a **new** experiment: propose a Datalog guard or richer copy detector,
choose it on development data, then evaluate it on fresh sequence groups and a new frozen test split. Retesting an
adapted rule on these same 24 cases would be validation, not new evidence of generalization.
