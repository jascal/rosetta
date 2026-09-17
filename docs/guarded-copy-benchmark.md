# Guarded copy on fresh sequence groups

This experiment follows the [initial copy benchmark](induction-benchmark.md). All previous Qwen cases, including its
old test split, are now **discovery data**. The fresh experiment uses seed 1 and excludes every token ID appearing
in that earlier dataset. There is no reuse of old cases as final-test evidence.

## Qwen result

**Proved on the frozen 96-case firing domain:** the selected guard exactly matches the local quantized
Qwen2.5-0.5B-Instruct oracle (`nmiss=0`, `nuncov=0`, no missing or invalid references). It uses suffix length 3 and
at least two agreeing sources. The same rule answered 48/48 covered train cases and 36/36 covered validation cases
correctly before being frozen. See the [run report](../reference/benchmarks/qwen25_05b_guarded_seed1/REPORT.md) and
[exact retained certificate](../reference/benchmarks/qwen25_05b_guarded_seed1/certificate-evidence/check-svsuedkp/certificate.json).

| Artifact | Fresh test correct | Wrong | Abstained | Exact certificate |
|---|---:|---:|---:|---|
| Unguarded copy | 137 | 7 | 0 | fails |
| Selected guard: suffix 3, sources ≥2 | 96 | 0 | 48 | passes on the frozen 96-case firing domain |
| Diagnostic strict guard: suffix 9, sources ≥3 | 16 | 0 | 128 | passes on its frozen 16-case firing domain |
| Lexical baseline | 0 | 0 | 144 | fails full-test coverage |

**Empirical:** the selected guard has 66.7% coverage and 100% precision on this sample. It abstains on all 16
negative controls. Its 1,823-byte `circuits.dl` runs in Soufflé alone. All seven unguarded-copy test errors were on
cases with only one prior sequence copy, which the selected guard excludes by its predeclared structural condition.
That pattern is a result of this experiment, not a universal guarantee. The eight test sequence groups and their
correlated variants do not establish population-wide reliability.

The 144-case full-test certificate for the selected guard fails with `nmiss=0`, `nuncov=48`: abstentions are retained
as uncovered obligations. **Open:** faithful coverage of that residual and generalization beyond this finite domain.

During review, the lexical emitter's last-position aggregate was found to drop shorter contexts in mixed-length
batches. It now binds the instance before aggregation. Original frozen artifacts remain intact; a separately labeled
[post-test baseline correction](../reference/benchmarks/qwen25_05b_guarded_seed1/baseline-correction/README.md) rebuilds
only from training facts and scores the existing references. Corrected train coverage is 72/72 with an exact train
certificate; fresh-test coverage remains 0/144. This correction does not change the guarded artifact or its certificate.

## Hypothesis and admission

The old failures occurred at prefix lengths 3, 6, and 9, so increasing the suffix length alone does not explain them.
The new **open** hypothesis is that repeated, agreeing evidence supports a more reliable copy circuit. Its runtime
guard requires all earlier occurrences of the current suffix to agree on the successor, with enough occurrences to
meet a support threshold. Otherwise it abstains. These are structural conditions on token facts, not a token whitelist
or an oracle confidence threshold.

`dl/copy_guard.dl` implements that rule. The six predeclared candidates combine suffix lengths `{3, 6, 9}` with
minimum source counts `{2, 3}`. `dl/copy_guard_select.dl` admits candidates only when:

- Every development case on which the guard fires matches the recorded oracle.
- Active causal pairs change the predicted token and the oracle follows the change, with no failed active pairs.
- Successful causal pairs occur in at least two sequence groups in **each** of train and validation.

Datalog selects the eligible candidate with greatest development coverage; ties prefer shorter suffixes, then fewer
required sources. If no candidate qualifies, the selected runtime is an explicit abstaining Datalog circuit.
The always-reported diagnostic strict guard uses suffix length 9 and three sources, regardless of admission.

## Dataset and freeze

Each sequence group has a unique 12-token nonce sequence. The dataset includes one, two, and three prior copies of
the sequence, followed by a prefix of length 3, 6, or 9. Each base case is paired with an intervention changing **all**
earlier copies of the target token. The final matching suffix stays unchanged. Datalog checks declared edits against
the staged token facts and checks the causal responses. Each group and all its variants stay together.

The seed-1 runs contain 72 train cases (4 groups), 54 validation cases (3 groups), 144 test cases (8 groups), and
16 negative controls (8 separate groups). Each negative group supplies one no-repeat context and one context with
conflicting successor tokens. These are correlated case variants, not 144 independent test groups. Their tokens are
sampled from `[200, 40000)` after excluding prior tokens; vocabularies are disjoint across all new sequence groups.

Before any oracle query, `protocol.json` records the hypothesis family, dataset hash, selection rule, and source
hashes. After development selection, `dl/guard_domain.dl` computes each candidate's test firing domain using **only
token facts**, with no reference relation. `frozen.json` records those exact instance IDs, artifact hashes, domain
query evidence, and the development-reference hash. Only then are test and negative-control references requested.

This separates two certificate obligations:

- **Full test:** equality on all 144 cases. Deliberate abstentions fail this obligation.
- **Frozen firing domain:** equality on every case where the frozen candidate fires. The domain is determined before
  test references exist, so wrong predictions cannot be removed. Empty firing domains fail certification.

Both obligations are checked by `dl/equiv.dl`. A passing second certificate is **proved only over that stated finite
domain**, relative to the recorded oracle. It does not establish faithfulness on arbitrary future inputs. Structural
coverage and precision are **empirical** measurements. Diagnostic artifacts remain hypotheses unless their exact
certificate passes, even when their corpus precision is high.

## Controls

The [authored copy control](../reference/benchmarks/copy_control_guarded_seed1/REPORT.md) selects suffix length 3 with
two agreeing sources. It answers 96/144 test cases correctly, abstains on 48, and earns an exact certificate on the
96-case frozen firing domain. It abstains on all 16 negative controls. This certifies the authored task on these
contexts; it is not evidence about an LLM.

The [constant-output control](../reference/benchmarks/constant_control_guarded_seed1/REPORT.md) rejects every guard.
Its selected circuit abstains everywhere, and its empty-domain certificate fails. The positive and negative controls
therefore test both successful admission and rejection without vacuous proofs.

## Reproduce

```bash
python3 py/benchmark_guarded_copy.py /tmp/guarded-copy --oracle copy-control
python3 py/benchmark_guarded_copy.py /tmp/guarded-constant --oracle constant-control

# Serve the local bundle in a separate terminal, then query it at build time:
fieldrun --bundle /path/to/bundle --serve 8187
python3 py/benchmark_guarded_copy.py /tmp/guarded-qwen --oracle fieldrun \
  --port 8187 --bundle /path/to/bundle
```

The default prior dataset is `reference/benchmarks/qwen25_05b/dataset.json`; `--prior` and `--seed` are explicit options.
Use a fresh output directory. On this Mac, Soufflé needs `DEVELOPER_DIR=/Library/Developer/CommandLineTools`.
The caller must ensure `--bundle` identifies the model actually served; recorded hashes are not server attestation.

Each report links its checks through `report.json` → `results` → artifact → `certificates`. Replay any retained check:

```bash
python3 py/certificate.py <run-directory>/certificate-evidence/<check-directory>
```

The emitted runtime directory has only `circuits.dl` and `run.dl`. Supply `tok.facts` and run Soufflé; no model, Python,
weights, or original whole-model program participates at runtime. Tests relocate these files into an otherwise empty
directory and exercise agreeing sources, conflicting sources, insufficient support, and changed targets.

## Next evidence

The fixed-artifact replication is implemented in the [generalization experiment](copy-generalization.md), with three
fresh seeds, varied lengths and gaps, and near-match distractors.

Keep this guard fixed and test additional seeds, sequence lengths, source distances, and distractors before widening
its claimed domain. Separately, treat the one-copy failures as discovery data for another detector; any new selection
needs a new untouched test split. No additional rule was tuned using this experiment's final references.
