# Matched MiMo-7B guarded copy, int8 (2026-09-26)

**Empirical, finite-domain comparison.** Both pinned tokenizer files are byte
identical. Both runs used seed 7, the same generated dataset, 126 development
cases, 144 untouched test cases, and 16 off-domain controls. Base int8 matched
the official FP32 reference on 32/32 earlier parity contexts. RL int8 matched
29/32, with the three disagreements on graph contexts. The result is therefore
a useful matched behavioral comparison, with an RL precision caveat.

| Checkpoint | Selected guard | Raw copy correct | Raw copy wrong | Strict guard correct / fired | Strict guard certificate |
|---|---:|---:|---:|---:|---|
| MiMo-7B-Base int8 | none | 86/144 | 58 | 12/16 | failed (4 mismatches) |
| MiMo-7B-RL int8 | none | 122/144 | 22 | 14/16 | failed (2 mismatches) |

By target case type, Base answered 46/72 unedited repeats and 40/72 causal
interventions correctly. RL answered 63/72 repeats and 59/72 interventions
correctly. On paired cases, both bundles matched 82; Base alone matched 4; RL
alone matched 40; both missed 18. Thus RL int8 is more accurate on this frozen
suite, including interventions. This does not establish why training changed
the outputs, and the RL int8 graph parity misses remain a precision limitation.

Datalog selected no guard for either checkpoint from development data. Each
selected artifact is a standalone abstaining circuit with zero test coverage;
neither earns an equivalence certificate. The train-only n-gram baseline also
has zero test coverage in both runs. The strict guard is diagnostic only and
fails exact equivalence on its frozen firing domain for both models.

Full reports and replayable evidence:

- [Base int8](guarded_copy_base_int8/REPORT.md)
- [RL int8](guarded_copy_rl_int8/REPORT.md)

The full checkpoint pair at f16 is the next precision-matched confirmation.

## Earlier Rosetta model references

On the earlier Qwen2.5-0.5B-Instruct seed-1 run, the selected guard fired on
96/144 cases and was exact on that frozen domain; the raw-copy diagnostic was
137/144. The current MiMo seed-7 runs select no guard, and raw-copy scores are
86/144 (Base) and 122/144 (RL). This suggests a substantial difference in
these synthetic copy tasks, with RL closer to Qwen than Base. The runs use
different seeds, model tokenizers, and token IDs, so this is descriptive
cross-model context, not a controlled checkpoint comparison. See the
[Qwen report](../../reference/benchmarks/qwen25_05b_guarded_seed1/REPORT.md).

For recursive graph probes, previous Pythia 70m and Llama 3.2 1b runs on the
40-case chain domain were close to chance and had zero successful edge-edit
pairs at every depth. Qwen 2.5 Coder 1.5b had 21/40 mismatches and paired
successes of 2/4, 1/4, then zero at depths 3–5. On the 200-case branched
domain, Base int8 has 61 model/rule disagreements and paired edge-edit success
of 18/20, 8/20, 5/20, 5/20, and 7/20 by depth. RL int8 has 94 disagreements
and 3/20, 2/20, 1/20, 0/20, and 0/20. Earlier Qwen2.5 Coder 1.5b on this
domain used a few-shot prompt: it had 74 disagreements and 13/20, 6/20, 3/20,
2/20, 2/20 paired successes. This makes Base's apparent graph advantage and
RL's weak edge-edit response visible, though the RL f16 run is still underway
because int8 has three graph parity flips on the 32-context suite. The Qwen
prompt differs, and the model families have different tokenizers. See
[`RECURSION_LADDER.md`](../../docs/RECURSION_LADDER.md).

No retained previous `probe_families.py` results were found for a like-for-like
family comparison. The current 10-trial Base/RL family screen is in
[`families/README.md`](families/README.md).
