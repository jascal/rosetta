# MiMo-7B full-checkpoint preflight (2026-09-26)

**Empirical, finite-domain parity gate.** This is a parity and
memory-fit preflight, not a circuit certificate or a claim about the model's
copy/reasoning behavior.

Checkpoint: `XiaomiMiMo/MiMo-7B-Base` at Hugging Face revision
`c72df4586cb8bdeebd65f36929cd3385a6566fbe` (15.7 GB of BF16 weights).
The official reference used Torch 2.11.0 and Transformers 5.10.2, with the
checkpoint's own `modeling_mimo.py`; `reference_probe.py` records its top two
logits. Fieldrun binary SHA256:
`8f5f2b72bc383d7489cf04ef4419e9fe0eea29473c443d8a7f5367cf641025cb`.
The checkpoint config and tokenizer SHA256 values are
`af0221e5da6d3458ba1463ff2901a1e504b8f6328a66b1bbe022ac0106e5dfcc`
and `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`.

The input is [`ids.json`](ids.json): one 34-token stream tokenized from a fixed
paragraph about capitals and colored balls. For each position 16–23, both
programs receive exactly the prior 16 IDs. This is a smoke domain; it does not
cover the planned guarded-copy or graph probes.

The expanded matched suite is [`matched_cases.json`](matched_cases.json), a
frozen 32-context set with 12 copy cases, 8 graph cases, 8 family probes, and
4 ordinary prompts. Its SHA256 is
`c6e10a434360004878a7c1c02a8203886112edbd4f0bfedc97bd497a027996fc`.

| Run | Agreement with official BF16 | Official FP16 | Official FP32 |
|---|---:|---:|---:|
| Official BF16 | 8/8 | 3/8 | 4/8 |
| Official FP16 | 3/8 | 8/8 | 6/8 |
| Fieldrun f16 | 4/8 | 6/8 | 8/8 |
| Fieldrun int8 | 4/8 | 6/8 | 8/8 |

On the expanded 32-context suite, fieldrun f16 matches the official FP32
reference **32/32 for MiMo-7B-Base and 32/32 for MiMo-7B-RL**. The Base int8
bundle also matches 32/32; the RL int8 bundle matches 29/32, with all three
errors in graph prompts. The RL f16 bundle recovers all three. These are
finite-domain empirical gates, not proofs for arbitrary contexts.

The complete comparisons are [`base_parity_summary.json`](base_parity_summary.json),
[`rl_parity_summary.json`](rl_parity_summary.json), and
[`rl_f16_parity_summary.json`](rl_f16_parity_summary.json).

## Exploratory RL int8 guarded copy

The [guarded-copy run](guarded_copy_rl_int8/REPORT.md) used the RL int8 bundle,
seed 7, 126 development cases, 144 untouched test cases, and 16 off-domain
controls. Datalog admitted **no guard** on development data. The selected
standalone circuit therefore abstained on all 144 test cases; its empty firing
domain does not earn an equivalence certificate. The train-only n-gram baseline
also abstained on the full test set. The diagnostic raw-copy circuit matched
the model on 122/144 test cases and failed on 22. The strict guard fired on
16/144, matched 14, and failed its exact finite-domain certificate on two.
These are **empirical** model measurements; no faithful copy circuit was
certified in this experiment.

The complete [report](guarded_copy_rl_int8/report.json),
[frozen domains](guarded_copy_rl_int8/frozen.json),
[references](guarded_copy_rl_int8/references.json), and
[replayable Soufflé evidence](guarded_copy_rl_int8/certificate-evidence) are
retained. The original run exposed two harness failures: Soufflé could not
compile an empty guard through unreachable copy rules, and the n-gram baseline
emitted tuples above its supported arity. The empty selection now emits an
explicit Datalog abstention; the baseline serializes variable-length suffix
facts and matches them in Datalog. This int8 result is exploratory because
RL int8 missed three graph contexts in the separate 32-context parity gate.

The matched Base/RL int8 comparison is summarized in
[`matched_guarded_copy_int8.md`](matched_guarded_copy_int8.md). On the same 144
test cases, Base raw copy matched 86 model answers and RL matched 122. RL was
right on 40 cases where Base was wrong; Base was right on 4 where RL was wrong.
Both selected no guard, and both strict diagnostic guards failed their exact
finite-domain certificates. The behavioral difference is empirical; an f16
matched run is still needed to confirm the RL quantization condition.

The [matched graph recursion run](mimo_graph_recursion.md) finds 61/200
Base int8 answers and 94/200 RL int8 answers disagreeing with the recursive
reachability rule. Paired edge-edit success falls from Base's 18/20 at depth 1
to 7/20 at depth 5; RL falls from 3/20 to 0/20. In a stratified four-pair per
depth follow-up, RL f16 disagrees on 19/40 cases versus 18/40 for RL int8, and
their answers flip on one case. The Base/RL graph difference persists in this
small precision check.

The matched [copy-distance and distractor stress test](copy_stress_mimo_comparison.md)
shows a modest RL gain in raw copy (130/288 vs 121/288) and in correct answers
on the fixed diagnostic guard's firing domain (35/48 vs 27/48). Neither
checkpoint selected a guard, and both strict diagnostic guards failed exact
equivalence. Near-match distractors reduce raw success to 2/72 for Base and
0/72 for RL. The earlier Qwen2.5 fixed guard did much better on its own similar
stress protocol, but its tokenizer and selected circuit differ; the contrast
is descriptive.

The [expanded guard search](rule_search_expansion.md) varied suffix length
1–12 and support 1–4 on a new paired Base/RL set. It selected no candidate for
either model. The fixed suffix-9/three-source diagnostic guard certified on
Base's 16 fired test cases and failed on RL's with one mismatch; both had
development failures, so neither was admitted. This search only expanded
parameter values, not the conceptual rule family. The original six-candidate
benchmark remains the default; use `--expanded-search` to reproduce the wider
grid.

As a search-capacity control, the [deterministic copy oracle](rule_search_control.md)
let the original family certify 96/144 held-out positive cases and the expanded
family certify 144/144; both abstained on all 16 negative controls. This
confirms the Datalog mechanism can represent and certify the authored copy
rule. It does not explain every MiMo mismatch, but it makes a pure
“Datalog cannot express copying” explanation unlikely for this circuit class.

The [successor-plurality search](successor_plurality_search.md) tests a new
source-aggregation concept on a fresh 102-case split corpus with one exact
12-token near-match source. It is exact on the deterministic control, but
neither MiMo model admits a rule. The fixed diagnostic plurality guard answers
on eight additional near-match cases and gets only 3/8 right there for Base
and RL; its total errors rise relative to consensus. This suggests the models
do not consistently follow a majority-source rule, even though Datalog
expresses and certifies it on the authored control.

The [post-hoc Base-to-RL transfer probe](guard_transfer_base_to_rl/REPORT.md) finds the
Base development near-miss (suffix 7, support 3) exact on Base's 16 fired test
cases but with one RL mismatch. The [margin audit](copy_behavior_stability.md)
shows lower median top-1 margins for task-wrong copy answers in both models,
with overlapping ranges. Margins are diagnostic; they do not become runtime
guard inputs.

The fieldrun f16 and int8 predictions are identical on all eight windows and
match the official model loaded with FP32 arithmetic on all eight. The official
model changes top-1 on five of eight windows between BF16 (the published weight
dtype) and FP16. Fieldrun's f16 storage is upcast for f32 forward arithmetic;
the FP32 reference uses the released BF16 weights upcast to f32, rather than
an unreleased f32 checkpoint. It is therefore the closer numerical comparison. The full
checkpoint mismatch on this domain is explained by arithmetic precision, with
no evidence here of a MiMo conversion or forward bug. It remains possible for
other contexts to reveal one. Neither bundle is faithful to the official BF16
execution on this eight-window domain; do not promote it as that oracle.

As a separate structural check, a two-layer slice of the published weights was
run in f32 in both systems: 8/8 top-1 agreement. The slice uses the original
embedding, first two layers, final norm, and output head; it does not establish
full-depth parity. Its raw predictions are in
[`two-layer-reference-fp32.json`](two-layer-reference-fp32.json) and
[`two-layer-fieldrun-f32.txt`](two-layer-fieldrun-f32.txt).

Memory fit is measured: the f32 bundle was converted but fieldrun was killed
with exit code 137 while loading its ~30.5 GB model on this 14 GiB RAM host.
The f16 bundle loaded but used about 9 GiB swap while scoring. The int8 bundle
loaded and completed the eight-position score. The official reference ran with
7 GiB CPU memory and disk offload. Full f32 parity therefore needs a larger
host or a fieldrun loading path with bounded memory.

## Reproduce on a suitable host

```bash
hf download XiaomiMiMo/MiMo-7B-Base \
  --revision c72df4586cb8bdeebd65f36929cd3385a6566fbe \
  --local-dir /tmp/rosetta-mimo7b-base
fieldrun convert --model /tmp/rosetta-mimo7b-base --dtype f16 \
  -o /tmp/rosetta-mimo7b-base-f16
fieldrun --bundle /tmp/rosetta-mimo7b-base-f16 --ids experiments/mimo_preflight/ids.json \
  --ctx 16 --n-eval 8 --dump /tmp/rosetta-mimo7b-f16-preds.txt
HF_HOME=/tmp/rosetta-hf-ref python experiments/mimo_preflight/reference_probe.py \
  /tmp/rosetta-mimo7b-base --dtype bf16 \
  --out /tmp/rosetta-mimo7b-bf16-reference.json
# Repeat with --dtype fp16 and --dtype fp32, keeping the same ids.json.

python experiments/mimo_preflight/make_slice.py \
  /tmp/rosetta-mimo7b-base /tmp/rosetta-mimo7b-two-layer
fieldrun convert --model /tmp/rosetta-mimo7b-two-layer --dtype f32 \
  -o /tmp/rosetta-mimo7b-two-layer-f32
fieldrun --bundle /tmp/rosetta-mimo7b-two-layer-f32 \
  --ids experiments/mimo_preflight/ids.json --ctx 16 --n-eval 8 \
  --dump /tmp/rosetta-mimo7b-two-layer-fieldrun-f32.txt
HF_HOME=/tmp/rosetta-hf-ref python experiments/mimo_preflight/reference_probe.py \
  /tmp/rosetta-mimo7b-two-layer --dtype fp32 \
  --out /tmp/rosetta-mimo7b-two-layer-reference-fp32.json
```

The raw reference top-two logits are in [`reference-bf16.json`](reference-bf16.json),
[`reference-fp16.json`](reference-fp16.json), and
[`reference-fp32.json`](reference-fp32.json); fieldrun predictions are in
[`fieldrun-f16.txt`](fieldrun-f16.txt) and [`fieldrun-int8.txt`](fieldrun-int8.txt).
The fieldrun f16 and int8 manifest SHA256 values are
`b208ba1d0eed4682e0bab16dff400b6d81bd5354e4a005510e37b1bda1226f6c`
and `2e9cadd6c791b2d1fcbcfe6c2d8bf3683616ab44350d1b99e09bf17e884a65ba`.
