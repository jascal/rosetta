# Successor-plurality guard search

**Empirical finite-domain search and certificate results.** This search adds a
distinct Datalog hypothesis to the existing unanimous-source guard. For each
repeated suffix, it counts the successor tokens from matching source positions
and predicts the unique most frequent successor when it has at least the
candidate's minimum support. It abstains on ties and insufficient support.
The search uses the expanded 48 suffix/support candidates. Normal admission
criteria are unchanged: zero development errors, all paired-edit checks, and
support in multiple groups in both train and validation.

The frozen corpus has 102 cases: 24 train, 24 validation, 48 test, and 6
off-domain controls. It uses one seed and sequence length 32, with plain,
32-token-noise, and a near-match distractor that repeats the full 12-token
query suffix once before a decoy. The Base and RL runs used the same dataset
SHA256:
`d730de85475d6e30cc2bfa78cae31c0055f356817b31a4ca2b520a239c796d72`.
Earlier Qwen and MiMo token IDs were excluded before generation.

## Deterministic rule control

On a deterministic oracle defined by the authored copy rule, both guards were
selected with suffix length 1 and support 1. The unanimous guard covered 32/48
test cases and abstained on the 16 near-match cases because the decoy source
conflicted with the true sources. The plurality guard covered all 48 and
matched all 48 labels; Soufflé certified its entire test firing domain with
zero misses. This verifies the new Datalog rule handles the intended single
conflicting-source pattern.

## MiMo results

Neither model admitted a normal selected guard under either policy. The
predeclared suffix-9/three-source diagnostic guard makes the policy difference
visible without changing admission:

| Model | Policy | Fired / 48 | Correct | Wrong | Abstained | Exact firing-domain certificate |
|---|---|---:|---:|---:|---:|---|
| Base int8 | Unanimous consensus | 16 | 11 | 5 | 32 | failed |
| Base int8 | Successor plurality | 24 | 14 | 10 | 24 | failed |
| RL int8 | Unanimous consensus | 16 | 10 | 6 | 32 | failed |
| RL int8 | Successor plurality | 24 | 13 | 11 | 24 | failed |

The development near-miss counts explain the rejection: the lowest-error
consensus candidates still had 1 Base mismatch / 1 failed pair (16 answers)
and 2 RL mismatches / 2 failed pairs (32 answers). The lowest-error plurality
candidates had 4 Base mismatches / 3 failed pairs and 5 RL mismatches / 4
failed pairs (24 answers each). No candidate passed zero-error admission.

Plurality adds eight fired cases by answering on the near-match layout, but
gets only 3/8 right there for each model. It therefore increases coverage and
errors together. The causal-copy candidate remains unadmitted; selected
artifacts abstain on all 48 test cases.

By layout, the fixed guard answers on eight plain and eight noise cases under
both policies. Base consensus gets 6/8 plain and 5/8 noise cases right;
plurality has the same counts on those layouts. RL consensus gets 6/8 plain
and 4/8 noise; plurality is identical there. In the near-match layout,
consensus abstains on all eight, while plurality fires on all eight and gets
3/8 right for both models. This is an exact-certificate failure for both
policies on each model's frozen firing domain, retained with counterexamples
in the evidence.

This cleanly separates **expressiveness** from **behavioral fit**: plurality
can encode and certify the desired majority rule on the authored control, but
MiMo's answers do not reliably follow that rule. The result does not show that
plurality is always a worse circuit; it shows it is not an admitted replacement
for these checkpoints on this domain.

Full runs, frozen domains, references, and Soufflé evidence are in
[`plurality_search_base_int8/`](plurality_search_base_int8/),
[`plurality_search_rl_int8/`](plurality_search_rl_int8/), and the deterministic
[`plurality_search_control_suffix12_v2/`](plurality_search_control_suffix12_v2/).
The exact generated input is
[`copy_guard_stress_seed37_len32_v2.json`](copy_guard_stress_seed37_len32_v2.json).
