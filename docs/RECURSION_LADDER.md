# Recursive graph reasoning ladder

This experiment asks whether a model's answers to novel directed-graph queries
are reproduced by **one recursive reachability rule**. It separates three claims:

1. **proved over the stated finite domain:** `dl/reach_circuit.dl` computes
   reachability on the generated graph instances, checked against the graph
   generator's answer labels by Soufflé (`nmiss=0 ∧ nuncov=0`).
2. **empirical:** each model's answer accuracy and response to an edge change at
   each required path length and iteration budget.
3. **proved over the stated finite domain, only if the model passes:** the same
   Datalog rule matches every *observed model answer*. A failed certificate is
   evidence of disagreement, not a software error.

The generator permutes node identifiers and edge order for each trial. Every
positive graph contains one path of exactly `depth` hops plus an unrelated
distractor chain. Its paired negative redirects one path edge into the distractor
chain, preserving the edge count, source outdegree, and query endpoints. The
negative is unreachable. A causal-follow score requires *both* answers in a
pair to be correct, guarding against constant yes/no responses.

The initial chain domain varies node identities and edge order while keeping
the same two-chain topology at a given depth. `--topology branched` adds
variable dead-end branches and distractor cycles. Further work should vary
graph size and use disjoint graph structures on holdout.
The saved branched domain has 20 pairs per depth (200 cases), certifies the
same recursive rule with `nmiss=0, nuncov=0`, and the symbolic control reaches
full equivalence at budget 5 (`branched_control.json`).
An additional exhaustive semantic check enumerates all 64 loop-free directed
graphs on three nodes and their six distinct ordered queries (384 cases,
maximum simple-path length 2). This includes cycles and certifies the rule
against an independent breadth-first reference. It is a proof on that finite
domain, separate from any model-equivalence claim.

Generate and certify a bounded domain:

```bash
python3 py/recursion_ladder.py --cases /tmp/graphs.jsonl --generate \
  --depths 1 2 3 4 5 --trials 20 --out /tmp/graph_certificate.json
# Exhaustive three-node semantic domain (384 graph/query pairs):
python3 py/recursion_ladder.py --cases /tmp/all3.jsonl \
  --exhaustive-nodes 3 --out /tmp/all3_certificate.json
```

Each JSONL case has `id`, `edges`, `query`, `depth`, `trial`, `positive`, and
`answer`. To compare recurrent models, collect **one response per case per
architecture and budget** in a JSONL file:

```json
{"id": 0, "architecture": "looped_model", "budget": 1, "answer": 1}
{"id": 0, "architecture": "looped_model", "budget": 2, "answer": 1}
```

`answer` is `1` for yes, `0` for no, or `null` for an unparseable response. A
group with missing cases is rejected; a group with nulls has no model-equivalence
certificate. This prevents a selected subset from being called a model proof.
Use repeatable `--responses responses.jsonl --out result.json` options to
compute the matrix across architectures and budgets.
`--symbolic-budgets 1 2 3 4 5` adds a clearly labeled, non-LLM control that
propagates graph reachability one edge per iteration. It should first certify
the full domain at budget 5; a different result signals a harness problem.
The candidate Datalog program is `dl/reach_circuit.dl`: it consumes `edge`
and `query` facts and needs no model, Python, or fieldrun at runtime.
`dl/recursive_reach.dl` includes that program and adds `ref` plus the verifier.

`dl/reach_search.dl` makes rule extraction explicit. Soufflé compares five
candidates—constant no, constant yes, direct edge, at most two hops, and
recursive reachability—against training references. It reports both an exact
zero-training-error selection and the simplest minimum-error candidate, then
measures the latter on held-out cases. The host only stages graph, query,
reference, and train/test facts; it can group response files across budgets:

```bash
python3 py/recursion_extract.py \
  --cases experiments/recursion_ladder/branched_cases.jsonl \
  --out experiments/recursion_ladder/branched_extraction.json
```

On the 200-case branched truth domain, trials 0–9 (100 cases) select the
recursive rule: both constant candidates have 50 errors, direct edge has 40,
at-most-two-hops has 30, and recursion has zero training and zero errors on
trials 10–19. This is **selection of a recursive rule from a bounded library**,
not synthesis of arbitrary Datalog syntax. On model responses, exact fits
often select a constant yes/no rule, reflecting degenerate predictions. On the
200-case set, few-shot HRM-Text's best fit is constant no (30/100 training
errors, 37/100 holdout errors); Qwen's is constant yes (15/100 training, 11/100
holdout). Per-model, per-budget audits are in `model_rule_search.json` and
`branched_model_rule_search.json`.

A second, structurally disjoint extraction check is saved as
`structural_holdout_result.json`. Soufflé selects the recursive candidate from
eight four-node training queries (the two-hop candidate makes one error), then
gets all six held-out queries right on five- and six-node graphs with cycles,
branches, a shortcut, and a disconnected component. The recursive circuit
certificate covers all 14 cases (`nmiss=0`, `nuncov=0`). This is a stronger
cross-size and topology check than the randomized trial split, while still a
small finite domain; the six holdout cases are not a broad statistical estimate.
Recreate its cases, rule search, and certificate with:

```bash
python3 py/recursion_structural_holdout.py
```

For an available conventional transformer, start a resident `fieldrun` server
and use `--fieldrun-port PORT --tokenizer PATH --architecture NAME
--record-responses PATH`. This obtains yes/no logits from `/topk`; the larger
logit determines the recorded binary response. This is a **constrained binary
choice from the model's logits**, not its unrestricted generated answer. It
records **budget 1 only**,
because the bundle does not expose a latent iteration parameter. Recurrent
architectures should export answers at every actual iteration budget through
the JSONL interface. Do not substitute repeated calls to a one-pass model for
internal recurrence.

The official [LoopFormer 3-block/8-iteration FineWeb checkpoint](https://huggingface.co/armenjeddi/LoopFormer-3block-8iterations-FineWeb300K)
has an explicit `steps` input. Download it outside the repo and collect its
budget sweep with `py/recursion_loopformer.py`:

```bash
hf download armenjeddi/LoopFormer-3block-8iterations-FineWeb300K --local-dir /tmp/rosetta-loopformer
HF_HOME=/tmp/rosetta-hf .venv/bin/python py/recursion_loopformer.py \
  --cases experiments/recursion_ladder/cases.jsonl \
  --model-dir /tmp/rosetta-loopformer --budgets 1 2 4 8 \
  --out experiments/recursion_ladder/loopformer_3x8.responses.jsonl
```

For budget `k`, the adapter supplies `k` equal steps of size `1/k`, preserving
the model's normalized trajectory interval. This is an inference budget sweep
on a pretrained language model, with **no graph-task fine-tuning**. Its published
checkpoint saves only the token embedding for the tied output head; the adapter
restores that weight tie after Transformers 5.x loads the custom model code.

The official [HRM-Text-1B checkpoint](https://huggingface.co/sapientinc/HRM-Text-1B)
is a pretrained language model with high- and low-level recurrent stacks. The
adapter evaluates `(H_cycles,L_cycles)=(1,1),(1,3),(2,3)`, or 2, 4, and 8
stack applications, with `use_cache=False` so each prompt is recomputed under
the requested recurrence. The 8-step configuration is the released default.

```bash
hf download sapientinc/HRM-Text-1B --local-dir /tmp/rosetta-hrm-text
HF_HOME=/tmp/rosetta-hf .venv/bin/python py/recursion_hrm_text.py \
  --cases experiments/recursion_ladder/cases.jsonl \
  --model-dir /tmp/rosetta-hrm-text \
  --out experiments/recursion_ladder/hrm_text_1b.responses.jsonl
```

The current text prompt is produced by `prompt(case)` in the driver. The model
score is therefore prompt-sensitive. The rule certificate is behavioral over
this bounded graph set; even a clean model certificate would not prove that
the model uses the same internal mechanism or generalizes to unbounded depth.
The optional `--prompt-style fewshot` prepends one positive and one negative
two-hop example using node IDs absent from the test graphs. Prompt styles have
separate response groups and separate model certificates.

## First run: 2026-09-23

Raw instances, constrained yes/no logits, and the full Datalog verdict are in
`experiments/recursion_ladder/`. Domain: seed 17, depths 1–5, four paired
trials per depth (40 cases). The semantic circuit certifies **40/40** with
`nmiss=0, nuncov=0`. The runnable rule is `dl/reach_circuit.dl`; the
verifier is `dl/recursive_reach.dl`.

| model or control | available budgets | model mismatch at budget 1 | paired edge-change successes by depth 1→5 |
|---|---:|---:|---|
| Pythia 70m, NeoX | 1 | 20/40 | 0/4, 0/4, 0/4, 0/4, 0/4 |
| Llama 3.2 1b, RoPE | 1 | 20/40 | 0/4, 0/4, 0/4, 0/4, 0/4 |
| Qwen 2.5 Coder 1.5b | 1 | 19/40 | 2/4, 1/4, 0/4, 0/4, 0/4 |
| LoopFormer 3-block/8-iteration | 1, 2, 4, 8 | 20/40 at budgets 1, 2, 8; 21/40 at 4 | 0/4 at every depth and budget |
| HRM-Text-1B | 2, 4, 8 stack calls | 20/40 at budgets 2, 8; 21/40 at 4 | 0/4 at every depth and budget |
| symbolic frontier control | 1–5 | 16/40 | at budget `k`, passes exactly depths ≤ `k` |

Balanced few-shot prompt control, same 40 graphs:

| model | best tested budget | mismatches | paired edge-change successes by depth 1→5 |
|---|---:|---:|---|
| Pythia 70m | 1 | 21/40 | 0/4, 0/4, 0/4, 0/4, 0/4 |
| Llama 3.2 1b | 1 | 20/40 | 0/4, 0/4, 0/4, 0/4, 0/4 |
| Qwen 2.5 Coder 1.5b | 1 | 13/40 | 4/4, 1/4, 1/4, 1/4, 0/4 |
| LoopFormer | 1, 2, 4, 8 | 20/40 at each | 0/4 at every depth and budget |
| HRM-Text-1B | 8 | 16/40 | 2/4, 1/4, 2/4, 0/4, 0/4 |

Qwen and HRM-Text show a few-shot benefit on the small domain, especially at
shorter depths. The model-equivalence certificate still fails in every row.
The larger branched domain is the guard against reading a 4-pair cell as a
general depth capability.

The symbolic control certifies all 40 cases at budget 5. None of the five models is
equivalent to the recursive rule on this domain. These are small, prompt-specific
measurements: Pythia chose “no” on 38/40 and Llama chose “yes” on 40/40, so
their 50% accuracies are not evidence of graph reasoning. LoopFormer and
HRM-Text also fail every paired edge change at every tested budget, despite
large budget-dependent shifts in their yes/no logits. The original HRM puzzle
checkpoint expects a puzzle representation rather than these text queries.
TRM is also evaluated below through a maze representation of an undirected
relation graph; that is a separate input and task interface from the text
prompt results above.

The 200-case branched follow-up is in `branched_results.json`. LoopFormer has
100, 100, 97, and 100 mismatches at budgets 1, 2, 4, and 8. HRM-Text has
100, 99, and 100 at budgets 2, 4, and 8. A few middle-budget pairs pass the
edge perturbation, but the gain does not persist at the larger budget or grow
with required path length. This is an empirical failure to extract the
reachability rule from these pretrained checkpoints under this prompt, not a
claim that the architectures cannot learn recursive reachability.

Under the few-shot prompt on the branched set, HRM-Text at eight stack calls
has **81/200 mismatches** and paired edge-change successes of **8, 5, 2, 4, 0**
out of 20 at depths 1–5. Qwen at one pass has **74/200 mismatches** and
**13, 6, 3, 2, 2** paired successes. Short-depth sensitivity is real on this
domain, but a one-pass model does at least as well. Neither model certifies the
recursive rule; the five-hop tail remains mostly unresolved.

The saved branched graphs also isolate cuts strictly inside the path: 8, 10,
and 12 pairs at depths 3, 4, and 5. Under the few-shot prompt, HRM-Text at
eight stack calls follows both answers in 1/8, 1/10, and 0/12 of those pairs;
Qwen at one pass follows 0/8, 0/10, and 0/12. These counts are included in
`branched_results.json` as `interior_edge_pairs` and `interior_edge_follow`.

## TRM on maze-encoded relation graphs

The [Maze-Hard TRM reproduction checkpoint](https://huggingface.co/Sanjin2024/TinyRecursiveModel-Maze-Hard)
was loaded with the [upstream TRM implementation](https://github.com/SamsungSAILMontreal/TinyRecursiveModels).
Its config is H=3, L=4, with two shared reasoning layers; each outer ACT
iteration applies the shared reasoning stack 15 times. The tested budgets
1, 2, 4, 8, and 16 count these outer answer updates. The checkpoint uses
maze-grid inputs, so this follow-up encodes an **undirected relation graph** as
open cell nodes joined by one-cell corridors. A positive case has a path of 1–5
relation edges. Its paired negative closes one corridor on that path and opens
a disconnected distractor corridor, preserving the relation-edge and open-cell
counts. For depths 3–5, the changed corridor is strictly interior to the path.

The driver [py/recursion_trm_maze.py](../py/recursion_trm_maze.py) evaluates
each actual recurrent state, decoding “yes” only when the checkpoint's predicted
`o` path connects S to G. The saved graph labels independently certify against
the recursive Datalog circuit on all 40 cases (`nmiss=0`, `nuncov=0`).
Model responses and per-budget certificates are in
`experiments/recursion_ladder/trm_maze_results.json`.

Reproduce the generated cases and sweep from the downloaded checkpoint/source:
the optional TRM driver needs PyTorch, `pydantic`, and `einops` in `.venv`.

```bash
.venv/bin/pip install pydantic einops
hf download Sanjin2024/TinyRecursiveModel-Maze-Hard --local-dir /tmp/rosetta-trm-maze
git clone --depth 1 https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git /tmp/rosetta-trm-source
.venv/bin/python py/recursion_trm_maze.py \
  --cases experiments/recursion_ladder/trm_maze_cases.jsonl --generate \
  --checkpoint /tmp/rosetta-trm-maze/step_9765 \
  --source-dir /tmp/rosetta-trm-source \
  --budgets 1 2 4 8 16 \
  --out experiments/recursion_ladder/trm_maze.responses.jsonl
```

| TRM outer updates | graph-answer accuracy | Datalog mismatches | paired edge changes followed | interior cuts followed |
|---:|---:|---:|---:|---:|
| 1 | 21/40 | 19 | 1/20 | 1/12 |
| 2 | 35/40 | 5 | 15/20 | 10/12 |
| 4 | 34/40 | 6 | 14/20 | 9/12 |
| 8 | 33/40 | 7 | 14/20 | 8/12 |
| 16 | 34/40 | 6 | 14/20 | 9/12 |

This gives a clear budget effect on this maze representation, but the rule does
not exactly match any budget's answers. The bounded candidate search selects
no zero-training-error rule from TRM answers. The best fit is constant no at
budget 1 and recursive reachability at budgets 2, 4, 8, and 16. Recursive-rule
training errors on the 20 training cases are 9, 2, 2, 3, and 3; those best fits
make 3, 4, 4, and 3 errors on the 20 held-out cases at budgets 2, 4, 8, and 16.
These results apply to this checkpoint, its maze-task training, and the
maze-encoded **undirected** graph domain. They do not establish performance on
arbitrary directed relations or natural-language graph prompts.

The newer [T-LoopFormer paper](https://arxiv.org/abs/2609.15160) adds token-level
loop routing. As of this run, its [official repository](https://github.com/YuMingQian1234/T-LoopFormer)
documents the architecture and training code but does not identify a released
checkpoint; its README describes a six-GPU training launch. It therefore has no
checkpoint result in this matrix. The tested LoopFormer is the released 3-block,
8-iteration checkpoint described above.
