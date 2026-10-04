# MiMo-7B family probe screen

**Empirical screening run**, 10 randomized trials per family for each checkpoint.
Both int8 bundles passed through the same frozen text stimuli and tokenizer
files were byte-identical. Succession used one comma-separated representation
for both models. The results come from the existing `probe_families.py` score:
argmax for IOI and succession, top-five membership for most open-answer
families. They are not exact Datalog certificates, and 10 trials are too few
for a population claim.

| Family | Base detect / causal | RL detect / causal |
|---|---:|---:|
| Copy/name-mover (IOI) | 10% / 10% | 0% / 10% |
| Succession (days) | 100% / 100% | 100% / 100% |
| Succession (months) | 100% / 100% | 100% / 100% |
| Coreference (gender) | 100% / 100% | 100% / 70% |
| Capital relation | 100% / 100% | 100% / 100% |
| Antonym relation | 100% / 100% | 100% / 100% |
| Is-a transitivity | 100% / 100% | 100% / 100% |
| Modus ponens | 100% / 100% | 100% / 100% |
| Temporal ordering | 100% / 100% | 100% / 100% |
| Syllogism | 100% / 100% | 100% / 100% |
| Spatial relation | 100% / 100% | 100% / 100% |
| Set membership | 90% / 90% | 80% / 60% |
| Defeasible exception | 100% / 100% | 100% / 100% |
| Analogy | 100% / 100% | 90% / 90% |
| Causal do-vs-see | 100% / 100% | 90% / 100% |

Both models pass the script's 80%/80% screen on 12/15 families. Base also
passes coreference and set membership, where RL's causal-follow rates are
below threshold. Both are weak on IOI name-mover copying. The plausible
checkpoint differences need a larger preregistered sample and top-k margin
checks; the current results are a lead for follow-up, not evidence of a
training mechanism.

Raw aggregate outputs are in [`base_int8.txt`](base_int8.txt) and
[`rl_int8.txt`](rl_int8.txt). The script does not retain trial-level examples.
