# Qwen3.5 competing-source intervention

**Open:** do departures from the designated long-source successor causally track
the competing short source's successor? The completed copying inventory showed
26 competing-source answers and 10 answers absent from the input among 144
positive training cases. Observational agreement cannot establish source use.

This development study keeps each selected context's final query and long-source
answer unchanged and replaces only its competing source's successor. The 36
contexts and source code were hashed before requesting their references. A
Datalog integrity query separately verifies that every context differs from its
original in exactly the declared token, and that the replacement is fresh within
that original context.

`dl/qwen35_decoy_plan.dl` chooses one original target-edit pair per training
vocabulary group, repetition count (1/2/3), and source order (first/last).
It prefers the lowest-ID pair with a designated-source departure in either
member; otherwise it chooses the lowest-ID control pair. The three groups and
six repetition/order combinations give 18 pairs. Each pair contributes two new
contexts: original and edited long-source answers, with the same new decoy in
both. Seed2100 draws distinct decoy replacements from the original role pool,
excluding every decoy token used in either inventory split. No string content or
new output participates in that draw.

Selection uses training references only. This is a deliberately enriched
mechanism sample, not an unbiased estimate of model performance. Its contrasts
are old/new decoy under each of two long-answer states. The old contexts already
have complete, replayable references. New references use the pinned official
Qwen3.5-4B FP32 graph, with every vocabulary score retained and argmax computed in
Datalog as in the inventory.

`dl/qwen35_decoy_summary.dl` reports the new answer's relationship to the new
decoy, long target, old decoy, or neither. Separate relations report any answer
change, old-decoy to new-decoy tracking, and paired tracking of long versus
competing sources. Changing an answer establishes sensitivity to the edited
source value for that context; it does not alone identify an internal routing
circuit. Stability likewise cannot prove absence of an internal contribution.

No rule is admitted from these observations. No final-test input or extracted
artifact has been selected. A successful mechanism observation would motivate a
separate internal intervention and eventually a frozen token-only circuit with
fresh disjoint confirmation. Failed contrasts remain in the evidence.

Evidence: [frozen protocol](../reference/benchmarks/qwen35_decoy_counterfactual/protocol.json),
[contexts](../reference/benchmarks/qwen35_decoy_counterfactual/dataset.json), and
[edit integrity](../reference/benchmarks/qwen35_decoy_counterfactual/edit-integrity.json).
After collection, `python3 py/evaluate_qwen35_decoy_counterfactual.py` replays all
36 witnesses and the three queries and retains a closed manifest. It refuses an
incomplete collection.

## Completed outcome

**Empirical, finite development sample:** all 36 new references are collected.
The complete argmax witnesses and all three queries replay unchanged. The
summary reports 30 long-source answers and six new-decoy answers, with no old
decoy or other answers. Thirteen answers change relative to the original input;
23 remain unchanged. Five changes track the old decoy to the new decoy, and eight
switch to the unchanged long-source target. Fifteen of 18 pairs follow their two
different long targets; three pairs follow their shared new decoy.

Changing only the decoy successor can therefore alter source choice as well as
supply a different copied value. A fixed answer determined solely by the final
query cannot explain the 13 changed cases. This does not identify which internal
component implements that sensitivity, nor prove a general source-selection
rule. The deliberately selected contexts and three shared vocabulary groups
limit generalization.

The next internal contrast should exchange MLP products between an original
context that chose the decoy (or another answer) and its decoy-edited context
that chose the long target. The final query and long target stay fixed. Forward
and reverse donor exchanges can test whether an internal state transfers this
routing change; identity and zero interventions distinguish substitution from
hook artifacts and broad disruption. Layer0 remains a hypothesis from the old
model, not an established Qwen3.5 localization.

[Summary](../reference/benchmarks/qwen35_decoy_counterfactual/summary.json),
[replay audit](../reference/benchmarks/qwen35_decoy_counterfactual/replay-audit.json).
No new standalone circuit or final certificate results from this study.
