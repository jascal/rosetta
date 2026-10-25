# Categorical readout of localized source values

Open before new projection-score extraction. Use all twelve historical source
sites and their frozen eight-token windows from the source-window study.
Datalog deduplicates identical token windows before looking at numeric values.
Retain every original row, including wrong-source and normalized-output cases.

For each unique window take its saved layer16/21 value projections. For a head
set, sum each selected head's source value projected through that head's o_proj
columns, multiply by the learned final RMSNorm gain, and project through the
complete lm_head vocabulary. The omitted final inverse-RMS factor is positive
and common to tokens, so it cannot change this component's ranking. No query
attention mass, residual stream or subsequent nonlinear block is included.
This is a fixed component-readout hypothesis, not a full-model forward.

Compare the six previously frozen treated/control head sets plus individually
specified copy heads 16/9 and 16/12. No head, normalization, interpolation or
score cutoff is selected after scores are seen. The joint set adds unpatched
source values from two layers with unit weights, an explicitly artificial
component sum rather than the model's final decision.

Freeze source windows, canonical representatives, head sets, source code,
checkpoint identity and previous tensor/evidence hashes before extracting
scores. Stage every vocabulary score, not a Python-selected top-k. Datalog
checks complete vocabulary coverage, computes global argmax with lowest-token
ID tie breaking, and reports ranks of historical intended/model outputs and
agreement on all twelve historical rows. Serialize only its categorical maps
as standalone .dl facts; numerical tensors, matrices and projection scores
remain build-time evidence. Emitting facts is not an equivalence certificate.

Report treated and control nulls, all disagreement examples and the finite
window alphabet. A stable output on these few fixed windows does not prove a
context-free token map or generalization to new source tokens. No fresh final
references are requested here, and the failed prior-guard final test is not
used to fit this readout. Any later routed circuit must be frozen before fresh
disjoint confirmation and pass dl/equiv.dl on its nonempty incremental domain.
