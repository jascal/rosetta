# Context-free continuation prior: discovery diagnostic

Before any requests, use every positive context from the rejected 801/802
long-consensus study. Datalog extracts the twelve distinct final query tokens
and all successor tokens of their earlier matches of length at least 12. Query
the pinned native model once on each **one-token** context, recording logits for
all those successors and the model's unconstrained top-1. This does not query a
new full-context final-test answer.

Report the prior's preferred in-context successor versus the long-consensus
prediction and the already recorded full-context answer, retaining ties,
uncovered predictions and every disagreement. The prior is a feature hypothesis,
not an asserted source mechanism. In particular, it may prefer a common token
that the full-context model correctly overrides.

No production rule is selected from these observations. A future semantic guard
would need a frozen model-free representation derived independently of final
contexts, new development data and untouched final references. Runtime model
calls, score lookups from a live model, whitelists of these known successful
targets, and promotion of pilot agreement to a certificate are forbidden.
