# Source-value locality test

Open before new requests. Keep all twelve historical recipients, paired donors,
six treated/control head sets and registered edit sites from the factorization
study. The two observed transfers are not singled out for input selection.

Test whether the contextual source values can be replaced by a build-time
token or bigram lookup. Extract v_proj values at layers 16 and 21 by running
the pinned model on [T] and [Q,T], where T is each registered source successor
and Q its immediately preceding token. Datalog chooses the complete required
feature-key set from frozen context/edit facts, with no answer-conditioned
selection. Use the final feature position and account for GQA value sharing.
These lexical tensors are build-time probes, not a runtime circuit.

At each selected source-edit position, substitute one of:
contextual recipient values (identity); contextual paired-donor values
(source_values); [T] recipient values (token_self); [T] donor values
(token_donor); [Q,T] recipient values (bigram_self); [Q,T] donor values
(bigram_donor). Hold each selected head's recipient attention probabilities
fixed. Other source-position values remain those of the unpatched recipient.
Patch only the final-position o_proj head slices. Preserve all six head sets
and an unpatched baseline: 444 full-context requests, plus the finite feature
extraction forwards. Joint patches use unpatched products independently at
both layers, as in the previous study.

Freeze code, contexts, feature requests, intervention requests, checkpoint and
previous captured factors/journal before new model requests. Use FP32/eager,
two threads, no cache. Retain lexical value tensors and all raw observations.
Require exact baseline/identity/contextual-donor argmax replay and declared
0.002+0.002*abs(reference) logit tolerance against the previous journal; retain
any failure rather than reinterpreting it as a lexical result. Datalog checks
request completeness and reports per-mode flips, preservation, donor-reference
and donor-target transfers. Self-feature substitutions are explicitly fidelity
tests, not donor transfer tests. A lexical mapping that happens to preserve
some answers does not establish equality of internal values.

No threshold, case, head set or lexical context length may be changed after
these observations. Results are empirical on a deliberately selected historical
panel, not fresh generalization. Even a successful lexical substitution leaves
attention routing and the output decision to extract. No runtime model,
captured contextual activations or weights may enter a delivered circuits.dl.
The original fresh, incremental equivalence goal remains unchanged.
