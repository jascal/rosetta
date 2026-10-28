# Frozen routing/value factorization discovery

Open, before new intervention requests. Use every one of the twelve recipients,
six head sets and paired donors from the completed answer-position exchange
study. Do not select only the successful transfers. This is historical
mechanism discovery, not a replacement final test for the rejected copy rule.

For each unpatched context capture, at layers 16 and 21, the last-position
attention probabilities A, the full source-position value projections V, and
the products entering o_proj. Account explicitly for grouped-query attention:
query head h uses value head floor(h / num_key_value_groups). Check A*V against
every selected captured product, maximum coordinate error at most 0.0001.
Retain the captured float32 tensors as build-time evidence.

For every recipient/head set, patch only final-position o_proj input slices:

- identity: recipient A times recipient V;
- exchange: donor A times donor V;
- values: recipient A times all donor V;
- routing: donor A times recipient V;
- source_values: recipient A times recipient V with donor V substituted only at
  the original registered source-successor edit positions.

Also retain the unpatched baseline. The 12 recipients and six head sets imply
372 requests. The joint patch always uses separately captured, unpatched
products from each layer; it does not pretend these form a natural trajectory.
Matched control head sets remain unchanged from the prior study.

Use the same checkpoint, float32/eager, two threads, no cache. Datalog checks
complete requests, native unpatched parity, identity parity, A*V reconstruction
and baseline/identity/exchange replay against the previous intervention journal
(argmax exact; logits tolerance 0.002 + 0.002*abs(reference)). A reconstruction
failure invalidates factorization interpretation; do not change its tolerance
after observing outputs. Report donor-reference and donor-target transfers
separately for every mode, including nulls and collateral answers.

Freeze sources, plan, context/edit inputs, previous observations and checkpoint
identity before requests. Model-free replay checks all retained Datalog
verdicts. Numeric tensor reconstruction is an instrument identity, not evidence
that a token-only Datalog circuit has been extracted. Values here can depend on
the whole preceding context; a value-channel transfer alone does not establish
a context-free lexical mapping. The source_values intervention distinguishes
the registered edited sites from broader context-mediated value changes.

Success of any discovery arm does not complete the goal. An eventual standalone
token-only router/value rule still needs positive incremental coverage and a
clean dl/equiv.dl certificate after freezing before fresh final references.
