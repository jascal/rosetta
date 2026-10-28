# Next discovery question: long exact context versus a recent short match

Status: **open**, no model requests for this pilot yet.

The corrupted-query pilot suggests preserving the final-token anchor matters
more than tolerating a longer approximate alignment. The outstanding
computation is source selection when the same suffix has conflicting successors.

Use four historical run-C background groups (lengths 12/24, two groups each),
including their paired target substitutions. Construct **one** true source
sequence and one distractor source. The distractor ends with the query's last
one or three tokens followed by a different, fresh successor, with distinct
fresh surrounding tokens. Place the distractor before or after the true source.
Append the original query prefix (six or twelve tokens).

This is eight target/background variants × two distractor match lengths × two
orders = 32 contexts. Fixed seed 502 supplies group-disjoint distractor tokens,
excluding every token in prior copy/native-trace datasets, the short-prefix
study and corrupted-query pilot. Pair target substitutions within each layout;
keep the distractor and query identical across each pair. These backgrounds are
discovery data and cannot establish fresh holdout generalization.

The hypothesis predicts the successor of the **uniquely longest exact context
match**, provided it has at least six (or, separately, twelve) matching tokens,
beats every different-successor match by at least three tokens, and its predicted
successor occurs exactly once in the input. Require a conflicting source match
of at least three tokens for the proposed new circuit. Otherwise abstain.
Ties between conflicting successors abstain. All matching, comparison, guarding
and prediction must be Datalog.

The three-token distractor should make the baseline's consensus-three route
abstain. Its nearest/longest routes also require repeated target tokens; both
candidate successors here occur once. Verify actual baseline abstention rather
than inferring it. One-token distractors are baseline-covered controls, and
their successes cannot count as incremental coverage.

Report every order/match-length cell, wrong output and abstention. Compare the
long-source hypothesis to the simple most-recent-source hypothesis; a real
context-selection computation must be tested against recency. This pilot
selects no production rule. Any admission or confirmation requires new groups,
vocabularies, explicit gates and the original goal's frozen finite-domain
certificates, positive incremental coverage, standalone execution and replay.
