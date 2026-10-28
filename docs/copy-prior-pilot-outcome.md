# Prior agreement is a promising guard hypothesis, not a replacement predictor

**Empirical historical-development diagnostic**, twelve one-token model queries
and 36 requested successor logits, fixed by Datalog before collection. No new
full-context or final-test references were collected. The unmodified historical
model executable and f32 bundle supplied every prior observation.

For each old context, Datalog selects the highest-prior successor among its
matching sources (length at least 12), using the model's logits after the final
query token **alone**. Ties abstain. This is deliberately different from running
the model on the full context.

| Minimum match / support | Old firing cases | Literal rule correct | Prior alone correct | Literal/prior agree | Errors where they agree |
|---|---:|---:|---:|---:|---:|
| 16 / 2 | 328 | 283 | 139 | 123 | 10 |
| 16 / 3 | 160 | 139 | 67 | 60 | 5 |
| 24 / 2 | 272 | 243 | 113 | 102 | 8 |
| 24 / 3 | 128 | 115 | 53 | 48 | 4 |
| 32 / 2 | 160 | 151 | 81 | 74 | 0 |
| 32 / 3 | 64 | 61 | 35 | 32 | 0 |

The prior alone is substantially worse than literal long-context consensus on
these cases. It cannot replace source selection. Its agreement with longer
supported matches is a promising *abstention guard*: the two longest variants'
agreement subsets contain no recorded error. This is post-hoc discovery on
already observed development data, with correlated variants and few groups.
It is neither an untouched test nor an equivalence certificate.

The lower-match variants still make errors even when both predictions agree,
so agreement is not a general guarantee. No threshold or production rule was
selected from this pilot, and no `dl/equiv.dl` success is claimed.

## Required next boundary

A runtime model call for this feature would violate the standalone invariant.
Before a fresh development experiment, define and freeze a finite lexical
feature alphabet and compile context-free prior observations to discrete
Datalog facts. The data generator must not select tokens by their feature value
or model success. Freeze this feature module before generating new contexts.
Final context vocabularies/groups must be disjoint from development; any shared
intrinsic lexical module must be declared explicitly, rather than calling its
alphabet unseen. All final full-context answers must remain unrequested until
the selected artifact and firing domains are frozen.

Such an artifact may contain an extracted finite lexical relation, but must
contain no neural parameter matrices, live model dependency or raw-logit
inference. Its table size and limited alphabet must be reported honestly. It
still needs fresh zero-error admission, positive incremental coverage, exact
finite-domain `dl/equiv.dl` certificates and isolated Souffle-only replay.

## Evidence

`reference/benchmarks/qwen25_copy_prior_pilot/` retains the frozen extraction
plan, contexts, requested scores, raw prior observations and Datalog report.
Its `certified()` relations establish completeness/instrument consistency only.
The two queries replay without the model. A synthetic test checks both agreeing
and opposing priors and rejects missing score observations. Incremental
certified coverage remains zero.
