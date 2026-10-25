# Source-resolved diagnostic after the failed long-match study

Status: **open**, fixed before native explanation requests. Historical
development data only; no final confirmation references or candidate selection.

`dl/copy_source_diagnosis.dl` classifies all 648 positive long-source rows by
their recorded answer: exact gold, a competing source successor, another input
token, or a token absent from the input. These are observable answer categories,
not anatomical or causal explanations. The query reports 526/65/13/44 rows.

For native explanations select the lowest-ID row in each independent group and
class, then the first two groups per class. The other-input class has only one
represented group, giving seven selected cases. Do not replace cases based on
their explanation output. Retain the complete classification evidence.

Use the pinned historical executable's existing `/explain` endpoint, with the
same restored f32 weight bundle. Record every returned head, neuron and logit
field. Check that the explanation predicts the previously recorded reference
and that each reported peak position/token matches the staged context.

The endpoint reports the six highest direct-logit-attribution heads after a
64-head prefilter, plus six neurons; it does not expose all attention heads,
full attention distributions or all source contributions. A head's peak is
only its maximum-weight read position. A peak at the gold/competing token is
descriptive, not a proof that it supplied the answer. The `promotes` lists are
top token projections of reported writes, not isolated causal predictions.

Datalog reports peak-token category, source-successor membership, and whether
each reported head promotes the gold or actual output token. No ranked head is
automatically converted into a candidate rule or asserted necessary. Any later
necessity/sufficiency experiment requires an explicit intervention plan,
identity and matched controls, then fresh data before a new certification claim.
