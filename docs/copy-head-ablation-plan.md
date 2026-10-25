# Whole-sequence attention-head deletion: fixed discovery plan

Status: **open**, before any model request for this experiment. This is
historical-case mechanism discovery, not final circuit confirmation.

Use the seven already frozen source-diagnosis contexts unchanged. Their native
explanations motivate layer-16 heads 7/9/12 (copy-related writes on both correct
examples) and layer-21 heads 2/13 (competing-source/semantic writes in errors).
These are attribution-derived hypotheses, not established mechanisms.

Zero specified query-head output slices before the attention output projection
at **every sequence position**. Downstream attention and MLP layers recompute.
This is neither answer-position-only patching nor a sufficiency/transplant test.
The API wraps fieldrun's existing `logits_ablated` kernel in an isolated checkout;
the original sibling checkout and historical probe binary stay unchanged.

The fixed 19 arms are:

- no deletion (an empty-head control, not activation write-back);
- each of copy heads 7/9/12, pair 9+12, and triplet 7+9+12 at layer 16;
- three matched single-head controls, their 9+12-sized pair, and their triplet;
- each semantic head 2/13 and their pair at layer 21;
- two matched single-head controls and their pair at layer 21;
- copy triplet plus semantic pair, and the five-head matched control.

Control heads are chosen in Datalog from a complete randomized ranking of all
14 heads in each treated layer, excluding the treated heads in that layer.
Seed **503** fixes the ranking. Reuse the same matched controls across arms.
No control is chosen using an intervention result. This is **133 requests**.

## Instrument gate before interpretation

Compare all 133 native interventions against the upstream checkpoint using
FP32/eager PyTorch and pre-hooks on the corresponding `o_proj` inputs. Mask the
same query-head slices at every position. Require identical argmax and both
recorded logits within `0.002 + 0.002*abs(reference)` for every arm/row. Empty-head
answers must also reproduce the historical model answers, with logits within
the same tolerance. Retain both raw observation sets, versions, source hashes,
and failures. If parity fails, report the instrument failure; do not interpret
the experimental contrast or silently drop failed cases.

The entire arm plan, sample and sources are frozen before either upstream or
native requests. A code correction after freeze must be preserved explicitly;
it cannot change treatments, cases, controls or interpretation using known
outcomes. Existing tiny-model tests must exercise the API schema, bounds,
duplicate rejection and agreement with the direct ablation call.

## Readouts and scope

Datalog checks completeness, parity, scope and empty-head consistency. It
reports, by arm and diagnostic class, answer flips relative to each row's
unpatched answer, gold answers, rescues of previously wrong answers and losses
of previously correct answers, plus both logit deltas. Compare treated groups
with equal-size, layer-matched control groups. Retain all single and joint arms;
a null single-head result does not refute a redundant group computation.

A deletion effect establishes a load-bearing contribution on these cases, not
a unique mechanism, an extracted algorithm, or sufficiency. A rescue after
deleting competing heads is evidence to investigate, not a certified repair.
The seven cases are deliberately selected historical examples, all from the
shortest long-context stratum. No population claim is supported. Every result
is **empirical**. The original goal's standalone Datalog circuit and positive
incremental finite-domain certification remain separate, unachieved obligations.
