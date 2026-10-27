# Qwen3.5 gate-subset bisection protocol

This is adaptive development search, not a final test or equivalence proof.
The previous six directed contrasts and their 30-neuron remaining-top32 payloads
are the fixed roots. Three roots transfer the donor answer; three do not. The
root members are ordered by the previous frozen absolute-product-change rank.
Datalog compresses that order into contiguous ranks without changing membership.

For each successful root, split its ordered set into floor(N/2) and ceil(N/2)
consecutive members. Freeze both child payloads before querying either. Observe
both children before choosing a branch. Follow the left child if it transfers,
otherwise the right child if it transfers. If both transfer, retain both results
and follow the left branch. If neither transfers, stop at the successful parent
and report an interaction at this partition. A successful singleton is a leaf.
An unsuccessful root is reported and receives no further search. At most five
splits per successful root, two children each, give at most 30 new forwards.

All choices and stopping conditions are in `dl/qwen35_gate_bisection.dl`.
Python stages the current observations, freezes the resulting round plan,
invokes the model for precisely its pending children, and stages complete score
facts. Datalog computes each argmax and verifies requested/applied substitutions
with the established FP32 narrowing tolerance. Both children remain retained,
including failed and unchosen children. Resume uses the same frozen round plan.

An interaction stop is not proof of irreducibility: different partitions or
combinations might work. A singleton transfer is sufficiency in that selected
context, not a token-computable rule, necessity, generalization, or standalone
replacement. No fresh confirmation labels are used. A candidate extracted from
this development work must still freeze its rule/guard before fresh disjoint
references and pass a nonempty incremental `dl/equiv.dl` certificate.
