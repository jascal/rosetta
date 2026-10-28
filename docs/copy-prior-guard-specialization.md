# Metadata-only specialization of the frozen lexical module

This implementation note is fixed before new development contexts or their
references. It does not change pools, bands, candidate guards, seeds or admission
criteria in `copy-prior-guard-plan.md`.

The complete frozen module contains 512 × 2048 categorical facts. A deployed
artifact may specialize this module to the input alphabet of its stated finite
domain: retain the cross product of final query tokens and possible matched
successor tokens appearing in the staged inputs. Select keys using token facts
and exact matching only, never using band values, firing success or references.
Datalog must check that every required key exists in the complete module and
that copied labels are unchanged. Retain this reference-free specialization
query and its evidence before any corresponding full-context reference request.

This removes unused lexical data, not wrong predictions. Its predictions on the
stated inputs must equal the unspecialized rule's predictions. Unknown keys at
runtime cause abstention. Report the resulting bounded alphabet, facts and
artifact size; do not claim all-vocabulary conversion. The original complete
feature module stays available as build-time evidence, with its hashes intact.
