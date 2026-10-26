# Actual-checkpoint MLP instrument validation

Before any probe request, freeze the pinned 4B checkpoint and the existing
five-token smoke context, all code, observation sites and interventions.
This is an instrument check, not a copy-localization or final generalization
study. Do not inherit Qwen2.5 head/neuron findings.

Observe last-position units 0/1 at layers 0,3,15,31: one Gated DeltaNet layer
and three full-attention layers. Collect seven forwards in fixed order:
baseline, identity, answer-position zero, source-position zero, independent
answer-position zero, independent source-position zero, unpatched cleanup.
Identity values come from the baseline via a Datalog query. Answer-position
zeros affect the eight observed units; source-position zeros affect layer0
units0/1 at position0. Retain the previously recorded unpatched scores as
another baseline comparator.

The primary instrument patches the official lazy forward graph. The comparison
instrument independently installs the declared substitutions inside each
block of the explicit streamed graph. Both use the same installed Transformers
operators and FP32 checkpoint values; this is not native fieldrun validation.

Datalog checks complete vocabulary/observation/patch coverage, exact applied
patch values, nonzero intervention doses, identity/cleanup observation parity,
baseline replay and full-logit/argmax parity between the paired paths.
Logit/observation tolerance is 0.002 absolute + 0.002 relative. Identity and
cleanup full-logit hashes must equal baseline exactly. Every failure remains
in the evidence; do not change units, values or tolerances after responses.

Success establishes these operations on this finite smoke context only. It
does not validate full-layer ablations, head interventions, recurrent-state
patching or a copy mechanism. A new baseline/localization study follows only
after this gate. No minimized Datalog circuit or fresh incremental coverage is
claimed by instrument consistency.
