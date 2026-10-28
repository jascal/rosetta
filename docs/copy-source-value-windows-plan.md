# Fixed source-window value test

Open before new requests. The prior bigram lookup preserved self answers but
lost the demonstrated literal donor-target transfer. Test source windows of
**4, 8 and 16 tokens**, including the source successor as the final token.
Clamp the window at the original context start. Keep the same twelve historical
recipients, paired source edits and six treated/control head sets. Do not
select only successful transfers, expand the lengths after answers or change
the rejected final-test circuit.

Datalog emits the exact window tokens and feature keys for every source site
and all three sizes, then emits self/donor substitutions for each size plus
contextual identity, contextual donor and unpatched controls. This yields
588 full-context requests. Every feature forward runs only its emitted window;
capture layer16/21 value projections at its last position. Substitute values
only at the registered source sites, hold recipient routing fixed, and patch
only selected final-position head products. The original captured factors and
the previous contextual identity/donor journal are frozen dependencies.

Use the same pinned checkpoint, FP32/eager, two threads, no cache. Freeze
contexts, windows, requests, source hashes, checkpoint and previous evidence
before any new model request. Retain all feature tensors and observations.
Datalog checks complete requests and baseline/identity/contextual-donor replay
(argmax exact; logits 0.002+0.002*abs(reference)); report preservation, flips,
donor-reference and literal-target transfer for every size/head set.

This is historical mechanism discovery, not a new final equivalence test.
If a window contains the complete causal prefix ending at a source site,
reproducing that site's value is a causal-prefix control, not evidence for
context compression. Explicitly report which features truncate real preceding
context. A successful transfer on an untruncated feature cannot establish
a bounded-memory implementation. Even a useful truncated window leaves source
routing and the output decision to extract. No tensor, model or weights become
a runtime dependency; the original standalone fresh-certification goal remains
unchanged.
