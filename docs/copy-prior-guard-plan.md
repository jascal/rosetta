# Fresh prior-agreement circuit: feature and test boundaries

Status: **open**, before feature-pool generation or model requests. The prior
pilot is discovery only and supplies no admission or final-test contexts.

## Feature alphabet, fixed before contexts

Seed **910** samples disjoint token-role pools uniformly from `[200,40000)`,
excluding every historical input token from the copy/native-trace studies and
all recent experiments. Pool sizes: 512 query-anchor tokens, 1024 source-target
tokens, 1024 competing-successor tokens, and 8192 background tokens. These are
random role assignments, not lexical classes or successful-token whitelists.
Do not resample a pool based on model outputs.

For each of the 512 query tokens, run the pinned model on that token alone,
requesting logits for every source-target/competing-successor pool token. Before
any new full contexts are generated, Datalog compiles these observations into
categorical prior bands: `min(63, floor(max_logit − token_logit))`, with one
logit-unit bands and a common clipped tail. Check complete observations and
that no readout exceeds the reported maximum. Retain raw scores at build time;
freeze the complete band relation, pool, transformer query and source hashes.

The runtime will contain only the discrete lexical relation and Datalog rules,
with no original neural matrices, raw-logit evaluation, model executable,
Python or live prior query. Report the lexical table's rows and bytes. This is
a finite extracted lexical feature module, not a claim that a million facts
are a tiny human-written algorithm. The alphabet is bounded; unknown features
cause abstention. Do not call final tokens unseen by feature extraction: they
are unseen in development *contexts*, while the intrinsic lexical module is
shared and frozen before those contexts exist.

## Fresh development and final contexts

Use seeds **911/912** for development and **913/914** for final confirmation,
three groups per source length. Reuse the complete factor matrix of the
long-consensus study: lengths 32/48/64, prefixes 16/24/32, two/three copies,
12/16-token decoys, both source orders, zero/eight-token gaps, all-source target
substitutions, and independent no-repeat/tied-source controls. This is 612
contexts in each phase. Each group draws unused tokens from the corresponding
role pool: query's last token from the query pool, target/substitute from the
target pool, decoy successor from its pool, all other tokens from background.
All four splits have disjoint groups and context vocabularies. Token selection
must not inspect prior bands or model answers. The final matrix stays sealed
until development admits a candidate; no final full-context references exist
during feature extraction or development selection.

Six candidates retain minimum long-match lengths 16/24/32 and support 2/3,
the competitor-length/margin guards from the long-consensus family, and add one
fixed semantic guard: the literal winner's prior band must be strictly better
than every different successor of a matching source of length at least 12.
If a required band is absent or ties with a competitor, abstain. No band-width
search, answer normalization, boundary guard or target whitelist is allowed.

Datalog admission requires zero mismatches on every firing case in each
development split, positive incremental coverage over the frozen behavior-only
baseline in each split, and active successful target-substitution pairs in at
least two groups per split. Select greatest combined incremental coverage;
ties prefer longer matches then fewer sources. Report baseline overlap,
conflicts, baseline errors and every candidate abstention. No admitted candidate
means no final reference requests for this family.

## Final success obligations

Emit the selected **standalone guarded circuit** and explicit abstention runner;
the baseline is the frozen comparison, not an implicitly certified fallback.
Before final requests freeze the selected artifact, feature relation, all final
inputs, exclusions, and token-only firing/incremental domains and predictions.
Every domain member must remain an obligation, including wrong answers.

Require `dl/equiv.dl` success on the nonempty final firing and incremental
domains: zero mismatches, uncovered, missing references and invalid data, plus
positive incremental coverage on each final seed. Also retain full-final-domain
failures caused by abstention. Evaluate every final context; never select the
domain using reference agreement or retune after final answers. Run relocated
artifacts from token/domain facts in Souffle alone, checking both decisions and
abstentions against frozen predictions. Hash-check and replay all verdicts
without the model. Report final coverage and all baseline-answered contexts on
which the new circuit abstains; do not claim a global coverage superset.

A passing certificate proves the stated finite domains relative to the pinned
oracle. It does not establish universal faithfulness, whole-model conversion,
or anatomical sufficiency of the previously ablated heads. The native head
findings motivate this feature decomposition but do not prove this circuit.
