# Development protocol: consensus beyond the baseline's 12-token horizon

Status: **open**, before new inputs or model requests. This is a new candidate
family, not a repair of rejected artifacts using final answers. No final-test
references from this research sequence exist.

Inspection of the frozen behavior baseline shows a concrete bound: its longest
source matcher stops at 12 tokens and requires repeated target tokens. A later
12-token decoy with a unique successor can therefore defeat its longest route,
while conflicting short successors prevent its consensus routes from firing.
Repeated *longer* evidence was absent from the earlier longest-match study,
which required each predicted successor to occur once.

The next computation hypothesis is consensus among longest-context matches
past that horizon. Six candidates cross minimum match length `{16,24,32}` with
minimum number of long agreeing sources `{2,3}`. Require every maximum-length
match to agree on its successor, and at least a four-token winning-length gap
over every different-successor source. Count distinct source positions for that
successor whose matches meet both the candidate minimum and the competitor's
maximum plus four; shorter agreeing sources need not equal the absolute maximum.
The competitor must match at least 12
tokens. Abstain on ties/conflicts or insufficient support. Matching is bounded
at 64 tokens; all selection and routing decisions are Datalog. No target-token
whitelist, source-boundary guard, trace threshold or answer normalization is used.

## Fresh development matrix

Seeds **801/802** are disjoint train/validation splits. Use two independent
groups per source length `{32,48,64}`, query prefixes `{16,24,32}` respectively.
Cross two/three true source copies, decoy suffix lengths `{12,16}`, both orders
(decoy before or after all true copies), zero/eight-token gaps, and paired edits
changing the target in every true source copy while preserving query/decoy.
Use independent negative groups for no-repeat inputs and tied full-context
sources with conflicting successors. Every source, decoy and gap token is fresh
except declared source repetition and matched suffixes. Exclude all historical
copy/native-trace inputs and every input from the recent studies.

This gives 384 positive variants and 24 controls across both splits: **408
contexts**, with six positive and six negative groups per split. Variants are
correlated. Groups and vocabularies are disjoint across splits. Keep all rows,
including baseline-covered decoy-first controls, candidate abstentions and ties.
Compute candidate/baseline/incremental domains from tokens without references.

Admit a candidate only if all firing cases in each split match the actual model
with zero errors, each split has positive incremental coverage, and at least
two independent groups per split have successful active source-target pairs.
Select greatest combined incremental coverage; ties prefer longer minimum
matches and then fewer required sources. Datalog makes this decision. If none
qualifies, retain all failures and make no final requests for this family.

## Confirmation if admitted

Before any final requests, fix fresh seeds and a full counterbalanced matrix,
exclude every discovery/development group and token, emit the standalone rule
and baseline-priority composite, and freeze their bytes plus reference-free
firing/incremental domains. Require `dl/equiv.dl` success on nonempty candidate,
composite and incremental domains, with positive incremental coverage per seed,
complete references and no errors or uncovered obligations. Preserve full-domain
failures due to abstention. Execute relocated token/domain-only runtime programs
and replay all checks without the model. No final-answer filtering or retuning.

The head-deletion study established a contribution on selected historical
contexts, not this symbolic rule. A successful finite-domain certificate would
remain scoped to the pinned oracle and stated domains; coverage and causal
source-substitution results are empirical. Anatomical sufficiency remains open.
