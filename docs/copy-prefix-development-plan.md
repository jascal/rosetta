# Next computation hypothesis: short-suffix single-source induction

Status: **open**, development plan only. No observations for this new matrix
exist yet. This is not the final confirmation freeze.

The [layer-0 exchange experiment](copy-mechanism-discovery-outcome.md) does not
locate answer-specific computation in that MLP. It therefore supplies no
neuron-derived symbolic rule. The next candidate is explicitly a **behavioral
computation hypothesis**, supported or rejected by counterfactual tests; do not
call it a mechanistic extraction from layer 0.

Run B's frozen behavior-only composite already covers ordinary three-token
suffix consensus, and its shorter/longest-match routes require repeated target
tokens. The untested extension is copying a singly occurring successor from
one- or two-token query suffixes. This predicts fresh output tokens from their
role in the input rather than from a stored vocabulary table.

## Bounded development family

Six candidates: suffix length `{1,2}` crossed with minimum context length
`{0,16,32}`. A candidate fires only if:

- an earlier complete suffix match has a successor;
- all complete matches of that suffix agree on that successor;
- the successor occurs exactly once in the entire context;
- no earlier complete three-token suffix match exists;
- the context meets the candidate's length threshold.

Every condition and the prediction are Datalog. These conditions are intended
to identify baseline abstentions, but that must be checked against the actual
frozen baseline by a token-only query, rather than assumed. Baseline conflicts,
candidate disagreements and any baseline answers on these cases are reported.

Select only a candidate with nonempty firing in both development splits, zero
model mismatches on every firing case in each, and successful source-target
counterfactual pairs in at least two independent groups per split. Choose the
eligible candidate with greatest combined development coverage; ties prefer
the longer suffix, then the smaller length threshold. No successful candidate
means rejection, with the failed proposals retained. Selection is Datalog.

## Development inputs

Use seeds 401 and 402 as distinct development splits, with four groups per
sequence length `{8,12,24}`. Each group has distinct fresh tokens and supplies
query-prefix lengths `{1,2,3}`, paired all-source target substitutions, and four
layouts: plain, noise8, noise32, and a one-token near match followed by a decoy
inside an eight-token gap. Each source appears once. The three-token prefix is
a baseline-covered comparison, not counted as novel coverage by construction.

Independent negative groups supply no-repeat contexts and conflicting
two-token-suffix sources. All groups and vocabularies are disjoint across
splits and exclude the historical copy datasets and native-trace runs A–C used
in this line of work. Record and check the exact exclusion files and token
sets before collection. Do not reinterpret authored gold as a model reference.

The candidate is evaluated against model references, with authored gold and
paired model responses reported separately. Retain every mismatch. Development
failure is not a license to change this six-candidate family in place.

## Confirmation boundary

If a candidate is admitted, specify a fresh test matrix and seeds before any
final reference requests. Retain a copy of the baseline, candidate and composite
router, freeze their bytes, compute all firing and incremental domains from
tokens alone, then freeze the inputs/domains/verifiers. Test must retain varied
lengths, gaps, distractors, substitutions and controls, with disjoint groups and
vocabularies. No domain may be selected by final oracle agreement.

The requirements in `copy-mechanism-plan.md` still apply: complete references,
`dl/equiv.dl` certificates for the nonempty candidate and composite firing
domains, positive incremental coverage, full-domain failures retained, runtime
isolation, and model-free replay. A passing finite-domain behavioral circuit
does not by itself settle the anatomical mechanism question left open by the
layer-0 experiment.
