# The rosetta expert package v3 — additive raw-count evidence

v3 adds machine-readable `schema_version: 3` and, for energy-mode packages only, the top-level
shrinkage constant `alpha` plus optional raw per-key counts. It does not replace v2 confidence
fields or change serving. v2 unified origins and strata; v1 defined the original package and
runtime protocol. Old manifests without any v3 field remain valid.

**One package, three origins, one contract.** An expert package is a set of arbitrating rules
over a bounded domain; every answer is cited to its rule, its origin, and its evidence; anything
else is an abstention. Packages are the SUPERSET of what any pipeline produces — a package with
one origin populated is a degenerate case of this spec, not a different species.

## The two-axis trust model

Every rule carries two independent coordinates:

**`origin`** — where the knowledge came from (optional; default `teacher` for wyly emissions,
`document` for classic/normrule conversions):

| origin | producer | evidence |
|---|---|---|
| `document` | normrules / grounding pipelines over source text | span references into the grounding sidecar; verbatim-quotable |
| `teacher` | wyly v5 distillation from an LLM's decisions (manifest `origin_model` names it) | admission record (marginals, folds, calibration) — the only origin that can produce answers present in NO source text (e.g. estate registers) |
| `feedback` | attributable-feedback patches (human/eval corrections) | the patch provenance; C9-guarded no-regression |

**`stratum`** — how the answer is held at serve time (arbitration pools, lexicographic
fall-through at `strata_tau`):

| stratum | label | semantics |
|---|---|---|
| 0 | **attested** | the served (canonical query + answer) is verbatim-checkable against the grounding sidecar — assigned AT SERVE TIME, not stored |
| 1 | **certified** | full admission (deployment-calibrated marginals, fold-stable) |
| 2 | **supported** | calibrated non-winners (query fired-accuracy ≥ 0.5); serve only when stratum-1 confidence < τ |

An answer can be teacher-origin AND attested (compiled rule whose output happens to be verbatim
in the source — the strongest case), or teacher-origin and merely certified (estate answers,
necessarily). Decision payloads carry `origin`, `stratum`, `citation`, and (when canonicalized)
`canonical` + bindings — consumers never need to know which pipeline built the package.

## The document-origin ingestion route (`py/convert_classic.py`)

Classic content packages (normative `knowledge.tsv` items + word vocabulary) convert into this
spec: a WordLevel `bundle.tokenizer.json` built THROUGH the runtime's own pretokenizer, gated
ngram rules with `origin:"document"` and norm-id citations, and the item texts as the grounding
sidecar. **Support 1 is valid for document origin** — the document saying it once is the
authority; determinism still gates ambiguous continuations. Known v1 limit: word-level
detokenization spacing can miss the attestation string-match on punctuation-fused tokens.

## The grounding sidecar

Manifest key `"grounding"`: a path (relative to the package) to plain text — the source corpus
or document set. Serve-time attestation: if the canonicalized query's (statement + answer)
string occurs in the sidecar, the decision is upgraded to stratum 0 (`attested`) and may quote
the containing span. The sidecar is optional; without it, stratum 0 is simply never assigned.

## Calibration requirements (normative)

Every arbitration participant MUST be calibrated on the deployment distribution where query
data exists: admitted rules (query-blended marginals), stratum-2 qualification (query
fired-accuracy), incumbents (eviction / champion restarts), and the counts tier (per-tail
calibration). Runtimes MUST assign `stratum` per idiom at parse time and pass it at every
arbitration site (see rosetta #43 / sgiandubh #28 for the failure mode this prevents).

---

# v1 schema and runtime protocol (unchanged, extended by the above)

The package is what rosetta emits and a thin runtime (`py/serve_package.py`, and the future sgiandubh C++) consumes —
the rosetta→sgiandubh convergence artifact. (For *who builds what* — rosetta is the sole builder, sgiandubh the thin
server+REPL — see [`CONVERGENCE.md`](./CONVERGENCE.md).) Two producers, one schema:

- **`idiom_learn.py --package` → `emit_expert_package`** (the strong path): causally-confirmed idioms + gated n-grams.
- **`abstain_emit.py`** (the observational path): gated n-grams only (a flat manifest; backward-compatible).

A package is a directory: `circuits.expert.dl` + `run.dl` (souffle decode, optional for a host-side runtime) +
**`manifest.json`** (the decision table the runtime actually needs). All are gitignored (reproducible via the flags).

## `manifest.json`

```json
{ "model": "...", "trusted_idioms": N, "gated_ngrams": M, "induction_ood": K,
  "minsupp": 3, "mindet": 1.0, "schema_version": 3, "alpha": 2.0,
  "rules": [ <rule>, ... ] }
```

`schema_version` and `alpha` are emitted only for energy-mode packages in v3, so classic
`support-weighted` package bytes are unchanged. `alpha` is the denominator pseudocount in the
shipped shrunk confidence, `cnt/(tot+alpha)`.

Every rule carries **`tier`** (`trusted` | `gated`) and **`basis`** (`causal` | `observational`) + `id` + provenance
(`cite` = supporting corpus positions; `citation` = resolved passage strings when a `corpus_meta.json` offset→citation map
was present at build). Three rule kinds:

| `kind` | tier / basis | fields | how the runtime fires it |
|---|---|---|---|
| `gate` | trusted / causal | `frame:{offset:token}`, `slot:k`, `table:{token:out}`, `causal`, `support`; energy-mode table rules may add `counts:{token:[cnt,tot]}` | frame matches (`ctx[-offset]==token` ∀) ∧ `ctx[-slot] in table` → `table[ctx[-slot]]` |
| `compose` | trusted / causal | `frame`, `operands:[k1,k2]`, `valmap:{token:value}`, `sum:{value-sum:out}`, `causal`, `extrapolate` | frame matches ∧ both operands in `valmap` ∧ `valmap[op1]+valmap[op2] in sum` → `sum[...]` |
| `ngram` | gated / observational | `ctx:[token ids]`, `out`, `support`, `determinism`; energy-mode rules may add `counts:[cnt,tot]` | longest suffix where `ctx` matches → `out` |
| `relation` | trusted / causal | `eq:[[i,j],...]`, `copy:k`, `confidence` | `ctx[-i]==ctx[-j]` ∀ pairs → `ctx[-k]` (routed after n-grams, above succession/induction; the learned repetition rule is `eq=[[1,2]], copy=1`) |
| `khop` | trusted / causal | `lo:int`, `hi:int`, `confidence` | 2-hop: rightmost earlier match of `ctx[-1]` (query, excluding the query's own position) → bridge = its successor; rightmost earlier match of `bridge` EXCLUDING the bridge's own site (`i != p1+1`, the load-bearing exclusion) → its successor is the prediction (routed OOD, after induction; mirrors pil `mir_khop2`/`_DL_KHOP` exactly) |

**Offsets are 1-based from the end** (offset 1 = the last context token, offset k = `ctx[-k]`). JSON keys are strings;
normalize to ints on load.

## Runtime protocol (`load_package` + `serve`, host-side — no souffle, no model)

1. **Tokenize** the query with the model's tokenizer (`bundle.tokenizer.json`) — the cover lives in BPE-token space.
2. **TRUSTED idioms first**, in manifest order — each is a structured lookup (frame-match + table/sum). Causally
   confirmed, so they're ungated.
3. **GATED n-grams** — longest matching suffix wins. Kept at build only by support/determinism ("fire only if confident",
   *not* gating inside an n-gram), so a match is already a confident match.
4. **ABSTAIN** if nothing fires → defer to a backstop, or refuse (the bounded expert).
   (Routed OOD circuits fire between 3 and 4, most-specific first: `relation` → `succession` → `induction` → `khop`.
   Trusted non-table kinds may ship a `confidence` field (held-out fired-accuracy) so a support-weighted
   runtime can arbitrate tiers per answer instead of by fixed priority.)

### Two-layer packages: derived predicates (`derived`) + the `dgate` kind

A manifest may carry a top-level `derived` array of FEATURE EXTRACTORS -- each a certified
program over the context, defined extensionally so the package stays host-side and exact:

```json
"derived": [{"id": "mate0", "kind": "bracket-mate", "openers": [9, 60, ...], "closers": [...]}]
```

The extractor REGISTRY (every kind Soufflé-certified against its tensor mirror, window-by-window
on real corpus samples — pil `wyly_mate_certify.py` 256/256 and `wyly_derived_certify.py`
192/192 per kind):

| kind | fields | feature value |
|---|---|---|
| `bracket-mate` | `openers`, `closers` | the innermost UNCLOSED opener (one shared stack), −1 if none |
| `recent-member` | `members` | the most recent token in the member set (e.g. clause openers), −1 if none |
| `recent-unique` | `members` | the most recent member occurring EXACTLY ONCE in the context (the distinguished-referent role), −1 if none |
| `bracket-depth` | `openers`, `closers`, `cap` | the balance counter: total depth clamped to [0, cap] |
| `prev-occ` | `of` (a derived-def id), `succ` | CHAINED role: the previous occurrence of the referenced feature's token; with `succ` the entity ECHO (what followed this referent last time) |
| `since-member` | `members`, `cap` | DISCOURSE: tokens since the most recent member (sentence enders → position-in-sentence; connectives → position-in-move); cap+1 if none |
| `member-parity` | `members` | DISCOURSE: count of members mod 2 (quote marks → the inside-quotation / attribution-scope indicator) |

All extractors may carry `succ: k` (role composition — the feature is read at position+k). Defs
are evaluated in listed order, so a def may reference any earlier def via `of`.

`dgate` rules gate on a derived feature jointly with the last token -- the first rules whose
guard is a computed ROLE rather than a token pattern (a two-layer program: derive the predicate,
gate on it):

```json
{"kind": "dgate", "feature": "mate0", "table": {"<mateTok>:<lastTok>": out, ...},
 "confs": {"<mateTok>:<lastTok>": 0.71, ...},
 "counts": {"<mateTok>:<lastTok>": [5, 7], ...}}
```

The optional energy-mode `counts` map is parallel to `table` and `confs`; `dgate` and `dgate2`
use the same colon-separated composite keys. Missing entries mean raw evidence is unavailable,
not `(0, 0)`.

First producer: pil wyly_lm_v5 (the sleep judge admitted the mate gate on the Isabelle corpus,
where it set the certified-core arc best). Runtimes: `py/serve_package.py` (support-weighted
cover) + sgiandubh `rosetta_package.h`.

### estate / estate2 (entity-state kinds)

`estate`: per-attribute ENTITY-STATE REGISTER — last-writer-wins (entity, value) fold with an
avoid-filter and an answer-slot signature; the EAV family is realized as one register per
attribute with arbitration selecting the attribute. `estate2`: the WORLD-STATE fold — loc[entity]
from movement verbs, holder/loc/history[object] from mined take/drop verbs (drops freeze
location); mode "is" answers current object location, mode "before" answers the history
predecessor of a reference location. All member sets (entities, locations, objects, verb
classes incl. take-vs-drop semantics) are self-grounded from the corpus.

### canon (query canonicalization)

Optional manifest section: a mined template inventory ({E}-slotted prefixes clustered into
properties by shared (entity, value) pairs) + entity list. Runtimes parse free-phrased queries
onto covered templates BEFORE key lookup (no parse -> clean abstention, never raw fragment
matching) and surface the canonical form + bindings in the decision payload — the
silent-substitution defense: consumers always see the question actually answered.

### Strata (labeled trust pools)

Any rule may carry `"stratum": n` (default 1) and the manifest `"strata_tau"` (default 0.35).
A support-weighted runtime arbitrates stratum 1 first; where its best confidence is below tau,
stratum-2 candidates may claim the answer (lexicographic (stratum, confidence) fall-through --
C10 applies per stratum). Stratum 2 holds calibrated non-winners of the admission race
(fired-accuracy-qualified): answers carry their stratum, so consumers can distinguish
certified from supported from tentative -- graceful degradation with labeled trust.

### The support-weighted cover (`"cover": "support-weighted"`)

A manifest may declare `"cover": "support-weighted"` at the top level. A conforming runtime then
replaces the fixed tier priority with per-answer ARBITRATION: every applicable rule fires and the
answer with the highest confidence wins (ties keep the first candidate — idioms in manifest order,
then n-grams longest-first). Confidences are what the package ships:

- `ngram` rules carry `confidence` = Laplace-shrunk per-key determinism `c/(t+α)`;
- `gate` rules carry `confs` = a per-content-key confidence map parallel to `table`;
- trusted kinds (`relation`, `induction`, `khop`, …) carry the scalar `confidence` (held-out fired-accuracy).

Energy-mode v3 emission carries raw evidence alongside those floats:

- a single-key `ngram` has `counts: [cnt, tot]`;
- table-shaped `gate`, `dgate`, and `dgate2` rules have a `counts` dictionary keyed exactly like
  `confs`, with `[cnt, tot]` values;
- the online counts-tier `ngram` also has `total` beside its existing `support` and redundantly
  carries `counts: [support, total]`, giving both runtimes one uniform n-gram parse shape.

All fields are optional on load: absence is unknown evidence, never zero. Counts are parsed but
are not used by v3 serving. A future multi-step energy beam may rank exact ratios with integer
cross-multiplication. It must not do so at the `M=1`, `beam_width=1` corner: manifest confidence
is rounded to four decimals, while integer comparison ranks the exact ratio and can differ on a
rounding tie. That corner therefore continues to delegate bit-exactly to float `serve_sw`.

This is the argmax policy whose dominance over every fixed priority is kernel-checked
(i-orca `examples/concept_grounding/Arbitration.thy`: `argmax_policy_optimal`), with CALIBRATION as
the stated premise — the shipped confidences must approximate true per-key accuracy, which is why
producers Laplace-shrink and support-gate them (`miscalibration_bound` quantifies the cost of
inflation: within 2ε·total-weight of optimal under ε-miscalibration). Reference implementation:
`py/serve_package.py` (`serve_sw`/`decide`); C++ port: sgiandubh `src/rosetta_package.h`. First
producer: pil `wyly_lm_v5` `WYLY_EMIT=1` (served-vs-learner parity verified to 4 decimals on
11.5k/12k held-out windows per dataset).
5. **Cite** the answer: the fired rule's `citation`/`cite`. (`circuits.expert.dl` additionally emits `cprov(inst,ruleid)`
   for the souffle path.)

The whole package — idioms and n-grams — is host-side consumable; souffle (`circuits.expert.dl`/`run.dl`) is an
equivalent engine path, not required for a host-side runtime.

## Two regimes

- **Exact backstop** (rosetta `whole.dl`): abstain → exact recompute; the package compresses the confident part.
- **Partial / no backstop** (sgiandubh bounded expert): abstain → honest refusal. The `trusted` (causal) tier is the
  safe core — a backstop-less expert must not trust purely `observational` rules, which is why the tier/basis tagging exists.
