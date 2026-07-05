# The rosetta expert package (schema + runtime protocol)

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
  "minsupp": 3, "mindet": 1.0, "rules": [ <rule>, ... ] }
```

Every rule carries **`tier`** (`trusted` | `gated`) and **`basis`** (`causal` | `observational`) + `id` + provenance
(`cite` = supporting corpus positions; `citation` = resolved passage strings when a `corpus_meta.json` offset→citation map
was present at build). Three rule kinds:

| `kind` | tier / basis | fields | how the runtime fires it |
|---|---|---|---|
| `gate` | trusted / causal | `frame:{offset:token}`, `slot:k`, `table:{token:out}`, `causal`, `support` | frame matches (`ctx[-offset]==token` ∀) ∧ `ctx[-slot] in table` → `table[ctx[-slot]]` |
| `compose` | trusted / causal | `frame`, `operands:[k1,k2]`, `valmap:{token:value}`, `sum:{value-sum:out}`, `causal`, `extrapolate` | frame matches ∧ both operands in `valmap` ∧ `valmap[op1]+valmap[op2] in sum` → `sum[...]` |
| `ngram` | gated / observational | `ctx:[token ids]`, `out`, `support`, `determinism` | longest suffix where `ctx` matches → `out` |
| `relation` | trusted / causal | `eq:[[i,j],...]`, `copy:k`, `confidence` | `ctx[-i]==ctx[-j]` ∀ pairs → `ctx[-k]` (routed after n-grams, above succession/induction; the learned repetition rule is `eq=[[1,2]], copy=1`) |

**Offsets are 1-based from the end** (offset 1 = the last context token, offset k = `ctx[-k]`). JSON keys are strings;
normalize to ints on load.

## Runtime protocol (`load_package` + `serve`, host-side — no souffle, no model)

1. **Tokenize** the query with the model's tokenizer (`bundle.tokenizer.json`) — the cover lives in BPE-token space.
2. **TRUSTED idioms first**, in manifest order — each is a structured lookup (frame-match + table/sum). Causally
   confirmed, so they're ungated.
3. **GATED n-grams** — longest matching suffix wins. Kept at build only by support/determinism ("fire only if confident",
   *not* gating inside an n-gram), so a match is already a confident match.
4. **ABSTAIN** if nothing fires → defer to a backstop, or refuse (the bounded expert).
   (Routed OOD circuits fire between 3 and 4, most-specific first: `relation` → `succession` → `induction`.
   Trusted non-table kinds may ship a `confidence` field (held-out fired-accuracy) so a support-weighted
   runtime can arbitrate tiers per answer instead of by fixed priority.)
5. **Cite** the answer: the fired rule's `citation`/`cite`. (`circuits.expert.dl` additionally emits `cprov(inst,ruleid)`
   for the souffle path.)

The whole package — idioms and n-grams — is host-side consumable; souffle (`circuits.expert.dl`/`run.dl`) is an
equivalent engine path, not required for a host-side runtime.

## Two regimes

- **Exact backstop** (rosetta `whole.dl`): abstain → exact recompute; the package compresses the confident part.
- **Partial / no backstop** (sgiandubh bounded expert): abstain → honest refusal. The `trusted` (causal) tier is the
  safe core — a backstop-less expert must not trust purely `observational` rules, which is why the tier/basis tagging exists.
