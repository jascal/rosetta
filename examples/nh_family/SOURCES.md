# NH family-law expert — sourcing, citation scheme, build plan

A **bounded legal slice** chosen as the first legal expert: small enough to run on the current runtime, valuable enough
to be real, and a clean fit for *cite-the-exact-source-and-abstain*. **Not legal advice** — an informational retrieval
tool over public-domain primary sources; every answer is verbatim text + its citation, and it abstains rather than
inventing law.

## Scope (the corpus)

**Statutes — NH RSA, Title XLIII "Domestic Relations"** (the family-law core):
- RSA 457 (Marriages), 458 (Divorce, Nullity, Separation), 458-A (UCCJEA — custody jurisdiction),
  458-C (Child Support Guidelines), 459 (Support of dependents), 461-A (Parental Rights & Responsibilities — custody),
  plus related: 460 (mediation), 173-B (domestic violence protective orders), 169-C (abuse & neglect) as needed.
- ~a few hundred sections, ~tens of thousands of words → **~RISC-V-to-5× scale** (fits the current in-memory runtime).

**Case law — NH Supreme Court family-law opinions** (add *after* statutes validate; the larger, freshness-sensitive,
lower-precision half): start with the last ~10 years of family-law opinions (custody, support, divorce, parental rights).

## Sourcing (all public)

- **RSA**: NH General Court — `gencourt.state.nh.us` (the RSA is public). Fetch the Title XLIII chapter texts.
- **Opinions**: public domain (US court opinions aren't copyrightable). Bulk via **CourtListener** (free API/bulk) or
  **Caselaw Access Project** (CAP, Harvard). Filter to NH Supreme Court, family-law topics.
- Both are committable (public domain) — but large; the *raw* + built index are gitignored (reproducible via the adapter).

## The citation-id scheme (the callable handle)

Reuses the runtime's `"id · Facet"` section convention (so the cite-as-handle + `/lookup` we built work unchanged):

| source | passage `section` | `citation_id` (the handle) |
|---|---|---|
| statute | `RSA 461-A:6 · Parental Rights & Responsibilities` | `RSA 461-A:6` |
| statute subsec | `RSA 458-C:3, II · Child Support Guidelines` | `RSA 458-C:3, II` |
| opinion | `In re R.A., 153 N.H. 82 (2005) · custody` | `In re R.A., 153 N.H. 82 (2005)` |

So an answer cites `[1] RSA 461-A:6`, and `GET /lookup?id=RSA 461-A:6` (or, via claymore, `?spoke=nh_family&id=…`)
refetches the exact provision verbatim. Legal citations *are* the canonical handles — pinpoint, verifiable, stable.

## The `nh_legal` adapter (TODO: `pack/adapters/nh_legal.py`)

Like `normrules`, but for legal structure (no model):
- **RSA** → one passage per section: `section = "RSA <chap>:<sec> · <chapter title>"`, `text = <section text>` (preserve
  subsection structure; emit subsections as their own pinpoint passages where useful).
- **opinions** → chunk each opinion (headnote / issue / holding paragraphs): `section = "<bluebook cite> · <topic>"`,
  `text = <chunk>`. Carry the year for currency.

## Retrieval at legal scale / quality

This *slice* runs on the current runtime. Two upgrades matter as it grows (and for legal precision even at this size):
- **Index**: the in-memory linear cosine scan is fine for a few thousand passages; the full NH corpus (millions) needs an
  ANN index (FAISS/HNSW) or BM25, sharded/mmap'd. (See the "all NH law" scaling discussion.)
- **Embeddings**: mean-pooled GloVe is the weak link; legal language wants a real embedding model (breaks model-free at
  runtime) or **BM25** (stays model-free) — likely hybrid. The scorecard (`min_precision 0.95`) will force this decision.

## Legal-specific guardrails

- **Bounded + cited + abstain** is the safety design: it returns the verbatim provision/opinion with its cite, or refuses.
- **No LLM synthesis** for law (claymore `deterministic` mode, not `llm`) — synthesis reintroduces hallucination, which is
  unacceptable for legal text. The optional `[reasoning]` tier uses *sound, authored* rules, not model guesswork.
- **Currency** — a build-time snapshot; version provisions by effective date and rebuild on amendments / new opinions.
- **Pinpoint citations** — the id carries chapter:section[:subsection] / case pincite so `/lookup` is exact.

## Build (once sourced + the adapter exists)

```bash
FIELDRUN unused (model-free).  .venv/bin/python build_expert.py examples/nh_family/expert.toml
# → examples/nh_family/package/ (knowledge.tsv + wordvec.txt + scorecard.json), then serve as a claymore spoke "nh_family".
```

## Pragmatic order
1. statutes only (RSA Title XLIII core) → build → scorecard → tune retrieval to hit `min_precision 0.95`.
2. add the authored `[reasoning]` facts (key cross-refs/definitions) — measure the lift.
3. add recent case law → re-measure (expect lower precision; this is where the index/embedding upgrades land).
