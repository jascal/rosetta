# Scaling document experts to ~100 documents

How the build, runtime, and hub combine to serve an expert assembled from ~100 documents — and the measured numbers
from a real run. Three pieces, each independent and back-compatible (a single source / `[adapter]` / `url` is just the
1-element case of each).

## 1. Build — N documents × M adapter types (rosetta)

`build_expert` composes one expert from a list of documents, each ingested by a registered adapter
(`pack.adapters`), then merged:

```toml
[[document]]
adapter = "pretext"      # M adapter types: normrules | riscv_prose | pretext | latexml
source  = "$AATA_SRC"
prefix  = "aata"         # namespaces this document's passage ids → no cross-document collisions
[[document]]
adapter = "latexml"
source  = "paper.html"
prefix  = "ax2606_30639"
```

`Extraction.merge` concatenates passages + items, dedupes defines/statements by entity (first document wins). One
unbuildable document is **skipped with a warning** (the build proceeds on the rest); an unknown adapter *name* is a hard
config error. Output is document-encounter order (deterministic); no downstream step depends on passage order.

## 2. Runtime — index + parallelism (sgiandubh)

- `g_by_id` (id → passage) built once after load — O(1) passage resolution (strategy answers, `/lookup`).
- `retrieve_answer`'s cosine scan — the dominant per-query cost — is **sharded across threads**
  (`hardware_concurrency`, capped at 8; serial below 4000 passages where thread overhead isn't worth it). Shards are
  index-ordered and reduced with strict `>`, so results are **bit-identical to the serial scan**.
- Globals are write-once at startup, read-only while serving — no locks.

## 3. Hub — redundant replicas (claymore)

A spoke fronts N identical expert copies (`urls`/`replicas`). `call_replica` does round-robin load-spread + failover
(only on transport/HTTP failure — a legitimate abstain is a valid response). `/health` reports per-replica up/down.
This multiplies throughput and survives a replica going down.

## Measured (true 100+ document run)

A merged expert over **121 recent arXiv papers** (LaTeXML HTML via the `latexml` adapter; a runtime benchmark — the
arXiv corpus was not committed):

| metric | value |
|--------|-------|
| documents | 121 |
| passages | 16,327 |
| build time | 13.7 s |
| serial latency | 6.18 ms/query (parallel cosine, 16 cores) |
| concurrent throughput | 284 queries/sec (16 clients, one instance) |

For comparison, a 2-book PreTeXt expert (4748 passages) serves ~0.5–1 ms/query. Latency grows ~linearly with passage
count and divides across cores; claymore replicas multiply throughput beyond a single instance.

**Caveat at scale:** with many diverse documents the effective domain is wide, so off-domain *abstain* triggers less
(e.g. "capital of France" matched a knowledge-graph paper using that exact edge as an example). That's a corpus-curation
property, not a runtime defect — compose documents into an expert with a coherent domain in mind.
