# FAILED independent finite-domain equivalence

Candidate: frozen option 5, minimum match 32, support 2, length margin 4,
strict categorical prior agreement. Seeds 921/922, all 612 preregistered contexts.

| domain | cases | mismatch | uncovered | missing reference | invalid | verdict |
|---|---:|---:|---:|---:|---:|---|
| frozen firing | 240 | 13 | 0 | 0 | 0 | FAIL |
| frozen incremental | 96 | 13 | 0 | 0 | 0 | FAIL |
| full trial | 612 | 13 | 372 | 0 | 0 | FAIL |

The authoritative verdicts are in `certificates.json`, computed by
`dl/equiv.dl`; `summary.json` also rejects the fixed confirmation gates.
All references are the global argmax of the pinned native Qwen2.5-0.5B-Instruct
host. This is a failed model-agreement claim, not a truth test.

`runtime.json` passes: copying only `artifact/circuits.dl`, `artifact/run.dl`
and token/domain facts to a fresh directory reproduces all frozen decisions
and abstentions in Souffle alone. Runtime contains 3,240 categorical logical
facts and no neural inference. `replay-audit.json` confirms exact model-free
replay of eight queries, including failures. These checks do not repair the
13 model disagreements. The artifact is retained as a rejected candidate.

See [outcome](../../../docs/copy-prior-guard-confirmation-outcome.md) and
the immutable `protocol.json`/`frozen.json` for scope and provenance.
