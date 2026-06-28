# CLAUDE.md

See [`AGENTS.md`](./AGENTS.md) for the canonical orientation. In brief: **rosetta** minimizes a whole LLM into
provably-faithful Datalog — it consumes `fieldrun`'s `whole.dl` and certifies each extracted circuit equivalent to the
model via a Datalog query (`dl/equiv.dl`). **Datalog is the implementation**; Python (`py/`) only drives `souffle`.

The one rule that governs everything: a circuit is `proved` only when `dl/equiv.dl` returns a clean certificate
(`nmiss=0 ∧ nuncov=0`) over a stated domain — otherwise it is `open`. The verdict is a query result, never a claim.

rosetta is the **minimization arm** of the PIC certified-compression loop (`i-orca` verifies · `fieldrun` analyzes ·
`pil` learns · rosetta minimizes); it does not modify `fieldrun` (emitter changes go in a `fieldrun` branch + PR).
