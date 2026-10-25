# Direct component readout: serializable, behaviorally inadequate

**Empirical**, six distinct frozen eight-token source windows covering twelve
historical diagnostic rows, eight head sets fixed before extraction. All
7,292,928 vocabulary scores are retained. Datalog verifies complete 151,936-token
coverage per feature/head set, computes global winners and ranks, and retains
every mismatch. No top-k or Python argmax selected the emitted facts.

| component | intended source targets matched / 12 | native model answers matched / 12 |
|---|---:|---:|
| copy head 16/9 | 5 | 3 |
| copy head 16/12 | 5 | 3 |
| copy pair | 5 | 3 |
| semantic pair | 2 | 1 |
| joint treated | 2 | 1 |
| matched copy control | 0 | 0 |
| matched semantic control | 0 | 0 |
| matched joint control | 0 | 0 |

This rejects the fixed source-value → output-projection → final-gain →
vocabulary-argmax hypothesis as a faithful output rule on this panel. Copy
components do carry target-related information, but their maxima do not
reproduce the full model's decisions. These selected, partly duplicated
historical rows are not a population accuracy estimate.

For example, source token ` tendency` projects to ` tangent` under head 16/9
and ` tí` under head 16/12. The intended token ranks 24th and 5th respectively.
For the `\ty` source, head 16/12 promotes `(y` first, actual ` y` second and
literal `\ty` third. These are retained counterexamples, not a prescription
to change the selected rank after observing the answers. Query-dependent
attention mixtures, residual information and later nonlinear blocks remain
outside this deliberately limited readout.

The emitted categorical mapping contains 48 eight-token-window facts. Its
two-file artifact is 4,054 bytes and runs in Souffle alone, with no neural
weights or numeric activations. Relocated execution reproduces all 96
categorical decisions for the twelve windows across eight head sets. This
proves serialization fidelity over that stated finite component domain; it
does not prove model equivalence. Unknown windows produce no readout.

All ten Datalog queries replay unchanged without a model. Evidence is in
[`qwen25_copy_value_readout`](../reference/benchmarks/qwen25_copy_value_readout/),
including frozen window representatives, complete projection scores, categorical
maps, raw component vectors, checkpoint identity and runtime comparison.

```bash
python3 py/audit_copy_value_readout.py
```

**Open:** no routed token-only copy circuit or fresh incremental equivalence
certificate results from this readout. The artifact is a rejected component
hypothesis, not the requested replacement. Per the user's subsequent steering,
the next model target is Qwen3.5-4B; see the
[migration preflight](qwen35-4b-migration.md).
