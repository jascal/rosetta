# Layer-0 exchange discovery outcome

**Empirical**, one pinned f32 Qwen2.5-0.5B-Instruct checkpoint, run C's historical
contexts, answer-position layer-0 MLP products only. This is discovery evidence,
not a fresh generalization test or a model-equivalence certificate.

The exact historical probe executable was rebuilt from fieldrun `0948cbe` in an
isolated checkout. Its SHA-256 is the original `54de8fc8…`; the reconstructed
weight file is also byte-identical (`3a2fb8e5…`). The current converter adds
sidecar metadata, so the sidecar differs. All 114 unpatched answers reproduce
the recorded references, and their reference logits pass the declared
0.002 absolute + 0.002 relative tolerance. Original numerical parity evidence
does not imply universal parity to the upstream implementation.

Datalog chose donors with the same final token and a different recorded answer,
using minimum donor ID to resolve ties. There were eligible donors for all 96
copy rows and six of the 18 control rows. The remaining controls are retained
as no-donor rows. Each paired recipient received identity, zero, and donor
exchange arms: **306 interventions**. Sources, observations and plans were
hashed in `frozen-discovery.json` before the first intervention.

| Arm | Copy rows preserving their answer | Copy rows adopting donor answer | Other answers |
|---|---:|---:|---:|
| Identity | 96/96 | 0 | 0 |
| Zero | 13/96 | 0 | 83 |
| Same-final-token exchange | 96/96 | 0 | 0 |

The six paired controls also preserve their own answer under identity and
exchange, and all six produce other answers under zeroing. Identity reference
and donor logits reproduce the unpatched logits exactly. Zero arms reproduce
the historical greedy answers, with reference logits within the predeclared
tolerance. The exchange vectors are numerically different; the query retains
per-pair L1 differences, maximum coordinate differences, relative L1 changes,
and both logit effects.

**Interpretation:** layer 0 contributes something necessary under deletion, but
this exchange experiment finds no transferable answer-specific computation in
its MLP output. A shared final-token transformation enabling downstream copying
is consistent with these results. It is not established: redundant codes,
unexchanged attention/residual information and the limited donor distribution
remain alternatives. No neuron subset, standalone computation, or additional
certified coverage has been extracted by this probe.

The unchanged run-B behavior baseline answers all 96 copy rows on these
discovery inputs, with one mismatch; it answers six controls correctly. The
other 12 controls are uncovered. This prevents counting a duplicate copy rule
as progress in coverage. A next computation hypothesis needs explicit tests on
the baseline's token-only abstention domain, alongside full-domain reporting.

## Evidence

Study: `reference/benchmarks/qwen25_copy_mechanism/`.
`summary.json` records `dl/copy_mechanism_summary.dl` outputs, and
`replay-audit.json` records five replayed queries with unchanged verdicts and
relations. One retained query is the initial preflight's syntax-error failure;
its checker was corrected before the preflight passed and before interventions
were frozen. This failure is not model evidence and was not removed.

The `certified` relations here certify **instrument/input integrity**, not
equivalence to the model. There is no `dl/equiv.dl` success for a new circuit.

```bash
python3 py/audit_copy_mechanism.py reference/benchmarks/qwen25_copy_mechanism
```

The model is absent from that replay. The observations, interventions, failed
preflight, passing preflight and all query sources/outputs are retained.
