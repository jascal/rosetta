# Qwen guarded copy: finite-domain certificate

**proved**, relative to the recorded local Qwen2.5-0.5B-Instruct fieldrun oracle: the
[selected standalone circuit](selected/circuits.dl) matches every reference on its **96-case frozen firing domain**.

The guard requires a matching three-token suffix, at least two earlier matches with successor tokens, and agreement
of all those successors. It copies the common successor; otherwise it abstains. Runtime requires only Soufflé and
token facts. No weights, Python, fieldrun, or `whole.dl` are used at runtime.

- [Frozen domain and artifact hashes](frozen.json): `artifacts.selected.domain` contains the original instance IDs,
  determined by Datalog without reference facts before any test oracle queries.
- [Dataset](dataset.json): 144 test cases in eight sequence groups; 96 satisfy the guard, 48 do not.
- [Exact proof evidence](certificate-evidence/check-svsuedkp/certificate.json): `ndomain=96`, `ncover=96`,
  `nmiss=0`, `nuncov=0`, `nmissing=0`, `ninvalid=0`, `certified=true`.
- [Full-test obligation](certificate-evidence/check-y1s1p0_l/certificate.json): `ndomain=144`, `nmiss=0`,
  `nuncov=48`, `certified=false`. This circuit is not a complete replacement on the full test domain.
- [Protocol](protocol.json), [recorded references](references.json), and [all results](report.json) retain provenance.

Replay without the model:

```bash
python3 py/certificate.py reference/benchmarks/qwen25_05b_guarded_seed1/certificate-evidence/check-svsuedkp
```

On this Mac, set `DEVELOPER_DIR=/Library/Developer/CommandLineTools` for Soufflé preprocessing.
The verifier checks every supplied obligation, but this is a proof over these **96 recorded contexts only**.
Faithfulness on arbitrary new inputs remains **open**; coverage and precision on this corpus are **empirical**.
