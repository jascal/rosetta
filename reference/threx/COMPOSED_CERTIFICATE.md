# Threx composed circuit — exact finite-domain certificate

**proved:** `circuit.dl` agrees with `whole.dl` over all 25 bearing pairs in contexts
`[0, 20, Bi, Bj, 19, 19, 7]`, with `Bi, Bj ∈ {21, 22, 23, 24, 25}`.

The fresh `dl/equiv.dl` query returned `certified()` with `ndomain=25`, `ncover=25`,
`nmiss=0`, `nuncov=0`, `nmissing=0`, and `ninvalid=0`.

[Retained evidence](certificate-evidence/check-_r8ai129/certificate.json) includes the exact circuit, verifier,
domain/token/reference facts, output relations, and SHA-256 hashes. The oracle provenance records the `whole.dl` hash.
This certificate is for the composed `circuit.dl`, not the temperature-dependent `circuits.dl` or unseen contexts.

Replay from the repository root without the model:

```bash
python3 py/certificate.py reference/threx/certificate-evidence/check-_r8ai129
```

Regenerate references and check against the model at build time:

```bash
python3 py/verify_threx.py --evidence-dir reference/threx/certificate-evidence
```
