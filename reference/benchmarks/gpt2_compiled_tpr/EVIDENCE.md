# Evidence retained

- **SVO**: complete, including `certificate-evidence/`. It holds the TPR layer's failed `equiv.dl` certificate
  (1 mismatch), which can be replayed with `python3 py/certificate.py <check-dir>`.
- **COPY**: `certificate-evidence/` is **omitted**. It is four copies (about 46 MB) of the 10 MB weight table, staged
  once per check. The COPY verdicts are:
  - the idiom layer's failed certificate: 122 mismatches. `report.json` holds the first 50; all 122 are in
    `audit.json` (`tasks.COPY.certificates.idiom.mismatches`), re-derived by `dl/equiv.dl` on the frozen domain;
  - an empty TPR domain.
  The weight table is kept compressed as `COPY/tpr/w.facts.gz`; run `gunzip -k` before re-running. Re-running
  `py/benchmark_compiled_tpr.py run` against the same bundle regenerates the evidence.
- `audit.json` is the post-run audit (`py/audit_compiled_tpr.py`; see the outcome doc). It needs `w.facts`
  unpacked as above.
