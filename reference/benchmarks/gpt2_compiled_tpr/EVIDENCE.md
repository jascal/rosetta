# Evidence retained

- **SVO**: complete, including `certificate-evidence/`. It holds the TPR layer's failed `equiv.dl` certificate
  (1 mismatch), which can be replayed with `python3 py/certificate.py <check-dir>`.
- **COPY**: `certificate-evidence/` is **omitted**. It is four copies (about 46 MB) of the 10 MB weight table, staged
  once per check. The COPY verdicts are:
  - the idiom layer's failed certificate: 122 mismatches, listed in `report.json`;
  - an empty TPR domain.
  The weight table is kept compressed as `COPY/tpr/w.facts.gz`; run `gunzip -k` before re-running. Re-running
  `py/benchmark_compiled_tpr.py run` against the same bundle regenerates the evidence.
