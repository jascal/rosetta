"""pack.cover — the ONE point where pack touches the minimization core.

The cover is the *smart* tier: causally-confirmed idioms (generalize OOD) + support/determinism-gated n-grams, with
provenance — smart WITHOUT sacrificing accuracy (causal confirmation + abstention) or speed (lookups). build-expert
builds it whenever a model bundle is available; it replaces the old bare `gram` tier entirely (CONVERGENCE.md).

Dependency direction is one-way: pack → core, never the reverse (tests/test_pack.py enforces it). For now we delegate
to the core's existing `--package` pipeline (idiom_learn, which calls emit_expert_package internally); the cleaner
direct-function-call refactor (expose core.build_package()) is a tracked follow-up — it touches the core API, not pack.
"""
import os
import subprocess
import sys

CORE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # rosetta/py — the minimization core
IDIOM_LEARN = os.path.join(CORE, "idiom_learn.py")


def build(model_dir, *, n=1400, w=8, minsupp=3, mindet=1.0, crisp=False):
    """Extract the cover for an already-prepared model dir → <model_dir>/package/manifest.json. Returns its path.

    `model_dir` must hold the rosetta extraction inputs (corpus.json + the oracle source: a fieldrun --serve via
    FIELDRUN_SERVE, or whole.dl, or cached logits) — the same inputs the core's CLI consumes."""
    cmd = [sys.executable, IDIOM_LEARN, str(n), str(w), model_dir, "--package",
           f"--minsupp={minsupp}", f"--mindet={mindet}"]
    if crisp:
        cmd.append("--crisp")
    subprocess.run(cmd, check=True)
    return os.path.join(model_dir, "package", "manifest.json")
