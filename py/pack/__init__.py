"""rosetta.pack — the expert-packaging layer (CONVERGENCE.md).

The *factory* that assembles a deployable bounded-expert package (cover + curated answers + grounding + gram +
citations) for the thin sgiandubh runtime to serve. This layer DEPENDS ON the minimization core (`py/`, via
`pack.cover`); the core must never import `pack` (enforced by tests/test_pack.py). Single entry: `build_expert`.
"""
from .build import build_expert

__all__ = ["build_expert"]
