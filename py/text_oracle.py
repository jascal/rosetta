"""Text-package oracle for the generic M-step beam engine."""
from __future__ import annotations

from beam_engine import Path, Step
from serve_package import enumerate_candidates, serve_sw


class TextOracle:
    """Pure branching oracle over the firing rules of a rosetta expert package."""

    def __init__(
        self,
        ctx,
        idioms,
        ngrams,
        W,
        m_derived=None,
        cmap=None,
        m_tau=None,
    ):
        self.ctx = tuple(ctx)
        self.idioms = idioms
        self.ngrams = ngrams
        self.W = W
        self.m_derived = m_derived
        self.cmap = cmap
        self.m_tau = m_tau

    def initial_state(self):
        return self.ctx

    def expand(self, ctx):
        candidates = enumerate_candidates(
            ctx,
            self.idioms,
            self.ngrams,
            self.W,
            m_derived=self.m_derived,
            cmap=self.cmap,
        )
        best_by_token = {}
        for candidate in candidates:
            incumbent = best_by_token.get(candidate.answer)
            if incumbent is None or candidate.conf > incumbent.conf:
                best_by_token[candidate.answer] = candidate

        children = []
        for token, candidate in best_by_token.items():
            meta = dict(candidate.meta)
            meta["conf"] = candidate.conf
            meta["origin"] = meta.get("origin", "teacher")
            meta["stratum"] = (
                1 if candidate.stratum <= 1 else candidate.stratum
            )
            step = Step(
                key=(len(ctx), token),
                margin=0.0,
                counts=candidate.counts,
                meta=meta,
            )
            children.append((tuple(ctx) + (token,), [step]))
        return children

    def extract_commit(self, winning_path: Path):
        if not winning_path.steps:
            return None
        first = winning_path.steps[0]
        return first.key[1], first.meta

    def fallback(self):
        return serve_sw(
            self.ctx,
            self.idioms,
            self.ngrams,
            self.W,
            m_derived=self.m_derived,
            cmap=self.cmap,
            m_tau=self.m_tau,
        )
