"""pack.spec — load a declarative expert.toml (EXPERTS.md) into a build config.

An expert build is a reproducible experiment; the spec captures corpus + experimental design + targets so a build is
re-runnable, not buried in a shell invocation. Sections:
  [corpus]      text, questions, citation                       (CORPUS DESIGN)
  [model]       bundle, fieldrun                                (omit + [adapter] → model-free)
  [adapter]     name, source                                    (model-free structured source)
  [experiment]  holdout, off_domain, testset                    (EXPERIMENTAL DESIGN)
  [[benchmark]] name, set, target                               (targets; first-class slot)
  [gate]        min_precision, max_leak                         (HARD-FAIL thresholds)
  [reasoning]   rules, facts, closure                           (OPT-IN authored deduction — REASONING.md)
$ENV references (e.g. bundle = "$BUNDLE") are resolved from the environment.
"""
import os
import tomllib


def _expand(x):
    if isinstance(x, str):
        return os.path.expandvars(x)
    if isinstance(x, dict):
        return {k: _expand(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_expand(v) for v in x]
    return x


def load_spec(path):
    """Parse expert.toml → dict, with $ENV expanded. Raises a clear error on a malformed/missing file."""
    try:
        with open(path, "rb") as f:
            spec = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise ValueError(f"expert spec: cannot read {path} — {e}")
    if "corpus" not in spec and "adapter" not in spec:
        raise ValueError(f"expert spec: {path} has neither [corpus] nor [adapter] — nothing to build from")
    return _expand(spec)


def to_build_kwargs(spec, base="."):
    """Map a loaded spec → build_expert(**kwargs). Relative paths resolve against `base` (the spec's directory).
    A model (`[model].bundle`) ⇒ cover=True (build the smart tier); no model ⇒ model-free."""
    c = spec.get("corpus", {})
    m = spec.get("model", {})
    a = spec.get("adapter", {})

    def p(v):
        return os.path.join(base, v) if v and not os.path.isabs(v) else v

    gr = spec.get("grounding", {})
    bundle = m.get("bundle") or None
    return {
        "corpus": p(c.get("text")),
        "questions": p(c.get("questions")),
        "citation": c.get("citation", ""),
        "model": spec.get("name") or "rosetta-expert",
        "bundle": bundle,
        "fieldrun": m.get("fieldrun"),
        "cover": bool(bundle),                                   # a model present ⇒ build a cover (the smart tier)
        "adapter": a.get("name"),
        "adapter_source": p(a.get("source")) if a.get("source") else None,
        "dim": gr.get("dim", 300),                              # grounding embedding dim (0 = lexical, no download)
        "corpus_vectors": gr.get("corpus_vectors", False),
        "no_split": gr.get("no_split", False),
    }
