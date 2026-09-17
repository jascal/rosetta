"""pack.spec — load a declarative expert.toml (EXPERTS.md) into a build config.

An expert build is a reproducible experiment; the spec captures corpus + experimental design + targets so a build is
re-runnable, not buried in a shell invocation. Sections:
  [corpus]      text, prose, questions, citation                (CORPUS DESIGN; prose = extra citable passages)
  [model]       bundle, fieldrun                                (omit + [adapter] → model-free)
  [adapter]     name, source                                    (model-free structured source — single document)
  [[document]]  adapter, source, …opts                          (N documents of M adapter types → ONE expert)
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
    if not any(k in spec for k in ("corpus", "adapter", "document")):
        raise ValueError(f"expert spec: {path} has no [corpus]/[adapter]/[[document]] — nothing to build from")
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
    rs = spec.get("reasoning", {})                              # opt-in authored-reasoning tier (REASONING.md)
    bundle = m.get("bundle") or None

    # [[document]] — N documents of M adapter types composed into one expert (build merges their Extractions).
    docs = spec.get("document", [])
    documents = [{"adapter": d.get("adapter") or d.get("name"),
                  "source": p(d.get("source")),
                  "opts": {k: v for k, v in d.items() if k not in ("adapter", "name", "source")}}
                 for d in docs] or None

    return {
        "documents": documents,
        "corpus": p(c.get("text")),
        "prose": p(c.get("prose")) if c.get("prose") else None,   # extra citable passages concatenated into grounding
        "questions": p(c.get("questions")),
        "citation": c.get("citation", ""),
        "model": spec.get("name") or "rosetta-expert",
        "bundle": bundle,
        "fieldrun": m.get("fieldrun"),
        "cover": bool(bundle),                                   # a model present ⇒ build a cover (the smart tier)
        "adapter": a.get("name"),
        "adapter_source": p(a.get("source")) if a.get("source") else None,
        "adapter_opts": {k: v for k, v in a.items() if k not in ("name", "source")},   # e.g. prefix, chapters
        "dim": gr.get("dim", 300),                              # grounding embedding dim (0 = lexical, no download)
        "corpus_vectors": gr.get("corpus_vectors", False),
        "no_split": gr.get("no_split", False),
        "inventory": bool(rs.get("inventory", False)),          # [reasoning] inventory = true → count/list aggregates
        "inventory_label": rs.get("label", "instruction"),      # the thing counted ("term"/"section"/"option")
        "inventory_prefix": rs.get("prefix", "riscv:inventory"),  # the count/list citation-handle namespace (per domain)
        "reasoning_rules": rs.get("rules"),                     # e.g. "ergo:aggregate" → ../ergo/aggregate.dl
    }
