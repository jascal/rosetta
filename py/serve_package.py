#!/usr/bin/env python3
"""rosetta · serve_package.py — the REFERENCE THIN RUNTIME for a rosetta expert package, and the spec for sgiandubh's C++.

The convergence (rosetta = builder, sgiandubh = thin runtime): rosetta emits the package (circuits.abstain.dl + manifest.json,
see py/abstain_emit.py); a runtime consumes it. This reference consumer proves the package is servable **host-side with NO
souffle and NO model** — the manifest IS the compiled cover (a decision table). It does exactly what sgiandubh's runtime will:

    tokenize(query)  →  TRUSTED idioms (frame-match) → GATED n-grams (longest suffix) → {answer, tier, basis, citation}  or  ABSTAIN

The RUNTIME loads only manifest.json (load_package + serve) — no corpus, no souffle, no model. The corpus is used only by the
BUILDER (make_package, = py/abstain_emit) and by the held-out scorecard (eval), kept separate from serving.

KEY FINDING surfaced here (matters for sgiandubh): the cover is in the model's BPE-TOKEN space, so the runtime must tokenize
the query with the model's tokenizer (bundle.tokenizer.json). sgiandubh is currently word/text-based, so consuming the rosetta
cover requires a BPE tokenizer in its C++ runtime — the real Stage-2 dependency (this script is the port spec).

Deps: tokenizers (for the NL-query path). Usage: .venv/bin/python py/serve_package.py [model_dir] [W] [minsupp] [mindet]
"""
import json, os, sys, random
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abstain_emit import build_tab, confident_rules, emit, emit_manifest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_package(md, W, minsupp, mindet):
    """BUILDER side (= py/abstain_emit): build the cover from corpus.json, emit circuits.abstain.dl + manifest.json.
    Returns the held-out windows (for the eval scorecard only — NOT used by the runtime)."""
    ids = json.load(open(os.path.join(md, "corpus.json")))["ids"]
    N = min(len(ids) - 1, 40000)
    wins = [(tuple(ids[i - W:i]), ids[i], i) for i in range(W, N)]
    random.Random(0).shuffle(wins)
    cut = int(len(wins) * 0.7); train, hold = wins[:cut], wins[cut:]
    tab, cites = build_tab(train, W)
    rules = confident_rules(tab, cites, W, minsupp, mindet)
    ridmap = emit(os.path.join(md, "circuits.abstain.dl"), rules, W, minsupp, mindet)
    meta_p = os.path.join(md, "corpus_meta.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else None
    emit_manifest(os.path.join(md, "manifest.json"), ridmap, os.path.basename(md), W, minsupp, mindet, meta)
    return hold


def _cite(r):
    return r["citation"] if r.get("citation") else (f"corpus@{r['cite'][:3]}" if r.get("cite") else "")


def load_package(manifest_path):
    """THE THIN RUNTIME state from manifest.json ONLY (no corpus, no souffle, no model). Handles the TIERED package
    (idiom_learn --package: causal idioms + gated n-grams) and is backward-compatible with the flat n-gram manifest
    (abstain_emit: a rule with no 'kind' is an n-gram). Returns (idioms, ngrams, manifest):
      idioms = ordered TRUSTED rules (gate/compose), kept in priority order — all host-side matchable;
      ngrams[k][suffix] = (out, basis, cite) — the GATED tier (= 'fire only if confident', NOT gating inside the n-gram).
    JSON keys are strings; normalized to ints here so a C++ porter sees the same shape."""
    m = json.load(open(manifest_path))
    idioms, ngrams = [], defaultdict(dict)
    for r in m["rules"]:
        kind = r.get("kind", "ngram")
        if kind == "ngram":
            ctx = tuple(r["ctx"])
            ngrams[len(ctx)][ctx] = (r["out"], r.get("basis", "observational"), _cite(r))
        elif kind == "gate":
            idioms.append({"kind": "gate", "id": r["id"], "cite": _cite(r),
                           "frame": {int(o): int(t) for o, t in r["frame"].items()}, "slot": r["slot"],
                           "table": {int(k): int(v) for k, v in r["table"].items()}})
        elif kind == "compose":
            idioms.append({"kind": "compose", "id": r["id"], "cite": _cite(r),
                           "frame": {int(o): int(t) for o, t in r["frame"].items()}, "operands": r["operands"],
                           "valmap": {int(t): int(v) for t, v in r["valmap"].items()},
                           "sum": {int(s): int(o) for s, o in r["sum"].items()}})
    return idioms, ngrams, m


def serve(ctx, idioms, ngrams, W):
    """The runtime decision: TRUSTED idioms (frame-matched, in priority order) → GATED n-grams (longest matching suffix)
    → ABSTAIN. Host-side, no souffle — even the idioms (gate = frame + slot→table; compose = frame + operands→value-sum→
    output) are a structured lookup, so the whole package is consumable without an engine. (No min_det: the manifest holds
    only confident rules — gating happened at build.)"""
    for r in idioms:                                            # trusted tier (causal), first
        fr = r["frame"]
        if not all(len(ctx) >= o and ctx[-o] == t for o, t in fr.items()):
            continue
        if r["kind"] == "gate":
            k = r["slot"]
            if len(ctx) >= k and ctx[-k] in r["table"]:
                return {"answer": r["table"][ctx[-k]], "tier": "trusted", "basis": "causal", "citation": r["cite"], "rule": r["id"]}
        else:  # compose
            k1, k2 = r["operands"]
            if len(ctx) >= max(k1, k2) and ctx[-k1] in r["valmap"] and ctx[-k2] in r["valmap"]:
                ssum = r["valmap"][ctx[-k1]] + r["valmap"][ctx[-k2]]
                if ssum in r["sum"]:
                    return {"answer": r["sum"][ssum], "tier": "trusted", "basis": "causal", "citation": r["cite"], "rule": r["id"]}
    for k in range(min(len(ctx), W), 0, -1):                    # gated n-gram tier (longest suffix wins)
        s = tuple(ctx[-k:])
        if s in ngrams[k]:
            out, basis, cite = ngrams[k][s]
            return {"answer": out, "tier": "gated", "basis": basis, "citation": cite, "k": k}
    return None  # ABSTAIN


def main():
    md = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "models", "llama32_1b")
    md = md if os.path.isabs(md) else os.path.join(HERE, md)
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    minsupp = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    mindet = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    name = os.path.basename(md)
    hold = make_package(md, W, minsupp, mindet)                  # builder: corpus → package (circuits.abstain.dl + manifest.json)
    idioms, ngrams, manifest = load_package(os.path.join(md, "manifest.json"))  # RUNTIME: load manifest.json only
    meta_on = any("citation" in r for r in manifest["rules"][:50])
    print(f"=== serve_package · {name} · loaded {manifest['n_rules']} rules from manifest.json "
          f"(thin runtime: no corpus/souffle/model){' · citations ✓' if meta_on else ''} ===")
    ans = cor = 0
    for ctx, o, _ in hold:                                       # held-out scorecard (eval only; uses the corpus)
        r = serve(ctx, idioms, ngrams, W)
        if r is not None:
            ans += 1; cor += (r["answer"] == o)
    H = len(hold)
    print(f"scorecard: coverage {ans/H:.0%}  precision {cor/ans if ans else 0:.0%}  abstain {1-ans/H:.0%}  (n={H})")
    dec = lambda t: f"id{t}"
    tok = None
    tp = os.path.join(md, "bundle.tokenizer.json")
    if os.path.exists(tp):
        try:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(tp); dec = lambda t: repr(tok.decode([t]))
        except Exception:
            pass
    shown_ans = shown_ab = 0
    print("sample served decisions (answer → citation, or abstain):")
    for ctx, o, _ in hold:
        r = serve(ctx, idioms, ngrams, W)
        if r is not None and shown_ans < 3:
            mark = "✓" if r["answer"] == o else "✗"
            print(f"  ANSWER {dec(r['answer'])} {mark}  [{r['tier']}/{r['basis']}]  cite: {r['citation']}")
            shown_ans += 1
        elif r is None and shown_ab < 2:
            print("  ABSTAIN (no confident rule) — would defer to backstop / refuse")
            shown_ab += 1
        if shown_ans >= 3 and shown_ab >= 2:
            break
    if tok is not None:
        print("NL-query path (tokenize → serve) — the token-space bridge sgiandubh's runtime needs:")
        for q in ["The United States of America", "xylophone quantum tariff zzzqq"]:
            ctx = tok.encode(q, add_special_tokens=False).ids[-W:]
            r = serve(ctx, idioms, ngrams, W)
            print(f"  query {q!r} → " + (f"answer {dec(r['answer'])} (cite {r['citation']})" if r else "ABSTAIN"))
    else:
        print("(no bundle.tokenizer.json — NL path skipped; this model serves token contexts only)")
    # manifest shape, for C++ porters
    s0 = manifest["rules"][0]
    print(f"manifest.json rule shape (for the C++ runtime): {{id, ctx:[token ids], out, support, determinism, cite[, citation]}}")
    print(f"  e.g. {json.dumps({k: s0[k] for k in ('id', 'ctx', 'out', 'support', 'determinism') if k in s0})}")


if __name__ == "__main__":
    main()
