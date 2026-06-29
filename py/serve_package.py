#!/usr/bin/env python3
"""rosetta · serve_package.py — the REFERENCE THIN RUNTIME for a rosetta expert package, and the spec for sgiandubh's C++.

The convergence (rosetta = builder, sgiandubh = thin runtime): rosetta emits the package (circuits.abstain.dl + manifest.json,
see py/abstain_emit.py); a runtime consumes it. This reference consumer proves the package is servable **host-side with NO
souffle and NO model** — the manifest IS the compiled cover (a decision table). It does exactly what sgiandubh's runtime will:

    tokenize(query)  →  longest-suffix-match over the manifest  →  {answer, citation, confidence}  or  ABSTAIN

KEY FINDING surfaced here (matters for sgiandubh): the cover is in the model's BPE-TOKEN space, so the runtime must
tokenize the query with the model's tokenizer (bundle.tokenizer.json). sgiandubh is currently word/text-based, so consuming
the rosetta cover requires a BPE tokenizer in its C++ runtime — the real Stage-2 dependency (this script is the port spec).

Demonstrates: the bounded-expert scorecard on held-out token contexts + sample cited answers + abstentions, and (if a
tokenizer is present) an NL-query path. Deps: requirements-sae.txt has tokenizers (for the NL path).
Usage: .venv/bin/python py/serve_package.py [model_dir] [W] [minsupp] [mindet]
"""
import json, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abstain_emit import build_tab, confident_rules

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_manifest(md, W, minsupp, mindet):
    """Build the in-memory package (the manifest = ruleid → {ctx, out, support, det, cite}) the way abstain_emit emits it,
    + a longest-suffix index for serving. (A deployed runtime loads manifest.json instead — same structure.)"""
    ids = json.load(open(os.path.join(md, "corpus.json")))["ids"]
    N = min(len(ids) - 1, 40000)
    wins = [(tuple(ids[i - W:i]), ids[i], i) for i in range(W, N)]
    random.Random(0).shuffle(wins)
    cut = int(len(wins) * 0.7); train, hold = wins[:cut], wins[cut:]
    tab, cites = build_tab(train, W)
    rules = confident_rules(tab, cites, W, minsupp, mindet)      # {k: {suffix: (out, support, det, cite)}}
    meta_p = os.path.join(md, "corpus_meta.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else None
    return rules, hold, meta


def cite_str(cite, meta):
    if not meta:
        return f"corpus@{cite[:3]}"                              # raw provenance offsets
    hits = []
    for p in cite:
        for m in meta:
            if m["start"] <= p < m["end"]:
                hits.append(m["citation"]); break
    return ", ".join(sorted(set(hits))) or f"corpus@{cite[:3]}"


def serve(ctx, rules, W, meta, min_det=0.0):
    """The runtime decision: longest confident suffix wins → answer + citation + confidence; else ABSTAIN. Host-side, no souffle."""
    for k in range(min(len(ctx), W), 0, -1):
        s = tuple(ctx[-k:])
        if s in rules[k]:
            out, sup, det, cite = rules[k][s]
            if det >= min_det:
                return {"answer": out, "citation": cite_str(cite, meta), "support": sup, "determinism": det, "k": k}
    return None  # ABSTAIN


def main():
    md = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "models", "llama32_1b")
    md = md if os.path.isabs(md) else os.path.join(HERE, md)
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    minsupp = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    mindet = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    name = os.path.basename(md)
    rules, hold, meta = build_manifest(md, W, minsupp, mindet)
    nrules = sum(len(rules[k]) for k in rules)
    print(f"=== serve_package · {name} · {nrules} rules · host-side (no souffle, no model){' · corpus_meta ✓' if meta else ''} ===")
    # bounded-expert scorecard on held-out token contexts
    ans = cor = 0
    for ctx, o, _ in hold:
        r = serve(ctx, rules, W, meta)
        if r is not None:
            ans += 1; cor += (r["answer"] == o)
    H = len(hold)
    print(f"scorecard: coverage {ans/H:.0%}  precision {cor/ans if ans else 0:.0%}  abstain {1-ans/H:.0%}  (n={H})")
    # sample served decisions (answers with citations, and abstentions)
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
    print("sample served decisions:")
    for ctx, o, _ in hold:
        r = serve(ctx, rules, W, meta)
        if r is not None and shown_ans < 3:
            mark = "✓" if r["answer"] == o else "✗"
            print(f"  ANSWER {dec(r['answer'])} {mark}  (k={r['k']} support={r['support']} det={r['determinism']:.2f})  cite: {r['citation']}")
            shown_ans += 1
        elif r is None and shown_ab < 2:
            print(f"  ABSTAIN (no confident rule) — would defer to backstop / refuse")
            shown_ab += 1
        if shown_ans >= 3 and shown_ab >= 2:
            break
    # the NL-query path (the token-space bridge sgiandubh needs)
    if tok is not None:
        print("NL-query path (tokenize → serve) — the token-space bridge sgiandubh's runtime needs:")
        for q in ["The United States of America", "xylophone quantum tariff zzzqq"]:
            ctx = tok.encode(q, add_special_tokens=False).ids[-W:]
            r = serve(ctx, rules, W, meta)
            print(f"  query {q!r} → " + (f"answer {dec(r['answer'])} (cite {r['citation']})" if r else "ABSTAIN"))
    else:
        print("(no bundle.tokenizer.json — NL path skipped; this model serves token contexts only)")


if __name__ == "__main__":
    main()
