"""CLASSIC -> UNIFIED package converter (the document-origin ingestion route).

Consumes a classic sgiandubh package (knowledge.tsv: `id · section\ttext` normative items,
gram/vocab.txt word vocabulary) and emits a unified manifest.json package (PACKAGE.md v2):
gated ngram rules over a WordLevel tokenizer, every rule carrying origin:"document" and a
norm-id citation; the item texts become the grounding sidecar, so served statements attest at
stratum 0 with quotes. Usage: convert_classic.py <classic_dir> <out_dir> [W]"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    W = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    out.mkdir(parents=True, exist_ok=True)

    items = []
    for line in (src / "knowledge.tsv").read_text().splitlines():
        if "\t" not in line:
            continue                                      # document origin: the doc
        head, text = line.split("\t", 1)
        items.append((head.strip(), text.strip()))

    # word-level tokenizer over the package vocabulary (+ item words) -- built THROUGH the
    # same HF Whitespace pretokenizer the runtime will use (regex mismatches = silent misses)
    from tokenizers import pre_tokenizers
    pre = pre_tokenizers.Whitespace()
    vocab = Counter()
    tokenized = []
    for _hid, text in items:
        ws = [w for w, _sp in pre.pre_tokenize_str(text)]
        tokenized.append(ws)
        vocab.update(ws)
    words = ["[UNK]"] + [w for w, _ in vocab.most_common()]
    wid = {w: i for i, w in enumerate(words)}
    tok = {"version": "1.0", "truncation": None, "padding": None,
           "added_tokens": [], "normalizer": None,
           "pre_tokenizer": {"type": "Whitespace"},
           "post_processor": None, "decoder": {"type": "WordPiece", "prefix": "",
                                               "cleanup": False},
           "model": {"type": "WordLevel", "vocab": wid, "unk_token": "[UNK]"}}
    (out / "bundle.tokenizer.json").write_text(json.dumps(tok))

    # gated ngrams over the items, citation = the norm id (+ section)
    tab = defaultdict(Counter)
    cite_of = defaultdict(Counter)
    for (hid, _text), ws in zip(items, tokenized):
        ids = [wid[w] for w in ws]
        for i in range(1, len(ids)):
            for k in range(1, min(W, i) + 1):
                key = tuple(ids[i - k:i])
                tab[key][ids[i]] += 1
                cite_of[key][hid] += 1
    rules = []
    for key, nexts in tab.items():
        (best, n) = nexts.most_common(1)[0]
        tot = sum(nexts.values())
        det = n / tot
        if det < 0.66:                                    # support 1 IS valid for
            continue
        rules.append({"kind": "ngram", "tier": "gated", "basis": "observational",
                      "ctx": list(key), "out": best, "support": n,
                      "determinism": round(det, 4), "confidence": round(det, 4),
                      "origin": "document",
                      "citation": [cite_of[key].most_common(1)[0][0]]})

    grounding = "\n".join(text for _hid, text in items)
    (out / "grounding.txt").write_text(grounding)
    man = {"model": f"classic:{src.name}", "cover": "support-weighted", "W": W,
           "origin": "document", "grounding": "grounding.txt",
           "n_rules": len(rules),
           "rules": [dict(r, id=i) for i, r in enumerate(rules)]}
    (out / "manifest.json").write_text(json.dumps(man))
    print(f"{src.name}: {len(items)} document items -> {len(rules)} rules "
          f"(origin: document, W={W}, vocab {len(words)}) -> {out}")


if __name__ == "__main__":
    main()
