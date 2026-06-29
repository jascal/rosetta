"""pack.gram — the gram kernel (corpus n-gram + vocab) for an expert package.

The generative/generalizing half: where the cover/curated facts give exact decisions on seen contexts, the gram kernel
lets the expert continue in-domain text it wasn't distilled on — at n-gram fidelity, BOUNDED to the corpus (an
out-of-corpus context has no continuation → abstain). Built from corpus text alone — no model.

Emits into <out>/: grams.tsv (`prev <TAB> next <TAB> count`, orders 1..N) + vocab.txt (the in-domain bound).

NOTE (CONVERGENCE.md): this tier is slated to RETIRE into the cover's gated n-grams once the coverage-parity gate
passes. Ported faithfully for now so `build_expert` reproduces the current experts; do not extend it.
"""
import collections
import os
import re


def tokenize(text):
    # words + sentence punctuation, case-preserved (so generation reads naturally); unicode-aware
    return re.findall(r"\w+|[.,;:!?]", text, re.UNICODE)


def build(corpus, out, *, order=3):
    """corpus path → <out>/grams.tsv + <out>/vocab.txt. Returns (n_tokens, n_rows, n_vocab)."""
    words = tokenize(open(corpus, encoding="utf-8").read())
    counts = collections.defaultdict(collections.Counter)  # joined-prev -> Counter(next)
    for i, w in enumerate(words):
        for o in range(1, order + 1):
            if i >= o - 1:
                key = " ".join(words[i - (o - 1):i])
                counts[key][w] += 1

    os.makedirs(out, exist_ok=True)
    rows = 0
    with open(os.path.join(out, "grams.tsv"), "w", encoding="utf-8") as f:
        for key, c in counts.items():
            for nxt, n in c.items():
                f.write(f"{key}\t{nxt}\t{n}\n")
                rows += 1
    vocab = sorted(set(w.lower() for w in words if w.isalnum() or w.isalpha() or any(ch.isalnum() for ch in w)))
    open(os.path.join(out, "vocab.txt"), "w", encoding="utf-8").write("\n".join(vocab))
    return len(words), rows, len(vocab)
