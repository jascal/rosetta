"""pack.grounding — corpus → knowledge.tsv (+ optional wordvec.txt). Retrieval/citation data for the runtime.

Grounding is *data*, not a model: knowledge.tsv (section <TAB> passage — the owner's content, attached verbatim to
answers) + wordvec.txt (word embeddings restricted to the corpus vocab, so the runtime grounds by meaning (cosine)
rather than overlap; falls back to lexical Jaccard when absent). Default: pretrained GloVe (auto-fetched once),
restricted to the corpus vocab — one shared, calibrated space so cosines mean something and off-domain abstains.

Faithful port of the former sgiandubh/tools/build_grounding.py, as an importable `build(...)`.
"""
import os
import re
from collections import Counter

SEC = re.compile(r'^\s*\[\s*\xa7?\s*([^\]]+?)\s*\]\s*(.*)$')
SENT = re.compile(r'(?<=[.!?])\s+')
# matches the runtime's tokenizer: lowercase alnum runs, len>1, minus a small stoplist
STOP = set("the is are was were be been a an of to in on for and or but what which who how why when where do does "
           "did you your it its that this these those with as at by from about can could would should i we".split())


def _toks(s):
    return [w for w in re.findall(r'[a-z0-9]+', s.lower()) if len(w) > 1 and w not in STOP]


def _build_corpus_vectors(sentences, dim, cap):
    """PPMI + truncated SVD word vectors over the corpus. Returns [(word, vec)], actual_dim. Needs numpy/scipy."""
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import svds

    tokd = [_toks(s) for s in sentences]
    freq = Counter()
    for t in tokd:
        freq.update(set(t))
    vocab = [w for w, _ in freq.most_common(cap)]
    idx = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    if V < 4:
        return [], 0
    rows, cols = [], []
    for t in tokd:
        present = sorted(set(idx[w] for w in t if w in idx))
        for a in present:
            for b in present:
                if a != b:
                    rows.append(a)
                    cols.append(b)
    if not rows:
        return [], 0
    C = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(V, V)).tocsr().tocoo()
    total = C.data.sum()
    rowsum = np.asarray(coo_matrix((C.data, (C.row, C.col)), shape=(V, V)).sum(axis=1)).ravel()
    pmi = np.log((C.data * total) / (rowsum[C.row] * rowsum[C.col] + 1e-12) + 1e-12)
    P = coo_matrix((np.maximum(0.0, pmi), (C.row, C.col)), shape=(V, V)).tocsr()
    k = min(dim, V - 1)
    u, s, _ = svds(P, k=k)
    W = u * np.sqrt(np.maximum(s, 0.0))
    nrm = np.linalg.norm(W, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    W = W / nrm
    return list(zip(vocab, W)), k


def _ensure_glove(dim, cache=os.path.expanduser("~/.cache/sgiandubh")):
    """Path to glove.6B.<dim>d.txt, extracting from (and downloading once if needed) glove.6B.zip in the cache."""
    import zipfile
    import urllib.request
    txt = os.path.join(cache, f"glove.6B.{dim}d.txt")
    if os.path.exists(txt):
        return txt
    os.makedirs(cache, exist_ok=True)
    zp = os.path.join(cache, "glove.6B.zip")
    if not os.path.exists(zp):
        url = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
        print(f"downloading GloVe (~822MB, one-time) -> {zp}")
        urllib.request.urlretrieve(url, zp)
    print(f"extracting glove.6B.{dim}d.txt -> {cache}")
    with zipfile.ZipFile(zp) as z:
        z.extract(f"glove.6B.{dim}d.txt", cache)
    return txt


def _build_pretrained(sentences, vec_path, cap):
    """Restrict pretrained vectors (GloVe/fastText text format) to the corpus vocab, L2-normalize. Returns [(w,vec)], dim."""
    import numpy as np
    freq = Counter()
    for s in sentences:
        freq.update(set(_toks(s)))
    vocab = set(w for w, _ in freq.most_common(cap))
    out, dim = [], 0
    with open(vec_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < 3 or parts[0] not in vocab:
                continue
            vec = np.asarray(parts[1:], dtype=np.float32)
            if dim == 0:
                dim = len(vec)
            elif len(vec) != dim:
                continue
            n = float(np.linalg.norm(vec))
            out.append((parts[0], vec / n if n > 0 else vec))
    return out, dim


def build(corpus, out, *, dim=300, corpus_vectors=False, pretrained=None, glove_dim=300,
          no_split=False, min_len=20, vocab_cap=4000):
    """corpus path → <out>/knowledge.tsv (+ <out>/wordvec.txt unless dim==0). Returns (n_passages, n_words, k)."""
    os.makedirs(out, exist_ok=True)
    passages = []
    with open(os.path.join(out, "knowledge.tsv"), "w", encoding="utf-8") as w:
        for line in open(corpus, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            m = SEC.match(line)
            sec, text = (m.group(1).strip(), m.group(2).strip()) if m else ("", line)
            for sent in ([text] if no_split else SENT.split(text)):
                sent = sent.strip()
                if len(sent) >= min_len:
                    w.write(f"{sec}\t{sent}\n")
                    passages.append(sent)

    if dim <= 0:
        return len(passages), 0, 0                                   # lexical-only grounding (no numpy/scipy needed)

    if corpus_vectors:
        vecs, k = _build_corpus_vectors(passages, dim, vocab_cap)
    else:                                                            # DEFAULT: pretrained GloVe, restricted to corpus vocab
        vpath = pretrained or _ensure_glove(glove_dim)
        vecs, k = _build_pretrained(passages, vpath, vocab_cap)
    if vecs:
        with open(os.path.join(out, "wordvec.txt"), "w", encoding="utf-8") as f:
            for word, vec in vecs:
                f.write(word + " " + " ".join(f"{x:.5f}" for x in vec) + "\n")
    return len(passages), len(vecs), k
