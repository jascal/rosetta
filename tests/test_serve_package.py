"""rosetta · the expert-package thin runtime must stay servable + high-precision (regression guard for the
rosetta→sgiandubh convergence). Builds a package from a committed reference corpus, loads ONLY the manifest, serves,
and asserts the bounded-expert behavior (high precision when it answers; it abstains on a real fraction). Pure Python —
no souffle, no model, no tokenizer needed."""
import os
import shutil
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "py"))
from serve_package import make_package, load_package, serve  # noqa: E402

CORPUS = os.path.join(HERE, "models", "llama32_1b", "corpus.json")


@pytest.mark.skipif(not os.path.exists(CORPUS), reason="needs models/llama32_1b/corpus.json")
def test_package_load_roundtrip_and_scorecard(tmp_path):
    md = tmp_path / "expert"
    md.mkdir()
    shutil.copyfile(CORPUS, md / "corpus.json")
    W, minsupp, mindet = 8, 3, 1.0
    hold = make_package(str(md), W, minsupp, mindet)                      # builder: corpus → package
    assert (md / "manifest.json").exists() and (md / "circuits.abstain.dl").exists()
    idioms, ngrams, manifest = load_package(str(md / "manifest.json"))    # runtime: load manifest only (abstain_emit = n-gram only)
    assert idioms == [] and sum(len(ngrams[k]) for k in ngrams) == manifest["n_rules"] > 0   # load roundtrip (all gated n-grams)
    ans = cor = 0
    for ctx, o, _ in hold:
        r = serve(ctx, idioms, ngrams, W)
        if r is not None:
            ans += 1
            cor += (r["answer"] == o)
    assert ans > 0, "served nothing"
    cov, prec = ans / len(hold), cor / ans
    assert prec >= 0.85, f"precision {prec:.2f} below bound (regression in the cover/serve path?)"
    assert 0.0 < cov < 0.9, f"coverage {cov:.2f} out of expected band (should be a high-precision abstaining expert)"
