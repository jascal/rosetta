"""pack.score_retrieval — scorecard for a MODEL-FREE retrieval expert (EXPERTS.md), graded against the REAL runtime.

A retrieval expert has no manifest/cover for pack.eval to serve, so we grade the deployable thing itself: serve the
package with sgiandubh and run a labeled testset through it. The testset (jsonl) labels each query expect:"answer" (an
in-domain question that should be answered) or "abstain" (off-domain), with an optional `contains` for precision. We
report the reject-option metrics — in-domain recall, off-domain leak, content precision — at given thresholds, so
--answer-cov/--answer-cos/--answer-margin can be CALIBRATED by sweeping them (measure, then tune; not eyeballing).

Usage: .venv/bin/python -m pack.score_retrieval <package_dir> <testset.jsonl> --serve-bin <sgiandubh> [--cov .6 --cos .5 --margin .2]
"""
import argparse
import json
import subprocess
import time
import urllib.request


def _serve(bin_path, package, port, *, cov, cos, margin):
    cmd = [bin_path, package, str(port), "--answer-from-corpus", "--require-citation",
           "--answer-cov", str(cov), "--answer-cos", str(cos), "--answer-margin", str(margin)]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):                                          # wait for /health (no foreground sleep elsewhere)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            return proc
        except Exception:
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError("sgiandubh did not come up")


def _ask(port, q):
    body = json.dumps({"messages": [{"role": "user", "content": q}],
                       "response_format": {"type": "json_object"}}).encode()
    r = urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", body, {"content-type": "application/json"}), timeout=20)
    d = json.loads(json.load(r)["choices"][0]["message"]["content"])
    return d.get("answer", ""), d.get("kind", ""), d.get("citation_id", "")


def score(package, testset, bin_path, *, cov=0.6, cos=0.5, margin=0.2, port=8155):
    rows = [json.loads(ln) for ln in open(testset, encoding="utf-8") if ln.strip()]
    proc = _serve(bin_path, package, port, cov=cov, cos=cos, margin=margin)
    try:
        ind = ind_hit = ind_prec_n = ind_prec_ok = off = off_leak = 0
        for r in rows:
            ans, kind, cite = _ask(port, r["q"])
            answered = kind != "abstain" and not ans.startswith("That isn't")
            if r["expect"] == "answer":
                ind += 1
                ind_hit += answered
                if answered and (r.get("contains") or r.get("cite_prefix")):
                    ind_prec_n += 1
                    ok = all(c.lower() in ans.lower() for c in r.get("contains", []))
                    ok = ok and (not r.get("cite_prefix") or cite.startswith(r["cite_prefix"]))
                    ind_prec_ok += ok
            else:
                off += 1
                off_leak += answered
    finally:
        proc.kill()
    return {
        "thresholds": {"cov": cov, "cos": cos, "margin": margin},
        "in_domain_recall": ind_hit / ind if ind else 0.0,       # answered / in-domain (the hybrid's recall win)
        "content_precision": ind_prec_ok / ind_prec_n if ind_prec_n else None,   # over the `contains`-labeled subset
        "off_domain_leak": off_leak / off if off else 0.0,       # answered / off-domain (must stay low)
        "n_in_domain": ind, "n_off_domain": off,
    }


def main():
    ap = argparse.ArgumentParser(prog="pack.score_retrieval")
    ap.add_argument("package")
    ap.add_argument("testset")
    ap.add_argument("--serve-bin", required=True, help="path to the sgiandubh binary")
    ap.add_argument("--cov", type=float, default=0.6)
    ap.add_argument("--cos", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--port", type=int, default=8155)
    a = ap.parse_args()
    sc = score(a.package, a.testset, a.serve_bin, cov=a.cov, cos=a.cos, margin=a.margin, port=a.port)
    print(json.dumps(sc, indent=1))


if __name__ == "__main__":
    main()
