#!/usr/bin/env python3
"""rosetta · oracle.py — the thin Datalog driver. Everything provable is in dl/*.dl; this just runs souffle and moves CSV.

Two jobs:
  decide(whole_dl, ctx)            run the faithful single-instance whole-model program on one context → its argmax.
  certify(circuit_dl, instances)   build the multi-instance facts, read each instance's answer off whole.dl (the
                                   oracle), then let dl/equiv.dl PROVE circuit == model over the whole instance set.
The faithfulness verdict comes out of Datalog (equiv.dl), not out of this file — Python only stages inputs.
"""
import os, subprocess, tempfile, shutil, glob

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EQUIV = os.path.join(HERE, "dl", "equiv.dl")
NGRAM = os.path.join(HERE, "dl", "ngram.dl")
MASTER = os.path.join(HERE, "dl", "master.dl")
INDUCTION = os.path.join(HERE, "dl", "induction.dl")
_COMPILED = {}   # whole_dl -> path of compiled native binary (or False if compilation isn't available)


def detect(insts, refs, w):
    """Run the Datalog n-gram detector (dl/ngram.dl): shortest model-deterministic suffix per context. Returns
    (orderhist {k:n_contexts}, minorder {inst:k}). The detection is the query — Python only stages the facts."""
    with tempfile.TemporaryDirectory() as d:
        ind, outd = os.path.join(d, "in"), os.path.join(d, "out")
        os.makedirs(ind); os.makedirs(outd)
        with open(os.path.join(ind, "ctx.facts"), "w") as cf, open(os.path.join(ind, "ref.facts"), "w") as rf:
            for i, ctx in enumerate(insts):
                if refs[i] is None:
                    continue
                for p, t in enumerate(ctx):
                    cf.write(f"{i}\t{p}\t{t}\n")
                rf.write(f"{i}\t{refs[i]}\n")
        open(os.path.join(ind, "wmax.facts"), "w").write(f"{w}\n")
        subprocess.run(["souffle", NGRAM, "-F", ind, "-D", outd], capture_output=True, text=True)
        def rows(name):
            p = os.path.join(outd, name)
            return [tuple(map(int, l.split("\t"))) for l in open(p).read().splitlines()] if os.path.exists(p) else []
        return {k: n for k, n in rows("orderhist.csv")}, {i: k for i, k in rows("minorder.csv")}


def serve_decide(port, ctx):
    """The model's argmax via a RESIDENT fieldrun server (`fieldrun --bundle <stem> --serve <port>`): POST /predict
    {"ids":[…]} -> {"next": id}. Loads the bundle ONCE (vs subprocess-per-call which reloads it every time) — essential
    for big models (a 1B int8 bundle is 1.2 GB to reload). Build-time refs oracle only; no runtime dependency."""
    import urllib.request, json
    data = json.dumps({"ids": [int(t) for t in ctx]}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/predict", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["next"]


def fieldrun_decide(bundle, ctx):
    """The model's argmax for one context via the fieldrun binary (build-time refs oracle; faithful == the model).
    bundle is the .fieldrun stem. Used for large models where the whole.dl forward is too slow to run in souffle —
    fieldrun is BUILD-TIME only; the minimized circuits.dl it certifies has no fieldrun dependency at runtime."""
    import re
    fr = os.environ.get("FIELDRUN_BIN") or "fieldrun"
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write('{"holdout_ids":[' + ",".join(map(str, ctx)) + "]}")
        path = f.name
    try:
        r = subprocess.run([fr, "--bundle", bundle, "--ids", path, "--explain"], capture_output=True, text=True)
    finally:
        os.unlink(path)
    m = re.search(r"model predicts \[(\d+)\]", r.stdout + r.stderr)
    return int(m.group(1)) if m else None


def run_induction(insts, refs, m):
    """Run dl/induction.dl at match length m. Returns {n_total, n_apply, n_hit, n_miss, hits:set(inst)}."""
    with tempfile.TemporaryDirectory() as d:
        ind, outd = os.path.join(d, "in"), os.path.join(d, "out")
        os.makedirs(ind); os.makedirs(outd)
        with open(os.path.join(ind, "ctx.facts"), "w") as cf, open(os.path.join(ind, "ref.facts"), "w") as rf:
            for i, ctx in enumerate(insts):
                if refs[i] is None:
                    continue
                for p, t in enumerate(ctx):
                    cf.write(f"{i}\t{p}\t{t}\n")
                rf.write(f"{i}\t{refs[i]}\n")
        open(os.path.join(ind, "mlen.facts"), "w").write(f"{m}\n")
        subprocess.run(["souffle", INDUCTION, "-F", ind, "-D", outd], capture_output=True, text=True)
        def scalar(name):
            p = os.path.join(outd, name + ".csv")
            s = open(p).read().strip() if os.path.exists(p) else ""
            return int(s) if s.isdigit() else 0
        def col(name):
            p = os.path.join(outd, name + ".csv")
            return {int(l) for l in open(p).read().splitlines()} if os.path.exists(p) else set()
        return {"n_total": scalar("n_total"), "n_apply": scalar("n_apply"), "n_hit": scalar("n_hit"),
                "n_miss": scalar("n_miss"), "hits": col("induct_hit")}


def run_master(insts, refs, w):
    """Run dl/master.dl: detection + cover + certificate in ONE Datalog program over staged ctx/ref/wmax facts.
    Returns {ncover, nmiss, nuncov, certified, orderhist}. Demonstrates the master-dl process (host only stages I/O)."""
    with tempfile.TemporaryDirectory() as d:
        ind, outd = os.path.join(d, "in"), os.path.join(d, "out")
        os.makedirs(ind); os.makedirs(outd)
        with open(os.path.join(ind, "ctx.facts"), "w") as cf, open(os.path.join(ind, "ref.facts"), "w") as rf:
            for i, ctx in enumerate(insts):
                if refs[i] is None:
                    continue
                for p, t in enumerate(ctx):
                    cf.write(f"{i}\t{p}\t{t}\n")
                rf.write(f"{i}\t{refs[i]}\n")
        open(os.path.join(ind, "wmax.facts"), "w").write(f"{w}\n")
        subprocess.run(["souffle", MASTER, "-F", ind, "-D", outd, "-I", os.path.join(HERE, "dl")],
                       capture_output=True, text=True)
        def scalar(name):
            p = os.path.join(outd, name + ".csv")
            s = open(p).read().strip() if os.path.exists(p) else ""
            return int(s) if s.isdigit() else (0 if not s else s)
        cert = os.path.join(outd, "certified.csv")
        return {"ncover": scalar("ncover"), "nmiss": scalar("nmiss"), "nuncov": scalar("nuncov"),
                "certified": os.path.exists(cert) and os.path.getsize(cert) > 0,
                "orderhist": {k: n for k, n in (tuple(map(int, l.split("\t")))
                              for l in (open(os.path.join(outd, "orderhist.csv")).read().splitlines()
                                        if os.path.exists(os.path.join(outd, "orderhist.csv")) else []))}}


def _run(dl, facts_dir, out_dir, includes=()):
    cmd = ["souffle", dl, "-F", facts_dir, "-D", out_dir]
    for inc in includes:
        cmd += ["-I", inc]
    return subprocess.run(cmd, capture_output=True, text=True)


def _compile_env():
    """Best-effort: locate souffle's runtime headers and provide a libtinfo.so dev symlink, so `souffle -c` can link.
    Returns (env, ok). Souffle's interpreter re-parses the whole program each call (~5s on a 22k-line model); a compiled
    binary runs in ~30ms — a ~140x speedup that makes large instance sets practical."""
    env = dict(os.environ)
    hdr = next(iter(glob.glob(os.path.expanduser("~/.local/include/souffle/CompiledSouffle.h"))
                     + glob.glob("/usr/include/souffle/CompiledSouffle.h")
                     + glob.glob("/usr/local/include/souffle/CompiledSouffle.h")), None)
    if hdr:
        inc = os.path.dirname(os.path.dirname(hdr))
        env["CPLUS_INCLUDE_PATH"] = inc + os.pathsep + env.get("CPLUS_INCLUDE_PATH", "")
    # libtinfo often ships only as libtinfo.so.6 (no dev symlink) — provide one in a cache dir
    libdir = os.path.join(tempfile.gettempdir(), "rosetta-souffle-lib")
    for cand in ("/usr/lib/x86_64-linux-gnu/libtinfo.so.6", "/lib/x86_64-linux-gnu/libtinfo.so.6"):
        if os.path.exists(cand):
            os.makedirs(libdir, exist_ok=True)
            link = os.path.join(libdir, "libtinfo.so")
            if not os.path.exists(link):
                os.symlink(cand, link)
            env["LIBRARY_PATH"] = libdir + os.pathsep + env.get("LIBRARY_PATH", "")
            break
    return env


def compiled(whole_dl):
    """Path to a cached native binary for whole_dl, compiling on first use. Returns None if compilation isn't available
    (caller then falls back to the interpreter). The binary IS the same program — same answers, ~140x faster."""
    whole_dl = os.path.abspath(whole_dl)
    if whole_dl in _COMPILED:
        return _COMPILED[whole_dl] or None
    exe = whole_dl + ".exe"
    if os.path.exists(exe) and os.path.getmtime(exe) >= os.path.getmtime(whole_dl):
        _COMPILED[whole_dl] = exe
        return exe
    r = subprocess.run(["souffle", "-c", "-o", exe, whole_dl], cwd=os.path.dirname(whole_dl),
                       capture_output=True, text=True, env=_compile_env())
    ok = os.path.exists(exe) and os.access(exe, os.X_OK)
    _COMPILED[whole_dl] = exe if ok else False
    return exe if ok else None


def decide(whole_dl, ctx):
    """The faithful model's argmax for one context. Splits weights into .facts data modules (so souffle bulk-loads them
    instead of re-parsing 99.96%-facts every call — ~100x faster and lets large models run at all), then evaluates the
    tiny forward.dl (compiled if possible, else interpreted) over the weights + this context's token.facts."""
    from split_facts import split
    forward, wdir = split(whole_dl)
    with tempfile.TemporaryDirectory() as d:
        ind, outd = os.path.join(d, "in"), os.path.join(d, "out")
        os.makedirs(ind); os.makedirs(outd)
        for fn in os.listdir(wdir):                      # stage weight modules (symlink, no copy)
            os.symlink(os.path.join(wdir, fn), os.path.join(ind, fn))
        with open(os.path.join(ind, "token.facts"), "w") as f:
            f.write("".join(f"{p}\t{t}\n" for p, t in enumerate(ctx)))
        exe = compiled(forward)                          # forward.dl is tiny now → small .cpp, fast compile
        if exe:
            subprocess.run([exe, "-F", ind, "-D", outd], capture_output=True, text=True)
        else:
            _run(forward, ind, outd)
        dc = os.path.join(outd, "decide.csv")
        if not os.path.exists(dc):
            return None
        s = open(dc).read().strip()
        return int(s) if s else None


def logits(whole_dl, ctx):
    """The faithful model's full logit scoreboard for one context: list of (token_id, logit). These are the T-INVARIANT
    incidence values — softmax(logits/T) is the model's next-token distribution at ANY temperature (T only scales them).
    Same machinery as decide(), but reads the whole.dl `logit` relation instead of collapsing to argmax."""
    from split_facts import split
    forward, wdir = split(whole_dl)
    with tempfile.TemporaryDirectory() as d:
        ind, outd = os.path.join(d, "in"), os.path.join(d, "out")
        os.makedirs(ind); os.makedirs(outd)
        for fn in os.listdir(wdir):
            os.symlink(os.path.join(wdir, fn), os.path.join(ind, fn))
        with open(os.path.join(ind, "token.facts"), "w") as f:
            f.write("".join(f"{p}\t{t}\n" for p, t in enumerate(ctx)))
        exe = compiled(forward)
        if exe:
            subprocess.run([exe, "-F", ind, "-D", outd], capture_output=True, text=True)
        else:
            _run(forward, ind, outd)
        lc = os.path.join(outd, "logit.csv")
        if not os.path.exists(lc):
            return None
        out = []
        for ln in open(lc).read().splitlines():
            if "\t" in ln:
                v, s = ln.split("\t")
                out.append((int(v), float(s)))
        return out


def run_equiv(circuit_dl, instances, refs):
    """Like certify(), but with the model answers ALREADY computed (refs aligned to instances). Lets the slow whole.dl
    oracle be paid once and cached, so the minimization loop can re-certify candidate circuits cheaply (equiv.dl only)."""
    with tempfile.TemporaryDirectory() as d:
        ind, outd = os.path.join(d, "in"), os.path.join(d, "out")
        os.makedirs(ind); os.makedirs(outd)
        inc_dir = os.path.join(d, "inc"); os.makedirs(inc_dir)
        shutil.copyfile(circuit_dl, os.path.join(inc_dir, "circuit.dl"))
        with open(os.path.join(ind, "tok.facts"), "w") as tf, open(os.path.join(ind, "ref.facts"), "w") as rf:
            for i, (ctx, out) in enumerate(zip(instances, refs)):
                for p, t in enumerate(ctx):
                    tf.write(f"{i}\t{p}\t{t}\n")
                if out is not None:
                    rf.write(f"{i}\t{out}\n")
        r = _run(EQUIV, ind, outd, includes=[inc_dir])
        def scalar(name):
            p = os.path.join(outd, f"{name}.csv")
            s = open(p).read().strip() if os.path.exists(p) else ""
            return int(s) if s else 0
        def rows(name):
            p = os.path.join(outd, f"{name}.csv")
            return [tuple(map(int, ln.split("\t"))) for ln in open(p).read().splitlines()] if os.path.exists(p) else []
        if not os.path.exists(os.path.join(outd, "ncover.csv")):
            return {"error": r.stderr.strip() or "equiv.dl produced no output"}
        return {"certified": r and os.path.getsize(os.path.join(outd, "certified.csv")) > 0
                if os.path.exists(os.path.join(outd, "certified.csv")) else False,
                "ncover": scalar("ncover"), "nmiss": scalar("nmiss"), "nuncov": scalar("nuncov"),
                "mismatches": rows("mismatch"), "uncovered": rows("uncovered")}


def certify(circuit_dl, whole_dl, instances):
    """Prove (in Datalog) that the circuit equals the model over ALL instances. instances: list of token-id lists.

    Returns dict {certified, ncover, nmiss, nuncov, mismatches, uncovered}. The circuit file must define
    cdecide(inst,out) over tok(inst,pos,id); it is #included by dl/equiv.dl (its directory is put on the -I path)."""
    with tempfile.TemporaryDirectory() as d:
        ind, outd = os.path.join(d, "in"), os.path.join(d, "out")
        os.makedirs(ind); os.makedirs(outd)
        # equiv.dl does `#include "circuit.dl"` — stage the candidate under that name on the include path
        inc_dir = os.path.join(d, "inc"); os.makedirs(inc_dir)
        shutil.copyfile(circuit_dl, os.path.join(inc_dir, "circuit.dl"))
        with open(os.path.join(ind, "tok.facts"), "w") as tf, open(os.path.join(ind, "ref.facts"), "w") as rf:
            for i, ctx in enumerate(instances):
                for p, t in enumerate(ctx):
                    tf.write(f"{i}\t{p}\t{t}\n")
                out = decide(whole_dl, ctx)          # the oracle: the model's own answer, from whole.dl
                if out is not None:
                    rf.write(f"{i}\t{out}\n")
        r = _run(EQUIV, ind, outd, includes=[inc_dir])
        def scalar(name):
            p = os.path.join(outd, f"{name}.csv")
            s = open(p).read().strip() if os.path.exists(p) else ""
            return int(s) if s else 0
        def rows(name):
            p = os.path.join(outd, f"{name}.csv")
            return [tuple(map(int, ln.split("\t"))) for ln in open(p).read().splitlines()] if os.path.exists(p) else []
        if not os.path.exists(os.path.join(outd, "ncover.csv")):
            return {"error": r.stderr.strip() or "equiv.dl produced no output"}
        return {
            "certified": os.path.exists(os.path.join(outd, "certified.csv"))
                          and bool(open(os.path.join(outd, "certified.csv")).read().strip() == "()" or
                                   os.path.getsize(os.path.join(outd, "certified.csv")) > 0),
            "ncover": scalar("ncover"), "nmiss": scalar("nmiss"), "nuncov": scalar("nuncov"),
            "mismatches": rows("mismatch"), "uncovered": rows("uncovered"),
        }
