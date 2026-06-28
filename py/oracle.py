#!/usr/bin/env python3
"""rosetta · oracle.py — the thin Datalog driver. Everything provable is in dl/*.dl; this just runs souffle and moves CSV.

Two jobs:
  decide(whole_dl, ctx)            run the faithful single-instance whole-model program on one context → its argmax.
  certify(circuit_dl, instances)   build the multi-instance facts, read each instance's answer off whole.dl (the
                                   oracle), then let dl/equiv.dl PROVE circuit == model over the whole instance set.
The faithfulness verdict comes out of Datalog (equiv.dl), not out of this file — Python only stages inputs.
"""
import os, subprocess, tempfile, shutil

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EQUIV = os.path.join(HERE, "dl", "equiv.dl")


def _run(dl, facts_dir, out_dir, includes=()):
    cmd = ["souffle", dl, "-F", facts_dir, "-D", out_dir]
    for inc in includes:
        cmd += ["-I", inc]
    return subprocess.run(cmd, capture_output=True, text=True)


def decide(whole_dl, ctx):
    """The faithful model's argmax for one context (single-instance whole.dl)."""
    with tempfile.TemporaryDirectory() as d:
        ind, outd = os.path.join(d, "in"), os.path.join(d, "out")
        os.makedirs(ind); os.makedirs(outd)
        with open(os.path.join(ind, "token.facts"), "w") as f:
            f.write("".join(f"{p}\t{t}\n" for p, t in enumerate(ctx)))
        _run(whole_dl, ind, outd)
        dc = os.path.join(outd, "decide.csv")
        if not os.path.exists(dc):
            return None
        s = open(dc).read().strip()
        return int(s) if s else None


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
