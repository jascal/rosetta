#!/usr/bin/env python3
"""Bounded graph-depth/iteration experiment and Datalog reachability certificate.

Generate paired positive/negative graph queries. The negative removes one
intermediate edge; graph names are freshly permuted per seed. Model responses
are loaded from JSONL so any architecture/iteration budget can be compared.
The host stages facts; Souffle computes the candidate and verdict.
"""
import argparse
import json
import random
import subprocess
import tempfile
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RULE = ROOT / "dl" / "recursive_reach.dl"


def make_cases(seed=17, depths=(1, 2, 3, 4), trials=8, topology="chains"):
    """One sole path per positive case; cut an edge for its negative twin."""
    if topology not in ("chains", "branched"):
        raise ValueError("topology must be chains or branched")
    cases = []
    for depth in depths:
        if depth < 1:
            raise ValueError("depths must be positive")
        for trial in range(trials):
            rng = random.Random((seed << 24) + depth * 100003 + trial)
            n = depth + 9
            names = list(range(n))
            rng.shuffle(names)
            path = names[:depth + 1]
            edges = list(zip(path, path[1:]))
            # Distractors include another chain but never link into the answer path.
            other = names[depth + 1:]
            edges += list(zip(other, other[1:]))
            if topology == "branched":
                # Dead-end branches and a distractor cycle vary graph structure
                # without creating a second path to the target. Reserve other[0]
                # for the perturbation's redirected edge.
                sources = rng.sample(path[:-1], rng.randint(1, min(depth, 3)))
                for source in sources:
                    edges.append((source, rng.choice(other[2:])))
                if rng.random() < 0.5:
                    edges.append((other[-1], other[rng.randrange(1, len(other) - 2)]))
            rng.shuffle(edges)
            cut = rng.randrange(depth)
            # For depth=1 this cuts the direct edge; for depth>=2 it tests a
            # genuine intermediate dependency at a randomly chosen hop.
            for positive in (True, False):
                # Preserve edge count and the source node's outdegree. The
                # perturbation changes only the destination of one path edge.
                variant = edges if positive else [
                    (path[cut], other[0]) if e == (path[cut], path[cut + 1]) else e
                    for e in edges
                ]
                cases.append({"id": len(cases), "seed": seed, "depth": depth, "trial": trial,
                              "positive": positive, "cut_hop": cut + 1,
                              "topology": topology,
                              "edges": [list(e) for e in variant], "query": [path[0], path[-1]],
                              "answer": int(positive)})
    return cases


def exhaustive_cases(nodes=3):
    """All loop-free directed graphs and distinct ordered queries on N nodes.

    This is a separate, exhaustive semantic domain. It includes branching and
    cycles; the answer label comes from an independent queue traversal.
    """
    if nodes < 2 or nodes > 4:
        raise ValueError("exhaustive domain supports 2..4 nodes")
    universe = [(a, b) for a in range(nodes) for b in range(nodes) if a != b]
    cases = []
    for mask in range(1 << len(universe)):
        edges = [edge for bit, edge in enumerate(universe) if mask & (1 << bit)]
        adjacency = defaultdict(list)
        for a, b in edges:
            adjacency[a].append(b)
        for source in range(nodes):
            lengths = {source: 0}
            queue = deque([source])
            while queue:
                u = queue.popleft()
                for v in adjacency[u]:
                    if v not in lengths:
                        lengths[v] = lengths[u] + 1
                        queue.append(v)
            for target in range(nodes):
                if target == source:
                    continue
                distance = lengths.get(target)
                cases.append({"id": len(cases), "edges": [list(e) for e in edges],
                              "query": [source, target], "answer": int(distance is not None),
                              "depth": distance if distance is not None else 0,
                              "topology": "exhaustive", "graph_mask": mask, "trial": mask})
    return cases


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as inp:
        return [json.loads(line) for line in inp if line.strip()]


def prompt(case, style="plain"):
    lines = [f"{a} -> {b}" for a, b in case["edges"]]
    question = "Directed edges:\n" + "\n".join(lines) + \
        f"\nIs there a path from {case['query'][0]} to {case['query'][1]}? Answer yes or no:"
    if style == "plain":
        return question
    if style == "fewshot":
        return ("Directed edges:\n90 -> 91\n91 -> 92\n"
                "Is there a path from 90 to 92? Answer yes or no: yes\n\n"
                "Directed edges:\n90 -> 91\n91 -> 93\n"
                "Is there a path from 90 to 92? Answer yes or no: no\n\n" + question)
    raise ValueError("prompt style must be plain or fewshot")


def collect_fieldrun(cases, port, tokenizer_path, architecture, budget=1, k=128, prompt_style="plain"):
    """Collect one-pass transformer predictions from a resident fieldrun server.

    A standard fieldrun bundle has no adjustable latent iteration count: budget=1
    records that fact rather than pretending to vary it. A model that does not
    place either yes/no token in top-K abstains for this argmax-style probe.
    """
    from tokenizers import Tokenizer
    from oracle import serve_topk

    if budget != 1:
        raise ValueError("fieldrun bundles expose one forward pass; use external responses for recurrent budgets")
    tok = Tokenizer.from_file(str(tokenizer_path))
    yes = {tuple(tok.encode(word, add_special_tokens=False).ids)[0] for word in (" yes", " Yes", "yes", "Yes")
           if len(tok.encode(word, add_special_tokens=False).ids) == 1}
    no = {tuple(tok.encode(word, add_special_tokens=False).ids)[0] for word in (" no", " No", "no", "No")
          if len(tok.encode(word, add_special_tokens=False).ids) == 1}
    if not yes or not no or yes & no:
        raise ValueError("tokenizer must have distinct single-token yes and no answers")
    rows = []
    for case in cases:
        ids = tok.encode(prompt(case, prompt_style)).ids
        scores = dict(serve_topk(port, ids, k=k))
        y = max((scores[t] for t in yes if t in scores), default=None)
        n = max((scores[t] for t in no if t in scores), default=None)
        answer = None if y is None and n is None else int(n is None or (y is not None and y > n))
        rows.append({"id": case["id"], "architecture": architecture, "budget": budget,
                     "prompt_style": prompt_style,
                     "answer": answer, "yes_logit": y, "no_logit": n})
    return rows


def symbolic_iterations(cases, budgets):
    """Controlled synchronous message-passing baseline; not an LLM measurement.

    Starting with only the source marked, each iteration propagates the mark
    across one edge. It lets us verify that the depth/iteration matrix can
    detect a genuine recurrence threshold before running looped checkpoints.
    """
    rows = []
    for budget in budgets:
        if budget < 1:
            raise ValueError("iteration budgets must be positive")
        for case in cases:
            reached = {case["query"][0]}
            for _ in range(budget):
                reached |= {b for a, b in case["edges"] if a in reached}
            rows.append({"id": case["id"], "architecture": "symbolic_frontier_control",
                         "budget": budget, "answer": int(case["query"][1] in reached)})
    return rows


def certify(cases, references):
    """Run the standalone Datalog circuit; refs map case id to binary label."""
    if set(references) != {c["id"] for c in cases}:
        raise ValueError("references must cover every case exactly once")
    with tempfile.TemporaryDirectory() as temp:
        base = Path(temp)
        facts, outputs = base / "facts", base / "out"
        facts.mkdir()
        outputs.mkdir()
        with open(facts / "edge.facts", "w") as f:
            for c in cases:
                for a, b in c["edges"]:
                    f.write(f"{c['id']}\t{a}\t{b}\n")
        with open(facts / "query.facts", "w") as f:
            for c in cases:
                f.write(f"{c['id']}\t{c['query'][0]}\t{c['query'][1]}\n")
        with open(facts / "ref.facts", "w") as f:
            for case_id, label in sorted(references.items()):
                if label not in (0, 1):
                    raise ValueError("references must be binary")
                f.write(f"{case_id}\t{label}\n")
        result = subprocess.run(["souffle", str(RULE), "-F", str(facts), "-D", str(outputs)],
                                capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr or result.stdout)

        def rows(name):
            return [[int(v) for v in row.split("\t")] for row in
                    (outputs / f"{name}.csv").read_text().splitlines() if row]

        return {"certified": bool((outputs / "certified.csv").read_text().strip()),
                "ncover": rows("ncover")[0][0], "nmiss": rows("nmiss")[0][0],
                "nuncov": rows("nuncov")[0][0], "mismatches": rows("mismatch"),
                "uncovered": rows("uncovered"),
                "predictions": {i: label for i, label in rows("cdecide")}}


def evaluate(cases, responses):
    """Responses: {id, architecture, budget, prompt_style, answer: 0/1/null}."""
    by_id = {c["id"]: c for c in cases}
    if len(by_id) != len(cases):
        raise ValueError("duplicate case ids")
    groups = defaultdict(dict)
    for row in responses:
        key = (row["architecture"], int(row["budget"]), row.get("prompt_style", "plain"))
        case_id = int(row["id"])
        if case_id not in by_id or case_id in groups[key]:
            raise ValueError(f"unknown or duplicate response: {key}, {case_id}")
        value = row["answer"]
        if value not in (None, 0, 1):
            raise ValueError("answer must be 0, 1, or null")
        groups[key][case_id] = value
    out = []
    for (arch, budget, style), answers in sorted(groups.items()):
        if set(answers) != set(by_id):
            raise ValueError(f"{arch} budget {budget} prompt {style} omitted cases")
        depths = {}
        for depth in sorted({c["depth"] for c in cases}):
            relevant = [c for c in cases if c["depth"] == depth]
            pos = {c["trial"]: c for c in relevant if c["positive"]}
            neg = {c["trial"]: c for c in relevant if not c["positive"]}
            interior = [t for t, c in pos.items() if 1 < c["cut_hop"] < depth]
            depths[str(depth)] = {"n": len(relevant),
                                  "accuracy": sum(answers[c["id"]] == c["answer"] for c in relevant) / len(relevant),
                                  "perturbation_follow": sum(answers[pos[t]["id"]] == 1 and
                                                             answers[neg[t]["id"]] == 0 for t in pos) / len(pos),
                                  "interior_edge_pairs": len(interior),
                                  "interior_edge_follow": (sum(answers[pos[t]["id"]] == 1 and
                                                               answers[neg[t]["id"]] == 0 for t in interior) / len(interior)
                                                           if interior else None),
                                  "abstentions": sum(answers[c["id"]] is None for c in relevant)}
        known = {i: value for i, value in answers.items() if value is not None}
        # The recursive circuit always predicts. Equivalence to a model requires
        # that the model give a binary answer on EVERY supplied instance.
        model_cert = certify(cases, known) if len(known) == len(cases) else None
        out.append({"architecture": arch, "budget": budget, "prompt_style": style,
                    "depths": depths,
                    "model_certificate": model_cert,
                    "model_certificate_status": "checked" if model_cert else "incomplete_model_answers"})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--exhaustive-nodes", type=int)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--topology", choices=("chains", "branched"), default="chains")
    parser.add_argument("--responses", type=Path, action="append", default=[])
    parser.add_argument("--fieldrun-port", type=int)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--architecture")
    parser.add_argument("--prompt-style", choices=("plain", "fewshot"), default="plain")
    parser.add_argument("--record-responses", type=Path)
    parser.add_argument("--symbolic-budgets", nargs="+", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.exhaustive_nodes:
        cases = exhaustive_cases(args.exhaustive_nodes)
        write_jsonl(args.cases, cases)
    elif args.generate:
        cases = make_cases(args.seed, args.depths, args.trials, args.topology)
        write_jsonl(args.cases, cases)
    else:
        cases = read_jsonl(args.cases)
    semantic = certify(cases, {c["id"]: c["answer"] for c in cases})
    response_rows = [row for path in args.responses for row in read_jsonl(path)]
    if args.symbolic_budgets:
        response_rows.extend(symbolic_iterations(cases, args.symbolic_budgets))
    if args.fieldrun_port:
        if not args.tokenizer or not args.architecture:
            parser.error("--fieldrun-port requires --tokenizer and --architecture")
        fieldrun_rows = collect_fieldrun(cases, args.fieldrun_port, args.tokenizer, args.architecture,
                                        prompt_style=args.prompt_style)
        response_rows.extend(fieldrun_rows)
        if args.record_responses:
            write_jsonl(args.record_responses, fieldrun_rows)
    trial_counts = {str(depth): len({c["trial"] for c in cases if c["depth"] == depth})
                    for depth in sorted({c["depth"] for c in cases})}
    result = {"domain": {"cases": len(cases), "depths": sorted({c["depth"] for c in cases}),
                         "trials_by_depth": trial_counts,
                         "topologies": sorted({c.get("topology", "chains") for c in cases}),
                         "paired_edge_cuts": all("cut_hop" in c for c in cases)},
              "semantic_certificate": semantic,
              "architectures": evaluate(cases, response_rows) if response_rows else []}
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
