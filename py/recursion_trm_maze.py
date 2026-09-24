#!/usr/bin/env python3
"""Generate graph mazes and evaluate a released TRM checkpoint over ACT budgets."""
import argparse
import json
import random
import sys
from pathlib import Path

from recursion_ladder import read_jsonl, write_jsonl


SIZE = 30
CHAR_ID = {"#": 1, " ": 2, "S": 3, "G": 4, "o": 5}


def validate_case(case):
    """Check bounds and ensure the maze encodes exactly the declared graph."""
    grid = case["grid"]
    if len(grid) != SIZE or any(len(row) != SIZE for row in grid):
        raise ValueError("maze grid must be 30x30")
    if any(not (0 <= node < SIZE * SIZE) for node in case["nodes"]):
        raise ValueError("graph node lies outside the maze")
    opened = {(y, x) for y, row in enumerate(grid) for x, char in enumerate(row) if char != "#"}
    observed = set()
    for a in case["nodes"]:
        ay, ax = divmod(a, SIZE)
        for b in case["nodes"]:
            by, bx = divmod(b, SIZE)
            if abs(ay - by) + abs(ax - bx) == 2 and ((ay + by) // 2, (ax + bx) // 2) in opened:
                observed.add((min(a, b), max(a, b)))
    expected = {(min(a, b), max(a, b)) for a, b in case["edges"]}
    if observed != expected:
        raise ValueError("maze corridors do not encode the declared relation graph")


def make_cases(seed=37, depths=(1, 2, 3, 4, 5), trials=4):
    """Encode an undirected relation path as a corridor graph in a 30x30 maze.

    Each relation edge is a one-cell corridor between graph-node cells. A
    paired negative closes one such corridor and opens a disconnected distractor
    corridor, preserving the number of relation edges and open cells.
    """
    cases = []
    for depth in depths:
        if depth < 1 or depth > 7:
            raise ValueError("maze path depths must be in 1..7")
        for trial in range(trials):
            rng = random.Random((seed << 24) + depth * 100003 + trial)
            vertical = bool(rng.randrange(2))
            reverse = bool(rng.randrange(2))
            axis_start = rng.randrange(4, SIZE - 2 * depth - 4)
            axis_start += 2 * depth if reverse else 0
            if vertical:
                row, col = axis_start, rng.randrange(8, 20)
            else:
                row, col = rng.randrange(8, 20), axis_start
            step = -1 if reverse else 1
            main_nodes = []
            for i in range(depth + 1):
                y = row + (step * 2 * i if vertical else 0)
                x = col + (0 if vertical else step * 2 * i)
                main_nodes.append(y * SIZE + x)
            # Reserve an isolated pair whose middle cell can become a distractor edge.
            if vertical:
                distractor_y = row
                distractor_x = col - 5 if col > 15 else col + 5
            else:
                distractor_y = row - 5 if row > 15 else row + 5
                distractor_x = col
            distractor_nodes = [distractor_y * SIZE + distractor_x,
                                distractor_y * SIZE + distractor_x + 2]
            corridor_cells = []
            for a, b in zip(main_nodes, main_nodes[1:]):
                corridor_cells.append((a + b) // 2)
            distractor_middle = (distractor_nodes[0] + distractor_nodes[1]) // 2
            # Choose a strict interior relation edge whenever one exists.
            cut_index = rng.randrange(1, depth - 1) if depth >= 3 else rng.randrange(depth)
            query = [main_nodes[0], main_nodes[-1]]
            for positive in (True, False):
                grid = [["#"] * SIZE for _ in range(SIZE)]
                for node in main_nodes + distractor_nodes:
                    grid[node // SIZE][node % SIZE] = " "
                for i, cell in enumerate(corridor_cells):
                    if positive or i != cut_index:
                        grid[cell // SIZE][cell % SIZE] = " "
                if not positive:
                    grid[distractor_middle // SIZE][distractor_middle % SIZE] = " "
                grid[query[0] // SIZE][query[0] % SIZE] = "S"
                grid[query[1] // SIZE][query[1] % SIZE] = "G"
                graph_edges = [[main_nodes[i], main_nodes[i + 1]] for i in range(depth)
                               if positive or i != cut_index]
                if not positive:
                    graph_edges.append(distractor_nodes)
                case = {"id": len(cases), "depth": depth, "trial": trial,
                        "positive": positive, "cut_hop": cut_index + 1,
                        "topology": "trm_grid_maze", "grid": ["".join(line) for line in grid],
                        "nodes": main_nodes + distractor_nodes,
                        "edges": graph_edges, "query": query, "answer": int(positive)}
                validate_case(case)
                cases.append(case)
    return cases


def path_reaches_goal(predicted, source, target):
    traversable = {node for node, token in enumerate(predicted) if token in (3, 4, 5)}
    if source not in traversable or target not in traversable:
        return False
    frontier = [source]
    reached = {source}
    while frontier:
        node = frontier.pop()
        if node == target:
            return True
        y, x = divmod(node, SIZE)
        for neighbor in (node - SIZE, node + SIZE, node - 1, node + 1):
            ny, nx = divmod(neighbor, SIZE)
            if (0 <= ny < SIZE and 0 <= nx < SIZE and
                    ((abs(ny - y) + abs(nx - x)) == 1) and neighbor in traversable and neighbor not in reached):
                reached.add(neighbor)
                frontier.append(neighbor)
    return False


def collect(cases, checkpoint, source_dir, budgets=(1, 2, 4, 8, 16), batch_size=4, threads=4):
    import torch

    torch.set_num_threads(threads)
    source_dir = str(Path(source_dir).resolve())
    sys.path.insert(0, source_dir)
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

    # Architecture parameters are from the released Maze-Hard checkpoint config.
    config = dict(batch_size=batch_size, seq_len=SIZE * SIZE, puzzle_emb_ndim=512,
                  num_puzzle_identifiers=1, vocab_size=6, H_cycles=3, L_cycles=4,
                  H_layers=0, L_layers=2, hidden_size=512, expansion=4, num_heads=8,
                  pos_encodings="rope", halt_max_steps=max(budgets),
                  halt_exploration_prob=0.1, forward_dtype="bfloat16", mlp_t=False,
                  puzzle_emb_len=16, no_ACT_continue=True)
    model = TinyRecursiveReasoningModel_ACTV1(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {key.removeprefix("_orig_mod.model."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()

    rows = []
    with torch.inference_mode():
        for start in range(0, len(cases), batch_size):
            batch_cases = cases[start:start + batch_size]
            actual = len(batch_cases)
            if actual < batch_size:
                batch_cases = batch_cases + [batch_cases[-1]] * (batch_size - actual)
            inputs = torch.tensor([[CHAR_ID[c] for line in case["grid"] for c in line]
                                   for case in batch_cases], dtype=torch.int32)
            batch = {"inputs": inputs,
                     "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.int32),
                     "labels": torch.zeros_like(inputs)}
            carry = model.initial_carry(batch)
            snapshots = {}
            for step in range(1, max(budgets) + 1):
                carry, output = model(carry, batch)
                if step in budgets:
                    predictions = output["logits"].argmax(dim=-1).tolist()
                    snapshots[step] = predictions[:actual]
            for i, case in enumerate(batch_cases[:actual]):
                for budget in budgets:
                    prediction = snapshots[budget][i]
                    answer = int(path_reaches_goal(prediction, *case["query"]))
                    rows.append({"id": case["id"], "architecture": "TRM-Maze-Hard",
                                 "budget": budget, "answer": answer,
                                 "prediction_grid": "".join("o" if token == 5 else
                                                               "S" if token == 3 else
                                                               "G" if token == 4 else
                                                               " " if token == 2 else "#"
                                                               for token in prediction)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--budgets", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases = make_cases() if args.generate else read_jsonl(args.cases)
    if args.generate:
        write_jsonl(args.cases, cases)
    if args.limit:
        cases = cases[:args.limit]
    rows = collect(cases, args.checkpoint, args.source_dir, args.budgets,
                   args.batch_size, args.threads)
    write_jsonl(args.out, rows)
    print(json.dumps({"cases": len(cases), "budgets": args.budgets, "responses": len(rows)}))


if __name__ == "__main__":
    main()
