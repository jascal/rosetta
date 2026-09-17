"""Prepare a document-group holdout BEFORE cover extraction; retain the exact split and hashes.

One nonempty source line is one document/passage. Whitespace-equivalent duplicates stay together.
This evaluates the cover tier against corpus continuations, not full-package retrieval or model equivalence.
"""
import hashlib
import json
import math
from pathlib import Path
import random


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def documents(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def group_id(text):
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def prepare(out, source, fraction=.2, seed=0, window=8):
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("holdout fraction must be in (0, 1)")
    if window < 1:
        raise ValueError("holdout window must be positive")
    lines = documents(source)
    groups = sorted({group_id(line) for line in lines})
    if len(groups) < 2:
        raise ValueError("holdout needs at least two distinct source lines/documents")
    random.Random(seed).shuffle(groups)
    count = max(1, min(len(groups) - 1, int(len(groups) * fraction)))
    held = set(groups[:count])
    directory = Path(out) / "evaluation"
    directory.mkdir(parents=True, exist_ok=True)
    train, test = directory / "train.txt", directory / "holdout.txt"
    train.write_text("\n".join(line for line in lines if group_id(line) not in held) + "\n")
    test.write_text("\n".join(line for line in lines if group_id(line) in held) + "\n")
    manifest = {"schema_version": 1, "source_sha256": file_hash(source), "seed": seed,
                "fraction": fraction, "window": window, "split_unit": "normalized source line",
                "train_groups": sorted(set(groups) - held), "holdout_groups": sorted(held),
                "train_sha256": file_hash(train), "holdout_sha256": file_hash(test)}
    (directory / "split.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return str(train)


def load(out):
    directory = Path(out) / "evaluation"
    manifest = json.loads((directory / "split.json").read_text())
    for key, filename in (("train", "train.txt"), ("holdout", "holdout.txt")):
        if file_hash(directory / filename) != manifest[f"{key}_sha256"]:
            raise ValueError(f"{key} corpus changed after split")
    train, test = documents(directory / "train.txt"), documents(directory / "holdout.txt")
    if not train or not test or {group_id(x) for x in train} & {group_id(x) for x in test}:
        raise ValueError("empty or overlapping train/holdout groups")
    return train, test, manifest
