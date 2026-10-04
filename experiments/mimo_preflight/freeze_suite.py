"""Freeze model-independent parity contexts for the matched MiMo-7B checkpoints."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "py"))
from recursion_ladder import make_cases, prompt  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    tok = Tokenizer.from_file(str(args.tokenizer))
    texts = []
    sequence = "amber cobalt jade ivory bronze crimson"
    for gap in (0, 2, 6):
        filler = "The archive records an unrelated morning. " * gap
        for copies in (1, 2):
            for distractor in (False, True):
                decoy = "amber cobalt jute ivory bronze crimson. " if distractor else ""
                texts.append(("copy", f"{(sequence + '. ') * copies}{filler}{decoy}amber cobalt jade"))
    for case in make_cases(seed=29, depths=(1, 2, 3, 4), trials=1, topology="branched"):
        texts.append(("graph", prompt(case)))
    texts.extend(("family", text) for text in (
        "When Alice and Bob went to the store, Bob gave a drink to",
        "When Bob and Alice went to the store, Alice gave a drink to",
        "Alice is a girl. Bob is a boy. The girl is named",
        "Alice is a boy. Bob is a girl. The girl is named",
        "Every robin is a bird. A robin is an animal. Therefore, a robin is",
        "Every robin is a bird. A robin is an animal. Therefore, a bird is",
        "A turns on B. B turns on C. We unplug B. Is C on? Answer:",
        "A turns on B. B turns on C. A is on. Is C on? Answer:",
    ))
    texts.extend(("ordinary", text) for text in (
        "The capital of France is",
        "The capital of Germany is",
        "def balanced(s):\n    stack = []\n    for ch in s:\n        if ch == '(': stack.append(ch)\n        elif ch == ')':",
        "The sequence is 2, 4, 6, 8,",
    ))
    rows = [{"id": i, "family": family, "text": text,
             "ids": tok.encode(text, add_special_tokens=False).ids}
            for i, (family, text) in enumerate(texts)]
    assert len(rows) == 32 and all(row["ids"] for row in rows)
    payload = {"tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(), "cases": rows}
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"frozen {len(rows)} contexts; lengths {min(len(r['ids']) for r in rows)}–{max(len(r['ids']) for r in rows)}")


if __name__ == "__main__":
    main()
