"""Collect resident fieldrun top-two logits on frozen matched MiMo cases."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    raw = args.cases.read_bytes()
    suite_sha256 = hashlib.sha256(raw).hexdigest()
    suite = json.loads(raw)
    manifest_sha256 = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    previous = [json.loads(line) for line in args.out.read_text().splitlines()] if args.out.exists() else []
    if any(row["suite_sha256"] != suite_sha256 or row["manifest_sha256"] != manifest_sha256 or
           row["id"] != i for i, row in enumerate(previous)):
        raise ValueError("existing output does not match this suite/bundle")
    with args.out.open("a") as output:
        for case in suite["cases"][len(previous):]:
            body = json.dumps({"ids": case["ids"], "k": 2}).encode()
            request = urllib.request.Request(f"http://127.0.0.1:{args.port}/topk", data=body,
                                             headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=600) as response:
                top = json.loads(response.read())["topk"]
            row = {"id": case["id"], "family": case["family"], "suite_sha256": suite_sha256,
                   "manifest_sha256": manifest_sha256, "top2_ids": [int(item[0]) for item in top],
                   "top2_logits": [float(item[1]) for item in top]}
            output.write(json.dumps(row) + "\n")
            output.flush()
            print(f"{case['id'] + 1}/{len(suite['cases'])}: {case['family']} {row['top2_ids'][0]}", flush=True)


if __name__ == "__main__":
    main()
