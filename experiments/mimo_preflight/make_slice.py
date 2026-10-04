"""Make a two-layer diagnostic checkpoint from the published MiMo-7B weights."""

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    for name in ("configuration_mimo.py", "modeling_mimo.py", "tokenizer.json", "tokenizer_config.json",
                 "generation_config.json"):
        shutil.copy2(args.checkpoint / name, args.out / name)
    config = json.loads((args.checkpoint / "config.json").read_text())
    config["num_hidden_layers"] = 2
    config["num_nextn_predict_layers"] = 0
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    selected = {key for key in index if key in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")
                or key.startswith(("model.layers.0.", "model.layers.1."))}
    weights = {}
    for shard in sorted({index[key] for key in selected}):
        with safe_open(args.checkpoint / shard, framework="pt", device="cpu") as file:
            for key in sorted(selected):
                if index[key] == shard:
                    weights[key] = file.get_tensor(key).contiguous()
    save_file(weights, args.out / "model.safetensors")


if __name__ == "__main__":
    main()
