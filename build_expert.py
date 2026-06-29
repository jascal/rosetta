#!/usr/bin/env python
"""build_expert.py — repo-root launcher for the pack builder (EXPERTS.md).

`pack` lives under py/, so `python -m pack.build` needs py/ on the path; this launcher puts it there so a build just
works from the repo root, with no PYTHONPATH and no -m re-import warning:

    .venv/bin/python build_expert.py examples/logic/expert.toml     # declarative (an expert.toml)
    .venv/bin/python build_expert.py <out> --corpus … --bundle …    # or the flag form

(The same args as `pack.build`; see py/pack/README.md.)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "py"))
from pack.build import main  # noqa: E402

if __name__ == "__main__":
    main()
