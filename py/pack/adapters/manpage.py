"""pack.adapters.manpage — a rendered Unix man page / CLI reference → citable passages + per-option defines + inventory.

A man page is already a structured reference: ALL-CAPS section headers at column 0 (NAME, SYNOPSIS, DESCRIPTION,
OPTIONS, EXIT STATUS, SEE ALSO, …) with indented bodies, and — anywhere in the body — one flag-tagged paragraph per
option (`-i, --ignore-case` then an indented description). Note the options are NOT always under a section literally
named OPTIONS: GNU coreutils lists them under DESCRIPTION. So this adapter finds an option wherever a paragraph BEGINS
with a flag, and renders the structure onto the uniform strategy table with the EXISTING intents (no new ergo cue, no
runtime change):
  * one citable passage per prose paragraph   "[man:<name>.<slug> · <SECTION>] <text>"
  * one citable passage per OPTION            "[man:<name>.opt.<slug> · OPTIONS] <flags> — <description>"
  * `defines(passage, entity)`                → define intent: the command name → DESCRIPTION; each LONG flag
                                                (`--ignore-case`) → its option passage ("what does --ignore-case do")
  * `items(canonical_flag, "options")`        → count/list ("how many options", "list the options")

Single-letter short flags are case-sensitive but the strategy runtime folds case, so they are NOT emitted as define
entities (they would be ambiguous); the case-stable `--long` flag is the reliable handle. Source = a RENDERED man page,
i.e. `man <name> | col -bx` (col -b strips backspace overstrike, -x expands tabs) — universally reproducible; the
DERIVED package is committable. Pass `name=` to override the command name (else taken from NAME).
"""
from __future__ import annotations

import os
import re

from .base import Extraction, register

_WS = re.compile(r"\s+")
_OVERSTRIKE = re.compile(r".\x08")                             # residual bold/underline overstrike ("x\bx") if any
_SECHEAD = re.compile(r"^([A-Z][A-Z0-9]*(?: [A-Z0-9]+)*)\s*$")   # a col-0 ALL-CAPS section header
_MANREF = re.compile(r"[A-Za-z0-9_.-]+\([0-9nx]\)")           # a "name(1)" man cross-reference (used to drop furniture)
_FLAG = re.compile(r"^(-[A-Za-z0-9]|--[A-Za-z0-9][\w-]*)")     # an option token: -i or --ignore-case
_SLUG = re.compile(r"[^a-z0-9]+")


def _slug(s: str) -> str:
    return _SLUG.sub("-", s.strip().lower()).strip("-") or "x"


def _opt_slug(flag: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", flag).strip("-") or "opt"   # CASE-PRESERVING so -V and -v get distinct ids


def _clean(s: str) -> str:
    return _WS.sub(" ", s).strip()


def _destrike(text: str) -> str:
    return _OVERSTRIKE.sub(lambda m: m.group(0)[-1], text)     # "g\bg" → "g" (defensive; col -b usually did this)


def _is_furniture(line: str) -> bool:
    """The running header/footer man renders per page: "GREP(1)  User Commands  GREP(1)" and "GNU coreutils 8.32  2019
    GREP(1)". Both END with a name(section) ref at the right margin; the header also STARTS with one. Body lines don't."""
    s = line.rstrip()
    if not s:
        return False
    return bool(_MANREF.match(s) or re.search(r"\s{2,}" + _MANREF.pattern + r"$", s))


def _sections(text: str):
    """Split rendered man text into [(SECTION, [body lines])], furniture removed, in document order."""
    out, cur, body = [], None, []
    for raw in _destrike(text).split("\n"):
        line = raw.rstrip()
        if _is_furniture(line):
            continue
        h = _SECHEAD.match(line)
        if h and (len(line) - len(line.lstrip()) == 0):       # a header sits at column 0
            if cur is not None:
                out.append((cur, body))
            cur, body = h.group(1), []
            continue
        if cur is not None:
            body.append(line)
    if cur is not None:
        out.append((cur, body))
    return out


def _blocks(body_lines):
    """A section body → blank-line-separated blocks (each a list of non-empty lines), in order."""
    blocks, cur = [], []
    for line in body_lines:
        if line.strip():
            cur.append(line)
        elif cur:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    return blocks


def _classify(block):
    """A block → ("option", flags_line, description) when it BEGINS with a flag (a `.TP` tagged paragraph), else
    ("prose", text). Leading flag lines are collected (a few options stack short+long across lines); the rest, plus any
    inline text after 2+ spaces on a flag line, is the description."""
    if not _FLAG.match(block[0].strip()):
        return ("prose", _clean(" ".join(ln.strip() for ln in block)), None)
    flags, desc, in_desc = [], [], False
    for ln in block:
        s = ln.strip()
        if not in_desc and _FLAG.match(s):
            m = re.match(r"^(.*?\S)(?:\s{2,}(.*))?$", s)      # split "-i, --ignore-case   Ignore case." if inline
            flags.append(m.group(1).strip())
            if m.group(2):
                desc.append(m.group(2).strip())
                in_desc = True
        else:
            in_desc = True
            desc.append(s)
    return ("option", "; ".join(flags), _clean(" ".join(desc)))


def _command_name(secs, override):
    if override:
        return override.strip()
    for name, body in secs:
        if name == "NAME":
            txt = _clean(" ".join(body))
            return _clean(re.split(r"\s[-–—]\s", txt, 1)[0]).split(",")[0].strip() or "command"
    return "command"


def _flag_tokens(flags_line):
    """"-i, --ignore-case=WORD" → ['-i', '--ignore-case'] (the callable option names, args dropped)."""
    toks = []
    for tok in re.split(r"[,;\s]+", flags_line):
        m = _FLAG.match(tok)
        if m:
            toks.append(m.group(1))
    return toks


def _unique(pid, seen):
    """A collision-free passage id (distinct options/sections must never share a citation handle)."""
    if pid not in seen:
        seen.add(pid)
        return pid
    i = 2
    while f"{pid}-{i}" in seen:
        i += 1
    seen.add(f"{pid}-{i}")
    return f"{pid}-{i}"


@register("manpage")
def adapt(source, *, name=None, section="1", citation=None, **_):
    """A rendered man page (`man x | col -bx` output) → Extraction (per-section passages, per-option defines + items)."""
    with open(source, encoding="utf-8", errors="replace") as f:
        secs = _sections(f.read())
    if not secs:
        raise ValueError(f"manpage adapter: no sections in {source} — is this `man <name> | col -bx` output?")
    cmd = _command_name(secs, name)
    citation = citation or f"{cmd}({section}) man page"
    passages, defines, items, seen_pid, seen_item = [], [], [], set(), set()
    desc_pid = None
    for sec, body in secs:
        prose_n = 0
        for block in _blocks(body):
            kind, a, b = _classify(block)
            if kind == "option":
                flags_line, desc = a, b
                toks = _flag_tokens(flags_line)
                if not (toks and desc):
                    continue
                longs = [t for t in toks if t.startswith("--")]
                canon = longs[0] if longs else toks[0]        # the option's canonical (unambiguous) name
                pid = _unique(f"man:{cmd}.opt.{_opt_slug(canon)}", seen_pid)
                passages.append((f"{pid} · OPTIONS", f"{flags_line} — {desc}"))
                if canon not in seen_item:                    # one inventory item per option (canonical name)
                    seen_item.add(canon)
                    items.append((canon, "options"))
                for t in longs:                               # define the LONG flags only (case-stable, unambiguous):
                    defines.append((pid, t.lower()))          #   "what does --ignore-case do" → this option passage
            else:
                prose_n += 1
                pid = _unique(f"man:{cmd}.{_slug(sec)}{'' if prose_n == 1 else '-' + str(prose_n)}", seen_pid)
                passages.append((f"{pid} · {sec}", a))
                if sec == "DESCRIPTION" and desc_pid is None:
                    desc_pid = pid
    if desc_pid:                                              # "what is <cmd>" → its DESCRIPTION
        defines.append((desc_pid, cmd.lower()))
    if not passages:
        raise ValueError(f"manpage adapter: {source} yielded no usable passages")
    return Extraction(passages, defines=defines, items=items, citation=citation)


def main():
    import argparse
    ap = argparse.ArgumentParser(prog="pack.adapters.manpage",
                                 description="a rendered man page (`man x | col -bx`) → passages + option defines/inventory")
    ap.add_argument("source", help="rendered man-page text file")
    ap.add_argument("out", help="output corpus path (.txt)")
    ap.add_argument("--name", default=None, help="command name (else taken from the NAME section)")
    a = ap.parse_args()
    ext = adapt(a.source, name=a.name)
    ext.write_corpus(a.out)
    print(f"[manpage] {os.path.basename(a.source)}: {ext.summary()} -> {a.out}")


if __name__ == "__main__":
    main()
