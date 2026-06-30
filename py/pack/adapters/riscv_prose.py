"""pack.adapters.riscv_prose — the RISC-V ISA manual's explanatory PROSE → citable passages (model-free).

The normrules adapter ([[normrules]]) emits the manual's *normative rules* (requirements). But definitional/conceptual
queries — "what is a hart?", "what is machine mode?" — have no normative rule that *defines* the term, so a rules-only
expert abstains on them. This adapter recovers the non-normative prose (the paragraphs the norm-rules were extracted
*out of*): one cleaned paragraph → one retrievable, citable passage, same shape as rules.txt so it concatenates straight
into the grounding corpus.

Source = the AsciiDoc src/ of github.com/riscv/riscv-isa-manual (CC BY 4.0 → committable). Like normrules, the DERIVED
corpus (prose.txt) is the committed artifact; the manual checkout is not. Emits into <out>/:
  prose.txt        "[manual:<slug> · <Chapter › Section>] <text>"   (grounding --no-split: one passage per paragraph)
  prose_plain.txt  "<text>"                                          (n-gram corpus)

Default chapter allowlist = the conceptual/definitional chapters (terminology, privilege modes, memory model, CSRs) —
not the extension-encoding chapters, which are lookup material the norm-rules already carry and which would only add
retrieval-leak surface. Pass `chapters=` to override.
"""
import os
import re

from .base import Extraction, register

# the conceptual chapters where the definitions live (paths relative to the manual's src/). The prefaces are excluded
# deliberately: they are version-changelog text ("Made the mstatus.mpp field WARL…"), not definitional prose — pure
# retrieval-leak surface.
DEFAULT_CHAPTERS = [
    "unpriv/intro.adoc", "unpriv/naming.adoc", "unpriv/zicsr.adoc",
    "unpriv/rvwmo.adoc", "unpriv/mm-explanatory.adoc",
    "priv/machine.adoc", "priv/supervisor.adoc", "priv/csrs.adoc",
]


def _macro(m):
    """Decode a RISC-V AsciiDoc spec macro `kind:name[arg]` to readable text (prose is read AS the answer):
    insn:mret[] → MRET, ext:sv39[] → SV39, csr:mstatus[mprv] → mstatus.mprv, csr:satp[] → satp."""
    kind, name, arg = m.group(1), m.group(2), (m.group(3) or "").strip()
    if not name:
        return arg                                              # csr::[sum] (malformed) → the field
    if kind in ("insn", "inst", "ext"):
        return name.upper()
    if kind == "csr":
        return f"{name}.{arg}" if arg and arg.isidentifier() else name
    return name                                                 # reg:/other → the name


_MACRO = re.compile(r"\b(insn|inst|ext|csr|reg)::?([A-Za-z0-9._]*)\[([^\]]*)\]")

# --- inline markup cleaners (applied to an already-assembled paragraph), in order ---
_INLINE = [
    (_MACRO, _macro),                                            # insn:/csr:/ext: spec macros → readable text
    (re.compile(r"\(\(\([^)]*\)\)\)"), ""),                       # (((index, entries))) → drop
    (re.compile(r"\[#[^\]]+\]#([^#]*)#"), r"\1"),                 # [#norm:anchor]#highlighted# → highlighted
    (re.compile(r"cite:\[[^\]]*\]"), ""),                        # cite:[ref] → drop
    (re.compile(r"<<[^,>]*,\s*([^>]*)>>"), r"\1"),               # <<anchor, display text>> → display text
    (re.compile(r"<<[^>]*>>"), ""),                              # <<anchor>> → drop (bare cross-ref)
    (re.compile(r"\[\[[^\]]*\]\]"), ""),                         # [[anchor]] → drop
    (re.compile(r'"`'), '"'), (re.compile(r'`"'), '"'),         # "`smart`" double-quotes → "
    (re.compile(r"'`"), "'"), (re.compile(r"`'"), "'"),         # '`smart`' single-quotes → '
    (re.compile(r"`([^`]+)`"), r"\1"),                          # `mono` → mono
    (re.compile(r"(?<![A-Za-z0-9])_([^_]+)_(?![A-Za-z0-9])"), r"\1"),   # _italic_ → italic (not mid-word)
    (re.compile(r"\*\*?([^*]+)\*\*?"), r"\1"),                  # *bold* / **bold** → bold
    (re.compile(r"footnote:\[[^\]]*\]"), ""),                   # footnote:[..] → drop
    (re.compile(r"\s*:::+\s*"), ": "),                          # labeled-list "Term:::" → "Term: "
    (re.compile(r"\s+([.,;:])"), r"\1"),                         # " ." left by a stripped xref → "."
    (re.compile(r"\s+"), " "),                                   # collapse whitespace
]

# block-level skips: a line that opens material we don't want as prose
_SKIP_LINE = re.compile(
    r"^\s*(:[\w-]+:|ifdef::|ifndef::|endif::|include::|image::|video::|"
    r"\[(source|cols|width|NOTE|TIP|IMPORTANT|WARNING|CAUTION|%|wavedrom|ditaa|"
    r"plantuml|format|options|align|stem|latexmath).*\]|\[\[|//)"
)
_FENCE = re.compile(r"^\s*(----+|====+|\|===|\.\.\.\.+|\*\*\*\*+|____+|\+\+\+\++)\s*$")
_HEADING = re.compile(r"^(=+)\s+(.*\S)\s*$")
_LISTMARK = re.compile(r"^\s*([*.]+|\d+\.)\s+")                  # bullet / numbered list marker


def _clean(text):
    for pat, rep in _INLINE:                                    # rep may be a string template or a callable (macros)
        text = pat.sub(rep, text)
    return text.strip()


def _is_prose(text):
    """Keep substantial natural-language paragraphs; drop fragments, encodings, and table/code residue."""
    if len(text) < 40:
        return False
    words = text.split()
    if len(words) < 8:
        return False
    alpha = sum(c.isalpha() or c.isspace() for c in text)
    return alpha / len(text) >= 0.65                            # mostly letters → prose, not an encoding/operand table


def chapter_to_passages(path, chapter_title):
    """One .adoc file → [(slug_suffix, section_path, text)]. Accumulates blank-line-separated prose blocks, tracks the
    heading path, and skips tables / source blocks / passthroughs / attribute & macro lines."""
    out = []
    section = chapter_title
    buf = []
    fenced = False                                             # inside a ---- / |=== / .... delimited block
    n = 0

    def flush():
        nonlocal n
        if not buf:
            return
        text = _clean(" ".join(buf))
        buf.clear()
        if _is_prose(text):
            n_local = n
            out.append((f"{n_local}", section, text))
            n += 1

    for raw in open(path, encoding="utf-8"):
        line = raw.rstrip("\n")
        if _FENCE.match(line):                                 # delimiter line: a NOTE/sidebar (====) keeps its prose,
            if line.strip().startswith("="):                  #   so DON'T toggle skipping for it — just drop the line
                flush()
                continue
            fenced = not fenced                               # ---- / |=== / .... → toggle skip (code/tables)
            flush()
            continue
        if fenced:
            continue
        h = _HEADING.match(line)
        if h:
            flush()
            title = _clean(h.group(2))
            section = title if h.group(1) == "==" else f"{chapter_title} › {title}"
            continue
        if not line.strip():                                  # blank line = paragraph boundary
            flush()
            continue
        if _SKIP_LINE.match(line):
            continue
        line = _LISTMARK.sub("", line)                        # strip a leading list marker, keep the item text
        buf.append(line.strip())
    flush()
    return out


def _slug(chapter_file):
    return os.path.splitext(os.path.basename(chapter_file))[0].replace("-", "_")


def _passages(manual_src, chapters=None):
    """Yield (section, text) prose passages — section = "manual:<file>_<n> · <Chapter › Section>"."""
    for chap in (chapters or DEFAULT_CHAPTERS):
        path = os.path.join(manual_src, chap)
        if not os.path.exists(path):
            print(f"[riscv_prose] skip (absent): {chap}")
            continue
        base = _slug(chap)
        title = base.replace("_", " ").title()
        for suffix, section, text in chapter_to_passages(path, title):
            yield f"manual:{base}_{suffix} · {section}", text


@register("riscv_prose")
def adapt(source, *, chapters=None, citation="RISC-V ISA Manual (CC BY 4.0)", **_):
    """manual src/ → Extraction: explanatory-prose passages + defines (the structural parenthesized-abbrev extractor)."""
    from .. import reasoning
    passages = list(_passages(source, chapters))
    if not passages:
        raise ValueError(f"riscv_prose adapter: extracted 0 passages from {source} (is this the manual's src/ dir?)")
    return Extraction(passages, defines=reasoning.extract_defines(passages), citation=citation)


def to_corpus(manual_src, out, *, chapters=None, citation="RISC-V ISA Manual (CC BY 4.0)"):
    """Back-compat: write <out>/prose.txt + prose_plain.txt. Returns (prose_txt, prose_plain, n_passages)."""
    os.makedirs(out, exist_ok=True)
    prose_txt = os.path.join(out, "prose.txt")
    prose_plain = os.path.join(out, "prose_plain.txt")
    total = 0
    with open(prose_txt, "w", encoding="utf-8") as ft, open(prose_plain, "w", encoding="utf-8") as fp:
        for section, text in _passages(manual_src, chapters):
            ft.write(f"[{section}] {text}\n")
            fp.write(text + "\n")
            total += 1
    if total == 0:
        raise ValueError(f"riscv_prose adapter: extracted 0 passages from {manual_src} "
                         f"(checked chapters — is this the manual's src/ dir?)")
    return prose_txt, prose_plain, total


def main():
    import argparse
    ap = argparse.ArgumentParser(prog="pack.adapters.riscv_prose",
                                 description="RISC-V ISA manual src/ → citable prose corpus (prose.txt)")
    ap.add_argument("manual_src", help="path to the manual's src/ directory")
    ap.add_argument("out", help="output dir (prose.txt + prose_plain.txt)")
    ap.add_argument("--chapters", nargs="*", help="override the default conceptual-chapter allowlist")
    a = ap.parse_args()
    _, _, n = to_corpus(a.manual_src, a.out, chapters=a.chapters)
    print(f"[riscv_prose] {n} prose passages -> {os.path.join(a.out, 'prose.txt')}")


if __name__ == "__main__":
    main()
