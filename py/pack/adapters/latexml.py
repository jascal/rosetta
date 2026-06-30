"""pack.adapters.latexml — LaTeXML HTML5 + MathML (arXiv HTML / ar5iv) → an Extraction.

The fourth document-adapter instance: arXiv now publishes a LaTeXML-generated HTML version of every LaTeX submission
(MathML for math), and ar5iv has the back-catalog. LaTeXML tags structure with stable `ltx_` classes — sections
(ltx_title_section), paragraphs (ltx_para/ltx_p), and theorem-family blocks (ltx_theorem ltx_theorem_<kind>) — and puts
the LaTeX of each formula in `<math alttext="…">`. We parse that into passages + named statements (+ any clearly named
definitions).

HONEST SCOPE vs PreTeXt: papers are NOT purpose-tagged like a textbook. Authors use arbitrary \\newtheorem names, and
definitions are rarely titled with the defined term — so this adapter is high-RECALL, lower-PRECISION (sections +
paragraphs + theorem blocks are reliable; clean `defines` are not). And LICENSING: arXiv's default license is not an
open reuse license — only run this over CC-BY / CC-BY-SA / CC0 papers (filter by the per-paper license first).
"""
import html
import re
from html.parser import HTMLParser

from .base import Extraction, register

_STATEMENT_KINDS = {"theorem", "proposition", "lemma", "corollary"}
_WS = re.compile(r"\s+")
_CLASS = re.compile(r"ltx_theorem_([a-z]+)")


class _LaTeXMLParser(HTMLParser):
    """Walk LaTeXML HTML: collect section titles (for citations), prose paragraphs, and theorem-family blocks. Math is
    reduced to its `alttext` LaTeX; all other tags are dropped."""

    def __init__(self, prefix):
        super().__init__(convert_charrefs=True)
        self.prefix = prefix
        self.passages, self.defines, self.statements = [], [], []
        self.section = prefix
        self._buf = []                                # current text accumulator
        self._mode = None                             # None | "para" | "title" | ("block", kind)
        self._skip_depth = 0                          # >0 while inside <math> (skip its MathML children)
        self._n = 0

    # --- helpers ---
    def _flush_para(self):
        text = _WS.sub(" ", "".join(self._buf)).strip()
        self._buf = []
        if len(text) >= 40:
            self._n += 1
            self.passages.append((f"{self.prefix}:p{self._n} · {self.section}", text))

    def _flush_block(self, kind):
        text = _WS.sub(" ", "".join(self._buf)).strip()
        self._buf = []
        if len(text) < 25:
            return
        self._n += 1
        pid = f"{self.prefix}:{kind}{self._n}"
        # the block usually opens with "Theorem 2.1 (Name)." — pull a parenthesized name if present
        nm = re.match(r"\s*[A-Za-z]+[\s0-9.]*\(([^)]{2,60})\)", text)
        name = nm.group(1).strip().lower() if nm else ""
        self.passages.append((f"{pid} · {kind.capitalize()}{': ' + name if name else ''}", text))
        if kind in _STATEMENT_KINDS and name:
            self.statements.append((pid, name))
        if kind == "definition" and name:
            self.defines.append((pid, name))

    # --- HTMLParser callbacks ---
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class", "")
        if tag == "math":                             # keep the LaTeX (alttext), skip the MathML subtree
            self._buf.append(" " + html.unescape(a.get("alttext", "")) + " ")
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if "ltx_theorem" in cls:
            self._flush_para()
            m = _CLASS.search(cls)
            self._mode = ("block", m.group(1) if m else "theorem")
            self._buf = []
        elif "ltx_title_section" in cls or "ltx_title_subsection" in cls:
            self._flush_para()
            self._mode = "title"
            self._buf = []
        elif tag in ("p", "div") and "ltx_p" in cls and self._mode is None:
            self._mode = "para"

    def handle_endtag(self, tag):
        if tag == "math" and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if isinstance(self._mode, tuple) and tag == "div":
            self._flush_block(self._mode[1])
            self._mode = None
        elif self._mode == "title" and tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.section = _WS.sub(" ", "".join(self._buf)).strip()[:80] or self.prefix
            self._buf = []
            self._mode = None
        elif self._mode == "para" and tag in ("p", "div"):
            self._flush_para()
            self._mode = None

    def handle_data(self, data):
        if not self._skip_depth and self._mode is not None:
            self._buf.append(data)


@register("latexml")
def adapt(source, *, prefix="paper", citation="arXiv (LaTeXML HTML)", **_):
    """A LaTeXML HTML file (arXiv/ar5iv) → Extraction: sectioned paragraphs + theorem/definition blocks (+ named
    statements/defines where present). `source` is a path to the .html file."""
    p = _LaTeXMLParser(prefix)
    p.feed(open(source, encoding="utf-8", errors="replace").read())
    if not p.passages:
        raise ValueError(f"latexml adapter: 0 passages from {source} — is this a LaTeXML (arXiv/ar5iv) HTML file?")
    return Extraction(p.passages, defines=p.defines, statements=p.statements, citation=citation)
