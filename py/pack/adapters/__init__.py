"""pack.adapters — the document-adapter system (model-free expert sources).

A "document expert" is built from a SOURCE through a pluggable adapter that yields a canonical `Extraction` (citable
passages + the structural facts that drive the strategy tables). The builder and runtime consume only the Extraction,
so every source type is just a registered adapter — they are all INSTANCES of one contract:

    normrules     a spec's machine-readable normative rules            (RISC-V ISA: norm-rules.json)
    riscv_prose   an AsciiDoc spec manual's explanatory prose          (RISC-V ISA manual)
    pretext       a PreTeXt / MathBook-XML textbook (tagged structure) (aata, fcla, …)
    latexml       LaTeXML HTML5 + MathML (arXiv HTML / ar5iv)          (CC-licensed papers)
    glossary      a term→definition controlled vocabulary              (.json / .tsv / .txt)
    rfc           an IETF RFC / internet-standard plain-text spec      (rfc-editor.org .txt)
    manpage       a rendered Unix man page / CLI reference             (`man x | col -bx`)
    nh_legal      a statute / legal code (RSA-style §-numbered text)    (NH RSA Title XLIII)

Add a source type = write `adapt(source, **opts) -> Extraction` and `@register("name")`. See base.Extraction.
"""
from .base import Extraction, get, names, register

# import the adapter modules so their @register(...) runs (the registry is populated as a side effect)
from . import normrules, riscv_prose, pretext, latexml, pedagogy  # noqa: E402,F401
from . import glossary, rfc, manpage, nh_legal  # noqa: E402,F401

__all__ = ["Extraction", "get", "names", "register"]
