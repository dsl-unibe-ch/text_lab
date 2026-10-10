"""What a PDF text block is: equation, figure label, or prose.

Heuristics on PyMuPDF's ``page.get_text("dict")`` blocks, shared by the page
checks, the native extraction for Markdown and the PDF reconstruction.
"""

from __future__ import annotations

import re

# --- Math detection --------------------------------------------------------
#
# Research PDFs commonly render equations either in a math-specific font
# (Computer Modern math families, STIX Math, Cambria Math, MTMain,
# Latin Modern Math, ...) or as high-density blocks of math operators.
# Sending such content through a NMT model destroys the equation and
# often produces garbled fallback glyphs after re-insertion because the
# target font lacks the original math symbols. We therefore detect
# "math-y" blocks and leave them untouched: no translation, no redaction.
#
# IMPORTANT: the font pattern is deliberately narrow. Earlier versions
# matched "CMR" (Computer Modern Roman) and "Latin Modern" — the body
# fonts of most LaTeX papers — which caused every text block to be
# flagged as math, producing an unchanged output document.
_MATH_FONT_PATTERN = re.compile(
    r"\bCMSY\d*"  # Computer Modern Symbols
    r"|\bCMMI\d*"  # Computer Modern Math Italic
    r"|\bCMEX\d*"  # Computer Modern Math Extension
    r"|\bMSAM\d*|\bMSBM\d*"  # AMS math fonts
    r"|\bMTMain\b|\bMTSym\b|\bMTEx\b|\bMTExtra\b"  # MathTime
    # ONLY the Math variant (not Roman/Mono/Italic)
    r"|LatinModern-?Math\b"
    r"|Cambria\s?Math\b"  # OpenType math
    r"|(?:STIX|STIXTwo)\s?Math\b"  # STIX Math families only
    r"|MathJax"
    r"|Lucida(?:New)?Math\b"
    r"|\bSymbol(?:MT)?\b"  # Adobe Symbol / SymbolMT
    r"|Euler(?:Fraktur|Script|Extra)?\b",
    re.IGNORECASE,
)


def _is_math_char(ch: str) -> bool:
    """Identify characters used almost exclusively in equations."""
    code = ord(ch)
    # NOTE: intentionally excludes the Greek block (0x0370-0x03FF) and the
    # Arrows block (0x2190-0x21FF). Both appear in body text often enough
    # (proper nouns, bullets, decorative arrows) to make them unreliable.
    return (
        0x2200 <= code <= 0x22FF  # Math Operators (strong)
        or 0x2A00 <= code <= 0x2AFF  # Supplemental Math Operators
        or 0x27C0 <= code <= 0x27EF  # Miscellaneous Math Symbols-A
        or 0x2980 <= code <= 0x29FF  # Miscellaneous Math Symbols-B
        or 0x1D400 <= code <= 0x1D7FF  # Math Alphanumeric Symbols
        or 0x2100 <= code <= 0x214F  # Letterlike Symbols (ℕ ℤ ℝ ∑ ...)
    )


def _is_math_span(span: dict) -> bool:
    """Heuristic: does a pymupdf span look like part of an equation?"""
    font = str(span.get("font", ""))
    if font and _MATH_FONT_PATTERN.search(font):
        return True
    text = (span.get("text", "") or "").strip()
    if len(text) < 3:
        # Ignore tiny spans (bullets, subscripts, single punctuation).
        return False
    math_count = sum(1 for c in text if _is_math_char(c))
    if math_count == 0:
        return False
    # Require a strong ratio *and* at least two math characters so isolated
    # symbols in a sentence never flip a whole span.
    return math_count >= 2 and math_count / len(text) >= 0.30


def is_math_block(block: dict) -> bool:
    """Return True if a block is an equation, to be left untouched.

    A strict majority of its spans must look like math.
    """
    spans = [
        s for line in block.get("lines", []) for s in line.get("spans", [])
    ]
    if not spans:
        return False
    math_spans = sum(1 for s in spans if _is_math_span(s))
    return math_spans * 2 > len(spans)  # STRICT > 50%


# --- Image-overlap detection ----------------------------------------------


def _rect_area(rect: tuple[float, float, float, float]) -> float:
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])


def rect_intersect_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Return the area two ``(x0, y0, x1, y1)`` rectangles share."""
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    return max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)


def extract_image_bboxes(
    page_dict: dict,
) -> list[tuple[float, float, float, float]]:
    """Return image-block bboxes on a page (block ``type == 1``)."""
    out: list[tuple[float, float, float, float]] = []
    for block in page_dict.get("blocks", []):
        if block.get("type") == 1:
            bbox = block.get("bbox")
            if bbox and len(bbox) == 4:
                out.append(tuple(bbox))
    return out


def text_block_overlaps_image(
    text_bbox: tuple[float, float, float, float],
    image_bboxes: list[tuple[float, float, float, float]],
    min_overlap: float = 0.40,
) -> bool:
    """Return True if an image covers much of a text block.

    "Much" is ``min_overlap`` of the text block's area, 40% by default.

    Redacting such blocks tends to leave white bands over figures and
    corrupt visual context, so we skip translation for them.
    """
    ta = _rect_area(text_bbox)
    if ta <= 0:
        return False
    for ib in image_bboxes:
        if rect_intersect_area(text_bbox, ib) / ta >= min_overlap:
            return True
    return False
