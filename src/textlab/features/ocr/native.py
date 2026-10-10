"""The born-digital lane: pages read from a PDF's own text layer.

A page whose text layer is complete and usable is read with PyMuPDF (text
and embedded images) instead of being recognized by the vision model,
which is much faster. Pages with equations, a mis-encoded text layer or
too little text go to the PaddleOCR-VL lane instead.
"""

from __future__ import annotations

import base64
from pathlib import Path

from textlab.features.ocr import doc_ir
from textlab.features.ocr.marks import apply_markup, write_mark_debug_overlay
from textlab.features.ocr.rasters import decode_bgr, downscale_png_b64

#: Rasterization DPI of the page preview for native pages.
PREVIEW_DPI = 150
#: A page needs at least this many words to skip the vision model.
MIN_WORDS_NATIVE = 6
MIN_CHARS_NATIVE = 20


def page_has_text_layer(page) -> bool:
    """Return True if a PDF page has enough real text to skip recognition."""
    text = page.get_text("text") or ""
    if len(text.strip()) < MIN_CHARS_NATIVE:
        return False
    try:
        words = page.get_text("words")
    except Exception:
        words = []
    return len(words) >= MIN_WORDS_NATIVE


# Where a PDF with a broken ``ToUnicode`` CMap dumps its raw glyph indices:
# it emits indices instead of characters, and they surface as private-use
# codepoints (or, below, as control bytes).
_PUA_RANGES = ((0xE000, 0xF8FF), (0xF0000, 0xFFFFD), (0x100000, 0x10FFFD))
#: Private-use glyphs are the one signal with a legitimate use — journal
#: headers and corporate templates map a logo or a bullet into the PUA — so
#: they need company before they condemn a page. Control characters and
#: U+FFFD have no legitimate use at all and count from the first one.
PUA_GLYPH_MIN = 4


def text_layer_is_corrupt(page) -> bool:
    """Heuristic: is this page's text layer mis-encoded beyond recovery?

    Some publisher PDFs ship a ``ToUnicode`` CMap that maps glyphs to control
    bytes and private-use codepoints rather than to characters. The extracted
    text then looks like ``MMD½Hk; P; Q ¼`` instead of ``MMD[Hk; P, Q] =``,
    which no downstream consumer can undo — the mapping back to real characters
    simply is not in the file. Re-reading the rendered page with the VL model
    is the only way to recover it, so such a page is routed there even when the
    caller asked for the born-digital fast lane.

    Deliberately narrow: it fires only on codepoints that cannot be text.
    Mojibake proper (``ðX; YÞ`` for ``(X, Y)``) is left alone, because
    every character in it is a legitimate letter in some language.
    """
    text = page.get_text("text") or ""
    if not text:
        return False
    n_pua = 0
    for ch in text:
        cp = ord(ch)
        if cp < 0x20 and ch not in "\t\n\r":
            return True  # C0 control: an unambiguous CMap failure
        if cp == 0xFFFD:
            return True  # the decoder already gave up on this glyph
        if any(lo <= cp <= hi for lo, hi in _PUA_RANGES):
            n_pua += 1
            if n_pua >= PUA_GLYPH_MIN:
                return True
    return False


# Unicode blocks that signal mathematical notation in a text layer.
_MATH_CHAR_RANGES = (
    (0x2200, 0x22FF),  # mathematical operators
    (0x27C0, 0x27EF),  # misc mathematical symbols-A
    (0x2980, 0x29FF),  # misc mathematical symbols-B
    (0x2A00, 0x2AFF),  # supplemental mathematical operators
    (0x2070, 0x209F),  # super/subscripts
    (0x0391, 0x03C9),  # Greek (equation variables)
    (0x1D400, 0x1D7FF),  # mathematical alphanumeric symbols
)
_MATH_FONT_HINTS = ("cmmi", "cmsy", "cmex", "msam", "msbm", "math", "stix")
MATH_GLYPH_MIN = 8  # math chars on a page before it is routed to the VL lane


def _is_math_char(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _MATH_CHAR_RANGES)


def page_has_math(page) -> bool:
    """Heuristic: does this born-digital page contain equations?

    PyMuPDF extracts equation glyphs as junk text, so pages with math must go
    through the VL lane even when they have a text layer. Signals: a minimum
    count of math-block Unicode chars, or spans set in known math fonts.
    """
    text = page.get_text("text") or ""
    n_math = sum(1 for ch in text if _is_math_char(ch))
    if n_math >= MATH_GLYPH_MIN:
        return True
    try:
        data = page.get_text("dict")
    except Exception:
        return False
    n_math_spans = 0
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                font = str(span.get("font", "")).lower()
                if any(hint in font for hint in _MATH_FONT_HINTS):
                    n_math_spans += 1
                    if n_math_spans >= 2:
                        return True
    return False


#: A figure has to be big enough to look at. Below either bar an image block
#: cannot carry information: ``MIN_FIGURE_PT`` is smaller than one character of
#: body text on the page, and ``MIN_FIGURE_PX`` is a raster too small to show a
#: shape. Publisher PDFs build a diagram out of hundreds of such fragments —
#: gradient tiles and hairline rules — and one real page has been seen to yield
#: 208 image blocks, of which the size bars alone rule out 192.
MIN_FIGURE_PT = 6.0
MIN_FIGURE_PX = 8


#: Greyscale standard deviation below which an image carries no detail. Solid
#: panel fills measure 0.0 and a near-solid one 1.3, while the faintest real
#: image seen measures 6.6 and most sit above 50, so this sits inside a wide
#: gap. Note it must be measured on *luminance*: counting distinct colours
#: would discard a bilevel scan, which is real content with only two of them.
MAX_FLAT_FILL_STD = 3.0


def is_figure_sized(block) -> bool:
    """Is this image block big enough to be a figure rather than a fragment?

    Checked in both spaces, because either one alone can be fooled: a gradient
    tile can be a large raster squeezed into a hairline box, and a decorative
    rule can be a 2x2 raster stretched across the page.
    """
    x0, y0, x1, y1 = block.get("bbox", (0, 0, 0, 0))
    if min(x1 - x0, y1 - y0) < MIN_FIGURE_PT:
        return False
    raster = (block.get("width"), block.get("height"))
    if not all(isinstance(v, int) and v > 0 for v in raster):
        return True  # no raster dimensions to judge by: keep it
    return min(raster) >= MIN_FIGURE_PX


def is_flat_fill(img_bytes: bytes | None) -> bool:
    """Is this image a solid or near-solid block of colour?

    The coloured rectangles behind a diagram's panels are embedded as images
    just like its photographs are, and they are far too big for any size test
    to catch. What separates them is that they hold no detail at all.

    Undecodable images are reported as *not* flat, so a failure to read one
    never silently drops content.
    """
    arr = decode_bgr(img_bytes)
    if arr is None or arr.size == 0:
        return False
    import cv2

    # Subsample first so the cost does not grow with the raster: a fill stays
    # flat under striding, and anything with detail keeps it.
    step_y = max(1, arr.shape[0] // 64)
    step_x = max(1, arr.shape[1] // 64)
    grey = cv2.cvtColor(arr[::step_y, ::step_x], cv2.COLOR_BGR2GRAY)
    return float(grey.std()) <= MAX_FLAT_FILL_STD


def native_page(
    fitz_page, page_number: int, debug_dir: Path | None = None
) -> doc_ir.Page:
    """Extract text + embedded images from a born-digital page into IR."""
    scale = PREVIEW_DPI / 72.0
    data = fitz_page.get_text("dict")
    regions: list[doc_ir.Region] = []
    order = 0
    skipped_fragments = 0
    for block in data.get("blocks", []):
        bbox = [c * scale for c in block.get("bbox", [0, 0, 0, 0])]
        if block.get("type") == 0:  # text block
            lines = []
            for line in block.get("lines", []):
                spans = [
                    span.get("text", "") for span in line.get("spans", [])
                ]
                joined = "".join(spans).strip()
                if joined:
                    lines.append(joined)
            text = "\n".join(lines).strip()
            if not text:
                continue
            regions.append(
                doc_ir.Region(
                    id=f"p{page_number}_r{order}",
                    type=doc_ir.TEXT,
                    bbox=bbox,
                    reading_order=order,
                    content={"text": text, "markdown": text},
                    confidence={"layout": 1.0, "ocr": None},
                    source="native",
                )
            )
            order += 1
        elif block.get("type") == 1:  # image block
            img_bytes = block.get("image")
            # Size first: it is free, and it rules out the bulk of the noise
            # before anything has to be decoded.
            if not is_figure_sized(block) or is_flat_fill(img_bytes):
                skipped_fragments += 1
                continue
            asset = None
            if img_bytes:
                asset = {
                    "b64": base64.b64encode(img_bytes).decode("ascii"),
                    "ext": (block.get("ext") or "png"),
                }
            regions.append(
                doc_ir.Region(
                    id=f"p{page_number}_r{order}",
                    type=doc_ir.FIGURE,
                    bbox=bbox,
                    reading_order=order,
                    content={"text": ""},
                    confidence={"layout": 1.0, "ocr": None},
                    asset=asset,
                    source="native",
                )
            )
            order += 1

    if skipped_fragments:
        # Worth a line: it is the difference between a 3-figure page and a
        # bundle with 200 unusable PNGs in it.
        figures = sum(1 for r in regions if r.type == doc_ir.FIGURE)
        print(
            f"[ocr] page {page_number}: skipped {skipped_fragments} "
            f"decorative image block(s), kept {figures} figure(s)",
            flush=True,
        )

    page = doc_ir.Page(
        page_number=page_number,
        regions=regions,
        source="native",
    )
    pix = fitz_page.get_pixmap(dpi=PREVIEW_DPI)
    raster_bytes = pix.tobytes("png")
    page_bgr = decode_bgr(raster_bytes)
    # The geometric markup baseline is diagnostic only unless the explicit
    # survey enrichment is requested (survey pages use the VL lane below).
    debug_collect = [] if debug_dir is not None else None
    if debug_dir is not None:
        apply_markup(page, page_bgr, debug_collect=debug_collect)
    if debug_dir is not None:
        write_mark_debug_overlay(
            page_bgr,
            page,
            debug_collect,
            Path(debug_dir) / f"page_{page_number}_marks.png",
        )
    page.image_b64 = downscale_png_b64(
        raster_bytes, regions, form_groups=page.form_groups
    )
    page.width = fitz_page.rect.width * scale
    page.height = fitz_page.rect.height * scale
    return page
