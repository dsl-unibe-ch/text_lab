"""
Format-preserving document translation.

For each supported document type this module returns bytes of the **same**
file type with the natural-language text translated in place, leaving all
structural markup (headings, lists, tables, images, links, code, math)
intact wherever possible.

Public API
----------

    translate_markdown(md_text, translate_fn, progress_cb=None) -> str
    translate_docx(docx_bytes, translate_fn, progress_cb=None) -> bytes
    translate_pdf(pdf_bytes, translate_fn, progress_cb=None,
                  target_iso=None) -> bytes

Every ``translate_fn`` is a plain callable ``str -> str`` — usually a
partially-applied :func:`core.translation.engine.translate`. The functions in
this module handle:

* **Markdown** — line-by-line prose translation with structure detection
  (headings, lists, blockquotes) plus :mod:`core.translation.shield` masking to
  protect links, math, code, and inline HTML.
* **DOCX** — raw XML walk (stdlib only) of ``word/document.xml`` (plus any
  headers/footers). Paragraph runs are translated in place while keeping
  paragraph styles, run styles, tables, images, and lists intact.
* **PDF** — pymupdf block extraction, semantic reflow (hyphen joining, line
  merging, cross-block paragraph detection), redaction of the original text
  bounding boxes preserving images, and reinsertion of the translated text
  at the same location with auto-shrinking font size.

The module is Streamlit-free. All progress reporting is via ``progress_cb``
callbacks with signature ``(done: int, total: int, stage: str)``.
"""

from __future__ import annotations

import functools
import io
import os
import re
import zipfile
from typing import Callable, Dict, List, Mapping, Optional, Tuple
from xml.etree import ElementTree as ET

from .shield import shielded_translate, shielded_translate_many

TranslateFn = Callable[[str], str]
ProgressCb = Optional[Callable[[int, int, str], None]]
Glossary = Optional[Mapping[str, str]]

# Scanned-PDF threshold: total non-whitespace characters in the text layer
# below which the PDF is treated as image-only and routed through OCR.
SCANNED_PDF_TEXT_THRESHOLD = 200


def _report(cb: ProgressCb, done: int, total: int, stage: str) -> None:
    if cb is not None:
        try:
            cb(done, total, stage)
        except Exception:
            pass


# ===========================================================================
# Markdown
# ===========================================================================

# Structural line matchers. We only translate the *prose* portion of a line
# and re-attach the structural prefix afterwards.
_MD_HEADING_RE = re.compile(r"^(\s{0,3}#{1,6}\s+)(.*?)(\s+#+\s*)?$")
_MD_LIST_RE = re.compile(r"^(\s*(?:[-*+]|\d+[\.\)])\s+(?:\[[ xX]\]\s+)?)(.+)$")
_MD_BLOCKQUOTE_RE = re.compile(r"^(\s*>+\s*)(.*)$")
_MD_FENCE_RE = re.compile(r"^\s*(?:```|~~~)")
_MD_HR_RE = re.compile(r"^\s*(?:-{3,}|_{3,}|\*{3,})\s*$")
_LITERAL_BLOCKS = (("$$", "$$"), (r"\[", r"\]"), ("<!--", "-->"))


def _literal_block(line: str):
    for opening, closing in _LITERAL_BLOCKS:
        if line.lstrip().startswith(opening):
            tail = line.lstrip()[len(opening):]
            return True, None if closing in tail else closing
    return False, None


def translate_markdown(
    md_text: str,
    translate_fn: TranslateFn,
    progress_cb: ProgressCb = None,
    glossary: Glossary = None,
    *,
    glossary_case_sensitive: bool = False,
) -> str:
    """
    Translate a Markdown document while preserving structure.

    A paragraph the source hard-wrapped over several lines is rejoined first
    (:func:`reflow_soft_wraps`), so the model sees whole sentences; markdown
    treats those breaks as cosmetic anyway. Structure -- headings, lists,
    tables, block quotes, fenced code -- keeps its own lines.

    Fenced code blocks are passed through untouched. Every other line is
    routed through :func:`shielded_translate` so links, math, inline code,
    HTML, and placeholders survive the round-trip. The optional
    ``glossary`` maps source-language terms to forced target-language
    replacements.

    All translatable line bodies are collected first and translated in a
    single batched call (:func:`shielded_translate_many`), which on GPU is
    dramatically faster than one model call per line.
    """
    # Rejoin sentences the source hard-wrapped across lines. Without this each
    # line goes to the model on its own, with no subject and no verb, and the
    # translation is as broken as the fragment it came from.
    lines = reflow_soft_wraps(md_text).splitlines(keepends=False)
    out: List[Optional[str]] = []
    in_fence = False
    literal_end = None
    total = len(lines)

    # Gather translatable bodies; each slot records where/how to reinsert.
    bodies: List[str] = []
    slots: List[Tuple[int, str, str]] = []  # (out_index, prefix, suffix)

    def _defer(prefix: str, body: str, suffix: str) -> None:
        out.append(None)
        slots.append((len(out) - 1, prefix, suffix))
        bodies.append(body)

    for i, line in enumerate(lines, start=1):
        _report(progress_cb, i, total, "parsing markdown")

        if literal_end:
            out.append(line)
            if literal_end in line:
                literal_end = None
            continue
        if not in_fence:
            protected, literal_end = _literal_block(line)
            if protected:
                out.append(line)
                continue

        # Code fences: toggle and passthrough (fence + contents).
        if _MD_FENCE_RE.match(line):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue

        # Blank line / HR / structural-only lines: passthrough.
        if not line.strip() or _MD_HR_RE.match(line):
            out.append(line)
            continue

        # Heading:  ## Title
        m = _MD_HEADING_RE.match(line)
        if m:
            prefix, body, suffix = m.group(1), m.group(2), (m.group(3) or "")
            _defer(prefix, body, suffix)
            continue

        # Blockquote:  > text
        m = _MD_BLOCKQUOTE_RE.match(line)
        if m:
            prefix, body = m.group(1), m.group(2)
            if body.strip():
                _defer(prefix, body, "")
            else:
                out.append(line)
            continue

        # List item:  - text  |  1. text  |  * [x] text
        m = _MD_LIST_RE.match(line)
        if m:
            prefix, body = m.group(1), m.group(2)
            _defer(prefix, body, "")
            continue

        # Regular paragraph line.
        _defer("", line, "")

    _report(progress_cb, total, total, "translating markdown")
    translated = shielded_translate_many(
        bodies, translate_fn, glossary=glossary,
        glossary_case_sensitive=glossary_case_sensitive,
    )
    for (idx, prefix, suffix), tr in zip(slots, translated):
        out[idx] = f"{prefix}{tr}{suffix}"

    return "\n".join(s if s is not None else "" for s in out)


# ===========================================================================
# DOCX (raw XML — no python-docx dependency)
# ===========================================================================

_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_XML_NS = "http://www.w3.org/XML/1998/namespace"

# ElementTree namespace registration so output XML keeps the ``w:`` prefix.
ET.register_namespace("w", _W_NS)


def _q(tag: str) -> str:
    """Return the Clark-notation qualified name for a WordProcessingML tag."""
    return f"{{{_W_NS}}}{tag}"


def _iter_paragraphs(root: ET.Element):
    """Yield every ``<w:p>`` element (paragraphs) inside the given root, in
    document order. Tables are covered because they contain ``<w:p>`` too."""
    yield from root.iter(_q("p"))


def _paragraph_text(p: ET.Element) -> str:
    """Concatenate all ``<w:t>`` children of a paragraph, in order."""
    parts: List[str] = []
    for t in p.iter(_q("t")):
        if t.text:
            parts.append(t.text)
    return "".join(parts)


def _rewrite_paragraph(p: ET.Element, translated: str) -> None:
    """
    Put ``translated`` back into the paragraph.

    Strategy: place the whole translated string in the first ``<w:t>`` and
    empty the rest, preserving each run's parent ``<w:r>`` (and therefore
    its style — bold/italic/color of the first run wins for the paragraph).
    Also sets ``xml:space="preserve"`` so leading/trailing whitespace is
    kept by Word.
    """
    ts = list(p.iter(_q("t")))
    if not ts:
        return
    first, rest = ts[0], ts[1:]
    first.text = translated
    first.set(f"{{{_XML_NS}}}space", "preserve")
    for t in rest:
        t.text = ""


def _translate_docx_part(
    xml_bytes: bytes,
    translate_fn: TranslateFn,
    progress_cb: ProgressCb,
    stage: str,
    counter: List[int],
    total_est: int,
    glossary: Glossary = None,
    *,
    glossary_case_sensitive: bool = False,
) -> bytes:
    """Translate a single DOCX XML part and return new bytes."""
    if not xml_bytes.strip():
        return xml_bytes
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return xml_bytes

    paragraphs = list(_iter_paragraphs(root))
    texts = [_paragraph_text(p) for p in paragraphs]
    counter[0] += len(paragraphs)
    _report(progress_cb, counter[0], total_est, stage)

    translated_list = shielded_translate_many(
        texts, translate_fn, glossary=glossary,
        glossary_case_sensitive=glossary_case_sensitive,
    )
    for p, text, translated in zip(paragraphs, texts, translated_list):
        if not text.strip():
            continue
        _rewrite_paragraph(p, translated)

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


# Parts of a .docx zip that contain user-visible prose.
_DOCX_TRANSLATABLE_PARTS = (
    "word/document.xml",
    "word/footnotes.xml",
    "word/endnotes.xml",
    "word/comments.xml",
)
_DOCX_TRANSLATABLE_PREFIXES = ("word/header", "word/footer")


def _is_translatable_docx_part(name: str) -> bool:
    if name in _DOCX_TRANSLATABLE_PARTS:
        return True
    return name.endswith(".xml") and any(
        name.startswith(pfx) for pfx in _DOCX_TRANSLATABLE_PREFIXES
    )


def translate_docx(
    docx_bytes: bytes,
    translate_fn: TranslateFn,
    progress_cb: ProgressCb = None,
    glossary: Glossary = None,
    *,
    glossary_case_sensitive: bool = False,
) -> bytes:
    """
    Translate a .docx file, returning a new .docx file.

    Preserves: paragraph styles (headings, list styles), tables (each cell
    is a paragraph in the XML), inline images, hyperlinks, footnotes/
    endnotes, comments, headers and footers.

    Best-effort limitations:
    * If a paragraph is split into multiple styled runs (e.g. a bold word in
      the middle of a sentence), the whole paragraph is translated as one
      unit and re-inserted into the first run. Intra-paragraph run styling
      is therefore lost; paragraph-level style is preserved.
    """
    src = zipfile.ZipFile(io.BytesIO(docx_bytes), "r")
    out_buf = io.BytesIO()
    dst = zipfile.ZipFile(out_buf, "w", zipfile.ZIP_DEFLATED)

    # First pass: estimate paragraph count for progress reporting.
    total_paragraphs = 0
    parts_to_translate = []
    for info in src.infolist():
        if _is_translatable_docx_part(info.filename):
            data = src.read(info.filename)
            try:
                root = ET.fromstring(data)
                total_paragraphs += sum(1 for _ in _iter_paragraphs(root))
            except ET.ParseError:
                pass
            parts_to_translate.append(info.filename)

    total_paragraphs = max(1, total_paragraphs)
    counter = [0]

    for info in src.infolist():
        data = src.read(info.filename)
        if info.filename in parts_to_translate:
            data = _translate_docx_part(
                data,
                translate_fn,
                progress_cb,
                "translating docx",
                counter,
                total_paragraphs,
                glossary=glossary,
                glossary_case_sensitive=glossary_case_sensitive,
            )
        dst.writestr(info, data)

    src.close()
    dst.close()
    return out_buf.getvalue()


# ===========================================================================
# PDF (pymupdf)
# ===========================================================================

_ENDS_SENTENCE_RE = re.compile(
    r"[\.\!\?\u3002\uFF01\uFF1F\u203C\u2049\uFF0E]\s*[\"'\)\]]?\s*$")


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
    r"\bCMSY\d*"                       # Computer Modern Symbols
    r"|\bCMMI\d*"                       # Computer Modern Math Italic
    r"|\bCMEX\d*"                       # Computer Modern Math Extension
    r"|\bMSAM\d*|\bMSBM\d*"             # AMS math fonts
    r"|\bMTMain\b|\bMTSym\b|\bMTEx\b|\bMTExtra\b"       # MathTime
    # ONLY the Math variant (not Roman/Mono/Italic)
    r"|LatinModern-?Math\b"
    r"|Cambria\s?Math\b"                # OpenType math
    r"|(?:STIX|STIXTwo)\s?Math\b"       # STIX Math families only
    r"|MathJax"
    r"|Lucida(?:New)?Math\b"
    r"|\bSymbol(?:MT)?\b"               # Adobe Symbol / SymbolMT
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
        0x2200 <= code <= 0x22FF        # Math Operators (strong)
        or 0x2A00 <= code <= 0x2AFF     # Supplemental Math Operators
        or 0x27C0 <= code <= 0x27EF     # Miscellaneous Math Symbols-A
        or 0x2980 <= code <= 0x29FF     # Miscellaneous Math Symbols-B
        or 0x1D400 <= code <= 0x1D7FF   # Math Alphanumeric Symbols
        or 0x2100 <= code <= 0x214F     # Letterlike Symbols (ℕ ℤ ℝ ∑ ...)
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


def _is_math_block(block: dict) -> bool:
    """
    Return True if a block should be treated as an equation / formula and
    left untouched. Requires a strict majority of spans to look like math.
    """
    spans = [
        s for line in block.get(
            "lines",
            []) for s in line.get(
            "spans",
            [])]
    if not spans:
        return False
    math_spans = sum(1 for s in spans if _is_math_span(s))
    return math_spans * 2 > len(spans)  # STRICT > 50%


# --- Image-overlap detection ----------------------------------------------


def _rect_area(rect: Tuple[float, float, float, float]) -> float:
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])


def _rect_intersect_area(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    return max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)


def _extract_image_bboxes(
        page_dict: dict) -> List[Tuple[float, float, float, float]]:
    """Return image-block bboxes on a page (block ``type == 1``)."""
    out: List[Tuple[float, float, float, float]] = []
    for block in page_dict.get("blocks", []):
        if block.get("type") == 1:
            bbox = block.get("bbox")
            if bbox and len(bbox) == 4:
                out.append(tuple(bbox))
    return out


def _text_block_overlaps_image(
    text_bbox: Tuple[float, float, float, float],
    image_bboxes: List[Tuple[float, float, float, float]],
    min_overlap: float = 0.40,
) -> bool:
    """
    Return True if the text block's bbox is significantly enclosed by an
    image bbox (default 40% of the text-block area).

    Redacting such blocks tends to leave white bands over figures and
    corrupt visual context, so we skip translation for them.
    """
    ta = _rect_area(text_bbox)
    if ta <= 0:
        return False
    for ib in image_bboxes:
        if _rect_intersect_area(text_bbox, ib) / ta >= min_overlap:
            return True
    return False


# --- Unicode font selection for text re-insertion -------------------------
#
# Helvetica (pymupdf's "helv") is a Base-14 font limited to WinAnsi. It
# cannot render most Greek/math symbols, CJK, Arabic, Devanagari, etc.
# When available we register a system TrueType font and use it instead.
_UNICODE_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
)


@functools.lru_cache(maxsize=1)
def _unicode_font_path() -> Optional[str]:
    """Return the first available Unicode TrueType font, or ``None``."""
    for path in _UNICODE_FONT_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None


# Structure that markdown marks with a line, on top of the _MD_* patterns
# above: a table row, and the four-space indent of a code block.
_MD_TABLE_ROW_RE = re.compile(r"^\s*\|")
_INDENTED_CODE_RE = re.compile(r"^(?: {4,}|\t)")


def _is_soft_wrap(line: str, following: str) -> bool:
    """Did *line* stop mid-sentence, with *following* continuing it?

    A hard-wrapped paragraph is one sentence the writer's editor happened to
    break in the middle. Translating the halves separately gives the model no
    subject, no verb and no context, and the output shows it -- which is the
    whole reason this exists.

    Conservative on purpose: a line break that follows a completed sentence is
    left alone, so deliberately line-structured prose keeps its shape.
    """
    if not line.strip() or not following.strip():
        return False
    if _ENDS_SENTENCE_RE.search(line.rstrip()):
        return False
    # An explicit markdown hard break ("two trailing spaces") is deliberate.
    # Checked before any rstrip, which would erase the evidence.
    if line.endswith("  "):
        return False
    # Structural lines are breaks in their own right, whatever they end with.
    # Probed with their indentation intact: that is what marks an indented
    # code block, and what tells a list continuation from a new paragraph.
    for probe in (line, following):
        if (
            _MD_HEADING_RE.match(probe)
            or _MD_LIST_RE.match(probe)
            or _MD_BLOCKQUOTE_RE.match(probe)
            or _MD_FENCE_RE.match(probe)
            or _MD_HR_RE.match(probe)
            or _MD_TABLE_ROW_RE.match(probe)
            or _INDENTED_CODE_RE.match(probe)
        ):
            return False
    return True


def reflow_soft_wraps(text: str) -> str:
    """Rejoin sentences that a hard-wrapped source split across lines.

    Blank lines, and any break that follows a finished sentence, are kept, so
    paragraph structure survives. A word hyphenated across the break is put
    back together.

    Fenced code blocks pass through untouched: the lines inside one are not
    prose, and nothing there ends in a full stop.

    Not safe for line-oriented formats -- subtitles, for one, where every line
    break carries meaning -- so callers opt in rather than get this for free.
    """
    if not text or "\n" not in text:
        return text

    lines = text.split("\n")
    out: List[str] = []
    in_fence = False
    literal_end = None
    previous_protected = False
    for i, line in enumerate(lines):
        was_protected = previous_protected
        previous_protected = False
        if literal_end:
            out.append(line)
            previous_protected = True
            if literal_end in line:
                literal_end = None
            continue
        if not in_fence:
            protected, literal_end = _literal_block(line)
            if protected:
                out.append(line)
                previous_protected = True
                continue
        if _MD_FENCE_RE.match(line):
            # The fence markers themselves are structural, so _is_soft_wrap
            # already refuses them; this is about the arbitrary code between.
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        if out and not was_protected and _is_soft_wrap(lines[i - 1], line):
            previous = out.pop().rstrip()
            if previous.endswith("-") and line.lstrip()[:1].islower():
                out.append(previous[:-1] + line.strip())  # de-hyphenate
            else:
                out.append(f"{previous} {line.strip()}")
        else:
            out.append(line)
    return "\n".join(out)


def _reflow_lines(lines: List[str]) -> str:
    """
    Join lines belonging to the same paragraph.

    * Trailing hyphen at end of a line + next line starts lowercase =>
      remove hyphen and join with no space.
    * Otherwise join with a single space, collapsing whitespace.
    """
    out_parts: List[str] = []
    for i, raw in enumerate(lines):
        line = raw.rstrip()
        if not line:
            continue
        if out_parts and out_parts[-1].endswith("-") and line[:1].islower():
            out_parts[-1] = out_parts[-1][:-1] + line
        else:
            if out_parts and not out_parts[-1].endswith(" "):
                out_parts.append(" ")
            out_parts.append(line)
    return "".join(out_parts).strip()


def _block_median_fontsize(block: dict) -> float:
    """Return a reasonable font size to use when inserting translated text."""
    sizes: List[float] = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            s = span.get("size")
            if s:
                sizes.append(float(s))
    if not sizes:
        return 10.0
    sizes.sort()
    return sizes[len(sizes) // 2]


def _semantic_paragraphs(
    all_blocks: List[Tuple[int, tuple, str, float]],
) -> List[List[int]]:
    """
    Group block indices into semantic paragraphs.

    Two adjacent blocks belong to the same paragraph when the first block
    does not end with sentence-terminating punctuation. This handles column
    breaks and page breaks that cut a paragraph in half.
    """
    groups: List[List[int]] = []
    current: List[int] = []
    for idx, (_page, _bbox, text, _size) in enumerate(all_blocks):
        if not text.strip():
            if current:
                groups.append(current)
                current = []
            continue
        if not current:
            current = [idx]
            continue
        prev_text = all_blocks[current[-1]][2]
        if _ENDS_SENTENCE_RE.search(prev_text):
            groups.append(current)
            current = [idx]
        else:
            current.append(idx)
    if current:
        groups.append(current)
    return groups


def translate_pdf(
    pdf_bytes: bytes,
    translate_fn: TranslateFn,
    progress_cb: ProgressCb = None,
    glossary: Glossary = None,
    *,
    glossary_case_sensitive: bool = False,
    warnings: Optional[List[str]] = None,
) -> bytes:
    """Reconstruct a native PDF with every translated block placed in full.

    Scanned/corrupt pages, empty translations and unsupported glyphs block
    this deliverable. Equations, and text drawn over images (figure labels),
    keep their source text. A translation longer than its box is shrunk to
    fit, never shortened. Such layout compromises are appended to
    ``warnings`` rather than blocking the whole document. The independently
    translated Markdown output is not affected by a reconstruction failure.
    """
    import fitz

    from .pdf_checks import (
        PDFIntegrityError, inspect_pdf, require_native_coverage,
    )

    require_native_coverage(inspect_pdf(pdf_bytes))
    figure_pages = set()
    equation_pages = set()
    shrunk_pages = set()
    with fitz.open(stream=pdf_bytes, filetype="pdf") as src:
        all_blocks = []
        protected_boxes = {}
        for page_index, page in enumerate(src):
            data = page.get_text("dict")
            image_boxes = _extract_image_bboxes(data)
            protected_boxes[page_index] = []
            for block in data.get("blocks", []):
                if block.get("type", 0) != 0:
                    continue
                text = _reflow_lines([
                    "".join(span.get("text", "")
                            for span in line.get("spans", []))
                    for line in block.get("lines", [])
                ])
                if not text.strip():
                    continue
                bbox = tuple(block.get("bbox", (0, 0, 0, 0)))
                if _is_math_block(block):
                    protected_boxes[page_index].append(fitz.Rect(bbox))
                    continue
                if _text_block_overlaps_image(bbox, image_boxes):
                    # Labels drawn over a figure: redacting them would cut
                    # into the image, so they keep their source text.
                    figure_pages.add(page_index + 1)
                    continue
                all_blocks.append((
                    page_index, bbox, text, _block_median_fontsize(block),
                ))

        groups = _semantic_paragraphs(all_blocks)
        sources = [" ".join(all_blocks[i][2] for i in group)
                   for group in groups]
        _report(progress_cb, 0, len(groups), "translating pdf")
        translated = shielded_translate_many(
            sources, translate_fn, glossary=glossary,
            glossary_case_sensitive=glossary_case_sensitive,
        )
        if len(translated) != len(groups):
            raise PDFIntegrityError("The translator omitted text blocks.")
        translated_blocks = {}
        for group, text in zip(groups, translated):
            if not text.strip():
                raise PDFIntegrityError(
                    "The translator returned empty text for source prose.",
                    [all_blocks[i][0] + 1 for i in group],
                )
            total = sum(len(all_blocks[i][2]) for i in group)
            pieces = _split_by_ratio(
                text, [len(all_blocks[i][2]) / max(1, total) for i in group],
            )
            if "".join(pieces) != text:
                raise PDFIntegrityError("Text distribution lost characters.")
            if any(not piece.strip() for piece in pieces):
                # Too few word boundaries to spread across the boxes: put
                # the whole paragraph in its largest box, which shrinks it.
                largest = max(group, key=lambda i: fitz.Rect(
                    all_blocks[i][1]).get_area())
                pieces = [text if i == largest else "" for i in group]
            for index, piece in zip(group, pieces):
                translated_blocks[index] = piece

        font_path = _unicode_font_path()
        for page_index, page in enumerate(src):
            indexes = [i for i, block in enumerate(all_blocks)
                       if block[0] == page_index]
            placed = []
            for index in indexes:
                rect = fitz.Rect(all_blocks[index][1]) & page.rect
                if rect.is_empty:
                    continue
                if any(rect.intersects(box)
                       for box in protected_boxes[page_index]):
                    # Redacting here would erase part of an equation.
                    equation_pages.add(page_index + 1)
                    continue
                page.add_redact_annot(rect, fill=(1, 1, 1))
                placed.append((index, rect))
            if not placed:
                continue
            page.apply_redactions(images=0, graphics=0)
            for index, rect in placed:
                if not translated_blocks[index].strip():
                    continue
                if not _insert_autoshrink(
                    page, rect, translated_blocks[index],
                    all_blocks[index][3], font_path,
                ):
                    shrunk_pages.add(page_index + 1)
        _report(progress_cb, len(groups), len(groups), "validated pdf layout")
        if warnings is not None:
            for pages, message in (
                (figure_pages, "Text drawn over figures was left in the "
                               "source language"),
                (equation_pages, "Text touching equations was left in the "
                                 "source language"),
                (shrunk_pages, "Some translated text was set in a very "
                               "small font to fit its original box"),
            ):
                if pages:
                    warnings.append(
                        f"{message} (pages "
                        + ", ".join(map(str, sorted(pages))) + ")."
                    )
        return src.tobytes(garbage=3, deflate=True)


# ===========================================================================
# PDF -> Markdown (uses the OCR pipeline for structure recovery and scans)
# ===========================================================================


def detect_pdf_is_scanned(
    pdf_bytes: bytes,
    threshold: int = SCANNED_PDF_TEXT_THRESHOLD,
) -> bool:
    """Legacy whole-document text heuristic; not a completeness check.

    Kept for external callers. Translation routing uses ``inspect_pdf`` per
    page instead, so mixed documents and short covers are handled correctly.
    """
    import fitz

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        total = 0
        for page in doc:
            total += len(page.get_text("text").strip())
            if total >= threshold:
                return False
        return True
    finally:
        doc.close()


def pdf_needs_ocr(pdf_bytes: bytes) -> bool:
    """Use the same per-page plan as the actual Markdown extraction path."""
    from .pdf_checks import inspect_pdf

    return any(plan.route == "ocr" for plan in inspect_pdf(pdf_bytes))


def _ocr_progress_bridge(progress_cb: ProgressCb, stage_hint: str):
    """Adapt OCR's fraction/text callback to a done/total/stage callback."""
    if progress_cb is None:
        return None

    def _cb(frac: float, text: str) -> None:
        pct = int(round(max(0.0, min(1.0, frac)) * 1000))
        # Prepend the outer stage hint so the UI shows "OCR: parsing page
        # 3/12".
        try:
            progress_cb(pct, 1000, f"{stage_hint}: {text}")
        except Exception:
            pass

    return _cb


def pdf_to_markdown_bundle(
    pdf_bytes: bytes,
    *,
    pdf_type: str = "auto",
    source_name: str = "input.pdf",
    progress_cb: ProgressCb = None,
    free_translation_vram_first: bool = False,
) -> Tuple[str, Dict[str, bytes]]:
    """Extract a coverage-checked document and its assets.

    Blank and short native pages do not require OCR. Only the OCR subset
    evicts translation weights; ordinary PDF batches keep their model warm.
    """
    from core import doc_ir
    from .pdf_extract import extract_document

    document = extract_document(
        pdf_bytes, pdf_type=pdf_type, source_name=source_name,
        progress=_ocr_progress_bridge(progress_cb, "OCR"),
        free_translation_vram_first=free_translation_vram_first,
    )
    return (
        doc_ir.to_markdown(document, asset_dir="assets", embed_assets=True),
        dict(doc_ir.collect_assets(document)),
    )


def translate_pdf_to_markdown(
    pdf_bytes: bytes,
    translate_fn: TranslateFn,
    *,
    progress_cb: ProgressCb = None,
    glossary: Glossary = None,
    pdf_type: str = "auto",
    source_name: str = "input.pdf",
    glossary_case_sensitive: bool = False,
) -> Tuple[str, Dict[str, bytes]]:
    """OCR the PDF into markdown, translate that markdown.

    Returns ``(translated_markdown, assets)``. Callers that want a single
    downloadable artifact can wrap the result with
    :func:`pack_markdown_bundle`.

    The OCR extraction frees any resident translation model from the GPU
    first, so the PaddleOCR-VL worker and the translation model run
    sequentially rather than competing for VRAM. The translation model is
    reloaded automatically for the :func:`translate_markdown` step below.
    """
    from .engine import translation_session

    with translation_session():
        _report(progress_cb, 0, 1, "reading pdf")
        md_source, assets = pdf_to_markdown_bundle(
            pdf_bytes, pdf_type=pdf_type, source_name=source_name,
            progress_cb=progress_cb, free_translation_vram_first=True,
        )
        md_translated = translate_markdown(
            md_source, translate_fn, progress_cb=progress_cb,
            glossary=glossary,
            glossary_case_sensitive=glossary_case_sensitive,
        )
        return md_translated, assets


def pack_markdown_bundle(
    md_text: str,
    assets: Optional[Mapping[str, bytes]],
    *,
    stem: str,
) -> Tuple[bytes, str]:
    """Return bytes and filename: a ZIP with assets, otherwise Markdown.

    ZIP layout:  ``<stem>.md``  +  ``assets/<name>.png`` per crop.
    """
    if not assets:
        return md_text.encode("utf-8"), f"{stem}.md"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{stem}.md", md_text)
        for name, data in assets.items():
            zf.writestr(f"assets/{name}", data)
    return buf.getvalue(), f"{stem}.md.zip"


def _split_by_ratio(text: str, ratios: List[float]) -> List[str]:
    """
    Split ``text`` into ``len(ratios)`` pieces whose lengths approximate the
    given ratios, snapping to nearest word boundaries.
    """
    if len(ratios) <= 1:
        return [text]
    total = sum(ratios) or 1.0
    boundaries = [match.end() for match in re.finditer(r"\s+", text)]
    pieces = []
    cursor = 0
    cumulative = 0.0
    for ratio in ratios[:-1]:
        cumulative += ratio
        target = round(len(text) * cumulative / total)
        available = [end for end in boundaries if cursor < end < len(text)]
        # Without word boundaries, keep the remaining text intact. The PDF
        # validator blocks any resulting empty boxes rather than cutting words.
        end = min(available, key=lambda item: abs(item - target)) \
            if available else len(text)
        pieces.append(text[cursor:end])
        cursor = end
    pieces.append(text[cursor:])
    return pieces


def _insert_autoshrink(
    page,
    bbox,
    text: str,
    preferred_size: float,
    font_path: Optional[str] = None,
) -> bool:
    """Insert the entire translation; never clip or shorten it.

    Shrinks the font down to 4.5 pt. If the text still does not fit, it is
    laid out as HTML scaled to the box. Returns ``False`` in that case so
    the caller can tell the user that text became very small.
    """
    import html as html_lib

    import fitz

    from .pdf_checks import PDFIntegrityError

    page_number = page.number + 1
    font = fitz.Font(fontfile=font_path) if font_path else fitz.Font("helv")
    if any(not char.isspace() and not font.has_glyph(ord(char), fallback=False)
           for char in text):
        raise PDFIntegrityError(
            "The PDF font does not contain all translated characters. "
            "Use Markdown or install a suitable font.", [page_number],
        )
    options = (dict(fontname="tl_uni", fontfile=font_path)
               if font_path else dict(fontname="helv"))
    fontsize = max(6.0, min(preferred_size, 14.0))
    for _ in range(9):
        shape = page.new_shape()
        remaining = shape.insert_textbox(
            bbox, text, fontsize=fontsize, color=(0, 0, 0),
            align=0, expandtabs=4, **options,
        )
        if remaining >= 0:
            shape.commit()
            return True
        if fontsize <= 4.5:
            break
        fontsize = max(4.5, fontsize * 0.85)

    css = f"* {{font-size: {fontsize}pt; margin: 0; padding: 0;}}"
    archive = None
    if font_path:
        archive = fitz.Archive(os.path.dirname(font_path))
        css = (
            "@font-face {font-family: tl_uni; src: url("
            + os.path.basename(font_path) + ");} "
            + css[:-1] + " font-family: tl_uni;}"
        )
    body = html_lib.escape(text).replace("\n", "<br>")
    spare, _scale = page.insert_htmlbox(
        bbox, body, css=css, archive=archive, scale_low=0,
    )
    if spare >= 0:
        return False
    raise PDFIntegrityError(
        "Translated text does not fit its original PDF box, even at the "
        "minimum font size. No text was shortened or replaced by ellipses; "
        "use the Markdown output instead.", [page_number],
    )


# ===========================================================================
# XLSX (openpyxl)
# ===========================================================================


def translate_xlsx(
    xlsx_bytes: bytes,
    translate_fn: TranslateFn,
    progress_cb: ProgressCb = None,
    glossary: Glossary = None,
    *,
    glossary_case_sensitive: bool = False,
) -> bytes:
    """
    Translate a .xlsx file cell-by-cell, returning a new .xlsx file.

    Preserves:
    * Workbook / sheet structure, column widths, merged cells, styles.
    * Numeric, date, boolean, and formula cells (they are left untouched;
      formulas are never sent to the model).
    * Comments' contents are translated too.

    Limitations:
    * Charts and images embedded via drawings are preserved as-is (their
      textual labels are not translated).
    * Sheet *names* are not translated — Excel range references rely on
      them and translation risks breaking cross-sheet formulas.
    """
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise RuntimeError(
            "openpyxl is required to translate .xlsx files."
        ) from exc

    wb = load_workbook(io.BytesIO(xlsx_bytes))

    # First pass — count translatable strings for a smooth progress bar.
    translatable: list = []  # (cell, kind='cell' | 'comment')
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                val = cell.value
                if not isinstance(val, str):
                    continue
                # Skip formula strings (openpyxl gives data_type == 'f' or
                # the string starts with '=') and empty/whitespace cells.
                if getattr(cell, "data_type", None) == "f":
                    continue
                if val.startswith("="):
                    continue
                if not val.strip():
                    continue
                translatable.append((cell, "cell"))

                comment = getattr(cell, "comment", None)
                if (comment is not None and comment.text
                        and comment.text.strip()):
                    translatable.append((cell, "comment"))

    total = max(1, len(translatable))
    texts = [
        cell.value if kind == "cell" else cell.comment.text
        for cell, kind in translatable
    ]
    _report(progress_cb, 0, total, "translating xlsx")
    translated_list = shielded_translate_many(
        texts, translate_fn, glossary=glossary,
        glossary_case_sensitive=glossary_case_sensitive,
    )
    _report(progress_cb, total, total, "translating xlsx")

    for (cell, kind), translated in zip(translatable, translated_list):
        if kind == "cell":
            cell.value = translated
        else:
            comment = cell.comment
            comment.text = translated
            cell.comment = comment

    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()


# ===========================================================================
# PPTX (python-pptx)
# ===========================================================================


def _pptx_paragraph_text(paragraph) -> str:
    """Join all run texts inside a python-pptx paragraph."""
    return "".join(r.text or "" for r in paragraph.runs)


def _pptx_rewrite_paragraph(paragraph, translated: str) -> None:
    """
    Replace the paragraph's visible text with ``translated``.

    Puts the whole translated string into the first run and empties the
    remaining runs, preserving the first run's font/color and the
    paragraph-level style. Intra-paragraph run styling is lost (same
    trade-off as the DOCX pipeline).
    """
    runs = list(paragraph.runs)
    if not runs:
        return
    runs[0].text = translated
    for r in runs[1:]:
        r.text = ""


def _iter_pptx_text_frames(prs):
    """Yield every ``text_frame`` from slides, notes, and table cells."""
    for slide in prs.slides:
        for shape in slide.shapes:
            yield from _iter_shape_text_frames(shape)
        if slide.has_notes_slide:
            notes = slide.notes_slide
            for shape in notes.shapes:
                yield from _iter_shape_text_frames(shape)


def _iter_shape_text_frames(shape):
    """Recurse into groups/tables/text-frames and yield each text frame."""
    # Grouped shapes.
    if getattr(shape, "shape_type", None) is not None and hasattr(
            shape, "shapes"):
        try:
            for sub in shape.shapes:
                yield from _iter_shape_text_frames(sub)
            return
        except AttributeError:
            pass
    # Tables: iterate cells.
    if getattr(shape, "has_table", False):
        for row in shape.table.rows:
            for cell in row.cells:
                if cell.text_frame is not None:
                    yield cell.text_frame
        return
    # Regular text-bearing shapes.
    if getattr(shape, "has_text_frame", False):
        yield shape.text_frame


def translate_pptx(
    pptx_bytes: bytes,
    translate_fn: TranslateFn,
    progress_cb: ProgressCb = None,
    glossary: Glossary = None,
    *,
    glossary_case_sensitive: bool = False,
) -> bytes:
    """
    Translate a .pptx file, returning a new .pptx file.

    Preserves slide layout, shape positions, images, colors, tables,
    speaker notes, and paragraph-level style. Same intra-paragraph run
    styling caveat as :func:`translate_docx`.

    Raises ``RuntimeError`` if ``python-pptx`` is not available in the
    running Python environment.
    """
    try:
        from pptx import Presentation
    except ImportError as exc:
        raise RuntimeError(
            "python-pptx is required to translate .pptx files."
        ) from exc

    prs = Presentation(io.BytesIO(pptx_bytes))

    # Collect (text_frame, paragraph) pairs so we can report a total.
    paragraphs: list = []
    for tf in _iter_pptx_text_frames(prs):
        for para in tf.paragraphs:
            paragraphs.append(para)

    total = max(1, len(paragraphs))
    texts = [_pptx_paragraph_text(para) for para in paragraphs]
    _report(progress_cb, 0, total, "translating pptx")
    translated_list = shielded_translate_many(
        texts, translate_fn, glossary=glossary,
        glossary_case_sensitive=glossary_case_sensitive,
    )
    _report(progress_cb, total, total, "translating pptx")

    for para, text, translated in zip(paragraphs, texts, translated_list):
        if not text.strip():
            continue
        _pptx_rewrite_paragraph(para, translated)

    out = io.BytesIO()
    prs.save(out)
    return out.getvalue()
