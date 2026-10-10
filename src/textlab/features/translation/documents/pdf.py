"""Reconstruction of a native PDF with the translation in place.

The text layer is read with PyMuPDF, blocks are joined into paragraphs
across columns and pages, and each translated paragraph replaces the
original text in the same boxes, shrinking the font when it is longer.
Equations and text drawn over figures keep their source text. Anything that
cannot be placed in full blocks this output (``PDFIntegrityError``); text is
never shortened.
"""

from __future__ import annotations

import functools
import os
import re

from ..shield import shielded_translate_many
from .base import Glossary, ProgressCb, TranslateFn
from .base import report_progress as _report
from .markdown import ENDS_SENTENCE_RE
from .pdf_blocks import (
    extract_image_bboxes,
    is_math_block,
    text_block_overlaps_image,
)
from .pdf_checks import (
    PDFIntegrityError,
    inspect_pdf,
    require_native_coverage,
)

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
def _unicode_font_path() -> str | None:
    """Return the first available Unicode TrueType font, or ``None``."""
    for path in _UNICODE_FONT_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None


def _reflow_lines(lines: list[str]) -> str:
    """Join lines belonging to the same paragraph.

    * Trailing hyphen at end of a line + next line starts lowercase =>
      remove hyphen and join with no space.
    * Otherwise join with a single space, collapsing whitespace.
    """
    out_parts: list[str] = []
    for raw in lines:
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
    sizes: list[float] = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            s = span.get("size")
            if s:
                sizes.append(float(s))
    if not sizes:
        return 10.0
    sizes.sort()
    return sizes[len(sizes) // 2]


# A list item or bullet; a bare number would also match "12 participants".
_STARTS_ITEM_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*[.)]|[IVXLC]+\.|[a-zA-Z][.)]|[•▪◦‣–—*])\s+\S"
)
# Section numbers as headings print them: "2", "2.1", "2.1.", "IV.".
_SECTION_NUMBER_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?|[IVXLC]+\.)\s+\S")


def _looks_like_heading(text: str) -> bool:
    """A short, unpunctuated numbered or all-caps line ("2.1 Methods")."""
    words = text.split()
    if not words or len(words) > 12:
        return False
    letters = [char for char in text if char.isalpha()]
    return bool(_SECTION_NUMBER_RE.match(text)) or (
        len(letters) > 3 and all(char.isupper() for char in letters)
    )


def _semantic_paragraphs(
    all_blocks: list[tuple[int, tuple, str, float]],
) -> list[list[int]]:
    """Group block indices into semantic paragraphs.

    Two adjacent blocks belong to the same paragraph when the first block
    does not end with sentence-terminating punctuation. This handles column
    breaks and page breaks that cut a paragraph in half.
    """
    groups: list[list[int]] = []
    current: list[int] = []
    for idx, (_page, _bbox, text, size) in enumerate(all_blocks):
        if not text.strip():
            if current:
                groups.append(current)
                current = []
            continue
        if not current:
            current = [idx]
            continue
        _prev_page, _prev_bbox, prev_text, prev_size = all_blocks[current[-1]]
        if (
            ENDS_SENTENCE_RE.search(prev_text)
            # Headings, captions and footnotes rarely end with a period;
            # a font-size change or heading shape marks the boundary.
            or abs(size - prev_size) > 0.5
            or _looks_like_heading(prev_text)
            or _STARTS_ITEM_RE.match(text)
        ):
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
    warnings: list[str] | None = None,
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

    require_native_coverage(inspect_pdf(pdf_bytes))
    figure_pages = set()
    equation_pages = set()
    shrunk_pages = set()
    with fitz.open(stream=pdf_bytes, filetype="pdf") as src:
        all_blocks = []
        protected_boxes = {}
        for page_index, page in enumerate(src):
            data = page.get_text("dict")
            image_boxes = extract_image_bboxes(data)
            protected_boxes[page_index] = []
            for block in data.get("blocks", []):
                if block.get("type", 0) != 0:
                    continue
                text = _reflow_lines(
                    [
                        "".join(
                            span.get("text", "")
                            for span in line.get("spans", [])
                        )
                        for line in block.get("lines", [])
                    ]
                )
                if not text.strip():
                    continue
                bbox = tuple(block.get("bbox", (0, 0, 0, 0)))
                if is_math_block(block):
                    protected_boxes[page_index].append(fitz.Rect(bbox))
                    continue
                if text_block_overlaps_image(bbox, image_boxes):
                    # Labels drawn over a figure: redacting them would cut
                    # into the image, so they keep their source text.
                    figure_pages.add(page_index + 1)
                    continue
                all_blocks.append(
                    (
                        page_index,
                        bbox,
                        text,
                        _block_median_fontsize(block),
                    )
                )

        groups = _semantic_paragraphs(all_blocks)
        sources = [
            " ".join(all_blocks[i][2] for i in group) for group in groups
        ]
        _report(progress_cb, 0, len(groups), "translating pdf")
        translated = shielded_translate_many(
            sources,
            translate_fn,
            glossary=glossary,
            glossary_case_sensitive=glossary_case_sensitive,
        )
        if len(translated) != len(groups):
            raise PDFIntegrityError("The translator omitted text blocks.")
        translated_blocks = {}
        for group, text in zip(groups, translated, strict=False):
            if not text.strip():
                raise PDFIntegrityError(
                    "The translator returned empty text for source prose.",
                    [all_blocks[i][0] + 1 for i in group],
                )
            total = sum(len(all_blocks[i][2]) for i in group)
            pieces = _split_by_ratio(
                text,
                [len(all_blocks[i][2]) / max(1, total) for i in group],
            )
            if "".join(pieces) != text:
                raise PDFIntegrityError("Text distribution lost characters.")
            if any(not piece.strip() for piece in pieces):
                # Too few word boundaries to spread across the boxes: put
                # the whole paragraph in its largest box, which shrinks it.
                largest = max(
                    group, key=lambda i: fitz.Rect(all_blocks[i][1]).get_area()
                )
                pieces = [text if i == largest else "" for i in group]
            for index, piece in zip(group, pieces, strict=False):
                translated_blocks[index] = piece

        font_path = _unicode_font_path()
        for page_index, page in enumerate(src):
            indexes = [
                i
                for i, block in enumerate(all_blocks)
                if block[0] == page_index
            ]
            placed = []
            for index in indexes:
                rect = fitz.Rect(all_blocks[index][1]) & page.rect
                if rect.is_empty:
                    continue
                if any(
                    rect.intersects(box) for box in protected_boxes[page_index]
                ):
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
                    page,
                    rect,
                    translated_blocks[index],
                    all_blocks[index][3],
                    font_path,
                ):
                    shrunk_pages.add(page_index + 1)
        _report(progress_cb, len(groups), len(groups), "validated pdf layout")
        if warnings is not None:
            for pages, message in (
                (
                    figure_pages,
                    "Text drawn over figures was left in the source language",
                ),
                (
                    equation_pages,
                    "Text touching equations was left in the source language",
                ),
                (
                    shrunk_pages,
                    "Some translated text was set in a very "
                    "small font to fit its original box",
                ),
            ):
                if pages:
                    warnings.append(
                        f"{message} (pages "
                        + ", ".join(map(str, sorted(pages)))
                        + ")."
                    )
        return src.tobytes(garbage=3, deflate=True)


def _split_by_ratio(text: str, ratios: list[float]) -> list[str]:
    """Split ``text`` into pieces whose lengths follow ``ratios``.

    Pieces end at the word boundary nearest to each target length.
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
        end = (
            min(available, key=lambda item: abs(item - target))
            if available
            else len(text)
        )
        pieces.append(text[cursor:end])
        cursor = end
    pieces.append(text[cursor:])
    return pieces


def _insert_autoshrink(
    page,
    bbox,
    text: str,
    preferred_size: float,
    font_path: str | None = None,
) -> bool:
    """Insert the entire translation; never clip or shorten it.

    Shrinks the font down to 4.5 pt. If the text still does not fit, it is
    laid out as HTML scaled to the box. Returns ``False`` in that case so
    the caller can tell the user that text became very small.
    """
    import html as html_lib

    import fitz

    page_number = page.number + 1
    font = fitz.Font(fontfile=font_path) if font_path else fitz.Font("helv")
    if any(
        not char.isspace() and not font.has_glyph(ord(char), fallback=False)
        for char in text
    ):
        raise PDFIntegrityError(
            "The PDF font does not contain all translated characters. "
            "Use Markdown or install a suitable font.",
            [page_number],
        )
    options = (
        dict(fontname="tl_uni", fontfile=font_path)
        if font_path
        else dict(fontname="helv")
    )
    fontsize = max(6.0, min(preferred_size, 14.0))
    for _ in range(9):
        shape = page.new_shape()
        remaining = shape.insert_textbox(
            bbox,
            text,
            fontsize=fontsize,
            color=(0, 0, 0),
            align=0,
            expandtabs=4,
            **options,
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
            + os.path.basename(font_path)
            + ");} "
            + css[:-1]
            + " font-family: tl_uni;}"
        )
    body = html_lib.escape(text).replace("\n", "<br>")
    spare, _scale = page.insert_htmlbox(
        bbox,
        body,
        css=css,
        archive=archive,
        scale_low=0,
    )
    if spare >= 0:
        return False
    raise PDFIntegrityError(
        "Translated text does not fit its original PDF box, even at the "
        "minimum font size. No text was shortened or replaced by ellipses; "
        "use the Markdown output instead.",
        [page_number],
    )
