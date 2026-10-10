"""Translation of Word, Excel and PowerPoint files in their own format.

Each function returns a file of the same type with the text translated in
place, keeping styles, tables, images and layout. A paragraph is translated
as one unit and written into its first run, so styling inside a paragraph
(a bold word mid-sentence) takes the style of the first run.
"""

from __future__ import annotations

import io
import zipfile
from xml.etree import ElementTree as ET

from ..shield import shielded_translate_many
from .base import Glossary, ProgressCb, TranslateFn
from .base import report_progress as _report

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
    """Yield every paragraph (``<w:p>``) below ``root``, in document order.

    Table cells are covered, since they contain paragraphs too.
    """
    yield from root.iter(_q("p"))


def _paragraph_text(p: ET.Element) -> str:
    """Concatenate all ``<w:t>`` children of a paragraph, in order."""
    parts: list[str] = []
    for t in p.iter(_q("t")):
        if t.text:
            parts.append(t.text)
    return "".join(parts)


def _rewrite_paragraph(p: ET.Element, translated: str) -> None:
    """Put ``translated`` back into the paragraph.

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
    counter: list[int],
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
        texts,
        translate_fn,
        glossary=glossary,
        glossary_case_sensitive=glossary_case_sensitive,
    )
    for p, text, translated in zip(
        paragraphs, texts, translated_list, strict=False
    ):
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
    """Translate a .docx file, returning a new .docx file.

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
    """Translate a .xlsx file cell-by-cell, returning a new .xlsx file.

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
                if (
                    comment is not None
                    and comment.text
                    and comment.text.strip()
                ):
                    translatable.append((cell, "comment"))

    total = max(1, len(translatable))
    texts = [
        cell.value if kind == "cell" else cell.comment.text
        for cell, kind in translatable
    ]
    _report(progress_cb, 0, total, "translating xlsx")
    translated_list = shielded_translate_many(
        texts,
        translate_fn,
        glossary=glossary,
        glossary_case_sensitive=glossary_case_sensitive,
    )
    _report(progress_cb, total, total, "translating xlsx")

    for (cell, kind), translated in zip(
        translatable, translated_list, strict=False
    ):
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
    """Replace the paragraph's visible text with ``translated``.

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
        shape, "shapes"
    ):
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
    """Translate a .pptx file, returning a new .pptx file.

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
        texts,
        translate_fn,
        glossary=glossary,
        glossary_case_sensitive=glossary_case_sensitive,
    )
    _report(progress_cb, total, total, "translating pptx")

    for para, text, translated in zip(
        paragraphs, texts, translated_list, strict=False
    ):
        if not text.strip():
            continue
        _pptx_rewrite_paragraph(para, translated)

    out = io.BytesIO()
    prs.save(out)
    return out.getvalue()
