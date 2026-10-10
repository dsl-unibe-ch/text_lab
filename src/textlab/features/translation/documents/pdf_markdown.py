"""PDF to Markdown through the OCR feature, then Markdown translation.

Native pages are read from the PDF's text layer; only scanned pages (or
equation pages, when LaTeX is requested) go through OCR. This is the one
place where translation depends on the OCR feature (``pdf_extract``).
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Mapping

from .base import Glossary, ProgressCb, TranslateFn
from .base import report_progress as _report
from .markdown import translate_markdown


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
    math_ocr: bool = False,
) -> tuple[str, dict[str, bytes]]:
    """Extract a coverage-checked document and its assets.

    Blank and short native pages do not require OCR. Only the OCR subset
    evicts translation weights; ordinary PDF batches keep their model warm.
    """
    from textlab.features.ocr import doc_ir

    from .pdf_extract import extract_document

    document = extract_document(
        pdf_bytes,
        pdf_type=pdf_type,
        source_name=source_name,
        progress=_ocr_progress_bridge(progress_cb, "OCR"),
        free_translation_vram_first=free_translation_vram_first,
        math_ocr=math_ocr,
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
    math_ocr: bool = False,
) -> tuple[str, dict[str, bytes]]:
    """OCR the PDF into markdown, translate that markdown.

    Returns ``(translated_markdown, assets)``. Callers that want a single
    downloadable artifact can wrap the result with
    :func:`pack_markdown_bundle`.

    The OCR extraction frees any resident translation model from the GPU
    first, so the PaddleOCR-VL worker and the translation model run
    sequentially rather than competing for VRAM. The translation model is
    reloaded automatically for the :func:`translate_markdown` step below.
    """
    from ..gpu_memory import translation_session

    with translation_session():
        _report(progress_cb, 0, 1, "reading pdf")
        md_source, assets = pdf_to_markdown_bundle(
            pdf_bytes,
            pdf_type=pdf_type,
            source_name=source_name,
            progress_cb=progress_cb,
            free_translation_vram_first=True,
            math_ocr=math_ocr,
        )
        md_translated = translate_markdown(
            md_source,
            translate_fn,
            progress_cb=progress_cb,
            glossary=glossary,
            glossary_case_sensitive=glossary_case_sensitive,
        )
        return md_translated, assets


def pack_markdown_bundle(
    md_text: str,
    assets: Mapping[str, bytes] | None,
    *,
    stem: str,
) -> tuple[bytes, str]:
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
