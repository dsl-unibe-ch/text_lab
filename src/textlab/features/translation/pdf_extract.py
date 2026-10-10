"""Per-page PDF extraction with explicit coverage validation."""

from __future__ import annotations

import base64
from dataclasses import replace
from pathlib import Path
import tempfile

from .pdf_checks import (
    PDFIntegrityError,
    PDFPagePlan,
    inspect_pdf,
    validate_document_pages,
)


def _native_page(page, number):
    from textlab.features.ocr import doc_ir

    from .format import _is_math_block

    regions = []
    for block in page.get_text("dict").get("blocks", []):
        content = {}
        asset = None
        if block.get("type", 0) == 0:
            text = "\n".join(
                "".join(span.get("text", "")
                        for span in line.get("spans", []))
                for line in block.get("lines", [])
            ).strip()
            if not text:
                continue
            kind = doc_ir.TEXT
            if _is_math_block(block):
                # Verbatim, never translated: Markdown passes fences through.
                text = "~~~text\n" + text.replace("~~~", "~ ~ ~") + "\n~~~"
            content = {"text": text}
        elif block.get("type") == 1 and block.get("image"):
            kind = doc_ir.FIGURE
            asset = {
                "b64": base64.b64encode(block["image"]).decode("ascii"),
                "ext": block.get("ext", "png"),
            }
        else:
            continue
        regions.append(doc_ir.Region(
            id=f"p{number}_r{len(regions)}", type=kind,
            bbox=list(block.get("bbox", (0, 0, 0, 0))),
            reading_order=len(regions), content=content,
            asset=asset, source="native",
        ))
    return doc_ir.Page(page_number=number, regions=regions, source="native")


def extract_document(
    pdf_bytes: bytes,
    *,
    pdf_type: str = "auto",
    source_name: str = "input.pdf",
    progress=None,
    free_translation_vram_first: bool = False,
    math_ocr: bool = False,
):
    """Extract native/blank pages directly and OCR only the required subset.

    OCR page identities are validated before remapping to the source PDF;
    omitted OCR pages cannot silently disappear from the Markdown export.
    """
    import fitz
    from textlab.features.ocr import doc_ir

    if pdf_type not in {"auto", "force_ocr"}:
        raise ValueError("pdf_type must be auto or force_ocr.")
    plans = inspect_pdf(pdf_bytes, math_ocr=math_ocr)
    if pdf_type == "force_ocr":
        plans = [replace(plan, route="ocr") if plan.route != "blank"
                 else plan for plan in plans]
    pages = {}
    ocr_plans = [plan for plan in plans if plan.route == "ocr"]
    with fitz.open(stream=pdf_bytes, filetype="pdf") as source:
        for plan in plans:
            if plan.route == "blank":
                pages[plan.number] = doc_ir.Page(
                    page_number=plan.number, source="blank",
                )
            elif plan.route == "native":
                pages[plan.number] = _native_page(
                    source[plan.number - 1], plan.number,
                )
        if ocr_plans:
            from textlab.features.ocr import auto_ocr
            from .engine import free_translation_vram, translation_session
            from .gpu_profile import sequential_ocr_allowed

            with translation_session():
                if free_translation_vram_first:
                    free_translation_vram()
                if not sequential_ocr_allowed(
                    min_free_mb=10_240, device="cuda:0",
                ):
                    raise PDFIntegrityError(
                        "OCR needs an eligible allocated GPU with at least "
                        "10 GiB free after translation-model release. "
                        "Free GPU memory or relaunch on a larger GPU.",
                        [plan.number for plan in ocr_plans],
                    )
                with tempfile.TemporaryDirectory(
                    prefix="tl_translate_ocr_",
                ) as temporary:
                    root = Path(temporary)
                    input_path = root / "ocr_pages.pdf"
                    with fitz.open() as subset:
                        for plan in ocr_plans:
                            index = plan.number - 1
                            subset.insert_pdf(
                                source, from_page=index, to_page=index,
                            )
                        subset.save(input_path)
                    workspace = root / "workspace"
                    workspace.mkdir()
                    recognized = auto_ocr.process_document(
                        input_path, workspace, native_fast_lane=False,
                        progress=progress, source_name=source_name,
                    )
                    subset_plans = [
                        PDFPagePlan(index, "ocr", False)
                        for index in range(1, len(ocr_plans) + 1)
                    ]
                    try:
                        validate_document_pages(recognized, subset_plans)
                    except PDFIntegrityError as error:
                        affected = [
                            ocr_plans[number - 1].number
                            for number in error.pages
                            if 1 <= number <= len(ocr_plans)
                        ] or [plan.number for plan in ocr_plans]
                        raise PDFIntegrityError(
                            "OCR returned missing, duplicate, unexpected, "
                            "or empty pages in the extraction subset.",
                            affected,
                        ) from error
                    recognized.pages.sort(key=lambda page: page.page_number)
                    for plan, page in zip(ocr_plans, recognized.pages):
                        page.page_number = plan.number
                        for index, region in enumerate(page.regions):
                            region.id = f"p{plan.number}_r{index}"
                        pages[plan.number] = page
    document = doc_ir.Document(
        pages=[pages[number] for number in sorted(pages)],
        source_name=source_name,
    )
    validate_document_pages(document, plans)
    return document
