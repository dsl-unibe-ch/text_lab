"""PDF page coverage and extraction checks, independent of model inference."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .pdf_blocks import is_math_block, rect_intersect_area


class PDFIntegrityError(ValueError):
    """A PDF deliverable failed coverage, extraction, or layout validation."""

    def __init__(self, message: str, pages=()):
        """Store the affected pages and add them to the message.

        Args:
            message: What failed.
            pages: The 1-based numbers of the affected pages.
        """
        self.pages = tuple(sorted(set(pages)))
        suffix = ""
        if self.pages:
            suffix = " Pages: " + ", ".join(map(str, self.pages)) + "."
        super().__init__(message + suffix + " This output was blocked.")


@dataclass(frozen=True)
class PDFPagePlan:
    """How one PDF page is read.

    Attributes:
        number: The 1-based page number.
        route: ``"native"`` (text layer), ``"ocr"`` or ``"blank"``.
        native_safe: Whether the text layer covers the page, so the page
            can be rebuilt as a PDF.
        reason: Why the page needs OCR, for the report.
        has_images: Whether the page contains images.
    """

    number: int
    route: str
    native_safe: bool
    reason: str = ""
    has_images: bool = False


def _corrupt_text(text: str) -> bool:
    private_count = 0
    for char in text:
        code = ord(char)
        if (code < 32 and char not in "\t\n\r") or code == 0xFFFD:
            return True
        if (
            0xE000 <= code <= 0xF8FF
            or 0xF0000 <= code <= 0xFFFFD
            or 0x100000 <= code <= 0x10FFFD
        ):
            private_count += 1
    return private_count >= 4


def inspect_pdf(
    pdf_bytes: bytes,
    *,
    math_ocr: bool = False,
) -> list[PDFPagePlan]:
    """Classify every page; a document-wide text count cannot hide scans.

    With ``math_ocr``, pages with native equations are also sent to OCR so
    the Markdown gets LaTeX formulas. That costs a model swap and an OCR
    pass, so by default they stay native and equations are kept verbatim.

    Raster-dominated pages need OCR even when a header or an old hidden text
    layer is present. Truly empty pages can pass through without loading OCR.
    Small figures remain assets, with their untranslated labels disclosed by
    the output report; this cannot prove semantic OCR completeness.
    """
    import fitz

    plans = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        if document.needs_pass:
            raise PDFIntegrityError("The PDF requires a password.")
        if not len(document):
            raise PDFIntegrityError("The PDF contains no pages.")
        for number, page in enumerate(document, 1):
            text = page.get_text("text") or ""
            blocks = page.get_text("dict").get("blocks", [])
            image_boxes = [
                block["bbox"]
                for block in blocks
                if block.get("type") == 1 and block.get("bbox")
            ]
            area = max(1, page.rect.get_area())
            raster_area = sum(
                rect_intersect_area(tuple(page.rect), box)
                for box in image_boxes
            )
            drawings = page.get_drawings() if not text.strip() else []
            has_math = any(
                is_math_block(block)
                for block in blocks
                if block.get("type", 0) == 0
            )
            if not text.strip() and not image_boxes and not drawings:
                plans.append(PDFPagePlan(number, "blank", True))
                continue
            reason = ""
            if _corrupt_text(text):
                reason = "the text layer contains invalid character mappings"
            elif raster_area / area >= 0.60:
                reason = "a page-sized raster requires OCR"
            elif not text.strip():
                reason = "visible page content has no extractable text"
            if reason:
                plans.append(
                    PDFPagePlan(
                        number,
                        "ocr",
                        False,
                        reason,
                        bool(image_boxes),
                    )
                )
            elif has_math and math_ocr:
                plans.append(
                    PDFPagePlan(
                        number,
                        "ocr",
                        True,
                        "equations need structured extraction for Markdown",
                        bool(image_boxes),
                    )
                )
            else:
                plans.append(
                    PDFPagePlan(
                        number,
                        "native",
                        True,
                        has_images=bool(image_boxes),
                    )
                )
    return plans


def require_native_coverage(plans: list[PDFPagePlan]) -> None:
    """Refuse a PDF rebuild when a page has no usable text layer.

    Raises:
        PDFIntegrityError: Naming the pages that would stay untranslated.
    """
    incomplete = [plan.number for plan in plans if not plan.native_safe]
    if incomplete:
        raise PDFIntegrityError(
            "A reconstructed PDF would leave scanned or unreadable content "
            "untranslated. Use the OCR-derived Markdown output instead.",
            incomplete,
        )


def validate_document_pages(document, plans: list[PDFPagePlan]) -> None:
    """Check page identities and nonempty recognition before any export."""
    numbers = [page.page_number for page in document.pages]
    expected = [plan.number for plan in plans]
    if Counter(numbers) != Counter(expected):
        missing = set(expected) - set(numbers)
        duplicate = {
            number for number, count in Counter(numbers).items() if count > 1
        }
        raise PDFIntegrityError(
            "Document extraction returned missing, duplicate, or unexpected "
            "pages.",
            missing | duplicate | (set(numbers) - set(expected)),
        )
    by_number = {page.page_number: page for page in document.pages}
    empty = []
    for plan in plans:
        if plan.route == "blank":
            continue
        page = by_number[plan.number]
        text = " ".join(region.text for region in page.regions).strip()
        if not text or _corrupt_text(text):
            empty.append(plan.number)
    if empty:
        raise PDFIntegrityError(
            "Document extraction produced empty or corrupt text on a "
            "nonblank page; completeness cannot be verified.",
            empty,
        )
