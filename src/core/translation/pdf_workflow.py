"""Independent PDF deliverables: failures never discard a valid sibling."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json

from .format import (
    pack_markdown_bundle,
    translate_pdf,
    translate_pdf_to_markdown,
)
from .messages import describe_error
from .pdf_checks import PDFIntegrityError, inspect_pdf, require_native_coverage
from .shield import record_translations


@dataclass
class PDFTranslationResult:
    outputs: list[tuple[str, bytes]] = field(default_factory=list)
    blocked: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    pages: list[dict] = field(default_factory=list)
    # (source, translation) units for the side-by-side review file.
    pairs: list[tuple[str, str]] = field(default_factory=list)

    def report_bytes(self) -> bytes:
        report = {
            "status": "partial" if self.blocked else "passed_checks",
            "available_outputs": [name for name, _ in self.outputs],
            "blocked_outputs": self.blocked,
            "warnings": self.warnings,
            "pages": self.pages,
            "validation_scope": (
                "Page coverage, nonempty extraction, protected markers, "
                "and native PDF layout checks. These are not a guarantee "
                "of OCR accuracy or semantic translation completeness."
            ),
        }
        if not self.outputs:
            report["status"] = "blocked"
        return json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")


def translate_pdf_outputs(
    pdf_bytes: bytes,
    translate_fn,
    *,
    stem: str,
    source_name: str,
    glossary=None,
    glossary_case_sensitive: bool = False,
    progress_cb=None,
    ocr_allowed: bool | None = None,
    outputs=("markdown", "pdf"),
    math_ocr: bool = False,
    backend: str | None = None,
) -> PDFTranslationResult:
    """Attempt the requested outputs separately, returning only checked bytes.

    ``outputs`` selects ``"markdown"`` and/or ``"pdf"``; skipping one skips
    its extraction (and, for Markdown, any OCR). When both are built,
    ``translate_fn``'s sentence cache makes the second mostly free.
    ``math_ocr`` sends equation pages to OCR for LaTeX in the Markdown.
    """
    result = PDFTranslationResult()
    plans = inspect_pdf(pdf_bytes, math_ocr=math_ocr)
    result.pages = [asdict(plan) for plan in plans]
    if any(plan.has_images for plan in plans):
        result.warnings.append(
            "Figures are preserved as assets; labels inside embedded "
            "figures are not translated in place. Review OCR and figure "
            "content against the source."
        )
    result.warnings.append(
        "Equation blocks identified by the PDF parser are preserved rather "
        "than translated. Layout and OCR checks cannot verify meaning."
    )
    needs_ocr = [plan.number for plan in plans if plan.route == "ocr"]
    if "markdown" not in outputs:
        needs_ocr = []
    if ocr_allowed is None and needs_ocr:
        from .gpu_profile import sequential_ocr_allowed

        ocr_allowed = sequential_ocr_allowed(device="cuda:0")

    def blocked(output: str, error: Exception) -> None:
        result.blocked.append({
            "output": output,
            "message": describe_error(error, backend),
            "reason": str(error),
            "pages": list(getattr(error, "pages", ())),
        })

    if "markdown" in outputs:
        try:
            if needs_ocr and not ocr_allowed:
                raise PDFIntegrityError(
                    "Markdown requires OCR, but the allocated GPU is not "
                    "eligible for sequential OCR/translation. Relaunch on a "
                    "supported GPU; no pages will be silently skipped.",
                    needs_ocr,
                )
            with record_translations() as markdown_pairs:
                markdown, assets = translate_pdf_to_markdown(
                    pdf_bytes, translate_fn, progress_cb=progress_cb,
                    glossary=glossary,
                    glossary_case_sensitive=glossary_case_sensitive,
                    source_name=source_name, math_ocr=math_ocr,
                )
            result.pairs = markdown_pairs
            data, name = pack_markdown_bundle(markdown, assets, stem=stem)
            result.outputs.append((name, data))
        except Exception as error:
            blocked("Markdown", error)

    if "pdf" in outputs:
        try:
            require_native_coverage(plans)
            layout_warnings: list[str] = []
            with record_translations() as pdf_pairs:
                data = translate_pdf(
                    pdf_bytes, translate_fn, progress_cb=progress_cb,
                    glossary=glossary,
                    glossary_case_sensitive=glossary_case_sensitive,
                    warnings=layout_warnings,
                )
            # Whole reflowed paragraphs read better than Markdown lines.
            result.pairs = pdf_pairs
            result.warnings.extend(layout_warnings)
            result.outputs.append((f"{stem}.pdf", data))
        except Exception as error:
            blocked("Reconstructed PDF", error)
    return result
