"""Independent PDF deliverables: failures never discard a valid sibling."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json

from .format import (
    pack_markdown_bundle,
    translate_pdf,
    translate_pdf_to_markdown,
)
from .chunking import InputTooLongError, OutputTruncatedError
from .pdf_checks import PDFIntegrityError, inspect_pdf, require_native_coverage
from .shield import ProtectedContentError


def describe_error(error: Exception) -> str:
    """A short, user-facing explanation; technical detail stays in reports."""
    if isinstance(error, InputTooLongError):
        return ("Part of the text is too long for the selected model to "
                "translate in one piece. Try a different model.")
    if isinstance(error, OutputTruncatedError):
        return ("The model could not finish translating part of the text. "
                "Try a different model.")
    if isinstance(error, ProtectedContentError):
        return ("The model did not keep the document's formatting (links, "
                "code, tables) intact. Try a different model.")
    if isinstance(error, PDFIntegrityError):
        pages = getattr(error, "pages", ())
        where = (" (pages " + ", ".join(map(str, pages)) + ")") if pages else ""
        if "scanned or unreadable" in str(error):
            return ("This PDF contains scanned or image-only pages" + where
                    + ", so only the Markdown version can be produced.")
        if "password" in str(error):
            return "This PDF is password-protected. Remove the password first."
        if "font" in str(error):
            return ("The PDF's font cannot display some translated "
                    "characters" + where + ". Use the Markdown version.")
        if "OCR" in str(error):
            return ("Scanned pages" + where + " could not be read. Relaunch "
                    "the app on a larger GPU and try again.")
        return ("The translated text could not be placed back into the PDF "
                "layout" + where + ". Use the Markdown version instead.")
    return "An unexpected error occurred while translating this document."


@dataclass
class PDFTranslationResult:
    outputs: list[tuple[str, bytes]] = field(default_factory=list)
    blocked: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    pages: list[dict] = field(default_factory=list)

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
) -> PDFTranslationResult:
    """Attempt Markdown and PDF separately, returning only checked bytes."""
    result = PDFTranslationResult()
    plans = inspect_pdf(pdf_bytes)
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
    if ocr_allowed is None and needs_ocr:
        from .gpu_profile import sequential_ocr_allowed

        ocr_allowed = sequential_ocr_allowed(device="cuda:0")

    def blocked(output: str, error: Exception) -> None:
        result.blocked.append({
            "output": output,
            "message": describe_error(error),
            "reason": str(error),
            "pages": list(getattr(error, "pages", ())),
        })

    try:
        if needs_ocr and not ocr_allowed:
            raise PDFIntegrityError(
                "Markdown requires OCR, but the allocated GPU is not "
                "eligible for sequential OCR/translation. Relaunch on a "
                "supported GPU; no pages will be silently skipped.",
                needs_ocr,
            )
        markdown, assets = translate_pdf_to_markdown(
            pdf_bytes, translate_fn, progress_cb=progress_cb,
            glossary=glossary,
            glossary_case_sensitive=glossary_case_sensitive,
            source_name=source_name,
        )
        data, name = pack_markdown_bundle(markdown, assets, stem=stem)
        result.outputs.append((name, data))
    except Exception as error:
        blocked("Markdown", error)

    try:
        require_native_coverage(plans)
        layout_warnings: list[str] = []
        data = translate_pdf(
            pdf_bytes, translate_fn, progress_cb=progress_cb,
            glossary=glossary,
            glossary_case_sensitive=glossary_case_sensitive,
            warnings=layout_warnings,
        )
        result.warnings.extend(layout_warnings)
        result.outputs.append((f"{stem}.pdf", data))
    except Exception as error:
        blocked("Reconstructed PDF", error)
    return result
