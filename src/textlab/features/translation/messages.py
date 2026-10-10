"""Short, actionable user-facing error messages.

Technical detail stays in logs and validation reports; the UI shows one
sentence and, where it helps, a concrete backend to try next.
"""

from __future__ import annotations

from .chunking import InputTooLongError, OutputTruncatedError
from .documents.pdf_checks import PDFIntegrityError
from .shield import ProtectedContentError

_HF_SMALL = {"nllb", "opus-mt"}
_HF_LARGE = {"nllb-large", "madlad-3b"}


def suggest_backend(error: Exception, backend: str | None) -> str:
    """Name the backend most likely to succeed where ``backend`` failed."""
    if isinstance(error, InputTooLongError):
        if backend == "ollama":
            return "Shorten the text or split very long words."
        return "Try the LLM (Ollama) backend, which accepts much longer input."
    if isinstance(error, ProtectedContentError):
        if backend == "ollama":
            return "Try a larger LLM model."
        return (
            "Try the LLM (Ollama) backend, which keeps links, code and "
            "tables intact more reliably."
        )
    if isinstance(error, OutputTruncatedError):
        if backend in _HF_SMALL:
            return (
                "Try NLLB-200 3.3B, which handles long technical text better."
            )
        if backend in _HF_LARGE:
            return "Try the LLM (Ollama) backend."
        return "Try NLLB-200 3.3B."
    return ""


def describe_error(error: Exception, backend: str | None = None) -> str:
    """One readable sentence (plus a suggestion) for a translation failure."""
    hint = suggest_backend(error, backend)
    if isinstance(error, InputTooLongError):
        message = (
            "Part of the text is too long for the selected model to "
            "translate in one piece."
        )
    elif isinstance(error, OutputTruncatedError):
        message = "The model could not finish translating part of the text."
    elif isinstance(error, ProtectedContentError):
        message = (
            "The model did not keep the document's formatting (links, "
            "code, tables) intact."
        )
    elif isinstance(error, PDFIntegrityError):
        return _describe_pdf(error)
    elif (
        isinstance(error, ValueError | RuntimeError) and len(str(error)) < 300
    ):
        # Our own explicit messages (unsupported pair, missing package, OOM).
        return str(error)
    else:
        return "An unexpected error occurred while translating this document."
    return f"{message} {hint}".strip()


def _describe_pdf(error: PDFIntegrityError) -> str:
    text = str(error)
    pages = getattr(error, "pages", ())
    where = (" (pages " + ", ".join(map(str, pages)) + ")") if pages else ""
    if "scanned or unreadable" in text:
        return (
            "This PDF contains scanned or image-only pages"
            + where
            + ", so only the Markdown version can be produced."
        )
    if "password" in text:
        return "This PDF is password-protected. Remove the password first."
    if "font" in text:
        return (
            "The PDF's font cannot display some translated characters"
            + where
            + ". Use the Markdown version."
        )
    if "OCR" in text:
        return (
            "Scanned pages" + where + " could not be read. Relaunch the "
            "app on a larger GPU and try again."
        )
    return (
        "The translated text could not be placed back into the PDF "
        "layout" + where + ". Use the Markdown version instead."
    )
