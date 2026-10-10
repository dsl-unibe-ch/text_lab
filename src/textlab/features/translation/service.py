"""Translating texts and documents: the translation feature's API.

Interfaces call this module; it ties together the backends
(:mod:`.engine`), the protection of links, code, math and glossary terms
(:mod:`.shield`) and the document formats (:mod:`.documents`).

Unlike transcription, translation runs in the calling process. The text
editor depends on a model that stays loaded between clicks, which a worker
process would reload every time; the model's GPU memory is instead released
through :mod:`textlab.common.gpu_manager` when another feature needs the
GPU.

* :func:`translate_text` translates pasted text.
* :func:`translate_documents` translates uploaded files, one translator per
  source language so repeated sentences are translated once.
* :func:`load_backend`, :func:`backend_ready` and :func:`load_signature`
  load a backend ahead of use and check that it is still loaded.

The few names interfaces need from other modules (the backend table,
language detection, the GPU profile, error types and messages) are
re-exported here, so interfaces import only this module.
"""

from __future__ import annotations

import dataclasses
import io
import os
import time
import zipfile
from collections.abc import Callable, Iterable, Mapping

from textlab.common.progress import Progress, ProgressCallback, no_progress

from .chunking import TranslationLimitError
from .documents.markdown import reflow_soft_wraps, translate_markdown
from .documents.office import translate_docx, translate_pptx, translate_xlsx
from .documents.pdf_workflow import PDFTranslationResult, translate_pdf_outputs
from .engine import (
    BACKENDS,
    FORMALITY_CAPABLE_BACKENDS,
    FORMALITY_CHOICES,
    TRANSLATION_BACKENDS,
    backend_is_loaded,
    backend_load_signature,
    flores_to_iso2,
    make_translate_fn,
    preload_backend,
)
from .gpu_profile import GpuProfile, detect_gpu_profile
from .lang_detect import (
    DetectionResult,
    detect_document_language,
    detect_language,
)
from .messages import describe_error
from .review import build_review_docx, build_review_html
from .shield import (
    ProtectedContentError,
    record_translations,
    shielded_translate,
)

__all__ = [
    # Options and results
    "DocumentOptions",
    "DocumentResults",
    "NamedFile",
    "PDFTranslationResult",
    "TranslationOptions",
    "UnpackedUploads",
    # Backends
    "BACKENDS",
    "FORMALITY_CAPABLE_BACKENDS",
    "FORMALITY_CHOICES",
    "TRANSLATION_BACKENDS",
    "backend_ready",
    "load_backend",
    "load_signature",
    "supports_pair",
    # Translation
    "translate_document",
    "translate_documents",
    "translate_text",
    "unpack_uploads",
    # Downloads
    "mime_type",
    "outputs_zip",
    "review_files",
    # Re-exported for interfaces
    "DetectionResult",
    "GpuProfile",
    "ProtectedContentError",
    "TranslationLimitError",
    "describe_error",
    "detect_gpu_profile",
    "detect_language",
]

#: File types that can be translated, alone or inside a ZIP archive.
SUPPORTED_EXTENSIONS = (
    ".md",
    ".txt",
    ".srt",
    ".vtt",
    ".pdf",
    ".docx",
    ".xlsx",
    ".pptx",
)
#: The deliverables a PDF can be translated into.
PDF_OUTPUTS = ("markdown", "pdf")

#: Minimum seconds between two progress updates of the same stage.
PROGRESS_INTERVAL = 0.25

_MIME_TYPES = {
    ".pdf": "application/pdf",
    ".docx": (
        "application/vnd.openxmlformats-officedocument"
        ".wordprocessingml.document"
    ),
    ".pptx": (
        "application/vnd.openxmlformats-officedocument"
        ".presentationml.presentation"
    ),
    ".xlsx": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ),
    ".md": "text/markdown",
    ".zip": "application/zip",
    ".html": "text/html",
}

#: A file name and its content.
NamedFile = tuple[str, bytes]


@dataclasses.dataclass(frozen=True)
class TranslationOptions:
    """What to translate with, and into which language.

    Attributes:
        backend: A key of :data:`.engine.BACKENDS`.
        source_code: FLORES-200 code of the source language, such as
            ``"deu_Latn"``; for documents, the fallback when detection is
            off or uncertain.
        source_name: The source language's display name, used in prompts.
        target_code: FLORES-200 code of the target language.
        target_name: The target language's display name.
        ollama_model: The Ollama model, for the ``ollama`` backend.
        formality: ``"default"``, ``"formal"`` or ``"informal"``; only
            backends in :data:`.engine.FORMALITY_CAPABLE_BACKENDS` use it.
        glossary: Source terms mapped to the translations to force.
        glossary_case_sensitive: Match glossary terms only in the exact
            case given.
    """

    backend: str
    source_code: str
    source_name: str
    target_code: str
    target_name: str
    ollama_model: str | None = None
    formality: str = "default"
    glossary: Mapping[str, str] = dataclasses.field(default_factory=dict)
    glossary_case_sensitive: bool = False


@dataclasses.dataclass(frozen=True)
class DocumentOptions:
    """How to translate a batch of documents.

    Attributes:
        pdf_outputs: Which of :data:`PDF_OUTPUTS` to build for PDFs.
        math_ocr: Send PDF pages with equations to OCR so the Markdown gets
            LaTeX; slower.
        detect_source: Detect each document's language instead of using
            the source language of the options.
        review: Add a side-by-side review file (HTML, and Word when
            python-docx is available) for each document.
    """

    pdf_outputs: tuple[str, ...] = PDF_OUTPUTS
    math_ocr: bool = False
    detect_source: bool = True
    review: bool = True


@dataclasses.dataclass
class DocumentResults:
    """The outcome of :func:`translate_documents`.

    Attributes:
        file_outputs: For each translated source file, its output files.
        errors: Source files that failed, with a message for the user.
        pdf_reports: The validation report of each PDF.
        languages: Each source file's language, as shown to the user.
        total: Number of source files.
        target_code: FLORES-200 code of the target language.
        elapsed: Duration of the run in seconds.
    """

    file_outputs: list[tuple[str, list[NamedFile]]]
    errors: list[tuple[str, str]]
    pdf_reports: list[tuple[str, PDFTranslationResult]]
    languages: dict[str, str]
    total: int
    target_code: str
    elapsed: float = 0.0

    @property
    def outputs(self) -> list[NamedFile]:
        """All output files, in order."""
        return [item for _, items in self.file_outputs for item in items]


@dataclasses.dataclass
class UnpackedUploads:
    """Uploaded files ready for translation.

    Attributes:
        files: The translatable files; ZIP archives are replaced by their
            members.
        skipped: ZIP members left out because their type is not supported.
        invalid: Uploads that claimed to be ZIP archives but are not.
    """

    files: list[NamedFile]
    skipped: list[str]
    invalid: list[str]


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def load_signature(options: TranslationOptions) -> tuple:
    """Return what must be loaded to translate with these options.

    Two option sets with the same signature use the same loaded model, so
    an interface can tell whether a change of backend, language pair or
    model needs a new :func:`load_backend`.

    Args:
        options: The translation options.

    Returns:
        An opaque, comparable tuple.
    """
    return backend_load_signature(
        backend=options.backend,
        src_lang=options.source_code,
        tgt_lang=options.target_code,
        ollama_model=options.ollama_model,
    )


def backend_ready(options: TranslationOptions) -> bool:
    """Return True if the backend for these options is loaded right now.

    This checks the model itself, not a remembered state: Ollama may have
    unloaded it, or another feature may have freed the GPU.

    Args:
        options: The translation options.

    Returns:
        Whether a translation can start without loading a model.
    """
    return backend_is_loaded(
        options.backend,
        src_lang=options.source_code,
        tgt_lang=options.target_code,
        ollama_model=options.ollama_model,
    )


def load_backend(options: TranslationOptions) -> None:
    """Load the backend for these options onto the compute device.

    The first load of a model can take a minute or two; afterwards
    translations start at once.

    Args:
        options: The translation options.
    """
    preload_backend(
        backend=options.backend,
        src_lang=options.source_code,
        tgt_lang=options.target_code,
        ollama_model=options.ollama_model,
        src_lang_name=options.source_name,
        tgt_lang_name=options.target_name,
    )


def supports_pair(backend: str, source_code: str, target_code: str) -> bool:
    """Return False if a backend cannot translate a language pair at all.

    Only OPUS-MT is limited, to the languages it has bilingual models for.

    Args:
        backend: A key of :data:`.engine.BACKENDS`.
        source_code: FLORES-200 code of the source language.
        target_code: FLORES-200 code of the target language.

    Returns:
        Whether the pair can be attempted.
    """
    if backend != "opus-mt":
        return True
    return bool(flores_to_iso2(source_code) and flores_to_iso2(target_code))


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------


def translate_text(
    text: str,
    options: TranslationOptions,
    *,
    on_progress: ProgressCallback = no_progress,
    on_notice: Callable[[str], None] | None = None,
) -> str:
    """Translate pasted text.

    Pasted text is usually prose, often hard-wrapped by the PDF or e-mail
    it came from, so sentences split over lines are rejoined first; a break
    after a finished sentence stays.

    Args:
        text: The source text.
        options: The translation options.
        on_progress: Receives an update after each chunk.
        on_notice: Receives notices for the user, such as a retry with
            smaller chunks.

    Returns:
        The translation.

    Raises:
        TranslationLimitError: If the text cannot be translated completely
            within the model's limits.
        ProtectedContentError: If protected content (links, code, glossary
            terms) did not survive the translation.
    """

    def report(done: int, total: int) -> None:
        if total > 0:
            on_progress(
                Progress(
                    f"Translating chunk {done}/{total}",
                    min(done / total, 1.0),
                )
            )

    translate_fn = _make_translator(
        options,
        options.source_code,
        options.source_name,
        progress_cb=report,
        status_cb=on_notice,
    )
    return shielded_translate(
        reflow_soft_wraps(text),
        translate_fn,
        glossary=options.glossary,
        glossary_case_sensitive=options.glossary_case_sensitive,
    )


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


def unpack_uploads(uploads: Iterable[NamedFile]) -> UnpackedUploads:
    """Replace ZIP archives among the uploads by their supported members.

    Args:
        uploads: File names and contents, as uploaded.

    Returns:
        The files to translate, and what was left out.
    """
    result = UnpackedUploads([], [], [])
    for name, data in uploads:
        if not name.lower().endswith(".zip"):
            result.files.append((name, data))
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(data), "r") as archive:
                for entry in archive.namelist():
                    if entry.endswith("/"):
                        continue
                    extension = os.path.splitext(entry)[1].lower()
                    if extension not in SUPPORTED_EXTENSIONS:
                        result.skipped.append(entry)
                        continue
                    result.files.append((entry, archive.read(entry)))
        except zipfile.BadZipFile:
            result.invalid.append(name)
    return result


def translate_documents(
    files: list[NamedFile],
    options: TranslationOptions,
    document_options: DocumentOptions | None = None,
    *,
    on_progress: ProgressCallback = no_progress,
) -> DocumentResults:
    """Translate documents into the target language, each in its format.

    A failing document is reported in the results and does not stop the
    others. Documents that are already in the target language are reported
    rather than translated.

    Args:
        files: File names and contents, with ZIP archives already unpacked
            (:func:`unpack_uploads`).
        options: The translation options.
        document_options: PDF outputs, language detection and review
            files; the defaults of :class:`DocumentOptions` when omitted.
        on_progress: Receives updates; the message names the file and the
            step, the fraction is the share of that step done.

    Returns:
        The output files, errors and reports of the batch.
    """
    document_options = document_options or DocumentOptions()
    started = time.monotonic()
    progress = _DocumentProgress(on_progress)
    results = DocumentResults(
        file_outputs=[],
        errors=[],
        pdf_reports=[],
        languages={},
        total=len(files),
        target_code=options.target_code,
    )
    translators: dict[str, Callable[[str], str]] = {}

    def translator(code: str, name: str) -> Callable[[str], str]:
        # One per source language, so its sentence cache is shared by all
        # outputs of all files in that language.
        if code not in translators:
            translators[code] = _make_translator(
                options,
                code,
                name,
                progress_cb=progress.sentences,
                status_cb=progress.retrying,
            )
        return translators[code]

    for index, (name, data) in enumerate(files, start=1):
        prefix = f"[{index}/{len(files)}] {name}"
        progress.stage(prefix, (index - 1) / len(files))
        source_code, source_name = options.source_code, options.source_name
        if document_options.detect_source:
            progress.stage(f"{prefix}: detecting language")
            detection = detect_document_language(name, data)
            if detection is not None and detection.display_name:
                source_code = detection.flores_code
                source_name = detection.display_name
                results.languages[name] = f"{source_name} (detected)"
            else:
                results.languages[name] = (
                    f"{options.source_name} (not detected, using your "
                    "selection)"
                )
        if source_code == options.target_code:
            results.errors.append(
                (
                    name,
                    "This document already appears to be in "
                    f"{options.target_name}.",
                )
            )
            continue
        try:
            with record_translations() as pairs:
                outputs = translate_document(
                    name,
                    data,
                    translator(source_code, source_name),
                    options.target_code,
                    glossary=options.glossary,
                    glossary_case_sensitive=options.glossary_case_sensitive,
                    progress_cb=progress.step,
                    pdf_result_cb=lambda report, name=name: (
                        results.pdf_reports.append((name, report))
                    ),
                    pdf_outputs=document_options.pdf_outputs,
                    math_ocr=document_options.math_ocr,
                    backend=options.backend,
                )
            if document_options.review:
                if results.pdf_reports and results.pdf_reports[-1][0] == name:
                    # Whole PDF paragraphs read better than Markdown lines.
                    pairs = results.pdf_reports[-1][1].pairs
                outputs += review_files(
                    name,
                    pairs,
                    source_name,
                    options.target_name,
                    options.target_code,
                )
            results.file_outputs.append((name, outputs))
        except Exception as error:
            results.errors.append(
                (name, describe_error(error, options.backend))
            )

    progress.stage("done", 1.0)
    results.elapsed = time.monotonic() - started
    return results


def translate_document(
    name: str,
    data: bytes,
    translate_fn: Callable[[str], str],
    target_code: str,
    *,
    glossary: Mapping[str, str] | None = None,
    glossary_case_sensitive: bool = False,
    progress_cb: Callable[[int, int, str], None] | None = None,
    pdf_result_cb: Callable[[PDFTranslationResult], None] | None = None,
    pdf_outputs: tuple[str, ...] = PDF_OUTPUTS,
    math_ocr: bool = False,
    backend: str | None = None,
) -> list[NamedFile]:
    """Translate one file into the same format.

    PDFs give up to two outputs (Markdown and a reconstructed PDF), each
    checked separately, plus a validation report; a failed output never
    discards the other.

    Args:
        name: The file name; its extension selects the format.
        data: The file content.
        translate_fn: A translator from :func:`.engine.make_translate_fn`.
        target_code: FLORES-200 code of the target language, added to the
            output names (``report.deu_Latn.docx``).
        glossary: Source terms mapped to the translations to force.
        glossary_case_sensitive: Match glossary terms in exact case only.
        progress_cb: Receives ``(done, total, stage)`` updates.
        pdf_result_cb: Receives a PDF's validation result, also when no
            output passed its checks.
        pdf_outputs: Which of :data:`PDF_OUTPUTS` to build.
        math_ocr: Send PDF pages with equations to OCR for LaTeX.
        backend: The backend key, used to suggest another one in errors.

    Returns:
        The output files.

    Raises:
        ValueError: If the type is not supported, or no PDF output passed
            its checks.
    """
    report = progress_cb or _ignore_progress
    base, extension = os.path.splitext(name)
    extension = extension.lower()
    out_stem = f"{os.path.basename(base)}.{target_code}"
    glossary_options = {
        "glossary": glossary,
        "glossary_case_sensitive": glossary_case_sensitive,
    }

    if extension == ".md":
        report(0, 1, "parsing markdown")
        text = data.decode("utf-8", errors="replace")
        result = translate_markdown(
            text, translate_fn, progress_cb=report, **glossary_options
        )
        report(1, 1, "reconstructing markdown")
        return [(f"{out_stem}.md", result.encode("utf-8"))]

    if extension == ".pdf":
        pdf_result = translate_pdf_outputs(
            data,
            translate_fn,
            stem=out_stem,
            source_name=os.path.basename(name),
            progress_cb=report,
            outputs=pdf_outputs,
            math_ocr=math_ocr,
            backend=backend,
            **glossary_options,
        )
        if pdf_result_cb is not None:
            pdf_result_cb(pdf_result)
        if not pdf_result.outputs:
            reasons = " ".join(
                item.get("message", item["reason"])
                for item in pdf_result.blocked
            )
            raise ValueError(reasons or "No PDF deliverables passed checks.")
        return pdf_result.outputs + [
            (
                f"{out_stem}.translation-report.json",
                pdf_result.report_bytes(),
            ),
        ]

    office_translators = {
        ".docx": translate_docx,
        ".xlsx": translate_xlsx,
        ".pptx": translate_pptx,
    }
    if extension in office_translators:
        kind = extension[1:]
        report(0, 1, f"parsing {kind}")
        result = office_translators[extension](
            data, translate_fn, progress_cb=report, **glossary_options
        )
        report(1, 1, f"reconstructing {kind}")
        return [(f"{out_stem}{extension}", result)]

    if extension in (".txt", ".srt", ".vtt"):
        report(0, 1, f"translating {extension}")
        text = data.decode("utf-8", errors="replace")
        if extension == ".txt":
            # Plain prose: rejoin sentences the file hard-wrapped. Subtitles
            # are excluded deliberately: there every line break carries
            # timing, and joining lines destroys the cue structure.
            text = reflow_soft_wraps(text)
        result = shielded_translate(text, translate_fn, **glossary_options)
        report(1, 1, "done")
        return [(f"{out_stem}{extension}", result.encode("utf-8"))]

    raise ValueError(f"Unsupported file type: {extension}")


def review_files(
    source: str,
    pairs: list[tuple[str, str]],
    source_language: str,
    target_language: str,
    target_code: str,
) -> list[NamedFile]:
    """Build the side-by-side review files for a translated document.

    Args:
        source: The source file name.
        pairs: ``(source, translation)`` units, in document order.
        source_language: The source language's display name.
        target_language: The target language's display name.
        target_code: FLORES-200 code of the target language.

    Returns:
        An HTML file, and a Word file when python-docx is available; nothing
        when there are no pairs.
    """
    if not pairs:
        return []
    title = os.path.basename(source)
    stem = f"{os.path.splitext(title)[0]}.{target_code}.side-by-side"
    options = {
        "title": title,
        "source_language": source_language,
        "target_language": target_language,
    }
    files = [(f"{stem}.html", build_review_html(pairs, **options))]
    docx_bytes = build_review_docx(pairs, **options)
    if docx_bytes is not None:
        files.append((f"{stem}.docx", docx_bytes))
    return files


def outputs_zip(
    file_outputs: list[tuple[str, list[NamedFile]]],
    errors: list[tuple[str, str]],
) -> bytes:
    """Pack the outputs of a batch into one ZIP archive.

    Markdown bundles (``.md.zip``) are unpacked into the archive. A single
    source file's outputs go to the root; with several, each source gets a
    folder so their ``assets/`` folders cannot collide. Each failed file
    gets a ``<name>.ERROR.txt`` with its message.

    Args:
        file_outputs: For each source file, its output files.
        errors: Failed source files and their messages.

    Returns:
        The ZIP archive.
    """
    buffer = io.BytesIO()
    nested = len(file_outputs) + len(errors) > 1
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for source, outputs in file_outputs:
            folder = (
                os.path.splitext(os.path.basename(source))[0] + "/"
                if nested
                else ""
            )
            for name, data in outputs:
                if name.endswith(".md.zip"):
                    with zipfile.ZipFile(io.BytesIO(data)) as bundle:
                        for entry in bundle.namelist():
                            archive.writestr(
                                folder + entry, bundle.read(entry)
                            )
                else:
                    archive.writestr(folder + name, data)
        for name, message in errors:
            archive.writestr(
                f"{name}.ERROR.txt",
                f"Failed to translate: {message}".encode(),
            )
    return buffer.getvalue()


def mime_type(name: str) -> str:
    """Return the MIME type of an output file, for downloads.

    Args:
        name: The file name.

    Returns:
        The type; ``text/plain`` for text formats without their own.
    """
    extension = os.path.splitext(name.lower())[1]
    return _MIME_TYPES.get(extension, "text/plain")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_translator(
    options: TranslationOptions,
    source_code: str,
    source_name: str,
    *,
    progress_cb: Callable[[int, int], None] | None,
    status_cb: Callable[[str], None] | None,
) -> Callable[[str], str]:
    """Return a translator for one source language with these options."""
    return make_translate_fn(
        src_lang=source_code,
        tgt_lang=options.target_code,
        backend=options.backend,
        ollama_model=options.ollama_model,
        src_lang_name=source_name,
        tgt_lang_name=options.target_name,
        formality=options.formality,
        progress_cb=progress_cb,
        status_cb=status_cb,
    )


def _ignore_progress(done: int, total: int, stage: str) -> None:
    """Discard a ``(done, total, stage)`` progress update."""


class _DocumentProgress:
    """Turn the document translators' callbacks into progress updates.

    Parsers report every line; each update costs the interface a redraw,
    so updates of one stage are limited to one per
    :data:`PROGRESS_INTERVAL`. While sentences are translated, the message
    includes an estimate of the time left.
    """

    def __init__(
        self,
        on_progress: ProgressCallback,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._on_progress = on_progress
        self._clock = clock
        self._last_time = 0.0
        self._last_stage: str | None = None
        self._rate_start = 0.0
        self._rate_done = 0
        self._rate_total = 0

    def stage(self, message: str, fraction: float | None = None) -> None:
        """Report a new step, without limiting the rate."""
        self._on_progress(Progress(message, fraction))

    def step(
        self, done: int, total: int, stage: str, label: str | None = None
    ) -> None:
        """Report ``done`` of ``total`` units of a stage, rate-limited."""
        now = self._clock()
        if (
            stage == self._last_stage
            and done < total
            and now - self._last_time < PROGRESS_INTERVAL
        ):
            return
        self._last_time = now
        self._last_stage = stage
        fraction = min(done / total, 1.0) if total > 0 else None
        self._on_progress(Progress(label or stage, fraction))

    def sentences(self, done: int, total: int) -> None:
        """Report translated sentences, with the time left."""
        time_left = self._time_left(done, total)
        label = f"translating sentences {done}/{total}{time_left}"
        self.step(done, total, "translating", label)

    def retrying(self, message: str) -> None:
        """Report a retry with smaller pieces; retries are routine."""
        self.stage("retrying some text in smaller pieces")

    def _time_left(self, done: int, total: int) -> str:
        """Return an estimate such as ``" · about 3 min left"``, or ``""``."""
        now = self._clock()
        if total != self._rate_total or done < self._rate_done:
            self._rate_start = now
            self._rate_total = total
        self._rate_done = done
        elapsed = now - self._rate_start
        if done <= 0 or done >= total or elapsed < 5:
            return ""
        remaining = elapsed / done * (total - done)
        if remaining < 60:
            return " · less than a minute left"
        return f" · about {round(remaining / 60)} min left"
