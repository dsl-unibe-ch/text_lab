"""The Translation page: an editor for text, and jobs for documents.

Two ways to use it, as on the Streamlit page, sharing the languages, the
engine and the glossary:

- **Text**: the source on the left, the translation on the right. Each
  translation is a short job that is not listed with the others
  (``listed=False``): its card is the editor's output pane
  (``translation/editor.html``). It waits for a running job like any GPU
  job, and the page says which one. The engine's model stays loaded
  between translations until another feature needs the GPU, so only the
  first translation with an engine loads it. Short texts come back with
  the answer to the request itself; longer ones follow over the page's
  event stream, with the previous translation shown until the new one is
  ready.
- **Documents**: files and ZIP archives, translated in their own format by
  a job, with a ZIP of all outputs, each document's files, the checks of
  PDFs, and source and translation side by side in the page.

The page remembers what was last chosen and the text in the editor (in
memory, for the session), so a reload shows them again.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import Request
from fastapi.responses import HTMLResponse
from markupsafe import Markup
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import FormData

from textlab.common import gpu_manager
from textlab.common.progress import CancelledError, Progress
from textlab.features.translation import service
from textlab.ui.web import uploads
from textlab.ui.web.auth import Checked, protected_router
from textlab.ui.web.jobs import (
    Download,
    Job,
    JobContext,
    JobRunner,
    render_card,
    runner_of,
)
from textlab.ui.web.render import duration, renderer

router = protected_router()

#: The page's id, which also names its jobs.
PAGE = "translation"

#: The source language's value when it is to be detected.
DETECT = "auto"

#: Most characters the editor translates at once; longer texts go through
#: Documents, as a text file.
TEXT_LIMIT = 20_000

#: Most terms in the glossary.
GLOSSARY_LIMIT = 50

#: Most characters of a glossary term or its translation.
TERM_LIMIT = 200

#: Most files in one upload; a ZIP counts as one.
MAX_FILES = 100

#: Languages listed first, as the ones most used here.
COMMON_LANGUAGES = ("eng_Latn", "deu_Latn", "fra_Latn", "ita_Latn", "spa_Latn")

#: Seconds a text translation request waits for its job before answering,
#: so that short texts come back at once instead of over the event stream.
SETTLE_SECONDS = 1.5

#: How long submitting waits for the language model server, in attempts
#: half a second apart (Ollama engine only).
SERVER_ATTEMPTS = 4

#: Passages of a document shown side by side in the page; the review file
#: holds all of them.
COMPARE_LIMIT = 300

#: What each kind of output file is, by the end of its name.
FILE_KINDS = (
    (".side-by-side.html", "Side by side, HTML"),
    (".side-by-side.docx", "Side by side, Word"),
    (".translation-report.json", "Checks of the PDF"),
    (".md.zip", "Markdown with images"),
    (".md", "Markdown"),
    (".pdf", "PDF"),
    (".docx", "Word"),
    (".xlsx", "Excel"),
    (".pptx", "PowerPoint"),
    (".txt", "Text"),
    (".srt", "Subtitles"),
    (".vtt", "Subtitles"),
    (".zip", "ZIP archive"),
)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Settings:
    """What the form chose: languages, engine, glossary, document outputs.

    Attributes:
        source: FLORES-200 code of the source language; ``""`` to detect
            it (for documents: each document's).
        target: FLORES-200 code of the target language.
        backend: A key of ``service.BACKENDS``.
        ollama_model: The language model, for the ``ollama`` backend.
        formality: One of ``service.FORMALITY_CHOICES``.
        glossary: Terms and the translations to force, in the form's order.
        glossary_case_sensitive: Match terms only in the case given.
        pdf_outputs: The outputs to make of PDFs.
        math_ocr: Read equations in PDFs into LaTeX for the Markdown.
        review: Add a side-by-side file for each document.
    """

    source: str = ""
    target: str = "eng_Latn"
    backend: str = "nllb"
    ollama_model: str | None = None
    formality: str = "default"
    glossary: tuple[tuple[str, str], ...] = ()
    glossary_case_sensitive: bool = False
    pdf_outputs: tuple[str, ...] = service.PDF_OUTPUTS
    math_ocr: bool = False
    review: bool = True

    @property
    def engine(self) -> str:
        """The engine's short name, with the model for a language model."""
        name = service.BACKENDS[self.backend].name
        if self.backend == "ollama" and self.ollama_model:
            return f"{name} {self.ollama_model}"
        return name

    def options(self, source: str | None = None) -> service.TranslationOptions:
        """Return the service's options, from ``source`` or the chosen one.

        Args:
            source: The source language's code, such as a detected one;
                the chosen one if ``None``.

        Returns:
            The options.
        """
        code = self.source if source is None else source
        return service.TranslationOptions(
            backend=self.backend,
            source_code=code,
            source_name=service.language_name(code) if code else "",
            target_code=self.target,
            target_name=service.language_name(self.target),
            ollama_model=self.ollama_model,
            formality=self.formality,
            glossary=dict(self.glossary),
            glossary_case_sensitive=self.glossary_case_sensitive,
        )


@dataclasses.dataclass
class Editor:
    """What the page shows again after a reload, kept for the session.

    Attributes:
        settings: The settings last sent.
        text: The text last translated in the editor.
    """

    settings: Settings = dataclasses.field(default_factory=Settings)
    text: str = ""


def editor_of(request: Request) -> Editor:
    """Return the page's remembered state, made on first use.

    Args:
        request: The current request.

    Returns:
        The state, kept in memory with the app.
    """
    state = request.app.state
    editor = getattr(state, "translation_editor", None)
    if editor is None:
        editor = Editor()
        state.translation_editor = editor
    return editor


def read_settings(form: FormData) -> Settings:
    """Read the shared settings and the document outputs from the form.

    Args:
        form: The form.

    Returns:
        The settings.

    Raises:
        UploadError: If a value is not one offered.
    """
    codes = {code for code, _ in service.languages()}
    source = str(form.get("source") or DETECT)
    if source == DETECT:
        source = ""
    elif source not in codes:
        raise uploads.UploadError("Choose the language to translate from.")
    target = str(form.get("target") or "")
    if target not in codes:
        raise uploads.UploadError("Choose the language to translate into.")
    if source == target:
        raise uploads.UploadError(
            "From and To are the same language. Choose another language "
            "under To."
        )

    backend = str(form.get("backend") or "")
    if backend not in service.BACKENDS:
        raise uploads.UploadError("Choose an engine from the list.")
    ollama_model = None
    formality = "default"
    if backend == "ollama":
        ollama_model = str(form.get("llm") or "")
        if ollama_model not in service.ollama_models():
            raise uploads.UploadError("Choose a language model from the list.")
    if backend in service.FORMALITY_CAPABLE_BACKENDS:
        formality = str(form.get("formality") or "default")
        if formality not in service.FORMALITY_CHOICES:
            raise uploads.UploadError("Choose how formal to write.")
    if source and not service.supports_pair(backend, source, target):
        raise uploads.UploadError(
            f"OPUS-MT has no model from {service.language_name(source)} "
            f"into {service.language_name(target)}. Choose another engine."
        )

    return Settings(
        source=source,
        target=target,
        backend=backend,
        ollama_model=ollama_model,
        formality=formality,
        glossary=read_glossary(form),
        glossary_case_sensitive=bool(form.get("glossary_case")),
        pdf_outputs=tuple(
            output
            for output in service.PDF_OUTPUTS
            if form.get(f"pdf_{output}")
        ),
        math_ocr=bool(form.get("math_ocr")),
        review=bool(form.get("review")),
    )


def read_glossary(form: FormData) -> tuple[tuple[str, str], ...]:
    """Read the glossary's terms; rows without both halves are left out.

    Args:
        form: The form, with the lists ``term`` and ``translation``.

    Returns:
        ``(term, translation)`` pairs, each term once (the last wins).

    Raises:
        UploadError: If there are too many terms, or one is too long.
    """
    terms: dict[str, str] = {}
    for term, translation in zip(
        form.getlist("term"), form.getlist("translation"), strict=False
    ):
        term, translation = str(term).strip(), str(translation).strip()
        if not term or not translation:
            continue
        if len(term) > TERM_LIMIT or len(translation) > TERM_LIMIT:
            raise uploads.UploadError(
                f"Glossary entries can have at most {TERM_LIMIT} characters."
            )
        terms[term] = translation
    if len(terms) > GLOSSARY_LIMIT:
        raise uploads.UploadError(
            f"The glossary can hold at most {GLOSSARY_LIMIT} terms."
        )
    return tuple(terms.items())


async def _check_server(settings: Settings) -> None:
    """Refuse a language model engine whose server does not answer."""
    if settings.backend != "ollama":
        return
    ready = await run_in_threadpool(service.server_ready, SERVER_ATTEMPTS)
    if not ready:
        raise uploads.UploadError(
            "The language model server (Ollama) does not answer. Try again "
            "in a minute, or choose another engine; if it keeps failing, "
            "send ollama.log from the session's folder to support."
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/translation")
def page(request: Request) -> HTMLResponse:
    """Show the editor, the documents form and the session's translations.

    Args:
        request: The current request.

    Returns:
        The page, with the settings and text last used.
    """
    runner = runner_of(request)
    editor = editor_of(request)
    current = latest_text_job(runner)
    models = service.ollama_models()
    server_ready = service.server_ready()
    return renderer(request).response(
        request,
        "translation/page.html",
        page_id=PAGE,
        saved=editor.settings,
        text=editor.text,
        text_limit=TEXT_LIMIT,
        common_languages=[
            (code, service.language_name(code)) for code in COMMON_LANGUAGES
        ],
        all_languages=sorted(service.languages(), key=lambda item: item[1]),
        engines=engines(models, server_ready),
        models=models,
        formality=service.FORMALITY_CHOICES,
        glossary_limit=GLOSSARY_LIMIT,
        term_limit=TERM_LIMIT,
        document_accept=",".join([*service.SUPPORTED_EXTENSIONS, ".zip"]),
        document_formats=" ".join(
            [
                ext.lstrip(".").upper()
                for ext in [*service.SUPPORTED_EXTENSIONS, ".zip"]
            ]
        ),
        editor_card=(
            Markup(render_card(request, current)) if current else None
        ),
        cards=Markup(
            "".join(
                render_card(request, job)
                for job in runner.jobs(PAGE, listed=True)[::-1]
            )
        ),
    )


@router.post("/translation/text")
async def translate_text(request: Request, session: Checked) -> HTMLResponse:
    """Translate the editor's text.

    Earlier text translations are stopped and dropped: the editor shows one
    at a time.

    Args:
        request: The form: ``text`` and the shared settings (see
            :func:`read_settings`).
        session: The session, with its CSRF token checked.

    Returns:
        The output pane's card: the translation if it was ready within
        :data:`SETTLE_SECONDS`, otherwise the job waiting or running.

    Raises:
        UploadError: If there is no text, too much, or a setting is not
            valid.
    """
    form = await uploads.read_form(request, max_files=0, max_fields=150)
    settings = read_settings(form)
    text = str(form.get("text") or "")
    if not text.strip():
        raise uploads.UploadError("Type or paste the text to translate.")
    if len(text) > TEXT_LIMIT:
        raise uploads.UploadError(
            f"The editor translates up to {TEXT_LIMIT:,} characters at a "
            "time. Translate longer texts as a document: save them as a "
            ".txt file."
        )
    await _check_server(settings)

    runner = runner_of(request)
    editor = editor_of(request)
    editor.settings, editor.text = settings, text
    previous = latest_text_job(runner)
    for job in runner.jobs(PAGE, listed=False):
        if job.active:
            runner.cancel(job.id)
        else:
            runner.remove(job.id)

    job = runner.create(
        title=f"Text into {service.language_name(settings.target)}",
        page=PAGE,
        gpu=gpu_manager.TRANSLATION,
        ollama_model=settings.ollama_model,
        card_template="translation/editor.html",
        listed=False,
    )
    if previous is not None and isinstance(previous.result, TextTranslation):
        # Shown, dimmed, until the new translation is ready.
        job.preview = previous.result.text
    runner.submit(job, partial(translate_editor_text, text, settings))

    deadline = time.monotonic() + SETTLE_SECONDS
    while job.active and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    return HTMLResponse(render_card(request, job))


@router.post("/translation")
async def translate_documents(
    request: Request, session: Checked
) -> HTMLResponse:
    """Start a job that translates the uploaded documents.

    Args:
        request: The form: ``files`` (documents and ZIP archives), the
            shared settings, ``pdf_markdown``, ``pdf_pdf``, ``math_ocr``
            and ``review``.
        session: The session, with its CSRF token checked.

    Returns:
        The new job's card.

    Raises:
        UploadError: If there is no file, one of another type, or a
            setting is not valid.
    """
    form = await uploads.read_form(
        request, max_files=MAX_FILES, max_fields=150
    )
    settings = read_settings(form)
    found = uploads.files(form, "files")
    if not found:
        raise uploads.UploadError("Choose the documents to translate.")
    accepted = (*service.SUPPORTED_EXTENSIONS, ".zip")
    for upload in found:
        uploads.check_suffix(upload, accepted)
    if not settings.pdf_outputs and any(
        PurePosixPath(upload.filename or "").suffix.lower() in (".pdf", ".zip")
        for upload in found
    ):
        raise uploads.UploadError(
            "Choose at least one output for PDF files: Markdown, PDF, or both."
        )
    await _check_server(settings)
    editor_of(request).settings = settings

    names = [uploads.display_name(upload) for upload in found]
    runner = runner_of(request)
    job = runner.create(
        title=names[0] if len(names) == 1 else f"{len(names)} documents",
        page=PAGE,
        gpu=gpu_manager.TRANSLATION,
        ollama_model=settings.ollama_model,
        result_template="translation/result.html",
    )
    try:
        used: set[str] = set()
        paths = [
            await run_in_threadpool(uploads.save, upload, job.inputs, used)
            for upload in found
        ]
    except BaseException:
        runner.discard(job)
        raise
    runner.submit(job, partial(translate_files, paths, settings))
    return HTMLResponse(render_card(request, job))


def latest_text_job(runner: JobRunner) -> Job | None:
    """Return the editor's newest translation job, if there is one.

    Args:
        runner: The app's job runner.

    Returns:
        The job, or ``None``.
    """
    jobs = runner.jobs(PAGE, listed=False)
    return jobs[-1] if jobs else None


def engines(
    models: list[str], server_ready: bool
) -> list[tuple[str, Any, str]]:
    """Return the engines to choose from, with why one cannot be used.

    Args:
        models: The language models for the Ollama engine.
        server_ready: Whether the Ollama server answers.

    Returns:
        ``(key, backend, problem)`` triples; ``problem`` is empty for an
        engine that can be chosen.
    """
    result = []
    for key, backend in service.BACKENDS.items():
        problem = ""
        if key == "ollama" and not models:
            problem = "No language model is available on this GPU."
        elif key == "ollama" and not server_ready:
            problem = "Its server does not answer yet."
        result.append((key, backend, problem))
    return result


# ---------------------------------------------------------------------------
# Text: the editor's jobs
# ---------------------------------------------------------------------------


class TranslationFailedError(RuntimeError):
    """A translation that cannot be done; the message is for the user."""


@dataclasses.dataclass(frozen=True)
class TextTranslation:
    """A translation in the editor: what its output pane shows.

    Attributes:
        text: The translation.
        source: The source language's code.
        target: The target language's code.
        detected: Whether the source language was detected, with its
            confidence (0 to 1); ``None`` if it was chosen.
        engine: The engine's short name.
        glossary_terms: How many glossary terms were forced.
        notices: Remarks of the translation, such as a retry in smaller
            pieces.
        seconds: How long it took.
    """

    text: str
    source: str
    target: str
    detected: float | None
    engine: str
    glossary_terms: int
    notices: tuple[str, ...]
    seconds: float

    @property
    def source_name(self) -> str:
        """The source language's name."""
        return service.language_name(self.source)

    @property
    def target_name(self) -> str:
        """The target language's name."""
        return service.language_name(self.target)

    @property
    def words(self) -> int:
        """How many words the translation has."""
        return len(self.text.split())

    @property
    def took(self) -> str:
        """How long it took, to a tenth of a second when that was short."""
        if self.seconds < 10:
            return f"{self.seconds:.1f} s"
        return duration(self.seconds)


def translate_editor_text(
    text: str, settings: Settings, context: JobContext
) -> TextTranslation:
    """Translate the editor's text, detecting its language if asked to.

    The engine is loaded first if it is not, with a message that says so:
    that is what takes long the first time.

    Args:
        text: The text.
        settings: The settings it was sent with.
        context: The job's context.

    Returns:
        The translation.

    Raises:
        TranslationFailedError: If the language cannot be told, is the
            target already, or the translation fails.
    """
    started = time.monotonic()
    source, confidence = settings.source, None
    if not source:
        context.report(Progress("Detecting the language", None))
        detected = service.detect_language(text)
        source, confidence = _detected_source(detected, settings.target)
    if not service.supports_pair(settings.backend, source, settings.target):
        raise TranslationFailedError(
            f"OPUS-MT has no model from {service.language_name(source)} "
            f"into {service.language_name(settings.target)}. Choose "
            "another engine."
        )
    options = settings.options(source)
    notices: list[str] = []

    def notice(message: str) -> None:
        if message not in notices:
            notices.append(message)

    try:
        _load(options, settings, context)
        translation = service.translate_text(
            text,
            options,
            on_progress=context.report,
            on_notice=notice,
            cancel=context.cancel,
        )
    except CancelledError:
        raise
    except Exception as error:
        raise TranslationFailedError(
            service.describe_error(error, settings.backend)
        ) from error
    return TextTranslation(
        text=translation,
        source=source,
        target=settings.target,
        detected=confidence,
        engine=settings.engine,
        glossary_terms=len(settings.glossary),
        notices=tuple(notices),
        seconds=time.monotonic() - started,
    )


def _detected_source(
    detected: service.DetectionResult | None, target: str
) -> tuple[str, float]:
    """Return the detected language's code and confidence, or refuse it."""
    if detected is None or not detected.flores_code:
        raise TranslationFailedError(
            "The language of the text could not be told. Choose it under From."
        )
    name = detected.display_name or detected.iso639_1
    if not detected.is_confident:
        raise TranslationFailedError(
            f"The text may be in {name}, but that is not certain "
            f"({detected.confidence:.0%}). Choose its language under From."
        )
    if detected.flores_code == target:
        raise TranslationFailedError(
            f"The text is in {name} already. Choose another language under To."
        )
    return detected.flores_code, detected.confidence


def _load(
    options: service.TranslationOptions,
    settings: Settings,
    context: JobContext,
) -> None:
    """Load the engine for ``options`` if it is not loaded, saying so."""
    if service.backend_ready(options):
        return
    context.report(
        Progress(
            f"Loading {settings.engine}. The first translation with an "
            "engine loads it, which can take a minute or two.",
            None,
        )
    )
    service.load_backend(options)
    context.check_cancelled()


# ---------------------------------------------------------------------------
# Documents: the jobs
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class OutputFile:
    """One output file of a document, offered for download.

    Attributes:
        name: The download's name.
        kind: What it is, such as "Word" or "Side by side, HTML".
        size: Its size in bytes.
    """

    name: str
    kind: str
    size: int


@dataclasses.dataclass(frozen=True)
class Document:
    """One translated (or failed) document of a job.

    Attributes:
        name: The file's name, as uploaded or inside its ZIP archive.
        language: Its language, as found: "German (detected)".
        files: Its output files.
        error: Why it could not be translated; empty if it was.
        blocked: PDF outputs that could not be made, with the reason.
        warnings: Remarks from the checks of a PDF.
        pairs: Source and translation, passage by passage, for comparing;
            at most :data:`COMPARE_LIMIT`.
        passages: How many passages it has in all.
    """

    name: str
    language: str
    files: tuple[OutputFile, ...] = ()
    error: str = ""
    blocked: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    pairs: tuple[tuple[str, str], ...] = ()
    passages: int = 0


@dataclasses.dataclass(frozen=True)
class Translations:
    """The result of a documents job.

    Attributes:
        documents: Each document, in order.
        remarks: What was left out of the uploads, such as unsupported
            files in a ZIP archive.
        source: The source language's name, or ``""`` if each document's
            was detected.
        target: The target language's name.
        engine: The engine's short name.
        glossary_terms: How many glossary terms were forced.
    """

    documents: tuple[Document, ...]
    remarks: tuple[str, ...]
    source: str
    target: str
    engine: str
    glossary_terms: int

    @property
    def translated(self) -> int:
        """How many documents were translated."""
        return sum(1 for document in self.documents if not document.error)


def translate_files(
    paths: list[Path], settings: Settings, context: JobContext
) -> Translations:
    """Translate uploaded documents and offer their outputs.

    Args:
        paths: The uploads, in the job's inputs: documents and ZIP
            archives.
        settings: The settings they were sent with.
        context: The job's context.

    Returns:
        The documents' results.

    Raises:
        TranslationFailedError: If there is nothing to translate, or no
            document could be translated.
    """
    context.report(Progress("Reading the uploads", None))
    unpacked = service.unpack_uploads(
        (path.name, path.read_bytes()) for path in paths
    )
    remarks = [
        f"{name} was left out: Translation does not read this type."
        for name in unpacked.skipped
    ] + [f"{name} is not a ZIP archive." for name in unpacked.invalid]
    if not unpacked.files:
        raise TranslationFailedError(
            "There is no document to translate in the upload. "
            + " ".join(remarks)
        )
    options = settings.options()
    if settings.source or settings.backend != "opus-mt":
        # OPUS-MT's model depends on each document's language.
        try:
            _load(options, settings, context)
        except CancelledError:
            raise
        except Exception as error:
            raise TranslationFailedError(
                service.describe_error(error, settings.backend)
            ) from error

    results = service.translate_documents(
        unpacked.files,
        options,
        service.DocumentOptions(
            pdf_outputs=settings.pdf_outputs,
            math_ocr=settings.math_ocr,
            detect_source=not settings.source,
            review=settings.review,
        ),
        on_progress=context.report,
        cancel=context.cancel,
    )
    context.report(Progress("Saving the translations", 1.0))
    documents = _offer_outputs(results, unpacked.files, context)
    if not results.file_outputs:
        reasons = " ".join(
            f"{document.name}: {document.error}" for document in documents
        )
        raise TranslationFailedError(
            "No document could be translated. " + reasons
        )
    return Translations(
        documents=tuple(documents),
        remarks=tuple(remarks),
        source=service.language_name(settings.source)
        if settings.source
        else "",
        target=service.language_name(settings.target),
        engine=settings.engine,
        glossary_terms=len(settings.glossary),
    )


def _offer_outputs(
    results: service.DocumentResults,
    files: list[service.NamedFile],
    context: JobContext,
) -> list[Document]:
    """Save the outputs, offer them, and describe each document."""
    used: set[str] = set()
    outputs = dict(results.file_outputs)
    errors = dict(results.errors)
    reports = dict(results.pdf_reports)
    several = len(results.outputs) > 1
    documents = []
    for name, _ in files:
        offered = []
        for file_name, data in outputs.get(name, []):
            unique = _unique(file_name, used)
            path = context.outputs / unique
            path.write_bytes(data)
            kind = file_kind(file_name)
            context.offer(
                Download(
                    unique,
                    kind,
                    service.mime_type(file_name),
                    path=path,
                    listed=not several,
                )
            )
            offered.append(OutputFile(unique, kind, len(data)))
        report = reports.get(name)
        pairs = results.pairs.get(name, [])
        documents.append(
            Document(
                name=name,
                language=results.languages.get(name, ""),
                files=tuple(offered),
                error="" if name in outputs else errors.get(name, ""),
                blocked=tuple(
                    f"{issue['output']}: "
                    + issue.get("message", issue["reason"])
                    for issue in (report.blocked if report else [])
                ),
                warnings=tuple(report.warnings if report else ()),
                pairs=tuple(pairs[:COMPARE_LIMIT]),
                passages=len(pairs),
            )
        )
    if several:
        archive = f"translated_{results.target_code}.zip"
        path = context.outputs / _unique(archive, used)
        path.write_bytes(
            service.outputs_zip(results.file_outputs, results.errors)
        )
        context.offer(
            Download(path.name, "All files", "application/zip", path=path)
        )
    return documents


def _unique(name: str, used: set[str]) -> str:
    """Return ``name``, or a numbered variant if a file has it already."""
    candidate, number = name, 1
    while candidate.casefold() in used:
        number += 1
        candidate = f"{number}_{name}"
    used.add(candidate.casefold())
    return candidate


def file_kind(name: str) -> str:
    """Return what an output file is, from the end of its name.

    Args:
        name: The file's name.

    Returns:
        For example "Word" or "Side by side, HTML"; "File" for others.
    """
    lower = name.lower()
    for ending, kind in FILE_KINDS:
        if lower.endswith(ending):
            return kind
    return "File"
