"""The OCR page: text, tables and figures from scans, photos and PDFs.

One page for the OCR tools, chosen with a switcher; each has its own form
and route, and their jobs share the list below. Built so far:

- **Standard**, the default: one PDF or image, or a ZIP of them read as
  a batch, by the automatic pipeline (``features.ocr.service``): pages
  with a usable
  text layer are read from it, the others by the recognition model. The
  result shows each page beside its text, with the regions outlined on the
  page; the tables, the figures and the models used; and every download.
  A batch gives a ZIP with a folder per file, and an overview.

The manual engines, historic handwriting and Hugging Face models follow.
Survey extraction is left out of the web UI (see the web UI plan).
"""

from __future__ import annotations

import dataclasses
from functools import partial
from pathlib import Path, PurePosixPath

from fastapi import Request
from fastapi.responses import HTMLResponse
from markupsafe import Markup
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import FormData

from textlab.common import gpu_manager
from textlab.common.progress import Progress
from textlab.features.ocr import service
from textlab.ui.web import uploads
from textlab.ui.web.auth import Checked, protected_router
from textlab.ui.web.jobs import Download, JobContext, render_card, runner_of
from textlab.ui.web.render import renderer

router = protected_router()

#: The page's id, which also names its jobs.
PAGE = "ocr"

#: The OCR tools: their switcher value, name, and whether they are built.
OCR_TOOLS = (
    ("standard", "Standard", True),
    ("engine", "Choose an engine", False),
    ("handwriting", "Historic handwriting", False),
    ("huggingface", "Hugging Face model", False),
)

#: What the standard tool reads: documents and images, or a ZIP.
STANDARD_EXTENSIONS = (*service.INPUT_EXTENSIONS, ".zip")

#: The searchable PDF's language value when it is detected per page.
DETECT = "auto"

#: The region types: their name, and how a count of them reads.
REGION_LABELS = {
    "text": ("Text", "text block", "text blocks"),
    "title": ("Title", "title", "titles"),
    "table": ("Table", "table", "tables"),
    "figure": ("Figure", "figure", "figures"),
    "formula": ("Formula", "formula", "formulas"),
    "footnote": ("Footnote", "footnote", "footnotes"),
    "header": ("Header", "header", "headers"),
    "footer": ("Footer", "footer", "footers"),
    "seal": ("Seal", "seal", "seals"),
    "checkbox": ("Check box", "check box", "check boxes"),
    "list": ("List", "list", "lists"),
    "reference": ("Reference", "reference", "references"),
    "other": ("Other", "other region", "other regions"),
}


def region_label(region_type: str) -> str:
    """Return a region type's name.

    Args:
        region_type: The type, such as ``"table"``.

    Returns:
        For example "Table".
    """
    if region_type in REGION_LABELS:
        return REGION_LABELS[region_type][0]
    return region_type.capitalize()


def region_count(region_type: str, count: int) -> str:
    """Return how many regions of a type there are, for people.

    Args:
        region_type: The type, such as ``"text"``.
        count: How many there are.

    Returns:
        For example "12 text blocks" or "1 table".
    """
    _, one, many = REGION_LABELS.get(
        region_type, ("", region_type, region_type)
    )
    return f"{count} {one if count == 1 else many}"


def region_counts(counts: dict[str, int]) -> list[str]:
    """Return counts of regions, the most frequent first.

    Args:
        counts: How many regions there are of each type.

    Returns:
        For example ``["12 text blocks", "2 tables"]``.
    """
    return [
        region_count(kind, count)
        for kind, count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/ocr")
def page(request: Request) -> HTMLResponse:
    """Show the OCR tools and the session's OCR jobs.

    Args:
        request: The current request.

    Returns:
        The page.
    """
    jobs = runner_of(request).jobs(PAGE)
    return renderer(request).response(
        request,
        "ocr/page.html",
        page_id=PAGE,
        ocr_tools=OCR_TOOLS,
        accept=",".join(STANDARD_EXTENSIONS),
        formats=" ".join(
            sorted(
                {ext.lstrip(".").upper() for ext in service.INPUT_EXTENSIONS}
                - {"JPEG", "TIF"}
            )
        )
        + ", or a ZIP of them",
        languages=sorted(service.TESSERACT_LANGUAGES.items()),
        cards=Markup("".join(render_card(request, job) for job in jobs[::-1])),
    )


@router.post("/ocr/standard")
async def standard(request: Request, session: Checked) -> HTMLResponse:
    """Read a document, an image or a ZIP of them with the standard tool.

    Args:
        request: The form: ``file``, and the options ``searchable_pdf``,
            ``ocr_lang``, ``highest_quality`` and ``describe_images``.
        session: The session, with its CSRF token checked.

    Returns:
        The new job's card.

    Raises:
        UploadError: If the file is missing or of another type, or an
            option is not valid.
    """
    form = await uploads.read_form(request, max_files=1)
    upload = uploads.single_file(form, "file", suffixes=STANDARD_EXTENSIONS)
    options = read_options(form)
    name = uploads.display_name(upload)
    batch = PurePosixPath(name).suffix.lower() == ".zip"
    runner = runner_of(request)
    job = runner.create(
        title=name,
        page=PAGE,
        gpu=gpu_manager.OCR,
        result_template="ocr/batch_result.html"
        if batch
        else "ocr/result.html",
    )
    try:
        path = await run_in_threadpool(uploads.save, upload, job.inputs)
    except BaseException:
        runner.discard(job)
        raise
    work = read_archive if batch else read_document
    runner.submit(job, partial(work, path, name, options))
    return HTMLResponse(render_card(request, job))


def read_options(form: FormData) -> service.OcrOptions:
    """Read the automatic pipeline's options from the form.

    Args:
        form: The form.

    Returns:
        The options; survey extraction is never asked for.

    Raises:
        UploadError: If the language is not one offered.
    """
    searchable = bool(form.get("searchable_pdf"))
    language = service.DEFAULT_TESSERACT_LANG
    if searchable:
        language = str(form.get("ocr_lang") or DETECT)
        if language != DETECT and (
            language not in service.TESSERACT_LANGUAGES.values()
        ):
            raise uploads.UploadError(
                "Choose the document's language from the list."
            )
    return service.OcrOptions(
        native_fast_lane=not form.get("highest_quality"),
        describe_images=bool(form.get("describe_images")),
        searchable_pdf=searchable,
        ocr_lang=language,
    )


# ---------------------------------------------------------------------------
# The jobs' work
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Recognized:
    """A document read by the automatic pipeline: what its result shows.

    Attributes:
        document: The saved document: pages, regions, tables, figures.
        options: The options it was read with.
    """

    document: service.SavedDocument
    options: service.OcrOptions

    @property
    def counts(self) -> list[str]:
        """The regions found, most frequent first: "12 text blocks"."""
        return region_counts(self.document.summary.get("region_counts", {}))

    @property
    def notices(self) -> list[str]:
        """What to check in the result."""
        summary = self.document.summary
        found = []
        if summary.get("n_uncertain_marks"):
            count = summary["n_uncertain_marks"]
            found.append(
                f"{count} check box{'es' if count != 1 else ''} or mark"
                f"{'s' if count != 1 else ''} could not be read with "
                "certainty."
            )
        if summary.get("n_markup_disagreements"):
            count = summary["n_markup_disagreements"]
            found.append(
                f"For {count} mark{'s' if count != 1 else ''}, the text "
                "recognition and the shape of the mark disagree; the text "
                "was kept."
            )
        return found

    def label(self, region_type: str) -> str:
        """Return the name of a region type.

        Args:
            region_type: The type, such as ``"title"``.

        Returns:
            Its name, such as "Title".
        """
        return region_label(region_type)

    def types(self, saved_page: service.SavedPage) -> list[str]:
        """Return the region types outlined on a page, for its legend.

        Args:
            saved_page: The page.

        Returns:
            The types, in the order of :data:`REGION_LABELS`.
        """
        present = {region.type for region in saved_page.regions if region.box}
        return [kind for kind in REGION_LABELS if kind in present]


@dataclasses.dataclass(frozen=True)
class ReadArchive:
    """A ZIP read by the automatic pipeline: an overview of its files.

    Attributes:
        files: Each file's overview.
        seconds: How long the batch took.
        options: The options it was read with.
    """

    files: tuple[service.BatchFile, ...]
    seconds: float
    options: service.OcrOptions

    @property
    def pages(self) -> int:
        """How many pages the files have in all."""
        return sum(item.pages for item in self.files)

    def counts(self, item: service.BatchFile) -> str:
        """Return a file's regions, most frequent first.

        Args:
            item: The file.

        Returns:
            For example "12 text blocks, 2 tables".
        """
        return ", ".join(region_counts(item.region_counts))


def read_document(
    path: Path,
    name: str,
    options: service.OcrOptions,
    context: JobContext,
) -> Recognized:
    """Read one PDF or image and offer every output.

    Args:
        path: The upload, in the job's inputs.
        name: Its name as uploaded.
        options: How to read it.
        context: The job's context.

    Returns:
        The document to show.
    """
    context.report(Progress("Starting", 0.0))
    result = service.recognize_document(
        name,
        path.read_bytes(),
        options,
        on_progress=context.report,
        cancel=context.cancel,
    )
    context.report(Progress("Saving the results", 1.0))
    saved = service.save_document(result, context.outputs)
    for item in saved.files:
        context.offer(
            Download(
                item.name,
                item.label,
                item.media_type,
                path=item.path,
                listed=item.listed,
            )
        )
    return Recognized(document=saved, options=options)


def read_archive(
    path: Path,
    name: str,
    options: service.OcrOptions,
    context: JobContext,
) -> ReadArchive:
    """Read every document and image in a ZIP, and offer the result ZIP.

    Args:
        path: The ZIP, in the job's inputs.
        name: Its name as uploaded.
        options: How to read the files.
        context: The job's context.

    Returns:
        The overview of the files.
    """
    context.report(Progress("Unpacking the archive", 0.0))
    result = service.recognize_batch(
        path, options, on_progress=context.report, cancel=context.cancel
    )
    target = context.outputs / f"{PurePosixPath(name).stem}_ocr.zip"
    target.write_bytes(result.zip_bytes)
    context.offer(
        Download(target.name, "All results", "application/zip", path=target)
    )
    return ReadArchive(
        files=tuple(result.files), seconds=result.elapsed, options=options
    )
