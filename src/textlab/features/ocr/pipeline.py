"""The automatic OCR pipeline: one PDF or image in, one document out.

:func:`process_document` turns a PDF or image into a typed
:class:`~textlab.features.ocr.doc_ir.Document`, choosing the route per page:

* **Born-digital lane** (:mod:`.native`): PDF pages with a real, usable text
  layer are read directly with PyMuPDF. No vision model, no rasterization.
* **PaddleOCR-VL lane** (:mod:`.vl_session`): scanned pages and images are
  rasterized and recognized by the PaddleOCR-VL worker. Its layout blocks
  become regions; checkbox regions are classified (:mod:`.marks`).

Optional steps: figure descriptions by a local vision model
(:mod:`.vision_enrich`), survey response extraction
(:mod:`textlab.features.survey.form_extract`) and a searchable PDF
(:mod:`.searchable_pdf`).
"""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path

from textlab.features.ocr import doc_ir, vision_enrich
from textlab.features.ocr.marks import apply_markup, write_mark_debug_overlay
from textlab.features.ocr.native import (
    native_page,
    page_has_math,
    page_has_text_layer,
    text_layer_is_corrupt,
)
from textlab.features.ocr.rasters import decode_bgr, downscale_png_b64
from textlab.features.ocr.vl_session import VLWorkerSession, run_vl_worker
from textlab.features.survey import form_extract

#: Rasterization DPI of the PaddleOCR-VL lane.
VL_DPI = 200
#: Rasterization DPI when survey responses are extracted.
SURVEY_DPI = 300

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

#: Progress callback: ``(fraction, text)``.
ProgressFn = Callable[[float, str], None]


def _emit(progress: ProgressFn | None, frac: float, text: str):
    if progress is not None:
        try:
            progress(max(0.0, min(1.0, frac)), text)
        except Exception:
            pass


def _finalize_vl_page(
    page_json: dict,
    page_number: int,
    raster_path: Path | None,
    debug_dir: Path | None = None,
    *,
    extract_survey: bool = False,
    vision_client=None,
    same_layout_template=None,
    survey_contract=None,
    searchable_pdf: bool = False,
    ocr_lang: str = "eng",
) -> doc_ir.Page:
    page = doc_ir.from_paddle_vl(page_json)
    page.page_number = page_number
    for region in page.regions:
        region.id = f"p{page_number}_r{region.id.split('_r')[-1]}"

    raster_bytes = None
    page_bgr = None
    if raster_path and Path(raster_path).exists():
        raster_bytes = Path(raster_path).read_bytes()
        page_bgr = decode_bgr(raster_bytes)

    # Geometry analysis uses full-resolution crops and boxes.
    debug_collect = [] if debug_dir is not None else None
    if extract_survey or debug_dir is not None:
        apply_markup(page, page_bgr, debug_collect=debug_collect)
    if extract_survey:
        if vision_client is None:
            raise RuntimeError(
                "Survey extraction requires a configured vision client"
            )
        form_extract.extract_page_forms(
            page,
            page_bgr,
            vision_client,
            same_layout_template=same_layout_template,
            contract=survey_contract,
        )
    if searchable_pdf and page_bgr is not None:
        from textlab.features.ocr import searchable_pdf as _searchable_pdf

        lang = ocr_lang
        if lang == "auto":
            lang = _searchable_pdf.page_language(page)
        page.text_layer = _searchable_pdf.page_text_layer(
            page, page_bgr, lang=lang
        )
        page.raster_size = (page_bgr.shape[1], page_bgr.shape[0])
        page.text_layer_engine = _searchable_pdf.engine_citation(lang)
    if debug_dir is not None:
        write_mark_debug_overlay(
            page_bgr,
            page,
            debug_collect,
            Path(debug_dir) / f"page_{page_number}_marks.png",
        )
    if raster_bytes:
        page.image_b64 = downscale_png_b64(
            raster_bytes, page.regions, form_groups=page.form_groups
        )
    return page


def process_document(
    input_path,
    workspace_dir,
    *,
    backend_python: str | None = None,
    worker_path: Path | None = None,
    native_fast_lane: bool = True,
    progress: ProgressFn | None = None,
    source_name: str | None = None,
    debug_dir=None,
    describe_images: bool = False,
    extract_survey: bool = False,
    vision_client=None,
    same_layout_template=None,
    survey_contract=None,
    searchable_pdf: bool = False,
    ocr_lang: str = "eng",
    vl_session: VLWorkerSession | None = None,
) -> doc_ir.Document:
    """Parse one document into a typed IR, choosing the route per page.

    Args:
        input_path: A PDF or image file.
        workspace_dir: A directory for page images, owned and removed by the
            caller.
        backend_python: The PaddleOCR-VL environment's interpreter, if not
            the image's.
        worker_path: A worker script to run instead of the worker module;
            for tests.
        native_fast_lane: Read pages with a usable text layer directly;
            when False, every page is sent to PaddleOCR-VL.
        progress: Receives ``(fraction, text)`` progress updates.
        source_name: The document's name in the result; defaults to the
            file name.
        debug_dir: When set, a ``page_N_marks.png`` overlay per page shows
            every located mark colored by state, for validation.
        describe_images: Run the figure-description enrichment.
        extract_survey: Run question-level response extraction. Survey pages
            use the 300-DPI Paddle lane even when they have a text layer.
        vision_client: A shared vision-model client; batch callers pass one
            to keep the model loaded across files.
        same_layout_template: A template shared by a batch. The first
            document supplies normalized crop locations; later documents
            still have structure and responses read from their own pixels.
        survey_contract: A schema-free or repaired Paddle-ID contract name;
            defaults to ``TEXTLAB_SURVEY_CONTRACT`` or the schema-free one.
        searchable_pdf: Build a PDF of the original pages with an invisible
            text layer. Tesseract places the words; the text is still the
            VL lane's. Born-digital pages keep the text layer they already
            have, because the original PDF is reused as the carrier.
        ocr_lang: Tesseract language for the word positions, or ``"auto"``
            to detect it per page from the text the VL lane extracted.
        vl_session: A resident PaddleOCR-VL worker; batch callers pass one
            so the weights load once for the batch, not once per file.

    Returns:
        The document, with pages in their original order.
    """
    input_path = Path(input_path)
    workspace_dir = Path(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    raster_dir = workspace_dir / "rasters"
    raster_dir.mkdir(parents=True, exist_ok=True)
    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

    document = doc_ir.Document(source_name=source_name or input_path.name)
    suffix = input_path.suffix.lower()

    owned_client = False

    def _vision_client():
        nonlocal vision_client, owned_client
        if vision_client is None:
            vision_client = vision_enrich.OllamaVisionClient()
            owned_client = True
        return vision_client

    # ---- standalone image: always the VL lane -------------------------------
    if suffix in IMAGE_EXTS:
        _emit(progress, 0.1, "Analysing image...")

        def _image_stage(done: int, total: int):
            _emit(
                progress,
                0.15 if done == 0 else 0.7,
                "Loading the recognition model..."
                if done == 0
                else "Reading the image...",
            )

        pages_json = run_vl_worker(
            [input_path],
            backend_python=backend_python,
            worker_path=worker_path,
            on_page=_image_stage,
            session=vl_session,
        )
        try:
            if pages_json:
                page = _finalize_vl_page(
                    pages_json[0],
                    1,
                    input_path,
                    debug_dir=debug_dir,
                    extract_survey=extract_survey,
                    vision_client=_vision_client() if extract_survey else None,
                    same_layout_template=same_layout_template,
                    survey_contract=survey_contract,
                    searchable_pdf=searchable_pdf,
                    ocr_lang=ocr_lang,
                )
                document.pages.append(page)
                if describe_images:
                    _emit(
                        progress,
                        0.85,
                        "Describing figures and images "
                        "(the vision model may take a minute to load)...",
                    )
                    vision_enrich.describe_page_figures(page, _vision_client())
                if searchable_pdf:
                    _emit(progress, 0.95, "Building searchable PDF...")
                    document.searchable_pdf = _build_searchable_pdf(
                        document, input_path
                    )
            _emit(progress, 1.0, "Done")
            return document
        finally:
            # Keep the model warm until its configured expiry.
            if owned_client and vision_client is not None:
                vision_client.close()

    if suffix != ".pdf":
        raise RuntimeError(f"Unsupported input type: {suffix}")

    # ---- PDF: decide route per page -----------------------------------------
    import fitz  # PyMuPDF

    doc = fitz.open(str(input_path))
    n_pages = doc.page_count
    native_pages: dict[int, doc_ir.Page] = {}
    vl_jobs: list[tuple[int, Path]] = []  # (page_number, raster_path)

    for i in range(n_pages):
        page_number = i + 1
        _emit(
            progress,
            0.05 + 0.35 * (i / max(1, n_pages)),
            f"Routing page {page_number}/{n_pages}...",
        )
        fitz_page = doc.load_page(i)
        if (
            not extract_survey
            and native_fast_lane
            and page_has_text_layer(fitz_page)
        ):
            # Having a text layer is not enough: it also has to be usable. A
            # page that fails either check is worse than one with no text layer
            # at all, because the fast lane would "succeed" and hand back
            # silently wrong characters, so it goes to the VL model whatever
            # the caller asked for.
            if page_has_math(fitz_page):
                unusable = "equations extract as junk text"
            elif text_layer_is_corrupt(fitz_page):
                unusable = "text layer is mis-encoded"
            else:
                native_pages[page_number] = native_page(
                    fitz_page, page_number, debug_dir=debug_dir
                )
                continue
            print(
                f"[ocr] page {page_number}: {unusable}, "
                "using the VL model instead of the born-digital fast lane",
                flush=True,
            )
        raster_path = raster_dir / f"page_{page_number:04d}.png"
        pix = fitz_page.get_pixmap(
            dpi=SURVEY_DPI if extract_survey else VL_DPI
        )
        raster_path.write_bytes(pix.tobytes("png"))
        vl_jobs.append((page_number, raster_path))

    # ---- run the VL lane in one batch ---------------------------------------
    vl_pages: dict[int, doc_ir.Page] = {}
    if vl_jobs:
        _emit(
            progress,
            0.45,
            f"Running PaddleOCR-VL on {len(vl_jobs)} page(s)...",
        )

        def _vl_page_done(done: int, total: int):
            # 0.45 -> 0.65 across the pages; recognition is ~25-50 s each.
            _emit(
                progress,
                0.45 + 0.2 * (done / max(1, total)),
                f"Running PaddleOCR-VL: page {min(done + 1, total)} of "
                f"{total}..."
                if done < total
                else f"Recognised {total} page(s)...",
            )

        pages_json = run_vl_worker(
            [p for _, p in vl_jobs],
            backend_python=backend_python,
            worker_path=worker_path,
            on_page=_vl_page_done,
            session=vl_session,
        )
    try:
        for idx, page_json in enumerate(pages_json if vl_jobs else []):
            page_number, raster_path = vl_jobs[idx]
            # Finalising also runs the word-geometry pass, seconds per page.
            _emit(
                progress,
                0.65 + 0.2 * (idx / max(1, len(vl_jobs))),
                f"Extracting survey responses from page {page_number}..."
                if extract_survey
                else f"Assembling page {page_number} of {n_pages}...",
            )
            vl_pages[page_number] = _finalize_vl_page(
                page_json,
                page_number,
                raster_path,
                debug_dir=debug_dir,
                extract_survey=extract_survey,
                vision_client=_vision_client() if extract_survey else None,
                same_layout_template=same_layout_template,
                survey_contract=survey_contract,
                searchable_pdf=searchable_pdf,
                ocr_lang=ocr_lang,
            )

        # ---- reassemble in page order ---------------------------------------
        for page_number in range(1, n_pages + 1):
            if page_number in native_pages:
                document.pages.append(native_pages[page_number])
            elif page_number in vl_pages:
                document.pages.append(vl_pages[page_number])

        if describe_images:
            # Per page: the first call pays a ~60 s model load, each figure
            # more.
            client = _vision_client()
            n_pages_out = len(document.pages)
            for index, page in enumerate(document.pages, start=1):
                _emit(
                    progress,
                    0.85 + 0.05 * (index / max(1, n_pages_out)),
                    f"Describing figures on page {index} of {n_pages_out} "
                    "(the vision model may take a minute to load)...",
                )
                vision_enrich.describe_page_figures(page, client)

        if searchable_pdf:
            _emit(progress, 0.95, "Building searchable PDF...")
            document.searchable_pdf = _build_searchable_pdf(
                document,
                input_path,
                raster_dpi=SURVEY_DPI if extract_survey else VL_DPI,
            )

        _emit(progress, 1.0, "Done")
        return document
    finally:
        doc.close()
        # Kept warm; see the image lane above.
        if owned_client and vision_client is not None:
            vision_client.close()


def _build_searchable_pdf(
    document: doc_ir.Document, input_path, *, raster_dpi: int = VL_DPI
) -> bytes | None:
    """Assemble the searchable PDF from the per-page layers, then drop them.

    A PDF source is reused as the carrier, keeping scan quality and any
    existing text layer; an image becomes a one-page PDF.
    """
    from textlab.features.ocr import searchable_pdf as _searchable_pdf

    layers = {
        p.page_number: p.text_layer for p in document.pages if p.text_layer
    }
    page_sizes = {
        p.page_number: p.raster_size for p in document.pages if p.raster_size
    }
    input_path = Path(input_path)
    try:
        if input_path.suffix.lower() == ".pdf":
            blob = _searchable_pdf.build_searchable_pdf(
                layers,
                source_pdf=str(input_path),
                raster_dpi=raster_dpi,
                page_sizes=page_sizes,
            )
        else:
            from PIL import Image

            buf = io.BytesIO()
            Image.open(input_path).convert("RGB").save(buf, format="PNG")
            blob = _searchable_pdf.build_searchable_pdf(
                layers, rasters={1: buf.getvalue()}, page_sizes=page_sizes
            )
    except Exception:
        blob = None
    finally:
        # Raster coordinates are meaningless once the workspace is gone; the
        # engine name stays, for the provenance summary.
        engines = [
            p.text_layer_engine for p in document.pages if p.text_layer_engine
        ]
        if blob and engines:
            document.extra_tools["text_layer"] = engines[0]
        for page in document.pages:
            page.text_layer = None
            page.raster_size = None
    return blob


def document_summary(document: doc_ir.Document) -> dict:
    """Small stats bundle the UI shows above the tabs."""
    counts: dict = {}
    n_uncertain = 0
    n_marks = 0
    n_overridden = 0
    n_disagreements = 0
    n_form_groups = 0
    n_described_figures = 0
    routes = set()
    for page in document.pages:
        routes.add(page.source)
        n_form_groups += len(page.form_groups)
        for region in page.ordered_regions():
            counts[region.type] = counts.get(region.type, 0) + 1
            markup = region.markup or {}
            if region.type == doc_ir.CHECKBOX and markup:
                n_marks += 1
                if markup.get("state") == "uncertain":
                    n_uncertain += 1
                if markup.get("status") == "geometry_disagreement":
                    n_disagreements += 1
            elif markup.get("kind") == "glyph-marks":
                n_marks += len(markup.get("items", []))
                n_uncertain += markup.get("n_uncertain", 0)
                n_overridden += markup.get("n_overridden", 0)
                n_disagreements += markup.get("n_disagreements", 0)
                if markup.get("status") in (
                    "count_mismatch",
                    "geometry_disagreement",
                ):
                    n_uncertain += 1
            if region.visual_description:
                n_described_figures += 1
    return {
        "n_pages": len(document.pages),
        "region_counts": counts,
        "n_marks": n_marks,
        "n_uncertain_marks": n_uncertain,
        "n_overridden_marks": n_overridden,
        "n_markup_disagreements": n_disagreements,
        "n_form_groups": n_form_groups,
        "n_described_figures": n_described_figures,
        "routes": sorted(routes),
    }
