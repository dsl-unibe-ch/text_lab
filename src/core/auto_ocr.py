"""Automatic OCR orchestrator.

One entry point — :func:`process_document` — turns an uploaded PDF or image into
a single typed :class:`doc_ir.Document`, hiding the routing decision from the UI:

* **Born-digital fast lane** — PDF pages that already carry a real text layer are
  read directly with PyMuPDF (text + embedded images) and tagged
  ``source="native"``. No VLM, no rasterisation.
* **PaddleOCR-VL lane** — scanned pages (no text layer) and standalone images are
  rasterised at 200 DPI and sent to the ``paddle_vl_worker`` subprocess. The
  returned layout blocks become IR regions, and every ``checkbox`` region is run
  through :mod:`markup_detect`.

Runs in the ``text_lab_main`` conda env (Streamlit side); the VL worker runs in
the isolated ``paddle_backend`` env via subprocess.
"""

from __future__ import annotations

import base64
import io
import json
import os
import subprocess
from pathlib import Path
from typing import Callable, List, Optional, Tuple

try:  # normal import path (src on sys.path)
    from core import doc_ir, form_extract, markup_detect, vision_enrich
except ImportError:  # standalone / test import
    import doc_ir  # type: ignore
    import form_extract  # type: ignore
    import markup_detect  # type: ignore
    import vision_enrich  # type: ignore

# ---- tunables ---------------------------------------------------------------
VL_DPI = 200            # rasterisation DPI for the PaddleOCR-VL lane
SURVEY_DPI = 300        # high-resolution lane for explicitly requested forms
PREVIEW_DPI = 150       # rasterisation DPI for native-lane layout previews
MAX_PREVIEW_DIM = 1600  # cap the stored page raster (keeps session_state light)
MIN_WORDS_NATIVE = 6    # a page needs at least this many words to skip the VLM
MIN_CHARS_NATIVE = 20

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
RESULT_MARKER = "TEXTLAB_PADDLEVL_RESULT_JSON="

ProgressFn = Callable[[float, str], None]


def _default_backend_python() -> str:
    # The PaddleOCR-VL doc-parser lives in its own conda env so the legacy
    # PaddleOCR backend stays pinned; fall back to it only if VL is unset.
    return os.environ.get(
        "PADDLE_VL_BACKEND_PYTHON",
        os.environ.get("PADDLE_BACKEND_PYTHON", "/opt/conda/envs/paddle_vl_backend/bin/python"),
    )


def _worker_path() -> Path:
    return Path(__file__).resolve().parent / "paddle_vl_worker.py"


def _emit(progress: Optional[ProgressFn], frac: float, text: str):
    if progress is not None:
        try:
            progress(max(0.0, min(1.0, frac)), text)
        except Exception:
            pass


# ==========================================
#        PADDLEOCR-VL SUBPROCESS
# ==========================================


def run_vl_worker(
    image_paths: List[Path],
    *,
    backend_python: Optional[str] = None,
    worker_path: Optional[Path] = None,
    extra_labels: str = "",
) -> List[dict]:
    """Invoke the PaddleOCR-VL worker on *image_paths*; return per-page dicts."""
    if not image_paths:
        return []
    backend_python = backend_python or _default_backend_python()
    worker_path = worker_path or _worker_path()

    env = os.environ.copy()
    bin_dir = Path(backend_python).resolve().parent
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{bin_dir.parent / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
    env.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")
    env.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    cmd = [backend_python, str(worker_path), *[str(p) for p in image_paths]]
    if extra_labels:
        cmd += ["--extra-labels", extra_labels]

    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)
    if result.returncode != 0:
        raise RuntimeError(
            "PaddleOCR-VL backend failed.\n"
            f"stdout:\n{result.stdout[-4000:]}\n\nstderr:\n{result.stderr[-4000:]}"
        )
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(RESULT_MARKER):
            return json.loads(line[len(RESULT_MARKER):]).get("pages", [])
    raise RuntimeError(
        "PaddleOCR-VL backend did not return JSON.\n"
        f"stdout:\n{result.stdout[-4000:]}\n\nstderr:\n{result.stderr[-4000:]}"
    )


# ==========================================
#        IMAGE / PREVIEW HELPERS
# ==========================================


def _downscale_png_b64(
    png_bytes: bytes,
    regions: List["doc_ir.Region"],
    max_dim: int = MAX_PREVIEW_DIM,
    form_groups: Optional[List["doc_ir.FormGroup"]] = None,
) -> Optional[str]:
    """Encode a page raster as base64 PNG, capping its size and scaling bboxes."""
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(png_bytes))
        img.load()
    except Exception:
        return base64.b64encode(png_bytes).decode("ascii")
    w, h = img.size
    scale = min(1.0, max_dim / max(w, h)) if max(w, h) > 0 else 1.0
    if scale < 1.0:
        img = img.convert("RGB").resize((max(1, int(w * scale)), max(1, int(h * scale))))
        for region in regions:
            if region.bbox:
                region.bbox = [round(c * scale, 2) for c in region.bbox]
        for group in form_groups or []:
            if group.bbox:
                group.bbox = [round(c * scale, 2) for c in group.bbox]
            for row in group.rows:
                for option in row.options:
                    if option.bbox:
                        option.bbox = [round(c * scale, 2) for c in option.bbox]
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _apply_markup(page: "doc_ir.Page", page_bgr=None, debug_collect=None):
    """Detect and classify survey marks on *page*.

    Two paths:

    * ``checkbox``-typed regions: classified directly from their crop (as
      before, now with the stroke-aware geometric rule).
    * any other region whose *content* carries mark glyphs (``○``/``☐``/``☒``
      ... — the VL model usually folds survey rows into tables or text): the
      glyph states are compared with the actual ink in the region crop. A
      disagreement is retained as review evidence; OCR content is immutable.

    ``page_bgr`` must be in the same coordinate space as the region bboxes
    (i.e. the full-resolution raster, before the preview downscale). When
    *debug_collect* is a list, every located mark's absolute bbox/state is
    appended to it so a page overlay can be rendered for auditing.
    """
    for region in page.regions:
        if region.type == doc_ir.CHECKBOX:
            block_text = region.content.get("text") or region.content.get("markdown") or ""
            crop_b64 = (region.asset or {}).get("b64")
            crop = markup_detect.decode_crop_b64(crop_b64)
            if crop is None and page_bgr is not None and len(region.bbox) >= 4:
                crop = _crop_region(page_bgr, region.bbox)
            verdict = markup_detect.classify_checkbox(block_text, crop)
            region.markup = verdict
            if verdict.get("state") == "uncertain" and "unverified checkbox state" not in region.warnings:
                region.warnings.append("unverified checkbox state")
            if verdict.get("status") == "geometry_disagreement":
                region.warnings.append("checkbox OCR and geometry disagree")
            if debug_collect is not None and len(region.bbox) >= 4:
                debug_collect.append({
                    "bbox": [int(v) for v in region.bbox[:4]],
                    "state": verdict.get("state", "uncertain"),
                    "override": False,
                    "region_id": region.id,
                })
            continue

        # Glyph-borne marks inside tables / text
        source_text = (
            region.content.get("html")
            if region.type == doc_ir.TABLE
            else (region.content.get("text") or region.content.get("markdown") or "")
        ) or ""
        glyph_items = markup_detect.extract_mark_glyphs(source_text)
        if not glyph_items:
            continue

        geo_marks = []
        crop = None
        origin = (0, 0)
        if page_bgr is not None and len(region.bbox) >= 4:
            origin = (max(0, int(round(region.bbox[0]))), max(0, int(round(region.bbox[1]))))
            crop = _crop_region(page_bgr, region.bbox)
            if crop is not None:
                geo_marks = markup_detect.find_marks(crop, n_expected=len(glyph_items))

        items, status = markup_detect.reconcile_marks(glyph_items, geo_marks)
        counts = markup_detect.summarize_items(items)

        if debug_collect is not None:
            _collect_mark_debug(debug_collect, region, origin, geo_marks, items, status)

        # Evidence thumbnails for marks a human should be able to verify at a
        # glance: geometry disagreements and uncertain states.
        for item in items:
            geo_bbox = item.pop("geo_bbox", None)
            if crop is None or not geo_bbox:
                continue
            if item.get("needs_review") or item.get("state") == "uncertain":
                x1, y1, x2, y2 = [int(v) for v in geo_bbox]
                sub = crop[max(0, y1):y2, max(0, x1):x2]
                if sub.size:
                    item["crop_b64"] = _encode_png_b64(sub)

        region.markup = {
            "kind": "glyph-marks",
            "status": status,
            "items": [
                {
                    k: item.get(k)
                    for k in (
                        "glyph", "state", "method", "score", "geometry",
                        "needs_review", "crop_b64",
                    )
                }
                for item in items
            ],
            **counts,
        }
        if status == "count_mismatch":
            region.warnings.append("mark count mismatch between transcription and geometry")
        if status == "geometry_saturated":
            region.warnings.append("geometry found ink in every mark (non-discriminative); overrides skipped")
        if status == "geometry_disagreement":
            region.warnings.append("mark transcription and geometry disagree; OCR left unchanged")
        if counts["n_uncertain"]:
            region.warnings.append("uncertain mark state(s)")


def _crop_region(page_bgr, bbox):
    h, w = page_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox[:4]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return page_bgr[y1:y2, x1:x2]


# ==========================================
#        MARK DEBUG OVERLAY (validation)
# ==========================================

_DEBUG_STATE_COLORS = {  # BGR
    "checked": (0, 170, 0),
    "unchecked": (200, 130, 0),
    "uncertain": (0, 140, 255),
}


def _collect_mark_debug(debug_collect, region, origin, geo_marks, items, status):
    """Record every geometrically located mark, in absolute page coordinates."""
    ox, oy = origin
    # When reconciliation aligned marks 1:1, colour by the final (possibly
    # overridden) state; otherwise fall back to the raw geometric verdict.
    aligned = status == "matched" and len(items) == len(geo_marks)
    for i, mark in enumerate(geo_marks):
        gb = mark.get("bbox")
        if not gb:
            continue
        if aligned:
            state = items[i].get("state", mark["state"])
            override = items[i].get("method") == "geometric-override"
        else:
            state, override = mark["state"], False
        debug_collect.append({
            "bbox": [ox + int(gb[0]), oy + int(gb[1]), ox + int(gb[2]), oy + int(gb[3])],
            "state": state,
            "override": override,
            "region_id": region.id,
        })


def _write_mark_debug_overlay(page_bgr, page, debug_marks, out_path):
    """Draw region boxes + every located mark (coloured by state) and save."""
    try:
        import cv2
    except Exception:
        return
    if page_bgr is None:
        return
    vis = page_bgr.copy()
    for region in page.regions:
        if not (region.markup and len(region.bbox) >= 4):
            continue
        x1, y1, x2, y2 = [int(v) for v in region.bbox[:4]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (170, 170, 170), 1)
        cv2.putText(vis, region.id, (x1, max(10, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
    for mark in debug_marks:
        x1, y1, x2, y2 = mark["bbox"]
        color = _DEBUG_STATE_COLORS.get(mark["state"], (0, 140, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        if mark.get("override"):
            cv2.putText(vis, "OVR", (x1, max(9, y1 - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), vis)


# ==========================================
#        NATIVE (BORN-DIGITAL) LANE
# ==========================================


def _page_has_text_layer(page) -> bool:
    text = page.get_text("text") or ""
    if len(text.strip()) < MIN_CHARS_NATIVE:
        return False
    try:
        words = page.get_text("words")
    except Exception:
        words = []
    return len(words) >= MIN_WORDS_NATIVE


# Unicode blocks that signal mathematical notation in a text layer.
_MATH_CHAR_RANGES = (
    (0x2200, 0x22FF),  # mathematical operators
    (0x27C0, 0x27EF),  # misc mathematical symbols-A
    (0x2980, 0x29FF),  # misc mathematical symbols-B
    (0x2A00, 0x2AFF),  # supplemental mathematical operators
    (0x2070, 0x209F),  # super/subscripts
    (0x0391, 0x03C9),  # Greek (equation variables)
    (0x1D400, 0x1D7FF),  # mathematical alphanumeric symbols
)
_MATH_FONT_HINTS = ("cmmi", "cmsy", "cmex", "msam", "msbm", "math", "stix")
MATH_GLYPH_MIN = 8  # math chars on a page before it is routed to the VL lane


def _is_math_char(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _MATH_CHAR_RANGES)


def _page_has_math(page) -> bool:
    """Heuristic: does this born-digital page contain equations?

    PyMuPDF extracts equation glyphs as junk text, so pages with math must go
    through the VL lane even when they have a text layer. Signals: a minimum
    count of math-block Unicode chars, or spans set in known math fonts.
    """
    text = page.get_text("text") or ""
    n_math = sum(1 for ch in text if _is_math_char(ch))
    if n_math >= MATH_GLYPH_MIN:
        return True
    try:
        data = page.get_text("dict")
    except Exception:
        return False
    n_math_spans = 0
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                font = str(span.get("font", "")).lower()
                if any(hint in font for hint in _MATH_FONT_HINTS):
                    n_math_spans += 1
                    if n_math_spans >= 2:
                        return True
    return False


def _native_page(fitz_page, page_number: int, debug_dir: Optional[Path] = None) -> "doc_ir.Page":
    """Extract text + embedded images from a born-digital page into IR."""
    scale = PREVIEW_DPI / 72.0
    data = fitz_page.get_text("dict")
    regions: List[doc_ir.Region] = []
    order = 0
    for block in data.get("blocks", []):
        bbox = [c * scale for c in block.get("bbox", [0, 0, 0, 0])]
        if block.get("type") == 0:  # text block
            lines = []
            for line in block.get("lines", []):
                spans = [span.get("text", "") for span in line.get("spans", [])]
                joined = "".join(spans).strip()
                if joined:
                    lines.append(joined)
            text = "\n".join(lines).strip()
            if not text:
                continue
            regions.append(
                doc_ir.Region(
                    id=f"p{page_number}_r{order}",
                    type=doc_ir.TEXT,
                    bbox=bbox,
                    reading_order=order,
                    content={"text": text, "markdown": text},
                    confidence={"layout": 1.0, "ocr": None},
                    source="native",
                )
            )
            order += 1
        elif block.get("type") == 1:  # image block
            img_bytes = block.get("image")
            asset = None
            if img_bytes:
                asset = {
                    "b64": base64.b64encode(img_bytes).decode("ascii"),
                    "ext": (block.get("ext") or "png"),
                }
            regions.append(
                doc_ir.Region(
                    id=f"p{page_number}_r{order}",
                    type=doc_ir.FIGURE,
                    bbox=bbox,
                    reading_order=order,
                    content={"text": ""},
                    confidence={"layout": 1.0, "ocr": None},
                    asset=asset,
                    source="native",
                )
            )
            order += 1

    page = doc_ir.Page(
        page_number=page_number,
        regions=regions,
        source="native",
    )
    pix = fitz_page.get_pixmap(dpi=PREVIEW_DPI)
    raster_bytes = pix.tobytes("png")
    page_bgr = _decode_bgr(raster_bytes)
    # The geometric markup baseline is diagnostic only unless the explicit
    # survey enrichment is requested (survey pages use the VL lane below).
    debug_collect = [] if debug_dir is not None else None
    if debug_dir is not None:
        _apply_markup(page, page_bgr, debug_collect=debug_collect)
    if debug_dir is not None:
        _write_mark_debug_overlay(page_bgr, page, debug_collect,
                                  Path(debug_dir) / f"page_{page_number}_marks.png")
    page.image_b64 = _downscale_png_b64(raster_bytes, regions, form_groups=page.form_groups)
    page.width = fitz_page.rect.width * scale
    page.height = fitz_page.rect.height * scale
    return page


# ==========================================
#        MAIN ENTRY POINT
# ==========================================


def _finalize_vl_page(
    page_json: dict,
    page_number: int,
    raster_path: Optional[Path],
    debug_dir: Optional[Path] = None,
    *,
    extract_survey: bool = False,
    vision_client=None,
    same_layout_template=None,
    survey_contract=None,
    searchable_pdf: bool = False,
    ocr_lang: str = "eng",
) -> "doc_ir.Page":
    page = doc_ir.from_paddle_vl(page_json)
    page.page_number = page_number
    for region in page.regions:
        region.id = f"p{page_number}_r{region.id.split('_r')[-1]}"

    raster_bytes = None
    page_bgr = None
    if raster_path and Path(raster_path).exists():
        raster_bytes = Path(raster_path).read_bytes()
        page_bgr = _decode_bgr(raster_bytes)

    # Markup/form analysis needs full-resolution crops, so it runs BEFORE the
    # preview downscale rescales the region bboxes. Geometry is evidence only.
    debug_collect = [] if debug_dir is not None else None
    if extract_survey or debug_dir is not None:
        _apply_markup(page, page_bgr, debug_collect=debug_collect)
    if extract_survey:
        if vision_client is None:
            raise RuntimeError("Survey extraction requires a configured vision client")
        form_extract.extract_page_forms(
            page,
            page_bgr,
            vision_client,
            same_layout_template=same_layout_template,
            contract=survey_contract,
        )
    # The word-geometry pass also needs full-resolution boxes, so it runs before
    # the preview downscale rewrites Region.bbox into preview coordinates.
    if searchable_pdf and page_bgr is not None:
        from core import searchable_pdf as _searchable_pdf

        lang = ocr_lang
        if lang == "auto":
            lang = _searchable_pdf.page_language(page)
        page.text_layer = _searchable_pdf.page_text_layer(page, page_bgr, lang=lang)
        page.raster_size = (page_bgr.shape[1], page_bgr.shape[0])
    if debug_dir is not None:
        _write_mark_debug_overlay(page_bgr, page, debug_collect,
                                  Path(debug_dir) / f"page_{page_number}_marks.png")
    if raster_bytes:
        page.image_b64 = _downscale_png_b64(
            raster_bytes, page.regions, form_groups=page.form_groups
        )
    return page


def _encode_png_b64(img_bgr) -> Optional[str]:
    try:
        import cv2

        ok, enc = cv2.imencode(".png", img_bgr)
        return base64.b64encode(enc.tobytes()).decode("ascii") if ok else None
    except Exception:
        return None


def _decode_bgr(png_bytes: Optional[bytes]):
    if not png_bytes:
        return None
    try:
        import cv2
        import numpy as np

        arr = np.frombuffer(png_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def process_document(
    input_path,
    workspace_dir,
    *,
    backend_python: Optional[str] = None,
    worker_path: Optional[Path] = None,
    native_fast_lane: bool = True,
    progress: Optional[ProgressFn] = None,
    source_name: Optional[str] = None,
    debug_dir=None,
    describe_images: bool = False,
    extract_survey: bool = False,
    vision_client=None,
    same_layout_template=None,
    survey_contract=None,
    searchable_pdf: bool = False,
    ocr_lang: str = "eng",
) -> "doc_ir.Document":
    """Parse one document into a typed IR, choosing the route per page.

    Parameters
    ----------
    input_path : path to a PDF or image file.
    workspace_dir : a (caller-owned, self-cleaning) directory for temp rasters.
    native_fast_lane : when False, every page is sent to PaddleOCR-VL.
    describe_images : run the explicitly requested figure-description enricher.
    extract_survey : run targeted, question-level response extraction. Survey
        pages use the 300-DPI Paddle lane even when they have a text layer.
    vision_client : optional shared client; batch callers should pass one to
        keep the model loaded across files.
    same_layout_template : optional in-job template shared by a batch. The
        first document supplies normalized crop locations; subsequent documents
        still have their structure and response read independently from pixels.
    survey_contract : optional schema-free or repaired Paddle-ID contract name;
        omitted uses ``TEXTLAB_SURVEY_CONTRACT`` or the schema-free default.
    searchable_pdf : build a PDF of the original pages with an invisible text
        layer. Needs Tesseract for word geometry; the indexed text is still the
        VL lane's. Born-digital pages skipped by the VL lane keep the text layer
        they already have, because the original PDF is reused as the carrier.
    ocr_lang : Tesseract language code for the word-geometry pass, or ``"auto"``
        to detect it per page from the text the VL lane already extracted.
    progress : optional ``callback(fraction, text)`` for a Streamlit progress bar.
    debug_dir : when set, writes a ``page_N_marks.png`` overlay per page showing
        every located mark coloured by state (for validation/auditing only).
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
        pages_json = run_vl_worker(
            [input_path], backend_python=backend_python, worker_path=worker_path
        )
        try:
            if pages_json:
                page = _finalize_vl_page(
                    pages_json[0], 1, input_path, debug_dir=debug_dir,
                    extract_survey=extract_survey,
                    vision_client=_vision_client() if extract_survey else None,
                    same_layout_template=same_layout_template,
                    survey_contract=survey_contract,
                    searchable_pdf=searchable_pdf,
                    ocr_lang=ocr_lang,
                )
                document.pages.append(page)
                if describe_images:
                    vision_enrich.describe_page_figures(page, _vision_client())
                if searchable_pdf:
                    _emit(progress, 0.95, "Building searchable PDF...")
                    document.searchable_pdf = _build_searchable_pdf(document, input_path)
            _emit(progress, 1.0, "Done")
            return document
        finally:
            if owned_client and vision_client is not None:
                vision_client.close(unload_model=True)

    if suffix != ".pdf":
        raise RuntimeError(f"Unsupported input type: {suffix}")

    # ---- PDF: decide route per page -----------------------------------------
    import fitz  # PyMuPDF

    doc = fitz.open(str(input_path))
    n_pages = doc.page_count
    native_pages: dict[int, doc_ir.Page] = {}
    vl_jobs: List[Tuple[int, Path]] = []  # (page_number, raster_path)

    for i in range(n_pages):
        page_number = i + 1
        _emit(progress, 0.05 + 0.35 * (i / max(1, n_pages)), f"Routing page {page_number}/{n_pages}...")
        fitz_page = doc.load_page(i)
        if (
            not extract_survey
            and
            native_fast_lane
            and _page_has_text_layer(fitz_page)
            and not _page_has_math(fitz_page)
        ):
            native_pages[page_number] = _native_page(fitz_page, page_number, debug_dir=debug_dir)
        else:
            raster_path = raster_dir / f"page_{page_number:04d}.png"
            pix = fitz_page.get_pixmap(dpi=SURVEY_DPI if extract_survey else VL_DPI)
            raster_path.write_bytes(pix.tobytes("png"))
            vl_jobs.append((page_number, raster_path))

    # ---- run the VL lane in one batch ---------------------------------------
    vl_pages: dict[int, doc_ir.Page] = {}
    if vl_jobs:
        _emit(progress, 0.45, f"Running PaddleOCR-VL on {len(vl_jobs)} page(s)...")
        pages_json = run_vl_worker(
            [p for _, p in vl_jobs], backend_python=backend_python, worker_path=worker_path
        )
    try:
        for idx, page_json in enumerate(pages_json if vl_jobs else []):
            page_number, raster_path = vl_jobs[idx]
            if extract_survey:
                _emit(
                    progress,
                    0.65 + 0.2 * (idx / max(1, len(vl_jobs))),
                    f"Extracting survey responses from page {page_number}...",
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
            _emit(progress, 0.9, "Describing figures and images...")
            client = _vision_client()
            for page in document.pages:
                vision_enrich.describe_page_figures(page, client)

        if searchable_pdf:
            _emit(progress, 0.95, "Building searchable PDF...")
            document.searchable_pdf = _build_searchable_pdf(
                document, input_path, raster_dpi=SURVEY_DPI if extract_survey else VL_DPI
            )

        _emit(progress, 1.0, "Done")
        return document
    finally:
        doc.close()
        if owned_client and vision_client is not None:
            vision_client.close(unload_model=True)


def _build_searchable_pdf(
    document: "doc_ir.Document", input_path, *, raster_dpi: int = VL_DPI
) -> Optional[bytes]:
    """Assemble the searchable PDF from the per-page layers, then drop them.

    A PDF source is reused as the carrier so scan quality and any existing
    born-digital text layer survive; an image source becomes a one-page PDF.
    """
    from core import searchable_pdf as _searchable_pdf

    layers = {p.page_number: p.text_layer for p in document.pages if p.text_layer}
    page_sizes = {
        p.page_number: p.raster_size for p in document.pages if p.raster_size
    }
    input_path = Path(input_path)
    try:
        if input_path.suffix.lower() == ".pdf":
            blob = _searchable_pdf.build_searchable_pdf(
                layers, source_pdf=str(input_path), raster_dpi=raster_dpi,
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
        # Raster-space coordinates are meaningless once the job workspace is
        # cleaned up, so they are not carried into session state.
        for page in document.pages:
            page.text_layer = None
            page.raster_size = None
    return blob


# ==========================================
#        SUMMARY HELPERS (for the UI)
# ==========================================


def document_summary(document: "doc_ir.Document") -> dict:
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
                if markup.get("status") in ("count_mismatch", "geometry_disagreement"):
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
