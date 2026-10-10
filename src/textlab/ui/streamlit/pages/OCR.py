"""OCR page: text, tables and layout from PDFs and images.

The automatic pipeline (:mod:`textlab.features.ocr.service`) handles single
documents and ZIP batches; manual engine selection is in an expander below
(:mod:`textlab.ui.streamlit.ocr.manual`). Results and review are drawn by
the modules in :mod:`textlab.ui.streamlit.ocr`.
"""

import os

# Paddle and PyTorch crash in OpenBLAS with more than one thread.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

os.environ["VLLM_GPU_MEMORY_UTILIZATION"] = "0.6"
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault(
    "PADDLE_PDX_CACHE_HOME",
    os.environ.get("PADDLEX_HOME", os.path.expanduser("~/.paddlex")),
)
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.dirname(current_dir)
favicon_path = os.path.join(app_dir, "assets", "text_lab_logo.png")
favicon = Image.open(favicon_path)

st.set_page_config(page_title="OCR", page_icon=favicon, layout="wide")


from textlab.common import gpu_manager  # noqa: E402
from textlab.features.ocr import service  # noqa: E402
from textlab.ui.streamlit.auth import check_token  # noqa: E402
from textlab.ui.streamlit.components.gpu import free_gpu_for  # noqa: E402
from textlab.ui.streamlit.ocr.manual import (  # noqa: E402
    manual_engines_expander,
)
from textlab.ui.streamlit.ocr.results import (  # noqa: E402
    render_document_result,
)
from textlab.ui.streamlit.ocr.review import render_survey_review  # noqa: E402
from textlab.ui.streamlit.ocr.state import (  # noqa: E402
    SURVEY_EXTRACTION_UI_ENABLED,
    clear_results,
    init_state,
)

check_token()

st.title("Document & Image OCR")

AUTO_INPUT_TYPES = [extension[1:] for extension in service.INPUT_EXTENSIONS]

init_state()


# ==========================================
#      AUTOMATIC PIPELINE — UI
# ==========================================


def _searchable_pdf_language(key, enabled):
    """Tesseract language for the word-positioning pass.

    Only positions come from Tesseract, but the language still matters: on a
    scanned German questionnaire `deu` placed 87.6% of tokens on their exact
    box against 82.1% for `eng`. Mismatched tokens just highlight a wider span.
    """
    auto = "Detect automatically"
    names = [auto, *service.TESSERACT_LANGUAGES]
    # Greyed out rather than hidden, so ticking the box shifts nothing below
    # it.
    choice = st.selectbox(
        "Document language",
        names,
        index=0,
        key=key,
        disabled=not enabled,
        help=(
            "Detection reads the text Text Lab already extracted, per page, "
            "so "
            "mixed-language documents are handled page by page; pages with "
            "too "
            "little text keep the English default. Set this explicitly if a "
            "document is misdetected. Either way it only affects how "
            "precisely "
            "each word is located — the text itself always comes from Text "
            "Lab's OCR."
        ),
    )
    if not enabled:
        return service.DEFAULT_TESSERACT_LANG
    if choice == auto:
        return "auto"
    return service.TESSERACT_LANGUAGES[choice]


#: Rough per-option cost, shown before a job that may run for minutes.
_OPTION_COSTS = {
    "highest_quality": "every page through the vision model",
    "searchable_pdf": "a word-positioning pass per page",
    "describe_images": "one vision-model call per figure",
    "extract_survey": "one vision-model call per question",
    "survey_batch_mode": (
        "one pass to learn the form, then a second read per file"
    ),
}


def _parse_options(prefix, *, batch=False):
    """Shared parse-option panel for single and batch. Returns the settings."""
    options = {}
    with st.container(border=True):
        if batch:
            # A choice for batch, where the cost multiplies by the file count.
            c1, c2 = st.columns([1, 1])
            with c1:
                options["searchable_pdf"] = st.checkbox(
                    "Searchable PDF for each file",
                    value=True,
                    key=f"{prefix}_searchable_pdf",
                    help=(
                        "Adds `document_searchable.pdf` per file: the "
                        "original "
                        "pages with an invisible, selectable text layer."
                    ),
                )
        else:
            # Always on for a single document: seconds a page, and the most
            # useful output.
            options["searchable_pdf"] = True
            c2 = st.container()
        with c2:
            options["ocr_lang"] = _searchable_pdf_language(
                f"{prefix}_pdf_lang", options["searchable_pdf"]
            )

        st.caption("**Quality and AI analysis**")
        c3, c4 = st.columns([1, 1])
        with c3:
            options["highest_quality"] = st.checkbox(
                "Highest quality",
                value=False,
                key=f"{prefix}_hq",
                help=(
                    "Sends every page through PaddleOCR-VL, even pages that "
                    "already have a digital text layer. Best for equations, "
                    "forms and complex layouts. Pages with detected math are "
                    "routed to the vision model automatically either way."
                ),
            )
        with c4:
            options["describe_images"] = st.checkbox(
                "Describe figures",
                value=False,
                key=f"{prefix}_describe_images",
                help=(
                    "Adds a generated description and visible-text "
                    "transcription to each detected figure."
                ),
            )

        options["survey_batch_mode"] = False
        if batch:
            options["survey_batch_mode"] = st.checkbox(
                "Extract questionnaire responses",
                value=False,
                key=f"{prefix}_survey_batch",
                help=(
                    "For a batch of the **same** paper questionnaire filled "
                    "in by "
                    "different people. TextLab learns the blank form from the "
                    "batch itself, then reads every respondent against it and "
                    "adds a survey/ folder: one row per respondent, "
                    "TRUE/FALSE "
                    "per checkbox, and a certainty beside every answer. The "
                    "normal text extraction still runs for each file."
                ),
            )

        options["extract_survey"] = False
        options["same_template"] = False
        if SURVEY_EXTRACTION_UI_ENABLED:
            options["extract_survey"] = st.checkbox(
                "Extract survey/form responses (experimental)",
                value=False,
                key=f"{prefix}_extract_survey",
                help=(
                    "Question-level response extraction at 300 DPI. Original "
                    "OCR "
                    "text is preserved and every answer is flagged for review."
                ),
            )
            if batch and options["extract_survey"]:
                options["same_template"] = st.checkbox(
                    "All files use the same questionnaire layout",
                    value=False,
                    key=f"{prefix}_same_template",
                )

        # Only what was opted into: the searchable PDF is the baseline for a
        # single document, not a cost.
        chargeable = dict(_OPTION_COSTS)
        if not batch:
            chargeable.pop("searchable_pdf", None)
        enabled = [
            chargeable[name] for name in chargeable if options.get(name)
        ]
        if enabled:
            st.caption(f"Adds {'; '.join(enabled)}.")
        else:
            st.caption(
                "Fastest settings — scanned pages only go to the vision model."
            )
    return options


def _ocr_options(opts):
    """Turn the option panel's settings into service options."""
    return service.OcrOptions(
        native_fast_lane=not opts["highest_quality"],
        describe_images=opts["describe_images"],
        extract_survey=opts["extract_survey"],
        same_template=opts["same_template"],
        survey_batch_mode=opts["survey_batch_mode"],
        searchable_pdf=opts["searchable_pdf"],
        ocr_lang=opts["ocr_lang"],
    )


def _run_single(uploaded_file, options):
    """Recognize one uploaded file and keep the result in session state."""
    clear_results(reset_running=False)
    st.session_state.ocr_running = True
    progress_bar = st.progress(0.0, text="Parsing document...")

    def show(update):
        progress_bar.progress(update.fraction or 0.0, text=update.message)

    try:
        result = service.recognize_document(
            uploaded_file.name,
            uploaded_file.getvalue(),
            options,
            on_progress=show,
        )
        st.session_state.auto_document = result.document
        st.session_state.auto_summary = result.summary
        st.session_state.auto_downloads = result.downloads
        st.session_state.auto_complete = True
    except Exception as error:
        st.session_state.auto_error = f"Automatic OCR failed: {error}"
        st.exception(error)
    finally:
        progress_bar.empty()
        st.session_state.ocr_running = False


def _run_batch(batch_zip, options):
    """Recognize an uploaded ZIP archive and keep the result ZIP."""
    clear_results(reset_running=False)
    st.session_state.ocr_running = True
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def show(update):
        if update.fraction is not None:
            progress_bar.progress(update.fraction)
        status_text.text(update.message)

    try:
        result = service.recognize_batch(batch_zip, options, on_progress=show)
        if result.survey is not None and result.survey.warning:
            st.warning(result.survey.warning)
        st.session_state.batch_auto_elapsed = result.elapsed
        st.session_state.batch_auto_zip = result.zip_bytes
        st.session_state.batch_auto_complete = True
        if result.survey is not None:
            st.session_state.survey_batch = result.survey
    except Exception as error:
        st.session_state.auto_error = f"Batch automatic OCR failed: {error}"
        st.exception(error)
    finally:
        st.session_state.ocr_running = False


def auto_single_ui():
    """Single document: upload, options, run button and result."""
    st.markdown(
        "Upload a **PDF** or **image** and press **Parse document**. TextLab "
        "automatically detects layout, tables, figures and formulas. Optional "
        "AI analysis can describe detected figures and images."
    )
    uploaded_file = st.file_uploader(
        "Choose a PDF or image file",
        type=AUTO_INPUT_TYPES,
        on_change=clear_results,
        args=(True,),
        key="auto_single_upload",
    )
    if uploaded_file is not None:
        opts = _parse_options("auto_single")
        if st.session_state.ocr_running:
            st.warning(
                "A job is currently running. The button is disabled until "
                "completion."
            )
        if st.button(
            "Parse document",
            type="primary",
            disabled=st.session_state.ocr_running,
            key="auto_single_btn",
        ):
            free_gpu_for(gpu_manager.OCR)
            _run_single(uploaded_file, _ocr_options(opts))

    if st.session_state.get("auto_complete"):
        render_document_result()
    elif st.session_state.get("auto_error"):
        st.error(st.session_state.auto_error)


def auto_batch_ui():
    """Batch: ZIP upload, options, run button, download and review."""
    st.markdown(
        "Upload a **ZIP archive** of PDFs or images. Each file is parsed "
        "with the "
        "automatic pipeline; the result ZIP mirrors your folder structure "
        "with a "
        "`document.md`, `document.json`, `tables/` and `assets/` per file. "
        "For a "
        "batch of the same filled-in questionnaire, tick **Extract "
        "questionnaire "
        "responses** to also get a `survey/` folder with one row per "
        "respondent."
    )
    batch_zip = st.file_uploader(
        "Upload ZIP file",
        type=["zip"],
        on_change=clear_results,
        args=(True,),
        key="auto_batch_upload",
    )
    if batch_zip is not None:
        opts = _parse_options("auto_batch", batch=True)
        if st.session_state.ocr_running:
            st.warning(
                "A job is currently running. The button is disabled until "
                "completion."
            )
        if st.button(
            "Parse batch",
            type="primary",
            disabled=st.session_state.ocr_running,
            key="auto_batch_btn",
        ):
            free_gpu_for(gpu_manager.OCR)
            _run_batch(batch_zip, _ocr_options(opts))

    if st.session_state.get("batch_auto_complete"):
        elapsed = st.session_state.get("batch_auto_elapsed")
        st.success(
            "Batch parsing completed successfully!"
            + (f" ({elapsed / 60:.1f} min)" if elapsed else "")
        )
        st.download_button(
            "Download all results (ZIP)",
            st.session_state.batch_auto_zip,
            file_name="batch_auto_ocr_results.zip",
            mime="application/zip",
            use_container_width=True,
            type="primary",
        )
        render_survey_review()
    elif st.session_state.get("auto_error"):
        st.error(st.session_state.auto_error)


# ==========================================
#                 PAGE LAYOUT
# ==========================================

workflow_mode = st.radio(
    "Workflow",
    ["Single Document OCR", "Batch OCR (ZIP)"],
    index=0,
    horizontal=True,
    help="Choose to process a single file or batch process a ZIP archive",
    on_change=clear_results,
    args=(True,),
)

st.divider()

if workflow_mode == "Single Document OCR":
    auto_single_ui()
else:
    auto_batch_ui()

st.divider()
manual_engines_expander(workflow_mode)
