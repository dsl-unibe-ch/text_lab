"""Historic German handwriting on the OCR page.

The work is done by :mod:`textlab.features.handwriting.service`; this
module holds the model choice, the upload, progress and the results. The
OCR page draws it in an expander below the legacy engines
(:func:`handwriting_expander`). Its session-state keys start with ``hw_``,
so it never shares state with the rest of the page.
"""

import os

import streamlit as st

from textlab.common import gpu_manager
from textlab.features.handwriting import service
from textlab.ui.streamlit.components.gpu import free_gpu_for

#: Session-state keys of the results.
RESULT_KEYS = ("hw_run", "hw_batch", "hw_error", "hw_error_details")
INPUT_TYPES = [extension[1:] for extension in service.INPUT_EXTENSIONS]
HUGGING_FACE = "https://huggingface.co"
KRAKEN_DOCS = "https://kraken.re"


def clear_results():
    """Forget the previous run's results."""
    for key in RESULT_KEYS:
        st.session_state.pop(key, None)


def handwriting_expander():
    """Draw the historic handwriting tool in an expander."""
    with st.expander(
        "Historic German handwriting (Kurrent / medieval manuscripts)"
    ):
        render_handwriting()


def render_handwriting():
    """Draw the historic handwriting mode: model, models used, workflows."""
    if "hw_running" not in st.session_state:
        st.session_state.hw_running = False

    st.markdown(
        "Transcribe **handwritten historical German documents**, such as "
        "Kurrent script from the 19th century or medieval manuscripts. "
        "Text Lab finds the text lines on each page and reads them one by "
        "one with a handwriting recognition model of the Digital Humanities "
        "group of the University of Bern."
    )

    models = service.available_models()
    if not models:
        st.error("No handwriting models are configured.")
        return
    labels = {model.label: model for model in models}
    choice = st.selectbox(
        "Handwriting model",
        list(labels),
        key="hw_model",
        help="Choose the model trained on material closest to your documents.",
        on_change=clear_results,
    )
    model = labels[choice]
    _models_panel(model)

    workflow = st.radio(
        "Workflow",
        ["Single document", "Batch (ZIP)"],
        key="hw_workflow",
        horizontal=True,
        on_change=clear_results,
    )
    st.divider()

    if st.session_state.hw_running:
        st.warning(
            "A transcription is running. Please wait until it finishes."
        )
    if workflow == "Single document":
        _single_document(model)
    else:
        _batch(model)

    st.caption(
        "Every run first loads the model onto the GPU, which takes about a "
        "minute; the first use of a model also downloads it. Each line is "
        "read separately, so check the transcription against the page, "
        "especially for unusual hands."
    )


def _repo_link(repo):
    """A Markdown link to a Hugging Face repository."""
    return f"[`{repo}`]({HUGGING_FACE}/{repo})"


def _kraken_link():
    """A Markdown link to kraken, the line segmenter."""
    return f"[kraken]({KRAKEN_DOCS})'s baseline segmenter (`blla`)"


def _models_panel(model):
    """Show, before a run, which models will process the documents."""
    with st.container(border=True):
        st.markdown("**Models used**")
        recognition = _repo_link(model.base_model)
        if model.adapter:
            recognition = (
                f"{_repo_link(model.adapter)}, a fine-tuned adapter on "
                f"{recognition}"
            )
        st.markdown(
            f"1. **Line detection:** {_kraken_link()}, a segmenter "
            "trained on historical manuscripts, finds the text lines on "
            "each page, in reading order.\n"
            f"2. **Text recognition:** {recognition} reads each line."
        )
        st.caption(model.description)


def _run_with_progress(work):
    """Run ``work(on_progress)`` with a progress bar; keep errors."""
    clear_results()
    st.session_state.hw_running = True
    bar = st.progress(0.0, text="Starting...")
    last = {"fraction": 0.0}

    def show(update):
        if update.fraction is not None:
            last["fraction"] = update.fraction
        bar.progress(min(last["fraction"], 1.0), text=update.message)

    try:
        free_gpu_for(gpu_manager.HANDWRITING)
        return work(show)
    except Exception as error:
        st.session_state.hw_error = f"Transcription failed: {error}"
        st.session_state.hw_error_details = getattr(error, "details", "")
        return None
    finally:
        bar.empty()
        st.session_state.hw_running = False


def _show_error():
    """Show the last run's error, with details for a developer."""
    if st.session_state.get("hw_error"):
        st.error(st.session_state.hw_error)
        details = st.session_state.get("hw_error_details")
        if details:
            with st.expander("Error details"):
                st.code(details)


def _show_models_used(models_used):
    """List the models that produced a result, for citation."""
    with st.expander("Models used (for citation)"):
        st.markdown(
            f"- **Line detection:** {models_used['line_detection']}\n"
            f"- **Text recognition:** {models_used['recognition']}"
        )


def _single_document(model):
    """Upload one PDF or image, transcribe it and show the result."""
    upload = st.file_uploader(
        "Choose a PDF or image",
        type=INPUT_TYPES,
        on_change=clear_results,
        key="hw_single_upload",
    )
    if upload is not None and st.button(
        "Transcribe",
        type="primary",
        disabled=st.session_state.hw_running,
        key="hw_single_btn",
    ):
        options = service.HandwritingOptions(model=model.key)
        run = _run_with_progress(
            lambda show: service.transcribe_document(
                upload.name, upload.getvalue(), options, on_progress=show
            )
        )
        if run is not None:
            st.session_state.hw_run = run

    _show_error()
    run = st.session_state.get("hw_run")
    if run is None:
        return
    document = run.document
    st.success(
        f"Transcribed {document.line_count} lines on "
        f"{len(document.pages)} page(s) with {document.model.label}."
    )
    columns = st.columns(3)
    columns[0].download_button(
        "Text (.txt)", run.text, file_name=run.text_name, mime="text/plain"
    )
    columns[1].download_button(
        "JSON", run.json, file_name=run.json_name, mime="application/json"
    )
    columns[2].download_button(
        "All files with PAGE XML (.zip)",
        run.zip_bytes,
        file_name=f"{os.path.splitext(run.text_name)[0]}_handwriting.zip",
        mime="application/zip",
    )
    _show_models_used(document.models_used())

    page_number = 1
    if len(document.pages) > 1:
        page_number = st.number_input(
            "Page",
            min_value=1,
            max_value=len(document.pages),
            value=1,
            key="hw_page",
        )
    page = document.pages[page_number - 1]
    left, right = st.columns(2)
    with left:
        st.markdown("**Detected lines**")
        if run.previews:
            st.image(run.previews[page_number - 1], use_container_width=True)
    with right:
        st.markdown("**Transcription**")
        if page.lines:
            # Plain text, so characters such as * or # in a line are not
            # read as Markdown; numbers match the boxes on the left.
            st.code(
                "\n".join(
                    f"{number:>3}  {line.text}"
                    for number, line in enumerate(page.lines, start=1)
                ),
                language=None,
            )
        else:
            st.info("No text lines were found on this page.")


def _batch(model):
    """Upload a ZIP of PDFs and images and offer the result ZIP."""
    st.markdown(
        "Upload a **ZIP archive** of PDFs or images. The result ZIP mirrors "
        "your folders, with a text file, a JSON file and PAGE XML files for "
        "each document."
    )
    upload = st.file_uploader(
        "Upload ZIP file",
        type=["zip"],
        on_change=clear_results,
        key="hw_batch_upload",
    )
    if upload is not None and st.button(
        "Transcribe batch",
        type="primary",
        disabled=st.session_state.hw_running,
        key="hw_batch_btn",
    ):
        options = service.HandwritingOptions(model=model.key, previews=False)
        result = _run_with_progress(
            lambda show: service.transcribe_batch(
                upload, options, on_progress=show
            )
        )
        if result is not None:
            st.session_state.hw_batch = result

    _show_error()
    result = st.session_state.get("hw_batch")
    if result is None:
        return
    st.success(f"Transcribed {result.lines} lines in {result.files} file(s).")
    st.download_button(
        "Download all results (ZIP)",
        result.zip_bytes,
        file_name="handwriting_results.zip",
        mime="application/zip",
        type="primary",
    )
    _show_models_used(result.models_used)
