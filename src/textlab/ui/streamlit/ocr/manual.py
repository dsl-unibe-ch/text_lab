"""Manual engine selection on the OCR page, shown as "legacy engines".

The user picks one engine (EasyOCR, PaddleOCR, OlmOCR, GLM-OCR) and gets its
plain text per page. The engines run in the OCR feature
(:func:`textlab.features.ocr.service.recognize_with_engine`); this module
draws the options, runs them and shows the result.
"""

from __future__ import annotations

import streamlit as st

from textlab.common import gpu_manager, html_safety
from textlab.features.ocr import service
from textlab.ui.streamlit.components.gpu import free_gpu_for
from textlab.ui.streamlit.ocr.state import clear_results

INPUT_TYPES = ["pdf", "png", "jpg", "jpeg", "bmp", "tiff"]


def manual_engines_expander(workflow_mode):
    """Draw the expander with the engine choice and the chosen workflow.

    Args:
        workflow_mode: The page's workflow, single document or batch.
    """
    with st.expander(
        "Advanced: legacy engines (EasyOCR / PaddleOCR / OlmOCR / GLM-OCR)"
    ):
        st.caption(
            "The classic engine-picker workflow. Each engine returns plain "
            "text; the automatic pipeline above is recommended for structured "
            "documents."
        )
        engine, options = _engine_options()
        if workflow_mode == "Single Document OCR":
            _single_flow(engine, options)
        else:
            _batch_flow(engine, options)


def _engine_options():
    """Draw the engine choice and its settings.

    Returns:
        The engine's name and its :class:`service.EngineOptions`.
    """
    col_engine, col_setting = st.columns([1, 1])
    with col_engine:
        name = st.selectbox(
            "OCR engine",
            list(service.ENGINES),
            index=0,
            on_change=clear_results,
            args=(True,),
            key="legacy_engine_select",
            help=(
                "Select the OCR backend. GLM-OCR is best for complex layouts "
                "and tables."
            ),
        )
    engine = service.ENGINES[name]

    language = "en"
    mode = ""
    if engine.modes:
        with col_setting:
            mode = st.selectbox(
                f"{name} Mode",
                list(engine.modes),
                key="legacy_glm_mode",
                help=(
                    "Choose what specific aspect of the document you want to "
                    "extract."
                ),
            )
    elif engine.languages:
        labels = list(engine.languages)
        default = (
            labels.index(engine.default_language)
            if engine.default_language in labels
            else 0
        )
        with col_setting:
            label = st.selectbox(
                "Document Language",
                labels,
                index=default,
                on_change=clear_results,
                args=(True,),
                key=f"manual_language_{name}",
                help=f"Select the text language for {name}.",
            )
        language = engine.languages[label]
    return name, service.EngineOptions(language=language, mode=mode)


def _running_notice():
    """Warn that the run button is disabled while a job runs."""
    if st.session_state.ocr_running:
        st.warning(
            "OCR is currently running. The button is disabled until "
            "completion."
        )


def _single_flow(engine, options):
    """One document: upload, run button and result."""
    st.markdown(
        "Upload a **PDF** or **Image** to extract its text content and "
        "preview the results."
    )
    uploaded_file = st.file_uploader(
        "Choose a PDF or Image file",
        type=INPUT_TYPES,
        on_change=clear_results,
        args=(True,),
        key="legacy_single_upload",
    )
    if uploaded_file is not None:
        _running_notice()
        if st.button(
            "Run OCR",
            disabled=st.session_state.ocr_running,
            key="legacy_single_btn",
        ):
            free_gpu_for(gpu_manager.OCR)
            _run_single(uploaded_file, engine, options)

    if st.session_state.get("manual_run") is not None:
        _render_single_result(st.session_state.manual_run)
    elif "manual_error" in st.session_state:
        st.error(st.session_state.manual_error)
        details = st.session_state.get("manual_error_details")
        if details:
            st.text_area("Error Details", details, height=150)
    elif st.session_state.get("ocr_running"):
        st.info("OCR is running. Please wait...")


def _run_single(uploaded_file, engine, options):
    """Read one uploaded file and keep the result in session state."""
    clear_results(reset_running=False)
    st.session_state.ocr_running = True
    run_notice = st.empty()
    run_notice.info("OCR has started.")
    progress_bar = st.progress(0.0, text=f"Running {engine}...")

    def show(update):
        progress_bar.progress(update.fraction or 0.0, text=update.message)

    try:
        st.session_state.manual_run = service.recognize_with_engine(
            uploaded_file.name,
            uploaded_file.getvalue(),
            engine,
            options,
            on_progress=show,
        )
        st.session_state.manual_preview_page = 0
    except Exception as error:
        st.session_state.manual_error = (
            f"An unexpected error occurred: {error}"
        )
        st.session_state.manual_error_details = getattr(error, "details", "")
        st.exception(error)
    finally:
        progress_bar.empty()
        run_notice.empty()
        st.session_state.ocr_running = False


def _render_single_result(run):
    """Show the text, a detected table, downloads and page previews."""
    st.success("OCR complete!")
    stem = run.text_name.removesuffix(".txt")

    table = service.html_table(run.text)
    if table is not None:
        st.markdown("### Detected Table")
        st.dataframe(table, use_container_width=True)
        st.download_button(
            label="Download Table as CSV",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name=f"{stem}_table.csv",
            mime="text/csv",
            type="primary",
        )
    elif service.contains_html_table(run.text):
        # pandas could not parse it; show the sanitized HTML instead.
        st.markdown("### Detected Table")
        st.markdown(
            html_safety.sanitize_table_html(run.text), unsafe_allow_html=True
        )

    st.markdown("### Extracted Text / Code")
    st.text_area("Result", run.text, height=400, key="md_result")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "Download as .txt", run.text, run.text_name, "text/plain"
        )
    with c2:
        st.download_button(
            "Download as .jsonl", run.json, run.json_name, "application/json"
        )
    with c3:
        st.download_button(
            "Download all outputs (.zip)",
            run.zip_bytes,
            "ocr_outputs.zip",
            "application/zip",
        )

    if run.previews:
        st.markdown("---")
        st.markdown("### Document Preview")
        _render_previews(run.previews)


def _render_previews(previews):
    """Show one page's previews, with buttons to move between pages."""
    current = st.session_state.get("manual_preview_page", 0)
    current = max(0, min(current, len(previews) - 1))
    st.session_state.manual_preview_page = current

    c_prev, c_info, c_next = st.columns([1, 2, 1])
    if c_prev.button("Previous", disabled=current <= 0, key="legacy_prev"):
        st.session_state.manual_preview_page -= 1
        st.rerun()
    with c_info:
        st.caption(f"Page {current + 1} of {len(previews)}")
    if c_next.button(
        "Next", disabled=current >= len(previews) - 1, key="legacy_next"
    ):
        st.session_state.manual_preview_page += 1
        st.rerun()

    preview = previews[current]
    if preview.layout is not None:
        left, right = st.columns(2)
        with left:
            st.caption("Detected boxes")
            st.image(preview.image, use_container_width=True)
        with right:
            st.caption("OCR Layout")
            st.image(preview.layout, use_container_width=True)
    else:
        st.image(
            preview.image,
            caption="Original Document",
            use_container_width=True,
        )


def _batch_flow(engine, options):
    """A ZIP archive: upload, run button and the result ZIP."""
    st.markdown(
        "Upload a **ZIP archive** containing multiple PDFs or Images. They "
        "will be processed and returned as a single organized ZIP."
    )
    batch_zip = st.file_uploader(
        "Upload ZIP file",
        type=["zip"],
        on_change=clear_results,
        args=(True,),
        key="legacy_batch_upload",
    )
    if batch_zip is not None:
        _running_notice()
        if st.button(
            "Run Batch OCR",
            disabled=st.session_state.ocr_running,
            key="legacy_batch_btn",
        ):
            free_gpu_for(gpu_manager.OCR)
            _run_batch(batch_zip, engine, options)

    if st.session_state.get("manual_batch_zip") is not None:
        st.success("Batch OCR completed successfully!")
        st.download_button(
            "Download All OCR Results (ZIP)",
            st.session_state.manual_batch_zip,
            file_name="batch_ocr_results.zip",
            mime="application/zip",
            use_container_width=True,
            type="primary",
        )
    elif "manual_error" in st.session_state:
        st.error(st.session_state.manual_error)


def _run_batch(batch_zip, engine, options):
    """Read an uploaded ZIP archive and keep the result ZIP."""
    clear_results(reset_running=False)
    st.session_state.ocr_running = True
    run_notice = st.empty()
    run_notice.info(
        "Batch OCR has started. This may take a while depending on the "
        "number of files."
    )
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def show(update):
        if update.fraction is not None:
            progress_bar.progress(update.fraction)
        status_text.text(update.message)

    try:
        st.session_state.manual_batch_zip = (
            service.recognize_batch_with_engine(
                batch_zip, engine, options, on_progress=show
            )
        )
    except Exception as error:
        st.session_state.manual_error = f"Batch processing failed: {error}"
        st.exception(error)
    finally:
        run_notice.empty()
        st.session_state.ocr_running = False
