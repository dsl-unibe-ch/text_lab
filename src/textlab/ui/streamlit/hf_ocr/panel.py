"""Hugging Face OCR models on the OCR page.

The work is done by :mod:`textlab.features.hf_ocr.service`; this module
holds the model name, prompt, the user's downloaded models, the upload,
progress and the results. The OCR page draws it in an expander below
historic handwriting (:func:`hf_ocr_expander`). Its session-state keys
start with ``hfo_``, so it never shares state with the rest of the page.
"""

import streamlit as st

from textlab.common import gpu_manager
from textlab.features.hf_ocr import service, store
from textlab.ui.streamlit.components.gpu import free_gpu_for

#: Session-state keys of the results.
RESULT_KEYS = ("hfo_run", "hfo_batch", "hfo_error", "hfo_error_details")
INPUT_TYPES = [extension[1:] for extension in service.INPUT_EXTENSIONS]
EXAMPLE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


def clear_results():
    """Forget the previous run's results."""
    for key in RESULT_KEYS:
        st.session_state.pop(key, None)


def hf_ocr_expander():
    """Draw the Hugging Face OCR models tool in an expander."""
    with st.expander("Hugging Face OCR models (choose your own model)"):
        render_hf_ocr()


def render_hf_ocr():
    """Draw the tool: model, prompt, downloaded models, workflows."""
    if "hfo_running" not in st.session_state:
        st.session_state.hfo_running = False

    st.markdown(
        "Read documents with a **vision-language model of your choice** "
        "from [Hugging Face](https://huggingface.co/models?pipeline_tag="
        "image-text-to-text). The model reads each page as a whole. It is "
        "downloaded once into your own folder; your documents stay on "
        "the cluster."
    )
    st.caption(
        "Works with public models that Transformers supports directly "
        "(image-text-to-text), up to the size the GPU holds. Models that "
        "need their own code, an account or more GPU memory are refused "
        "with a message. Check the model's license before you use its "
        "output."
    )

    repo_text = st.text_input(
        "Hugging Face model",
        key="hfo_repo",
        placeholder=EXAMPLE_MODEL,
        help="The model's name as owner/name, or the link to its page.",
        on_change=clear_results,
    )
    repo_id = None
    if repo_text.strip():
        try:
            repo_id = service.parse_repo_id(repo_text)
        except service.HfOcrError as error:
            st.error(str(error))
        else:
            st.markdown(
                f"Model: [`{repo_id}`](https://huggingface.co/{repo_id})"
            )

    prompt = st.text_area(
        "Prompt",
        value=service.DEFAULT_PROMPT,
        key="hfo_prompt",
        help=(
            "The instruction sent with every page. A model trained with a "
            "specific prompt reads best with that one; its model card "
            "names it. Leave empty for models that take no instruction."
        ),
        on_change=clear_results,
    )
    max_new_tokens = st.number_input(
        "Most tokens per page",
        min_value=1,
        max_value=service.MAX_NEW_TOKENS_LIMIT,
        value=service.DEFAULT_MAX_NEW_TOKENS,
        step=256,
        key="hfo_max_tokens",
        help=(
            "Upper limit of the text generated for one page; a dense page "
            "needs about 1,000 to 2,000. Lower it if a model keeps "
            "repeating itself."
        ),
    )
    _stored_models()

    workflow = st.radio(
        "Workflow",
        ["Single document", "Batch (ZIP)"],
        key="hfo_workflow",
        horizontal=True,
        on_change=clear_results,
    )
    st.divider()

    if st.session_state.hfo_running:
        st.warning("A run is in progress. Please wait until it finishes.")
    options = None
    if repo_id is not None:
        try:
            options = service.HfOcrOptions(
                repo_id=repo_id,
                prompt=prompt.strip(),
                max_new_tokens=int(max_new_tokens),
            )
        except service.HfOcrError as error:
            st.error(str(error))
    if workflow == "Single document":
        _single_document(options)
    else:
        _batch(options)

    st.caption(
        "Every run first loads the model onto the GPU; the first use of a "
        "model also downloads it, which can take several minutes for a "
        "large one. Check the text against the page: a model can leave "
        "out or invent text."
    )


def _stored_models():
    """List the user's downloaded models, with their size and a delete."""
    models = store.stored_models()
    total = sum(model.size_bytes for model in models)
    title = "Your downloaded models"
    if models:
        title += f" ({len(models)}, {store.format_size(total)})"
    with st.expander(title):
        st.caption(
            f"Stored in `{store.models_dir()}`, which counts towards your "
            "storage quota. Delete a model you no longer need; it is "
            "downloaded again on its next use."
        )
        if not models:
            st.info("No models downloaded yet.")
        for model in models:
            name, size, action = st.columns([6, 2, 2])
            name.markdown(f"`{model.repo_id}`")
            size.markdown(store.format_size(model.size_bytes))
            if action.button(
                "Delete",
                key=f"hfo_delete_{model.repo_id}",
                disabled=st.session_state.hfo_running,
            ):
                store.delete_model(model.repo_id)
                st.rerun()


def _run_with_progress(work):
    """Run ``work(on_progress)`` with a progress bar; keep errors."""
    clear_results()
    st.session_state.hfo_running = True
    bar = st.progress(0.0, text="Starting...")
    last = {"fraction": 0.0}

    def show(update):
        if update.fraction is not None:
            last["fraction"] = update.fraction
        bar.progress(min(last["fraction"], 1.0), text=update.message)

    try:
        free_gpu_for(gpu_manager.HF_OCR)
        return work(show)
    except service.HfOcrError as error:
        st.session_state.hfo_error = str(error)
        return None
    except Exception as error:
        st.session_state.hfo_error = f"The run failed: {error}"
        st.session_state.hfo_error_details = getattr(error, "details", "")
        return None
    finally:
        bar.empty()
        st.session_state.hfo_running = False


def _show_error():
    """Show the last run's error, with details for a developer."""
    if st.session_state.get("hfo_error"):
        st.error(st.session_state.hfo_error)
        details = st.session_state.get("hfo_error_details")
        if details:
            with st.expander("Error details"):
                st.code(details)


def _show_model_used(model, prompt):
    """Show the model that produced a result, for citation."""
    with st.expander("Model used (for citation)"):
        st.markdown(
            f"- **Model:** [`{model.repo_id}`]({model.url})"
            + (f", revision `{model.revision}`" if model.revision else "")
        )
        st.markdown("- **Prompt:**")
        st.code(prompt or "(none)", language=None, wrap_lines=True)


def _start_button(label, key, options, upload):
    """Draw the start button; return whether it was pressed."""
    return st.button(
        label,
        type="primary",
        disabled=(
            st.session_state.hfo_running or options is None or upload is None
        ),
        key=key,
        help=None if options is not None else "Enter a model name first.",
    )


def _single_document(options):
    """Upload one PDF or image, read it and show the result."""
    upload = st.file_uploader(
        "Choose a PDF or image",
        type=INPUT_TYPES,
        on_change=clear_results,
        key="hfo_single_upload",
    )
    if _start_button("Read document", "hfo_single_btn", options, upload):
        run = _run_with_progress(
            lambda show: service.read_document(
                upload.name, upload.getvalue(), options, on_progress=show
            )
        )
        if run is not None:
            st.session_state.hfo_run = run

    _show_error()
    run = st.session_state.get("hfo_run")
    if run is None:
        return
    document = run.document
    st.success(f"Read {len(document.pages)} page(s) with {run.model.repo_id}.")
    columns = st.columns(3)
    columns[0].download_button(
        "Text (.txt)",
        document.text,
        file_name=f"{run.stem}.txt",
        mime="text/plain",
    )
    columns[1].download_button(
        "JSON",
        run.to_json(),
        file_name=f"{run.stem}.json",
        mime="application/json",
    )
    columns[2].download_button(
        "All files (.zip)",
        run.zip_bytes,
        file_name=f"{run.stem}_hf_ocr.zip",
        mime="application/zip",
    )
    _show_model_used(run.model, run.prompt)

    page_number = 1
    if len(document.pages) > 1:
        page_number = st.number_input(
            "Page",
            min_value=1,
            max_value=len(document.pages),
            value=1,
            key="hfo_page",
        )
    left, right = st.columns(2)
    with left:
        st.markdown("**Page**")
        if run.previews:
            st.image(run.previews[page_number - 1], width="stretch")
    with right:
        st.markdown("**Text**")
        text = document.pages[page_number - 1].text
        if text:
            # Plain text: models often answer in Markdown or HTML, which
            # is shown as written, not rendered.
            st.code(text, language=None, wrap_lines=True)
        else:
            st.info("The model returned no text for this page.")


def _batch(options):
    """Upload a ZIP of PDFs and images and offer the result ZIP."""
    st.markdown(
        "Upload a **ZIP archive** of PDFs or images. The result ZIP mirrors "
        "your folders, with a text file and a JSON file for each document."
    )
    upload = st.file_uploader(
        "Upload ZIP file",
        type=["zip"],
        on_change=clear_results,
        key="hfo_batch_upload",
    )
    if _start_button("Read batch", "hfo_batch_btn", options, upload):
        result = _run_with_progress(
            lambda show: service.read_batch(upload, options, on_progress=show)
        )
        if result is not None:
            st.session_state.hfo_batch = (result, options.prompt)

    _show_error()
    stored = st.session_state.get("hfo_batch")
    if stored is None:
        return
    result, prompt = stored
    st.success(
        f"Read {result.pages} page(s) in {result.files} file(s) with "
        f"{result.model.repo_id}."
    )
    st.download_button(
        "Download all results (ZIP)",
        result.zip_bytes,
        file_name="hf_ocr_results.zip",
        mime="application/zip",
        type="primary",
    )
    _show_model_used(result.model, prompt)
