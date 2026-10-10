"""Meeting Notes Generator page for Text Lab.

Combines the WhisperX transcription pipeline and Ollama-based summarization
into a single automated workflow. Users upload an audio file (or supply an
existing transcript) and receive both a full transcript and a structured
summary in one step.

Two workflow tabs are provided:
- From Audio: full pipeline (transcribe then summarize).
- From Transcript: summarize an existing transcript CSV or plain text.

Transcription runs in a worker process through
:mod:`textlab.features.transcription.service`; summarization uses
:mod:`textlab.features.meeting_notes.service`.
"""

import datetime
import io
import os
import traceback

import streamlit as st
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.dirname(current_dir)
favicon_path = os.path.join(app_dir, "assets", "text_lab_logo.png")

favicon = Image.open(favicon_path)

st.set_page_config(
    page_title="Meeting Notes Generator",
    page_icon=favicon,
    layout="wide",
)

from textlab.common import gpu_manager  # noqa: E402
from textlab.common.language_mappings import (  # noqa: E402
    TRANSCRIBE_LANGUAGE_CODE_TO_NAME as LANGUAGE_CODE_TO_NAME,
)
from textlab.common.language_mappings import (  # noqa: E402
    TRANSCRIBE_LANGUAGE_MAPPING as LANGUAGE_MAPPING,
)
from textlab.common.model_config import (  # noqa: E402
    get_available_models,
    is_high_memory_gpu,
)
from textlab.common.ollama import (  # noqa: E402
    check_ollama_server,
    estimate_tokens,
    get_loaded_models,
    is_model_loaded,
    unload_all_models,
)
from textlab.features.meeting_notes.service import (  # noqa: E402
    SUMMARY_MODES,
    apply_speaker_labels,
    build_speaker_context,
    extract_unique_speakers,
    format_summary_document,
    get_partial_notes,
    get_summary_stream,
    get_synthesis_stream,
    needs_chunking,
    transcript_csv_to_speaker_text,
)
from textlab.features.transcription.audio import (  # noqa: E402
    SAMPLE_RATE,
    detect_language_from_bytes,
)
from textlab.features.transcription.formats import (  # noqa: E402
    format_duration,
    transcription_csv,
    transcription_text_from_csv,
)
from textlab.features.transcription.models import (  # noqa: E402
    TranscriptionOptions,
)
from textlab.features.transcription.service import (  # noqa: E402
    run_transcription,
    staged_uploads,
)
from textlab.features.transcription.whisper_models import (  # noqa: E402
    converted_model_missing,
    default_model,
)
from textlab.ui.streamlit.auth import check_token  # noqa: E402
from textlab.ui.streamlit.components.gpu import free_gpu_for  # noqa: E402
from textlab.ui.streamlit.components.progress import StatusBox  # noqa: E402

check_token()

# ---------------------------------------------------------------------------
# Helper: Streamlit write_stream compatible generator wrapper
# ---------------------------------------------------------------------------


def _stream_or_write(container, generator, key: str) -> str:
    """Write a streaming generator into a container and return the full text.

    Args:
        container: A Streamlit container (e.g. st.empty() or st.container()).
        generator: A generator yielding string tokens.
        key: Unique widget key prefix (unused, kept for caller clarity).

    Returns:
        The fully accumulated response string.
    """
    buf = io.StringIO()
    placeholder = container.empty()
    pending_chars = 0
    render_threshold = 60  # re-render every ~60 chars, not every token
    for token in generator:
        buf.write(token)
        pending_chars += len(token)
        if pending_chars >= render_threshold:
            placeholder.markdown(buf.getvalue())
            pending_chars = 0
    result = buf.getvalue()
    if result:
        placeholder.markdown(result)
    return result


# ---------------------------------------------------------------------------
# Shared UI: summary results display
# ---------------------------------------------------------------------------


def _render_summary_results(
    summary: str,
    transcript_text: str,
    source_label: str,
    mode_key: str,
    duration_str: str,
    word_count: int,
    key_prefix: str,
) -> None:
    """Render the summary and transcript results with download options.

    Download buttons use ``on_click="ignore"`` so that saving a file does not
    rerun the page.

    Args:
        summary: The generated summary text (Markdown).
        transcript_text: The full plain/speaker-labeled transcript.
        source_label: Original filename or input label shown in exports.
        mode_key: Summary mode key used.
        duration_str: Human-readable audio duration string.
        word_count: Number of words in the transcript.
        key_prefix: Unique prefix for Streamlit widget keys.
    """
    mode_label = SUMMARY_MODES[mode_key]["label"]

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Summary type", mode_label)
    col_b.metric("Transcript words", f"{word_count:,}")
    col_c.metric("Audio duration", duration_str or "N/A")

    st.divider()

    tab_summary, tab_transcript = st.tabs(["Summary", "Full Transcript"])

    with tab_summary:
        st.markdown(summary)

    with tab_transcript:
        st.text_area(
            "Transcript",
            value=transcript_text,
            height=420,
            disabled=True,
            label_visibility="collapsed",
        )

    st.divider()
    st.write("**Download results**")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_name = (
        os.path.splitext(source_label)[0] if source_label else "summary"
    )

    full_doc = format_summary_document(
        summary=summary,
        transcript_text=transcript_text,
        source_label=source_label,
        mode_key=mode_key,
        duration_str=duration_str,
    )

    col_d1, col_d2, col_d3 = st.columns(3)

    with col_d1:
        st.download_button(
            label="Summary (.md)",
            data=summary,
            file_name=f"{base_name}_summary_{timestamp}.md",
            mime="text/markdown",
            use_container_width=True,
            on_click="ignore",
            key=f"{key_prefix}_summary_dl",
        )
    with col_d2:
        st.download_button(
            label="Transcript (.txt)",
            data=transcript_text,
            file_name=f"{base_name}_transcript_{timestamp}.txt",
            mime="text/plain",
            use_container_width=True,
            on_click="ignore",
            key=f"{key_prefix}_transcript_dl",
        )
    with col_d3:
        st.download_button(
            label="Full document (.md)",
            data=full_doc,
            file_name=f"{base_name}_full_{timestamp}.md",
            mime="text/markdown",
            use_container_width=True,
            on_click="ignore",
            key=f"{key_prefix}_full_dl",
        )


# ---------------------------------------------------------------------------
# Shared UI: model and mode configuration sidebar widgets
# ---------------------------------------------------------------------------


def _render_model_selector(gpu_name: str, key_prefix: str) -> str:
    """Render an Ollama model selection widget.

    Args:
        gpu_name: The current GPU name for filtering available models.
        key_prefix: Unique prefix for Streamlit widget keys.

    Returns:
        The selected model name string.
    """
    models = get_available_models(gpu_name)
    if not models:
        st.error(
            "No models configured. Check the model list "
            "(src/textlab/common/models.json)."
        )
        st.stop()

    selected_key = f"{key_prefix}_model"
    if selected_key not in st.session_state:
        st.session_state[selected_key] = models[0]
    if st.session_state[selected_key] not in models:
        st.session_state[selected_key] = models[0]

    selected = st.selectbox(
        "Summarization model",
        options=models,
        key=selected_key,
        help="Language model used to generate the summary.",
    )
    return selected


def _render_mode_selector(key_prefix: str) -> str:
    """Render a summary mode selection widget with visible descriptions.

    Args:
        key_prefix: Unique prefix for Streamlit widget keys.

    Returns:
        The selected summary mode key string.
    """
    mode_labels = {key: val["label"] for key, val in SUMMARY_MODES.items()}
    mode_keys = list(mode_labels.keys())
    labels = list(mode_labels.values())

    selected_label = st.selectbox(
        "Summary type",
        options=labels,
        key=f"{key_prefix}_mode_label",
        help="Choose how the transcript should be structured.",
    )
    selected_key = mode_keys[labels.index(selected_label)]

    desc = SUMMARY_MODES[selected_key]["description"]
    st.caption(desc)

    return selected_key


def _render_output_language_selector(
    key_prefix: str,
    transcript_language_name: str | None = None,
) -> str | None:
    """Render a language selector for the summary output.

    Offers two options: English, or the same language as the transcript.
    When the user selects the transcript language option, ``None`` is returned
    so the engine can instruct the model to match the source language.

    Args:
        key_prefix: Unique prefix for Streamlit widget keys.
        transcript_language_name: Human-readable name of the transcript
            language (e.g. ``"German"``). Used to label the second option.
            If ``None``, the second option is labelled generically.

    Returns:
        ``"English"`` if English output is selected, or ``None`` to indicate
        the model should match the transcript language.
    """
    if transcript_language_name and transcript_language_name not in (
        "Auto-detect",
        "English",
    ):
        transcript_option = f"Transcript language ({transcript_language_name})"
    else:
        transcript_option = "Transcript language (auto-detect)"

    choice = st.radio(
        "Summary language",
        options=["English", transcript_option],
        key=f"{key_prefix}_output_lang",
        horizontal=True,
        help=(
            "Choose whether the summary should be written in English "
            "or in the same language as the audio."
        ),
    )
    return "English" if choice == "English" else None


# ---------------------------------------------------------------------------
# GPU memory management
# ---------------------------------------------------------------------------


def _render_gpu_clear_section() -> None:
    """Render a GPU memory status indicator and an optional clear button.

    Checks which Ollama models are currently resident in VRAM. If any are
    found, a warning is displayed explaining that transcription may fail on
    GPUs with limited memory, and a button is offered to evict all loaded
    models before running a new transcription.

    This function is a no-op (renders nothing) when no models are loaded,
    so it has zero visual footprint during a fresh session.
    """
    loaded = get_loaded_models()
    if not loaded:
        return

    model_list = ", ".join(loaded)
    with st.container(border=True):
        col_info, col_btn = st.columns([3, 1])
        with col_info:
            st.warning(
                f"**GPU memory in use** - the following model(s) are still "
                "loaded "
                f"in VRAM from a previous run: `{model_list}`. "
                "On GPUs with limited memory this may cause the transcription "
                "worker to run out of memory. Clear VRAM before starting a "
                "new "
                "transcription.",
            )
        with col_btn:
            st.write("")  # vertical alignment spacer
            if st.button(
                "Clear GPU Memory",
                key="audio_sum_clear_gpu",
                use_container_width=True,
                help=(
                    "Unload all Ollama models from VRAM so the transcription "
                    "worker has full GPU access."
                ),
            ):
                with st.spinner("Unloading models from GPU..."):
                    unloaded = unload_all_models()
                if unloaded:
                    st.success(
                        f"Unloaded {len(unloaded)} model(s): "
                        f"{', '.join(unloaded)}. "
                        "GPU memory is now free."
                    )
                else:
                    st.info(
                        "No models were loaded or unload request was ignored."
                    )
                st.rerun()


# ---------------------------------------------------------------------------
# "From Audio" tab
# ---------------------------------------------------------------------------


def _run_transcription(
    audio_bytes: bytes,
    audio_name: str,
    language: str,
    whisper_model: str,
    min_speakers,
    max_speakers,
) -> dict:
    """Transcribe the recording in a worker process, showing its progress.

    The worker releases all GPU memory WhisperX used when it exits, so
    Ollama can use the GPU for the summarization step.

    Args:
        audio_bytes: Raw bytes of the audio file.
        audio_name: Original filename (used for display and file naming).
        language: Language code from LANGUAGE_MAPPING (or auto-detected code).
        whisper_model: WhisperX model name or path.
        min_speakers: Minimum speaker count hint for diarization, or None.
        max_speakers: Maximum speaker count hint for diarization, or None.

    Returns:
        A dict containing:
            - ``transcription_csv``: TSV string (WhisperX segment output).
            - ``has_speakers``: bool, True if diarization succeeded.
            - ``duration_str``: Human-readable audio duration.

    Raises:
        RuntimeError: If the audio could not be decoded.
        textlab.common.jobs.WorkerError: If the transcription failed.
    """
    options = TranscriptionOptions(
        model=whisper_model,
        language=language,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    with staged_uploads([(audio_name, audio_bytes)]) as files:
        with StatusBox(
            "Running transcription pipeline...",
            complete_label="Transcription complete.",
            error_label="Transcription failed.",
        ) as status:
            result = run_transcription(files, options, on_progress=status)
    for note in result.notes:
        st.info(note)
    if not result.transcripts:
        raise RuntimeError("The audio could not be decoded.")

    transcript = result.transcripts[0]
    return {
        "transcription_csv": transcription_csv(transcript),
        "has_speakers": transcript.has_speakers,
        "duration_str": format_duration(transcript.duration_seconds),
    }


def _run_summarization(
    model_name: str,
    transcript_text: str,
    mode_key: str,
    speaker_context,
    output_language: str | None = "English",
) -> str:
    """Run the summarization step with step-by-step UI feedback.

    Automatically chooses single-pass streaming for short transcripts and
    chunked map-reduce for long ones.

    Args:
        model_name: Ollama model identifier.
        transcript_text: The transcript text to summarize.
        mode_key: Key in SUMMARY_MODES.
        speaker_context: Optional speaker description string.
        output_language: Language for the summary. ``None`` instructs the
            model to match the transcript language. Defaults to ``"English"``.

    Returns:
        The generated summary as a string.
    """
    # The transcription worker has exited; leave only the summary LLM.
    free_gpu_for(gpu_manager.LLM, ollama_model=model_name)
    loading_hint = (
        f"Loading {model_name} into GPU memory... This first run may take "
        "1-2 minutes."
        if not is_model_loaded(model_name)
        else "Generating summary..."
    )

    summary = ""

    st.caption("This may take several minutes. To cancel, refresh the page.")

    if not needs_chunking(transcript_text):
        with st.status(loading_hint, expanded=False) as status:
            status.update(label="Generating summary...", state="running")
            result_container = st.container()
            stream = get_summary_stream(
                model_name,
                transcript_text,
                mode_key,
                speaker_context,
                output_language,
            )
            summary = _stream_or_write(
                result_container, stream, "summary_stream"
            )
            status.update(
                label="Summary complete.", state="complete", expanded=False
            )
    else:
        progress_bar = st.progress(
            0.0, text="Analyzing transcript in chunks..."
        )
        status_placeholder = st.empty()

        def _progress(progress) -> None:
            progress_bar.progress(
                progress.fraction or 0.0, text=progress.message
            )
            status_placeholder.info(progress.message)

        partial_notes = get_partial_notes(
            model_name,
            transcript_text,
            mode_key,
            speaker_context,
            on_progress=_progress,
            output_language=output_language,
        )
        progress_bar.progress(
            len(partial_notes) / (len(partial_notes) + 1),
            text="Synthesizing final summary...",
        )
        status_placeholder.info("Synthesizing final summary from all parts...")

        result_container = st.container()
        stream = get_synthesis_stream(
            model_name, partial_notes, mode_key, output_language
        )
        summary = _stream_or_write(result_container, stream, "synth_stream")

        progress_bar.progress(1.0, text="Summary complete.")
        status_placeholder.empty()

    return summary


def _render_audio_tab(gpu_name: str) -> None:
    """Render the 'From Audio' workflow tab.

    Handles audio upload, transcription configuration, optional speaker
    labeling, summarization, and result display.

    Args:
        gpu_name: Current GPU name for model filtering.
    """
    st.write(
        "Upload an audio file. The tool will transcribe it with WhisperX, "
        "then automatically generate a structured summary using your chosen "
        "language model - all in one step."
    )

    # --- Configuration ---
    col_cfg1, col_cfg2, col_cfg3 = st.columns([2, 2, 2])

    with col_cfg1:
        audio_file = st.file_uploader(
            "Audio file",
            type=["wav", "mp3", "flac", "m4a", "ogg", "webm"],
            key="audio_sum_upload",
        )

    with col_cfg2:
        language_options = ["Auto-detect"] + list(LANGUAGE_MAPPING.keys())
        language_name = st.selectbox(
            "Language",
            language_options,
            key="audio_sum_language",
            help=(
                "Auto-detect uses the first 30 s of audio to identify the "
                "language."
            ),
        )

        is_auto_detect = language_name == "Auto-detect"
        if not is_auto_detect:
            language = LANGUAGE_MAPPING[language_name]
        else:
            language = None

        # Swiss German uses its own models.
        whisper_model = default_model(language)
        if language == "ch_de":
            st.caption("Using Swiss German Whisper model.")
        elif language == "ch_de_flurin":
            st.caption("Using Flurin Swiss German Turbo model.")

    with col_cfg3:
        mode_key = _render_mode_selector("audio_sum")
        output_language = _render_output_language_selector(
            "audio_sum",
            transcript_language_name=language_name
            if not is_auto_detect
            else None,
        )

    model_name = _render_model_selector(gpu_name, "audio_sum")

    # Optional speaker count hints (collapsed by default)
    with st.expander("Speaker diarization settings (optional)"):
        col_spk1, col_spk2 = st.columns(2)
        with col_spk1:
            use_min = st.checkbox(
                "Set minimum speakers", key="audio_sum_min_chk"
            )
            min_speakers = (
                st.number_input(
                    "Minimum speakers",
                    min_value=1,
                    value=2,
                    step=1,
                    key="audio_sum_min_val",
                )
                if use_min
                else None
            )
        with col_spk2:
            use_max = st.checkbox(
                "Set maximum speakers", key="audio_sum_max_chk"
            )
            max_speakers = (
                st.number_input(
                    "Maximum speakers",
                    min_value=1,
                    value=4,
                    step=1,
                    key="audio_sum_max_val",
                )
                if use_max
                else None
            )

    # Early validation
    if converted_model_missing(language, whisper_model):
        st.warning(
            "The Flurin Swiss German Turbo model has not been converted yet. "
            "Run ct2-transformers-converter first and confirm the directory "
            "exists at: "
            f"`{whisper_model}`"
        )

    # --- GPU memory status and pre-flight clear ---
    _render_gpu_clear_section()

    # Compute a config signature to detect when inputs change
    audio_sig = (
        f"{audio_file.name}:{getattr(audio_file, 'size', 0)}"
        if audio_file
        else "none"
    )
    current_sig = f"{audio_sig}_{language}_{whisper_model}"

    if st.session_state.get("audio_sum_last_sig") != current_sig:
        # Config changed - clear previous transcription so it reruns
        st.session_state.pop("audio_sum_transcript", None)
        st.session_state.pop("audio_sum_summary_error", None)

    # --- Run button ---
    run_col, _ = st.columns([1, 3])
    with run_col:
        run_clicked = st.button(
            "Transcribe and Summarize",
            type="primary",
            use_container_width=True,
            key="audio_sum_run",
        )

    if run_clicked:
        if audio_file is None:
            st.error("Please upload an audio file before running.")
            st.stop()
        free_gpu_for(gpu_manager.TRANSCRIBE)

        try:
            # Read once; used for language detection and transcription.
            audio_bytes = audio_file.getvalue()

            # --- Step 1: auto-detect language if requested ---
            if is_auto_detect:
                with st.status(
                    "Detecting language...", expanded=False
                ) as det_status:
                    detected_code, detected_prob = detect_language_from_bytes(
                        audio_bytes, sr=SAMPLE_RATE
                    )
                    language = detected_code or "en"
                    if language in ("ch_de", "ch_de_flurin"):
                        language = "de"
                    lang_name = LANGUAGE_CODE_TO_NAME.get(language, language)
                    det_status.update(
                        label=(
                            f"Detected language: {lang_name} "
                            f"({detected_prob:.0%} confidence)."
                        ),
                        state="complete",
                        expanded=False,
                    )

            # --- Step 2: Transcribe ---
            transcription_result = _run_transcription(
                audio_bytes=audio_bytes,
                audio_name=audio_file.name,
                language=language,
                whisper_model=whisper_model,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )

            # Keep the transcription for the re-summarize workflow.
            st.session_state["audio_sum_transcript"] = transcription_result
            st.session_state["audio_sum_audio_name"] = audio_file.name
            st.session_state["audio_sum_last_sig"] = current_sig

        except Exception as exc:
            st.error(f"Transcription failed: {exc}")
            st.code(getattr(exc, "details", "") or traceback.format_exc())
            st.stop()

    # --- Speaker labeling (after transcription, before summarization) ---
    transcript_data = st.session_state.get("audio_sum_transcript")

    if transcript_data is not None:
        csv_text = transcript_data["transcription_csv"]
        has_speakers = transcript_data["has_speakers"]
        duration_str = transcript_data["duration_str"]
        audio_name = st.session_state.get("audio_sum_audio_name", "audio")

        speaker_text = transcript_csv_to_speaker_text(csv_text)
        unique_speakers = (
            extract_unique_speakers(speaker_text) if has_speakers else []
        )

        label_map: dict[str, str] = {}

        if has_speakers and unique_speakers:
            with st.expander(
                f"Name the speakers (optional) - {len(unique_speakers)} "
                "speaker(s) detected",
                expanded=False,
            ):
                st.caption(
                    "Assign readable names to each speaker. These names will "
                    "appear "
                    "in the summary and transcript export."
                )
                n_cols = min(len(unique_speakers), 3)
                speaker_cols = st.columns(n_cols)
                for idx, spk in enumerate(unique_speakers):
                    with speaker_cols[idx % n_cols]:
                        label = st.text_input(
                            spk,
                            value=spk,
                            key=f"audio_sum_spk_{spk}",
                            help=f"Custom display name for {spk}.",
                        )
                        label_map[spk] = label

        # Apply speaker labels to transcript text
        display_transcript = (
            apply_speaker_labels(speaker_text, label_map)
            if label_map
            else speaker_text
        )
        plain_transcript = transcription_text_from_csv(csv_text)
        word_count = len(plain_transcript.split())

        # Build speaker context for the LLM prompt
        labelled_speakers = [label_map.get(s, s) for s in unique_speakers]
        speaker_context = (
            build_speaker_context(labelled_speakers) if has_speakers else None
        )

        # --- Summarize on a button click, or when there is no summary yet ---
        # If settings (model/mode/language) change after a summary exists,
        # show the existing result with a notice and let the user decide to
        # re-summarize. After a failure, summarization only runs again when
        # the user asks for it, so ordinary widget interactions do not
        # restart a long LLM job.
        existing_summary = st.session_state.get("audio_sum_summary")
        summary_error = st.session_state.get("audio_sum_summary_error")
        existing_mode = st.session_state.get("audio_sum_mode_used")
        existing_model = st.session_state.get("audio_sum_model_used")
        existing_lang = st.session_state.get("audio_sum_lang_used")

        settings_changed = existing_summary is not None and (
            (existing_mode != mode_key)
            or (existing_model != model_name)
            or (existing_lang != output_language)
        )

        if run_clicked or (existing_summary is None and summary_error is None):
            st.divider()
            st.write("**Generating summary...**")
            try:
                summary = _run_summarization(
                    model_name=model_name,
                    transcript_text=display_transcript,
                    mode_key=mode_key,
                    speaker_context=speaker_context,
                    output_language=output_language,
                )
            except Exception as exc:
                # Drop any summary of a previous recording; keep the error so
                # the transcript stays available below.
                st.session_state.pop("audio_sum_summary", None)
                summary_error = {
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                st.session_state["audio_sum_summary_error"] = summary_error
                existing_summary = None
            else:
                st.session_state["audio_sum_summary"] = summary
                st.session_state["audio_sum_mode_used"] = mode_key
                st.session_state["audio_sum_model_used"] = model_name
                st.session_state["audio_sum_lang_used"] = output_language
                st.session_state["audio_sum_display_transcript"] = (
                    display_transcript
                )
                st.session_state["audio_sum_word_count"] = word_count
                st.session_state["audio_sum_duration_str"] = duration_str
                st.session_state["audio_sum_source_label"] = audio_name
                st.session_state.pop("audio_sum_summary_error", None)
                st.rerun()

        if existing_summary is None and summary_error is not None:
            st.divider()
            st.error(f"Summarization failed: {summary_error['message']}")
            with st.expander("Error details"):
                st.code(summary_error["traceback"])

            retry_col, _ = st.columns([1, 3])
            with retry_col:
                if st.button(
                    "Retry summarization",
                    key="audio_sum_retry",
                    use_container_width=True,
                    help=(
                        "Summarize the existing transcript again without "
                        "re-transcribing."
                    ),
                ):
                    st.session_state.pop("audio_sum_summary_error", None)
                    st.rerun()

            st.write("**Transcript**")
            st.text_area(
                "Transcript",
                value=display_transcript,
                height=420,
                disabled=True,
                label_visibility="collapsed",
                key="audio_sum_failed_transcript",
            )
            _render_raw_transcript_downloads(csv_text, audio_name)

        elif existing_summary:
            st.divider()

            re_col, _ = st.columns([1, 3])
            with re_col:
                if st.button(
                    "Re-summarize with current settings",
                    key="audio_sum_rerun",
                    use_container_width=True,
                ):
                    st.session_state.pop("audio_sum_summary", None)
                    st.rerun()

            if settings_changed:
                st.info(
                    "Settings have changed. Click **Re-summarize** above to "
                    "regenerate "
                    "with the new model, type, or language."
                )

            _render_summary_results(
                summary=st.session_state["audio_sum_summary"],
                transcript_text=st.session_state.get(
                    "audio_sum_display_transcript", display_transcript
                ),
                source_label=st.session_state.get(
                    "audio_sum_source_label", audio_name
                ),
                mode_key=st.session_state.get("audio_sum_mode_used", mode_key),
                duration_str=st.session_state.get(
                    "audio_sum_duration_str", duration_str
                ),
                word_count=st.session_state.get(
                    "audio_sum_word_count", word_count
                ),
                key_prefix="audio_sum",
            )
            _render_raw_transcript_downloads(csv_text, audio_name)


def _render_raw_transcript_downloads(csv_text: str, audio_name: str) -> None:
    """Render download buttons for the raw WhisperX transcription output.

    Args:
        csv_text: The WhisperX TSV transcription.
        audio_name: Original audio filename, used to name the downloads.
    """
    with st.expander("Download raw transcription files"):
        base_name = os.path.splitext(audio_name)[0]
        col_tc1, col_tc2 = st.columns(2)
        with col_tc1:
            st.download_button(
                label="Transcript CSV (WhisperX)",
                data=csv_text,
                file_name=f"{base_name}_transcription.csv",
                mime="text/plain",
                use_container_width=True,
                on_click="ignore",
                key="audio_sum_csv_dl",
            )
        with col_tc2:
            st.download_button(
                label="Plain transcript (.txt)",
                data=transcription_text_from_csv(csv_text),
                file_name=f"{base_name}_transcript.txt",
                mime="text/plain",
                use_container_width=True,
                on_click="ignore",
                key="audio_sum_txt_dl",
            )


# ---------------------------------------------------------------------------
# "From Transcript" tab
# ---------------------------------------------------------------------------


def _render_transcript_tab(gpu_name: str) -> None:
    """Render the 'From Transcript' workflow tab.

    Accepts a WhisperX CSV from the Transcribe page, a plain text file,
    or directly typed/pasted text. Runs only the summarization step.

    Args:
        gpu_name: Current GPU name for model filtering.
    """
    st.write(
        "Already have a transcript? Paste text or upload a file exported from "
        "the Transcribe page. The tool will generate a structured summary "
        "instantly."
    )

    input_method = st.radio(
        "Input method",
        [
            "Upload transcript CSV (from Transcribe page)",
            "Upload text file",
            "Paste text",
        ],
        horizontal=True,
        key="ts_input_method",
    )

    transcript_text = ""
    has_speakers = False
    source_label = "transcript"

    if input_method == "Upload transcript CSV (from Transcribe page)":
        uploaded_csv = st.file_uploader(
            "Transcript CSV/TSV",
            type=["csv", "tsv"],
            key="ts_csv_upload",
            help=(
                "Upload the _transcription.csv file produced by the Transcribe"
                " page."
            ),
        )
        if uploaded_csv is not None:
            raw = uploaded_csv.read().decode("utf-8", errors="replace")
            speaker_text = transcript_csv_to_speaker_text(raw)
            unique_speakers = extract_unique_speakers(speaker_text)
            has_speakers = bool(unique_speakers)
            source_label = uploaded_csv.name

            if has_speakers:
                with st.expander(
                    f"Name the speakers (optional) - {len(unique_speakers)} "
                    "detected",
                    expanded=False,
                ):
                    st.caption(
                        "Assign readable names. These will appear in the "
                        "summary."
                    )
                    n_cols = min(len(unique_speakers), 3)
                    cols = st.columns(n_cols)
                    label_map: dict[str, str] = {}
                    for idx, spk in enumerate(unique_speakers):
                        with cols[idx % n_cols]:
                            label = st.text_input(
                                spk,
                                value=spk,
                                key=f"ts_spk_{spk}",
                            )
                            label_map[spk] = label
                    transcript_text = apply_speaker_labels(
                        speaker_text, label_map
                    )
            else:
                transcript_text = speaker_text

    elif input_method == "Upload text file":
        uploaded_txt = st.file_uploader(
            "Text file",
            type=["txt"],
            key="ts_txt_upload",
        )
        if uploaded_txt is not None:
            transcript_text = uploaded_txt.read().decode(
                "utf-8", errors="replace"
            )
            source_label = uploaded_txt.name

    else:  # Paste text
        transcript_text = st.text_area(
            "Transcript text",
            placeholder="Paste your transcript here...",
            height=250,
            key="ts_paste",
        )
        source_label = "pasted_text"

    if transcript_text:
        word_count = len(transcript_text.split())
        st.caption(
            f"{word_count:,} words | ~{estimate_tokens(transcript_text):,} "
            "estimated tokens"
        )

    st.divider()

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        mode_key = _render_mode_selector("ts")
    with col_cfg2:
        model_name = _render_model_selector(gpu_name, "ts")
    with col_cfg3:
        output_language = _render_output_language_selector("ts")

    run_col, _ = st.columns([1, 3])
    with run_col:
        run_ts = st.button(
            "Generate Summary",
            type="primary",
            use_container_width=True,
            key="ts_run",
            disabled=(not transcript_text.strip()),
        )

    if run_ts:
        if not transcript_text.strip():
            st.error("Please provide a transcript before running.")
            st.stop()

        unique_spks = extract_unique_speakers(transcript_text)
        speaker_context = (
            build_speaker_context(unique_spks) if unique_spks else None
        )

        st.divider()
        st.write("**Generating summary...**")
        try:
            summary = _run_summarization(
                model_name=model_name,
                transcript_text=transcript_text,
                mode_key=mode_key,
                speaker_context=speaker_context,
                output_language=output_language,
            )
        except Exception as exc:
            st.error(f"Summarization failed: {exc}")
            st.code(traceback.format_exc())
            st.stop()

        # Keep the result across reruns so it survives later widget
        # interactions; the rerun replaces the streamed text with the results.
        st.session_state["ts_result"] = {
            "summary": summary,
            "transcript_text": transcript_text,
            "source_label": source_label,
            "mode_key": mode_key,
            "model_name": model_name,
            "output_language": output_language,
            "word_count": len(transcript_text.split()),
        }
        st.rerun()

    result = st.session_state.get("ts_result")
    if result is not None:
        st.divider()
        input_changed = (
            result["transcript_text"] != transcript_text
            or result["mode_key"] != mode_key
            or result["model_name"] != model_name
            or result["output_language"] != output_language
        )
        if input_changed:
            st.info(
                "The transcript or settings have changed since this summary "
                "was "
                "generated. Click **Generate Summary** to regenerate it."
            )
        _render_summary_results(
            summary=result["summary"],
            transcript_text=result["transcript_text"],
            source_label=result["source_label"],
            mode_key=result["mode_key"],
            duration_str="",
            word_count=result["word_count"],
            key_prefix="ts",
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Render the Meeting Notes Generator page.

    Validates the Ollama server connection, then presents two workflow tabs:
    'From Audio' for the full transcription-plus-summarization pipeline and
    'From Transcript' for summarizing existing text.
    """
    st.title("Meeting Notes Generator")

    st.markdown(
        """
        This tool combines **transcription** and **AI summarization** into
        a single automated pipeline. Instead of transcribing audio on one
        page and then copying the result into the Chat page, you can upload
        an audio file here and receive both a full transcript and a
        structured summary in one step.

        All processing happens entirely within the University network -
        your audio and text data never leave the infrastructure.

        **Available summary types**
        """
    )

    mode_col1, mode_col2 = st.columns(2)
    mode_items = list(SUMMARY_MODES.items())
    for idx, (_key, mode) in enumerate(mode_items):
        col = mode_col1 if idx % 2 == 0 else mode_col2
        with col:
            st.markdown(f"**{mode['label']}** - {mode['description']}")

    st.divider()

    # Ollama server check
    if not check_ollama_server():
        st.error(
            "Could not connect to the Ollama server. Please check ollama.log "
            "in this session's folder (see Reporting a Problem in the user "
            "guide)."
        )
        st.stop()

    gpu_name = gpu_manager.get_gpu_name()
    if is_high_memory_gpu(gpu_name):
        st.caption(f"High-performance mode ({gpu_name}).")
    else:
        st.caption(f"Standard mode ({gpu_name}). Large models are hidden.")

    tab_audio, tab_transcript = st.tabs(["From Audio", "From Transcript"])

    with tab_audio:
        _render_audio_tab(gpu_name)

    with tab_transcript:
        _render_transcript_tab(gpu_name)


if __name__ == "__main__":
    main()
