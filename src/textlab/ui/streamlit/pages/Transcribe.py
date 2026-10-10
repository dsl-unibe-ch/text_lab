"""Transcribe page: transcribe recordings with WhisperX and review them.

Three workflows share the page: transcribing one recording, transcribing a
ZIP batch, and reviewing an existing transcript against its audio. The
transcription itself runs in a worker process through
:mod:`textlab.features.transcription.service`; this page collects the
settings, shows progress and offers the results.
"""

import os
import traceback

import streamlit as st

from textlab.common import gpu_manager
from textlab.common.language_mappings import (
    TRANSCRIBE_LANGUAGE_CODE_TO_NAME as LANGUAGE_CODE_TO_NAME,
)
from textlab.common.language_mappings import (
    TRANSCRIBE_LANGUAGE_MAPPING as LANGUAGE_MAPPING,
)
from textlab.features.transcription import formats
from textlab.features.transcription.audio import (
    SAMPLE_RATE,
    convert_audio_to_wav_bytes,
    detect_language_from_bytes,
)
from textlab.features.transcription.models import TranscriptionOptions
from textlab.features.transcription.service import (
    AUDIO_EXTENSIONS,
    run_transcription,
    staged_uploads,
    staged_zip,
)
from textlab.features.transcription.whisper_models import (
    converted_model_missing,
    default_model,
)
from textlab.ui.streamlit.auth import check_token
from textlab.ui.streamlit.components.audio_player import (
    audio_data_url,
    build_player_html,
    create_wavesurfer_preview,
)
from textlab.ui.streamlit.components.gpu import free_gpu_for
from textlab.ui.streamlit.components.progress import StatusBox

#: Upload types for the file uploaders, without the leading dot.
AUDIO_TYPES = [extension.lstrip(".") for extension in AUDIO_EXTENSIONS]

#: How a Text Lab language code is described under the model settings.
MODEL_NOTES = {
    "ch_de": "Using Swiss German Whisper model",
    "ch_de_flurin": "Using Flurin Swiss German Whisper Turbo model (CT2)",
}
DEFAULT_MODEL_NOTE = "Using default Large v3 Turbo whisper model"


def main():
    """Render the page."""
    st.set_page_config(page_title="Transcribe", layout="wide")
    check_token()
    st.title("Transcribe")

    workflow_mode = st.radio(
        "Workflow",
        [
            "Transcribe audio",
            "Batch transcribe (ZIP)",
            "Upload existing transcription",
        ],
        index=0,
        horizontal=True,
        help=(
            "Choose to transcribe new audio, batch process a zip, or review "
            "existing files"
        ),
    )

    st.divider()

    if workflow_mode == "Upload existing transcription":
        _review_existing()
    elif workflow_mode == "Batch transcribe (ZIP)":
        _transcribe_batch()
    else:
        _transcribe_single()


# ---------------------------------------------------------------------------
# Shared settings widgets
# ---------------------------------------------------------------------------


def _model_settings(language, key_prefix=""):
    """Render the model and VAD settings.

    Args:
        language: The chosen Text Lab language code, or ``None`` for
            auto-detect.
        key_prefix: Prefix for widget keys; ``"batch_"`` on the batch tab.

    Returns:
        ``(whisper_model, vad_max_pause)``; ``vad_max_pause`` is ``None``
        when VAD pre-filtering is off.
    """
    st.write("**Model Configuration**")
    model = default_model(language)
    st.info(MODEL_NOTES.get(language, DEFAULT_MODEL_NOTE))

    keys = _setting_keys(key_prefix)
    if st.checkbox(
        "Use custom model path", value=False, key=keys["custom_model"]
    ):
        model = st.text_input(
            "Custom model path",
            value=model,
            help="Enter model name or path",
            key=keys["model_path"],
        )

    vad_max_pause = None
    if st.checkbox(
        "Use VAD pre-filtering",
        value=False,
        help="Use Silero VAD to filter speech segments before transcription",
        key=keys["vad"],
    ):
        vad_max_pause = st.slider(
            "VAD max pause (seconds)",
            min_value=0.1,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Maximum pause duration to merge VAD segments",
            key=keys["vad_pause"],
        )
    return model, vad_max_pause


def _speaker_settings(key_prefix=""):
    """Render the optional speaker-count settings for diarization.

    Args:
        key_prefix: Prefix for widget keys; ``"batch_"`` on the batch tab.

    Returns:
        ``(min_speakers, max_speakers)``, each ``None`` when not set.
    """
    st.write("**Speaker Diarization**")
    keys = _setting_keys(key_prefix)
    bounds = []
    for label, short, default, check_key, value_key in (
        ("Set min speakers", "Min", 2, keys["min_check"], keys["min_value"]),
        ("Set max speakers", "Max", 4, keys["max_check"], keys["max_value"]),
    ):
        col_check, col_value = st.columns([1, 1])
        with col_check:
            enabled = st.checkbox(label, value=False, key=check_key)
        with col_value:
            bounds.append(
                st.number_input(
                    short,
                    min_value=1,
                    value=default,
                    step=1,
                    label_visibility="collapsed",
                    key=value_key,
                )
                if enabled
                else None
            )
    return bounds[0], bounds[1]


def _setting_keys(key_prefix):
    """Return the widget keys of the settings.

    The batch tab uses explicit keys. The single-file tab has always let
    Streamlit derive keys from the labels, which ``None`` keeps doing.
    """
    if key_prefix == "batch_":
        return {
            "custom_model": "batch_custom_mod",
            "model_path": "batch_mod_path",
            "vad": "batch_vad",
            "vad_pause": "batch_vad_pause",
            "min_check": "batch_min_spk_chk",
            "min_value": "batch_min_spk",
            "max_check": "batch_max_spk_chk",
            "max_value": "batch_max_spk",
        }
    return dict.fromkeys(
        (
            "custom_model",
            "model_path",
            "vad",
            "vad_pause",
            "min_check",
            "min_value",
            "max_check",
            "max_value",
        )
    )


def _show_result_notes(result):
    """Show skipped files and notes from a transcription run."""
    for skipped in result.skipped:
        st.warning(f"Skipped {skipped.name} - {skipped.reason}.")
    for note in result.notes:
        st.info(note)


# ---------------------------------------------------------------------------
# Upload existing transcription
# ---------------------------------------------------------------------------


def _review_existing():
    """Play a recording alongside an existing transcript."""
    st.write(
        "Load audio and a WhisperX CSV/TSV to review alignment with playback."
    )

    col1, col2 = st.columns(2)
    with col1:
        audio_path = st.text_input(
            "Audio file path (server-side)", value="", key="upload_audio_path"
        )
        audio_upload = st.file_uploader(
            "Or upload audio", type=AUDIO_TYPES, key="upload_audio"
        )
    with col2:
        transcript_path = st.text_input(
            "Transcript CSV/TSV path (server-side)",
            value="",
            key="upload_transcript_path",
        )
        transcript_upload = st.file_uploader(
            "Or upload CSV/TSV", type=["csv", "tsv"], key="upload_transcript"
        )

    display_mode = st.radio(
        "Display transcript as",
        ["segments", "words"],
        index=0,
        horizontal=True,
        help="Segments = sentence-level view, Words = word-by-word view",
        key="upload_mode",
    )

    audio_bytes = audio_label = audio_wave = wav_bytes = wav_filename = None
    if audio_upload is not None:
        audio_bytes = audio_upload.read()
        audio_label = audio_upload.name
    elif audio_path:
        if os.path.exists(audio_path):
            with open(audio_path, "rb") as handle:
                audio_bytes = handle.read()
            audio_label = audio_path
        else:
            st.error("Audio path does not exist.")
    if audio_bytes:
        with st.spinner("Converting audio to WAV..."):
            wav_bytes, wav_filename, audio_wave = convert_audio_to_wav_bytes(
                audio_bytes, os.path.basename(audio_label)
            )

    transcript_text = transcript_label = None
    if transcript_upload is not None:
        transcript_text = transcript_upload.read().decode(
            "utf-8", errors="replace"
        )
        transcript_label = transcript_upload.name
    elif transcript_path:
        if os.path.exists(transcript_path):
            with open(
                transcript_path, encoding="utf-8", errors="replace"
            ) as handle:
                transcript_text = handle.read()
            transcript_label = transcript_path
        else:
            st.error("Transcript path does not exist.")

    if (
        audio_bytes
        and transcript_text
        and wav_bytes
        and audio_wave is not None
    ):
        st.success(f"Loaded audio: {audio_label}")
        st.success(f"Loaded transcript: {transcript_label}")
        items, has_speakers = formats.load_transcript_items(
            transcript_text, display_mode
        )
        if display_mode == "words":
            player_mode = "word"
        else:
            player_mode = "diarization" if has_speakers else "segments"

        player_wav_bytes, preview_seconds = create_wavesurfer_preview(
            wav_bytes, audio_wave, sr=SAMPLE_RATE
        )
        if preview_seconds is not None:
            st.warning(
                "Waveform preview limited to the first "
                f"{formats.format_duration(preview_seconds)} to avoid large "
                "in-browser audio. Full audio is unchanged."
            )
        _render_player(
            player_wav_bytes, wav_filename or audio_label, items, player_mode
        )


def _render_player(wav_bytes, file_name, items, player_mode):
    """Render the waveform player with the synchronized transcript."""
    st.components.v1.html(
        build_player_html(
            audio_data_url(wav_bytes, file_name), items, player_mode
        ),
        height=520,
        scrolling=True,
    )


# ---------------------------------------------------------------------------
# Batch transcribe (ZIP)
# ---------------------------------------------------------------------------


def _transcribe_batch():
    """Transcribe every recording in a ZIP file."""
    st.write(
        "Upload a ZIP file containing multiple audio files. They will all be "
        "transcribed securely in bulk."
    )

    col1, col2 = st.columns(2)
    with col1:
        batch_zip = st.file_uploader(
            "Upload ZIP file", type=["zip"], key="batch_zip_upload"
        )
    with col2:
        language_name = st.selectbox(
            "Language (applied to all files)",
            ["Auto-detect", *LANGUAGE_MAPPING],
            key="batch_language_name",
            help=(
                "Use Auto-detect to decipher each file dynamically, or force "
                "a specific language."
            ),
        )
    language = LANGUAGE_MAPPING.get(language_name)

    col_config1, col_config2 = st.columns(2)
    with col_config1:
        whisper_model, vad_max_pause = _model_settings(language, "batch_")
    with col_config2:
        min_speakers, max_speakers = _speaker_settings("batch_")

    if st.button("Start Batch Transcription", type="primary"):
        free_gpu_for(gpu_manager.TRANSCRIBE)
        if batch_zip is None:
            st.error("Please upload a ZIP file first.")
        elif converted_model_missing(language, whisper_model):
            st.error(
                f"Flurin CT2 model not found at `{whisper_model}`. Please "
                "convert the model first using `ct2-transformers-converter` "
                "before running batch transcription."
            )
        else:
            options = TranscriptionOptions(
                model=whisper_model,
                language=language,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                vad_max_pause=vad_max_pause,
            )
            _run_batch(batch_zip, options)

    if st.session_state.get("batch_out_zip"):
        st.success("Batch Transcription completed successfully!")
        st.download_button(
            "Download All Transcripts (ZIP)",
            st.session_state.batch_out_zip,
            file_name="batch_transcriptions.zip",
            mime="application/zip",
            use_container_width=True,
            type="primary",
        )


def _run_batch(batch_zip, options):
    """Transcribe a ZIP batch, showing progress, and store the result ZIP."""
    try:
        with staged_zip(batch_zip) as files:
            if not files:
                st.error(
                    f"No valid audio files ({', '.join(AUDIO_EXTENSIONS)}) "
                    "found in the ZIP."
                )
                return
            st.info(
                f"Found {len(files)} audio file(s). Starting batch process..."
            )
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def report(progress):
                status_text.text(progress.message)
                if progress.fraction is not None:
                    progress_bar.progress(progress.fraction)

            result = run_transcription(files, options, on_progress=report)
        _show_result_notes(result)
        progress_bar.progress(1.0)
        status_text.text("Batch transcription complete!")
        st.session_state.batch_out_zip = formats.batch_zip(result.transcripts)
    except Exception as exc:
        st.error(f"Batch transcription failed: {exc}")
        st.code(getattr(exc, "details", "") or traceback.format_exc())


# ---------------------------------------------------------------------------
# Transcribe audio
# ---------------------------------------------------------------------------


def _transcribe_single():
    """Transcribe one recording and show it in the player."""
    st.write(
        "Upload an audio file to transcribe using WhisperX, then review the "
        "results."
    )

    if "transcribe_language_name" not in st.session_state:
        st.session_state.transcribe_language_name = "German"

    col1, col2 = st.columns(2)
    with col1:
        upload = st.file_uploader(
            "Upload audio file", type=AUDIO_TYPES, key="transcribe_audio"
        )
    if upload is not None:
        _detect_upload_language(upload)

    with col2:
        language_name = st.selectbox(
            "Language",
            list(LANGUAGE_MAPPING.keys()),
            key="transcribe_language_name",
            help=(
                "Select language. Swiss German automatically uses the Swiss "
                "German model."
            ),
        )
        language = LANGUAGE_MAPPING[language_name]
        message = st.session_state.get("transcribe_lang_detect_msg")
        if upload is not None and message:
            if (
                st.session_state.get("transcribe_lang_detect_level")
                == "warning"
            ):
                st.warning(message)
            else:
                st.info(message)

    col_config1, col_config2 = st.columns(2)
    with col_config1:
        whisper_model, vad_max_pause = _model_settings(language)
    with col_config2:
        min_speakers, max_speakers = _speaker_settings()

    use_vad = vad_max_pause is not None
    current_config = (
        f"{upload.name if upload else 'none'}_{language}_{whisper_model}_"
        f"{use_vad}_{vad_max_pause if use_vad else 0.5}"
    )
    if st.session_state.get("last_config") != current_config:
        st.session_state.transcription_results = None

    if converted_model_missing(language, whisper_model):
        st.warning(
            "The Flurin Swiss German Turbo model has not been converted yet. "
            "Run the `ct2-transformers-converter` command described in the "
            "docs first, then verify that the output folder exists at:\n"
            f"`{whisper_model}`"
        )

    if st.button("Start Transcription", type="primary"):
        free_gpu_for(gpu_manager.TRANSCRIBE)
        if upload is None:
            st.error("Please upload an audio file first.")
        elif converted_model_missing(language, whisper_model):
            st.error(
                f"Flurin CT2 model not found at `{whisper_model}`. Please "
                "convert the model first - see the warning above."
            )
        else:
            options = TranscriptionOptions(
                model=whisper_model,
                language=language,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                vad_max_pause=vad_max_pause,
            )
            _run_single(upload, options, current_config)

    if st.session_state.get("transcription_results"):
        _render_single_results(st.session_state.transcription_results)


def _detect_upload_language(upload):
    """Detect a new upload's language once and preselect it."""
    signature = f"{upload.name}:{getattr(upload, 'size', 'unknown')}"
    if st.session_state.get("transcribe_lang_detect_sig") == signature:
        return
    with st.spinner("Detecting language..."):
        try:
            code, probability = detect_language_from_bytes(
                upload.getvalue(), sr=SAMPLE_RATE
            )
        except Exception as exc:
            st.session_state.transcribe_lang_detect_sig = signature
            st.session_state.transcribe_lang_detect_msg = (
                f"Could not auto-detect language ({exc}). Please set it "
                "manually."
            )
            st.session_state.transcribe_lang_detect_level = "warning"
            return
    st.session_state.transcribe_lang_detect_sig = signature
    if code in LANGUAGE_CODE_TO_NAME:
        name = LANGUAGE_CODE_TO_NAME[code]
        st.session_state.transcribe_language_name = name
        st.session_state.transcribe_lang_detect_msg = (
            f"Auto-detected language: {name} ({code}, confidence "
            f"{probability:.0%}). You can still change it manually."
        )
        st.session_state.transcribe_lang_detect_level = "info"
    else:
        st.session_state.transcribe_lang_detect_msg = (
            f"Detected language code '{code}' is not in the selectable list. "
            "Please select language manually."
        )
        st.session_state.transcribe_lang_detect_level = "warning"


def _run_single(upload, options, current_config):
    """Transcribe one upload and store everything the results view needs."""
    try:
        audio_bytes = upload.getvalue()
        with st.spinner("Converting audio to WAV..."):
            wav_bytes, wav_filename, samples = convert_audio_to_wav_bytes(
                audio_bytes, upload.name, sr=SAMPLE_RATE
            )
        player_wav_bytes, preview_seconds = create_wavesurfer_preview(
            wav_bytes, samples, sr=SAMPLE_RATE
        )

        with staged_uploads([(upload.name, audio_bytes)]) as files:
            with StatusBox(
                "Transcribing audio...",
                complete_label="Transcription complete.",
                error_label="Transcription failed.",
            ) as status:
                result = run_transcription(files, options, on_progress=status)
        _show_result_notes(result)
        if not result.transcripts:
            st.error("Transcription failed: the audio could not be decoded.")
            return

        st.session_state.transcription_results = {
            "transcript": result.transcripts[0],
            "wav_bytes": wav_bytes,
            "player_wav_bytes": player_wav_bytes,
            "preview_seconds": preview_seconds,
            "wav_filename": wav_filename,
            "audio_name": upload.name,
        }
        st.session_state.last_config = current_config
        st.success("Transcription completed!")
    except Exception as exc:
        st.error(f"Transcription failed: {exc}")
        st.code(getattr(exc, "details", "") or traceback.format_exc())


def _render_single_results(results):
    """Show the player and the downloads for a finished transcription."""
    transcript = results["transcript"]
    base_name = os.path.splitext(results["audio_name"])[0]
    files = formats.transcript_files(transcript, base_name)

    display_mode = st.radio(
        "Display transcript as",
        ["segments", "words"],
        index=0,
        horizontal=True,
        key="transcribe_mode",
        help="Segments = sentence-level view, Words = word-by-word view",
    )
    if display_mode == "words":
        items, _ = formats.load_transcript_items(
            files[f"{base_name}_words.csv"], "words"
        )
        player_mode = "word"
    else:
        items, _ = formats.load_transcript_items(
            files[f"{base_name}_transcription.csv"], "segments"
        )
        player_mode = "diarization" if transcript.has_speakers else "segments"

    if results["preview_seconds"] is not None:
        st.warning(
            "Waveform preview limited to the first "
            f"{formats.format_duration(results['preview_seconds'])} to avoid "
            "large in-browser audio. Transcription uses the full audio."
        )
    _render_player(
        results["player_wav_bytes"] or results["wav_bytes"],
        results["wav_filename"],
        items,
        player_mode,
    )

    archive = dict(files)
    if results["wav_bytes"]:
        archive[results["wav_filename"] or f"{base_name}.wav"] = results[
            "wav_bytes"
        ]
    text_output = files[f"{base_name}_text.txt"]

    col_text, col_zip = st.columns(2)
    with col_text:
        st.download_button(
            "Download text (.txt)",
            text_output,
            file_name=f"{base_name}_text.txt",
            mime="text/plain",
            key="download_text_txt",
            use_container_width=True,
        )
    with col_zip:
        st.download_button(
            "Download all outputs (ZIP)",
            formats.zip_bytes(archive),
            file_name=f"{base_name}_outputs.zip",
            mime="application/zip",
            key="download_outputs_zip",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
