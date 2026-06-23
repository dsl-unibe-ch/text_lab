"""
Standalone transcription worker for the Meeting Notes Generator.

Runs the full WhisperX pipeline (transcription, alignment, diarization) as an
isolated subprocess so all GPU/CUDA memory is forcibly released when this
process exits. The parent process (Streamlit page) can then hand the GPU back
to Ollama for the summarization step without any VRAM contention.

Usage (called by Meeting_Notes_Generator.py - do not invoke manually)::

    python transcribe_worker.py <config_json> <output_json> <status_json>

Arguments:
    config_json  Path to a JSON file containing all input parameters.
    output_json  Path where this script writes its JSON result on success.
    status_json  Path to a JSON file that this script overwrites with the
                 current progress message so the parent can poll it.

Exit codes:
    0  Success - output_json has been written.
    1  Failure - output_json contains an ``"error"`` key with the traceback.
"""

import json
import os
import sys
import traceback

# Must be set before importing torch / whisperx.
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"


def _write_status(status_path: str, message: str) -> None:
    """
    Overwrite the status file with the current progress message.

    Args:
        status_path: Filesystem path of the status JSON file.
        message: Human-readable progress message for the parent process.
    """
    try:
        with open(status_path, "w", encoding="utf-8") as fh:
            json.dump({"message": message}, fh)
    except OSError:
        pass  # Best-effort; parent will fall back to a static spinner.


def _write_result(output_path: str, payload: dict) -> None:
    """
    Write the final result or error payload to the output JSON file.

    Args:
        output_path: Filesystem path of the output JSON file.
        payload: Dict with keys ``transcription_csv``, ``has_speakers``,
            ``duration_str``, and optionally ``error``.
    """
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)


def main() -> None:
    """
    Entry point for the transcription worker subprocess.

    Reads configuration from ``sys.argv[1]``, executes the WhisperX pipeline,
    and writes results to ``sys.argv[2]``. Progress messages are written to
    ``sys.argv[3]`` for the parent to poll.
    """
    if len(sys.argv) < 4:
        print(
            "Usage: transcribe_worker.py <config_json> <output_json> <status_json>",
            file=sys.stderr,
        )
        sys.exit(1)

    config_path = sys.argv[1]
    output_path = sys.argv[2]
    status_path = sys.argv[3]

    # --- Load configuration ---
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    src_dir = cfg["src_dir"]
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    # Deferred imports so the path is set before the project modules load.
    import torch
    import whisperx
    from whisperx import alignment

    # Extend alignment registry for Ukrainian (mirrors the page module).
    alignment.DEFAULT_ALIGN_MODELS_HF["uk"] = "Yehor/w2v-xls-r-uk"

    from core.transcribe_engine import (
        convert_audio_to_wav_bytes,
        format_duration,
        generate_transcription_csv,
        read_hf_token,
    )

    try:
        audio_path: str = cfg["audio_path"]
        audio_name: str = cfg["audio_name"]
        language: str = cfg["language"]
        align_language: str = cfg["align_language"]
        whisper_model: str = cfg["whisper_model"]
        min_speakers = cfg.get("min_speakers")
        max_speakers = cfg.get("max_speakers")
        hf_token_path: str = cfg["hf_token_path"]

        sampling_rate = 16000
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "float32"

        # --- Step 1: decode audio ---
        _write_status(status_path, "Decoding and converting audio...")
        with open(audio_path, "rb") as fh:
            audio_bytes = fh.read()

        _wav_bytes, _wav_filename, audio = convert_audio_to_wav_bytes(
            audio_bytes, audio_name, sr=sampling_rate
        )
        if audio is None or audio.size == 0:
            raise ValueError(
                "Audio decoding produced an empty waveform. "
                "Check that the file is a valid audio format."
            )

        duration_secs = len(audio) / float(sampling_rate)
        duration_str = format_duration(duration_secs)

        # --- Step 2: transcribe ---
        # Use only the basename when the model is a local path so internal
        # storage paths are never surfaced in the UI status messages.
        whisper_model_label = os.path.basename(whisper_model) if os.sep in whisper_model else whisper_model
        _write_status(status_path, f"Loading WhisperX model ({whisper_model_label})...")
        model = whisperx.load_model(
            whisper_model,
            device=device,
            compute_type=compute_type,
            language=align_language,
        )

        _write_status(status_path, "Transcribing audio...")
        result = model.transcribe(audio, batch_size=16)

        # Release main model from VRAM before loading alignment model.
        del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Step 3: alignment ---
        _write_status(status_path, "Aligning word-level timestamps...")
        model_a, metadata = whisperx.load_align_model(
            language_code=align_language, device=device
        )
        aligned = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )

        del model_a
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Step 4: diarization (optional) ---
        has_speakers = False
        hf_token = read_hf_token(hf_token_path)
        if hf_token:
            _write_status(status_path, "Running speaker diarization...")
            diarize_model = whisperx.diarize.DiarizationPipeline(
                use_auth_token=hf_token, device=device
            )
            diarize_segments = diarize_model(
                audio, min_speakers=min_speakers, max_speakers=max_speakers
            )
            aligned = whisperx.assign_word_speakers(diarize_segments, aligned)
            has_speakers = True

            del diarize_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            _write_status(status_path, "HuggingFace token not found - skipping diarization.")

        # --- Step 5: generate CSV output ---
        _write_status(status_path, "Generating transcript...")
        transcription_csv = generate_transcription_csv(aligned, has_speakers=has_speakers)

        _write_result(
            output_path,
            {
                "status": "ok",
                "transcription_csv": transcription_csv,
                "has_speakers": has_speakers,
                "duration_str": duration_str,
                "error": None,
            },
        )
        _write_status(status_path, "Transcription complete.")

    except Exception:
        tb = traceback.format_exc()
        _write_result(
            output_path,
            {
                "status": "error",
                "transcription_csv": "",
                "has_speakers": False,
                "duration_str": "",
                "error": tb,
            },
        )
        _write_status(status_path, "Transcription failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
