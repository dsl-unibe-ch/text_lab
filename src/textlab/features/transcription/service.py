"""Transcribing recordings with WhisperX: the transcription feature's API.

There is one pipeline for every recording, whether it comes from the
Transcribe page, a ZIP batch or the Meeting Notes Generator:

1. decode the audio with ffmpeg,
2. detect the language if none was chosen,
3. transcribe with WhisperX, optionally only the speech found by voice
   activity detection,
4. align the words to the audio,
5. identify speakers, when a Hugging Face token is configured.

Interfaces call :func:`run_transcription`, which runs the pipeline in a
worker process (see :mod:`textlab.common.jobs`) so all GPU memory is
released afterwards. :func:`transcribe_files` is the pipeline itself; the
worker calls it, and so can a batch job that already runs in its own
process.

Uploaded recordings are first written to the job workspace with
:func:`staged_uploads` or :func:`staged_zip`, which remove them again.
"""

from __future__ import annotations

import contextlib
import gc
import os
import shutil
import threading
import zipfile
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

import numpy as np

from textlab.common.config import get_settings
from textlab.common.jobs import run_worker
from textlab.common.progress import (
    Progress,
    ProgressCallback,
    check_cancelled,
    no_progress,
)
from textlab.common.storage import get_workspace
from textlab.common.upload_safety import safe_upload_name
from textlab.features.transcription import audio
from textlab.features.transcription.models import (
    AudioFile,
    SkippedFile,
    Transcript,
    TranscriptionOptions,
    TranscriptionResult,
)
from textlab.features.transcription.whisper_models import (
    alignment_language,
    model_label,
)

#: Audio formats accepted in uploads and ZIP batches.
AUDIO_EXTENSIONS = (
    ".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm", ".oga", ".opus",
)  # fmt: skip

#: Module run by :func:`run_transcription`.
WORKER_MODULE = "textlab.features.transcription.worker"

#: Workspace area for staged recordings and worker jobs.
WORKSPACE_AREA = "transcription"

#: Language assumed when detection gives no answer.
FALLBACK_LANGUAGE = "en"

#: Alignment model for a language WhisperX has none for by default.
EXTRA_ALIGN_MODELS = {"uk": "Yehor/w2v-xls-r-uk"}

NO_TOKEN_NOTE = "HuggingFace token not found - skipping diarization."


# ---------------------------------------------------------------------------
# Running the pipeline
# ---------------------------------------------------------------------------


def run_transcription(
    files: Sequence[AudioFile],
    options: TranscriptionOptions,
    *,
    on_progress: ProgressCallback = no_progress,
    cancel: threading.Event | None = None,
) -> TranscriptionResult:
    """Transcribe recordings in a worker process.

    Args:
        files: The recordings, already on disk (see :func:`staged_uploads`).
        options: How to transcribe them.
        on_progress: Receives progress updates.
        cancel: Stops the worker when set.

    Returns:
        The transcripts, the files that could not be decoded, and notes.

    Raises:
        textlab.common.jobs.WorkerError: If the pipeline fails.
        textlab.common.progress.CancelledError: If ``cancel`` was set.
    """
    request = {
        "files": [{"path": str(f.path), "name": f.name} for f in files],
        "options": options.to_dict(),
    }
    on_progress(Progress("Starting transcription worker..."))
    result = run_worker(
        WORKER_MODULE,
        request,
        area=WORKSPACE_AREA,
        on_progress=on_progress,
        cancel=cancel,
    )
    return TranscriptionResult.from_dict(result)


def transcribe_files(
    files: Sequence[AudioFile],
    options: TranscriptionOptions,
    *,
    on_progress: ProgressCallback = no_progress,
    cancel: threading.Event | None = None,
) -> TranscriptionResult:
    """Run the WhisperX pipeline on recordings in this process.

    Models are loaded once for all files: the Whisper model and the
    diarization model up front, the alignment model when the language
    changes. Use :func:`run_transcription` from an interactive app.

    Args:
        files: The recordings.
        options: How to transcribe them.
        on_progress: Receives progress updates.
        cancel: Checked between files and steps.

    Returns:
        The transcripts, the files that could not be decoded, and notes.

    Raises:
        textlab.common.progress.CancelledError: If ``cancel`` was set.
    """
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    import torch
    import whisperx
    from whisperx import alignment

    alignment.DEFAULT_ALIGN_MODELS_HF.update(EXTRA_ALIGN_MODELS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "float32"
    result = TranscriptionResult()
    total = len(files)

    on_progress(
        Progress(
            f"Loading WhisperX model ({model_label(options.model)})...", 0.0
        )
    )
    model = whisperx.load_model(
        options.model, device=device, compute_type=compute_type
    )
    diarizer = None
    if options.diarize:
        token = _read_hf_token()
        if token:
            on_progress(Progress("Loading diarization model...", 0.0))
            diarizer = whisperx.diarize.DiarizationPipeline(
                use_auth_token=token, device=device
            )
        else:
            result.notes.append(NO_TOKEN_NOTE)
    aligner = _Aligner(whisperx, device)

    try:
        for index, audio_file in enumerate(files):
            check_cancelled(cancel)
            prefix = (
                f"({index + 1}/{total}) {audio_file.name}: "
                if total > 1
                else ""
            )

            def step(
                message: str, index: int = index, prefix: str = prefix
            ) -> None:
                on_progress(Progress(prefix + message, index / total))

            step("Decoding and converting audio...")
            try:
                samples = audio.load_audio(audio_file.path)
            except audio.AudioDecodeError:
                result.skipped.append(
                    SkippedFile(audio_file.name, "could not decode audio")
                )
                continue

            language = options.language
            if language is None:
                step("Detecting language...")
                detected, _ = audio.detect_language(samples)
                language = detected or FALLBACK_LANGUAGE
            align_language = alignment_language(language)

            step("Transcribing audio...")
            check_cancelled(cancel)
            segments = _transcribe(
                model, samples, align_language, options, result
            )

            step("Aligning word-level timestamps...")
            check_cancelled(cancel)
            aligned = aligner.align(segments, align_language, samples)

            has_speakers = False
            if diarizer is not None:
                step("Running speaker diarization...")
                check_cancelled(cancel)
                turns = diarizer(
                    samples,
                    min_speakers=options.min_speakers,
                    max_speakers=options.max_speakers,
                )
                aligned = whisperx.assign_word_speakers(turns, aligned)
                has_speakers = True

            result.transcripts.append(
                Transcript(
                    name=audio_file.name,
                    language=align_language,
                    duration_seconds=len(samples) / float(audio.SAMPLE_RATE),
                    segments=_plain(aligned["segments"]),
                    has_speakers=has_speakers,
                )
            )
    finally:
        del model, diarizer
        aligner.release()
        _empty_gpu_cache(torch)

    on_progress(Progress("Transcription complete.", 1.0))
    return result


def _transcribe(
    model: Any,
    samples: Any,
    language: str,
    options: TranscriptionOptions,
    result: TranscriptionResult,
) -> list[dict[str, Any]]:
    """Transcribe a recording, or only its speech when VAD is enabled."""
    if options.vad_max_pause is None:
        output = model.transcribe(
            samples, batch_size=options.batch_size, language=language
        )
        return output["segments"]

    segments = []
    for index, speech in enumerate(
        audio.vad_segments(samples, options.vad_max_pause)
    ):
        start, end = int(speech["start"]), int(speech["end"])
        piece = samples[start:end]
        if piece.size == 0:
            result.notes.append(
                f"Skipped empty VAD segment {index} ({start}:{end})."
            )
            continue
        output = model.transcribe(
            piece, batch_size=options.batch_size, language=language
        )
        offset = start / audio.SAMPLE_RATE
        for segment in output.get("segments", []):
            segment["start"] += offset
            segment["end"] += offset
            segments.append(segment)
    return segments


class _Aligner:
    """Aligns words to audio, keeping one alignment model in memory.

    WhisperX has one alignment model per language; it is swapped only when
    the language changes, so a batch in one language loads it once.
    """

    def __init__(self, whisperx: Any, device: str):
        self._whisperx = whisperx
        self._device = device
        self._language: str | None = None
        self._model: Any = None
        self._metadata: Any = None

    def align(
        self, segments: list[dict[str, Any]], language: str, samples: Any
    ) -> dict[str, Any]:
        """Return the segments with word-level timestamps."""
        if language != self._language:
            self.release()
            self._model, self._metadata = self._whisperx.load_align_model(
                language_code=language, device=self._device
            )
            self._language = language
        return self._whisperx.align(
            segments,
            self._model,
            self._metadata,
            samples,
            self._device,
            return_char_alignments=False,
        )

    def release(self) -> None:
        """Drop the loaded alignment model."""
        self._model = self._metadata = self._language = None
        gc.collect()


def _read_hf_token() -> str | None:
    """Return the Hugging Face token from the configured file, if any."""
    token_file = get_settings().hf_token_file
    if token_file is None or not token_file.is_file():
        return None
    try:
        token = token_file.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return token or None


def _empty_gpu_cache(torch: Any) -> None:
    """Release memory held by PyTorch's GPU allocator."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _plain(value: Any) -> Any:
    """Convert NumPy numbers in nested lists and dicts to Python numbers."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


# ---------------------------------------------------------------------------
# Staging uploads in the workspace
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def staged_uploads(
    uploads: Iterable[tuple[str, bytes]],
) -> Iterator[list[AudioFile]]:
    """Write uploaded recordings to the workspace for the worker.

    Args:
        uploads: ``(file name, content)`` pairs as received from the
            browser.

    Yields:
        The staged recordings. They are deleted when the block ends.
    """
    with get_workspace().temp_dir(WORKSPACE_AREA, prefix="upload-") as folder:
        used: set[str] = set()
        files = []
        for name, content in uploads:
            path = _unique_path(folder, safe_upload_name(name, "audio"), used)
            path.write_bytes(content)
            files.append(AudioFile(path, name))
        yield files


@contextlib.contextmanager
def staged_zip(archive: BinaryIO | Path) -> Iterator[list[AudioFile]]:
    """Extract the recordings in a ZIP archive to the workspace.

    Only files with an :data:`AUDIO_EXTENSIONS` extension are extracted;
    folders inside the archive are flattened and macOS metadata files
    (``._*``) are ignored. Files with the same name get a numbered suffix.

    Args:
        archive: The ZIP file, as a path or a binary file object.

    Yields:
        The recordings, named after their file names in the archive. They
        are deleted when the block ends.
    """
    with get_workspace().temp_dir(WORKSPACE_AREA, prefix="batch-") as folder:
        used: set[str] = set()
        files = []
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                name = PurePosixPath(info.filename.replace("\\", "/")).name
                if info.is_dir() or name.startswith("._"):
                    continue
                if not name.lower().endswith(AUDIO_EXTENSIONS):
                    continue
                path = _unique_path(
                    folder, safe_upload_name(name, "audio"), used
                )
                with zf.open(info) as source, path.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                files.append(AudioFile(path, path.name))
        yield files


def _unique_path(folder: Path, name: str, used: set[str]) -> Path:
    """Return ``folder / name``, numbered if the name is already taken."""
    stem, suffix = os.path.splitext(name)
    candidate, counter = name, 1
    while candidate.casefold() in used:
        counter += 1
        candidate = f"{stem}_{counter}{suffix}"
    used.add(candidate.casefold())
    return folder / candidate
