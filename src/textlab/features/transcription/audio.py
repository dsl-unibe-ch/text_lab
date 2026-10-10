"""Audio decoding, language detection and voice activity detection.

Audio is decoded with ``ffmpeg`` into 16 kHz mono float samples, the format
WhisperX expects. Heavy libraries (torch, faster-whisper, Silero VAD) are
imported inside the functions that need them, so this module can be
imported without a GPU.
"""

from __future__ import annotations

import functools
import io
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from textlab.common import gpu_manager

#: Sample rate WhisperX works with.
SAMPLE_RATE = 16000

#: Seconds of audio used to detect the language.
LANGUAGE_DETECTION_SECONDS = 30


class AudioDecodeError(RuntimeError):
    """``ffmpeg`` could not decode the audio, or it contained no samples."""


def decode_audio_bytes(
    audio_bytes: bytes, sr: int = SAMPLE_RATE
) -> np.ndarray:
    """Decode audio of any format ffmpeg reads into mono float samples.

    The audio is first piped to ffmpeg; formats that need seeking (some MP4
    and M4A files) are retried from a temporary file, which is removed
    straight away.

    Args:
        audio_bytes: The encoded audio.
        sr: Sample rate to resample to.

    Returns:
        The samples as ``float32`` in the range -1 to 1.

    Raises:
        AudioDecodeError: If both attempts fail or produce no samples.
    """
    piped = _run_ffmpeg(["-i", "pipe:0"], sr, stdin=audio_bytes)
    if piped.returncode == 0 and piped.stdout:
        return _samples(piped.stdout)

    with tempfile.NamedTemporaryFile(suffix=".audio") as handle:
        handle.write(audio_bytes)
        handle.flush()
        from_file = _run_ffmpeg(["-i", handle.name], sr)
    if from_file.returncode == 0 and from_file.stdout:
        return _samples(from_file.stdout)

    raise AudioDecodeError(
        "Failed to decode audio from both stdin and temp file.\n"
        f"stdin rc={piped.returncode}, file rc={from_file.returncode}\n"
        f"stdin stderr:\n{piped.stderr.decode(errors='replace')}\n\n"
        f"file stderr:\n{from_file.stderr.decode(errors='replace')}"
    )


def load_audio(path: Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode an audio file into mono float samples.

    Args:
        path: The audio file.
        sr: Sample rate to resample to.

    Returns:
        The samples as ``float32`` in the range -1 to 1.

    Raises:
        AudioDecodeError: If ffmpeg fails or produces no samples.
    """
    result = _run_ffmpeg(["-i", str(path)], sr)
    if result.returncode == 0 and result.stdout:
        return _samples(result.stdout)
    raise AudioDecodeError(
        f"Could not decode {path.name}:\n"
        f"{result.stderr.decode(errors='replace')}"
    )


def to_wav_bytes(audio: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    """Encode samples as a 16-bit PCM WAV file.

    Args:
        audio: Mono float samples.
        sr: Their sample rate.

    Returns:
        The WAV file's bytes.
    """
    import soundfile

    buffer = io.BytesIO()
    soundfile.write(buffer, audio, sr, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def convert_audio_to_wav_bytes(
    audio_bytes: bytes, original_filename: str, sr: int = SAMPLE_RATE
) -> tuple[bytes, str, np.ndarray]:
    """Decode uploaded audio and re-encode it as WAV for playback.

    Args:
        audio_bytes: The uploaded audio.
        original_filename: The upload's name.
        sr: Sample rate to resample to.

    Returns:
        ``(wav_bytes, wav_filename, samples)``, where ``wav_filename`` is the
        original name with a ``.wav`` extension.

    Raises:
        AudioDecodeError: If the audio cannot be decoded.
    """
    audio = decode_audio_bytes(audio_bytes, sr=sr)
    wav_filename = os.path.splitext(original_filename)[0] + ".wav"
    return to_wav_bytes(audio, sr), wav_filename, audio


def detect_language(
    audio: np.ndarray, sr: int = SAMPLE_RATE
) -> tuple[str, float]:
    """Detect the spoken language from the start of a recording.

    Uses Whisper "tiny" through faster-whisper on the first
    :data:`LANGUAGE_DETECTION_SECONDS` seconds.

    Args:
        audio: Mono float samples.
        sr: Their sample rate.

    Returns:
        ``(language_code, probability)``, e.g. ``("de", 0.97)``.

    Raises:
        ValueError: If there are no samples.
    """
    if audio is None or audio.size == 0:
        raise ValueError("Decoded waveform is empty.")
    snippet = audio[: int(sr * LANGUAGE_DETECTION_SECONDS)]
    _, info = _language_detector().transcribe(
        snippet,
        beam_size=1,
        language=None,
        task="transcribe",
        vad_filter=False,
        without_timestamps=True,
    )
    code = getattr(info, "language", None)
    probability = float(getattr(info, "language_probability", 0.0) or 0.0)
    return code, probability


def detect_language_from_bytes(
    audio_bytes: bytes, sr: int = SAMPLE_RATE
) -> tuple[str, float]:
    """Decode encoded audio and detect its language.

    Args:
        audio_bytes: The encoded audio.
        sr: Sample rate to decode at.

    Returns:
        ``(language_code, probability)``.

    Raises:
        AudioDecodeError: If the audio cannot be decoded.
        ValueError: If it contains no samples.
    """
    return detect_language(decode_audio_bytes(audio_bytes, sr=sr), sr=sr)


def vad_segments(
    audio: np.ndarray, max_pause: float, sr: int = SAMPLE_RATE
) -> list[dict[str, int]]:
    """Find the speech in a recording with Silero voice activity detection.

    Args:
        audio: Mono float samples.
        max_pause: Merge speech segments separated by shorter pauses, in
            seconds.
        sr: The sample rate.

    Returns:
        Speech segments as ``{"start": sample, "end": sample}``.
    """
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    timestamps = get_speech_timestamps(
        torch.from_numpy(audio),
        load_silero_vad(),
        return_seconds=False,
        sampling_rate=sr,
    )
    return merge_segments(timestamps, max_pause * sr)


def merge_segments(
    segments: list[dict[str, Any]], max_gap: float
) -> list[dict[str, Any]]:
    """Merge segments that are separated by less than ``max_gap``.

    Args:
        segments: Segments with ``start`` and ``end``, in order.
        max_gap: Largest gap to bridge, in the segments' unit.

    Returns:
        The merged segments; the input is not modified.
    """
    if not segments:
        return []
    merged = [dict(segments[0])]
    for segment in segments[1:]:
        if segment["start"] - merged[-1]["end"] < max_gap:
            merged[-1]["end"] = segment["end"]
        else:
            merged.append(dict(segment))
    return merged


@functools.lru_cache(maxsize=1)
def _language_detector() -> Any:
    """Load Whisper "tiny" for language detection, once per process."""
    import torch
    from faster_whisper import WhisperModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    return WhisperModel("tiny", device=device, compute_type=compute_type)


def _release_language_detector() -> bool:
    """Drop the language detector so another feature can use the GPU."""
    loaded = bool(_language_detector.cache_info().currsize)
    _language_detector.cache_clear()
    return loaded


def _run_ffmpeg(
    input_args: list[str], sr: int, stdin: bytes | None = None
) -> subprocess.CompletedProcess:
    """Run ffmpeg to produce 16-bit mono PCM at ``sr`` on standard output."""
    command = [
        "ffmpeg", "-nostdin", "-threads", "0", *input_args,
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(sr),
        "pipe:1",
    ]  # fmt: skip
    return subprocess.run(
        command, input=stdin, capture_output=True, check=False
    )


def _samples(pcm: bytes) -> np.ndarray:
    """Convert 16-bit PCM bytes to float samples in the range -1 to 1."""
    return np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0


gpu_manager.register(
    gpu_manager.TRANSCRIBE,
    "Unloaded audio language detector",
    _release_language_detector,
)
