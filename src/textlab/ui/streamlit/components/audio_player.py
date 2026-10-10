"""Audio player with a synchronized transcript, for the Transcribe page.

It is a small HTML and JavaScript component built on wavesurfer.js: the
waveform plays the recording, and the transcript segment or word being
spoken is highlighted; clicking one jumps to it.
"""

from __future__ import annotations

import base64
import html
import json
import mimetypes
import string
import uuid
from importlib import resources
from typing import Any

import numpy as np

from textlab.features.transcription.audio import SAMPLE_RATE, to_wav_bytes

#: Largest WAV the browser player is given; longer audio gets a preview.
PLAYER_MAX_BYTES = 75 * 1024 * 1024
#: Longest audio the browser player is given, in seconds.
PLAYER_MAX_SECONDS = 30 * 60

#: The player's markup, script and styles, filled in by build_player_html.
_TEMPLATE = string.Template(
    resources.files(__package__)
    .joinpath("audio_player.html")
    .read_text(encoding="utf-8")
)


def create_wavesurfer_preview(
    wav_bytes: bytes, audio: np.ndarray | None, sr: int = SAMPLE_RATE
) -> tuple[bytes, float | None]:
    """Shorten audio that is too large for the in-browser player.

    Args:
        wav_bytes: The full recording as WAV.
        audio: Its samples.
        sr: Their sample rate.

    Returns:
        ``(player_wav_bytes, preview_seconds)``. ``preview_seconds`` is
        ``None`` when the full recording fits, otherwise the length of the
        preview the player gets.
    """
    duration = len(audio) / float(sr) if audio is not None else 0.0
    too_large = bool(wav_bytes) and len(wav_bytes) > PLAYER_MAX_BYTES
    if audio is None or not (too_large or duration > PLAYER_MAX_SECONDS):
        return wav_bytes, None

    by_seconds = int(PLAYER_MAX_SECONDS * sr)
    by_bytes = max(int((PLAYER_MAX_BYTES - 44) / 2), 1)
    samples = min(len(audio), by_seconds, by_bytes)
    if samples >= len(audio):
        return wav_bytes, None
    return to_wav_bytes(audio[:samples], sr), samples / float(sr)


def audio_data_url(audio_path_or_bytes: str | bytes, path_hint: str) -> str:
    """Embed audio in a ``data:`` URL for the player.

    Args:
        audio_path_or_bytes: The audio, or the path of an audio file.
        path_hint: A file name used to guess the MIME type.

    Returns:
        The ``data:`` URL.
    """
    if isinstance(audio_path_or_bytes, str):
        with open(audio_path_or_bytes, "rb") as handle:
            audio_bytes = handle.read()
    else:
        audio_bytes = audio_path_or_bytes
    mime_type = mimetypes.guess_type(path_hint)[0] or "audio/wav"
    encoded = base64.b64encode(audio_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_player_html(
    audio_url: str, items: list[dict[str, Any]], mode: str
) -> str:
    """Build the player's HTML.

    Args:
        audio_url: The audio, as a URL (see :func:`audio_data_url`).
        items: Transcript items with ``start``, ``end``, ``text`` and
            optionally ``speaker``.
        mode: ``"word"``, ``"segments"`` or ``"diarization"``.

    Returns:
        HTML to render with ``st.components.v1.html``.
    """
    component_id = f"ws_{uuid.uuid4().hex}"
    items_json = json.dumps(items)
    safe_audio_url = html.escape(audio_url, quote=True)
    safe_mode = html.escape(mode, quote=True)

    return _TEMPLATE.substitute(
        component_id=component_id,
        items_json=items_json,
        safe_audio_url=safe_audio_url,
        safe_mode=safe_mode,
    )
