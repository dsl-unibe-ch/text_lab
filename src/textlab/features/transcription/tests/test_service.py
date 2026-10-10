"""The transcription pipeline, with WhisperX, torch and ffmpeg faked.

These tests check the pipeline's logic, not the models: which models are
loaded and when, how languages and speakers are handled, what is skipped and
what is reported. ``test_integration.py`` runs the real models.
"""

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from textlab.common.config import get_settings
from textlab.common.progress import CancelledError
from textlab.features.transcription import audio, service
from textlab.features.transcription.models import (
    AudioFile,
    TranscriptionOptions,
)

SECONDS = 2


class FakeWhisperX:
    """Records calls and returns small, predictable results."""

    def __init__(self):
        self.loaded_align = []
        self.transcribed = []
        self.alignment = SimpleNamespace(DEFAULT_ALIGN_MODELS_HF={})
        self.diarize = SimpleNamespace(DiarizationPipeline=self._diarizer)
        self.diarizers = 0

    def load_model(self, name, device, compute_type):
        return SimpleNamespace(transcribe=self._transcribe)

    def _transcribe(self, samples, batch_size, language):
        self.transcribed.append((len(samples), language))
        return {"segments": [{"start": 0.0, "end": 0.5, "text": " hi"}]}

    def load_align_model(self, language_code, device):
        self.loaded_align.append(language_code)
        return f"aligner-{language_code}", {}

    def align(self, segments, model, metadata, samples, device, **kwargs):
        return {
            "segments": [
                dict(
                    segment,
                    words=[
                        {
                            "word": "hi",
                            "start": np.float32(segment["start"]),
                            "end": np.float32(segment["end"]),
                        }
                    ],
                )
                for segment in segments
            ]
        }

    def _diarizer(self, use_auth_token, device):
        self.diarizers += 1
        return lambda samples, min_speakers, max_speakers: "turns"

    def assign_word_speakers(self, turns, aligned):
        for segment in aligned["segments"]:
            for word in segment["words"]:
                word["speaker"] = "SPEAKER_00"
        return aligned


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    whisperx = FakeWhisperX()
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False, empty_cache=None)
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "whisperx", whisperx)
    monkeypatch.setitem(sys.modules, "whisperx.alignment", whisperx.alignment)

    def load_audio(path):
        if "broken" in Path(path).name:
            raise audio.AudioDecodeError("not audio")
        return np.zeros(audio.SAMPLE_RATE * SECONDS, dtype=np.float32)

    monkeypatch.setattr(audio, "load_audio", load_audio)
    monkeypatch.setattr(audio, "detect_language", lambda samples: ("fr", 0.9))

    token = tmp_path / "hf_token.txt"
    token.write_text("hf_secret\n")
    monkeypatch.setenv("TEXT_LAB_HF_TOKEN_FILE", str(token))
    get_settings.cache_clear()
    yield whisperx
    get_settings.cache_clear()


def files(*names):
    return [AudioFile(Path("/workspace") / name, name) for name in names]


def test_one_recording_end_to_end(fakes):
    result = service.transcribe_files(
        files("a.wav"), TranscriptionOptions(model="m", language="ch_de")
    )
    (transcript,) = result.transcripts
    assert transcript.name == "a.wav"
    assert transcript.language == "de"  # Swiss German is aligned as German
    assert transcript.duration_seconds == SECONDS
    assert transcript.has_speakers
    word = transcript.segments[0]["words"][0]
    assert word["speaker"] == "SPEAKER_00"
    assert type(word["start"]) is float  # NumPy numbers are converted
    assert fakes.transcribed == [(audio.SAMPLE_RATE * SECONDS, "de")]
    assert result.notes == [] and result.skipped == []


def test_ukrainian_alignment_model_is_registered(fakes):
    service.transcribe_files(files("a.wav"), TranscriptionOptions(model="m"))
    assert fakes.alignment.DEFAULT_ALIGN_MODELS_HF["uk"] == (
        "Yehor/w2v-xls-r-uk"
    )


def test_without_a_token_diarization_is_skipped_with_a_note(
    fakes, monkeypatch
):
    monkeypatch.delenv("TEXT_LAB_HF_TOKEN_FILE")
    get_settings.cache_clear()
    result = service.transcribe_files(
        files("a.wav"), TranscriptionOptions(model="m", language="en")
    )
    assert result.notes == [service.NO_TOKEN_NOTE]
    assert not result.transcripts[0].has_speakers
    assert fakes.diarizers == 0


def test_diarization_can_be_switched_off(fakes):
    result = service.transcribe_files(
        files("a.wav"),
        TranscriptionOptions(model="m", language="en", diarize=False),
    )
    assert result.notes == []
    assert fakes.diarizers == 0


def test_language_is_detected_per_file_when_not_given(fakes):
    result = service.transcribe_files(
        files("a.wav"), TranscriptionOptions(model="m")
    )
    assert result.transcripts[0].language == "fr"


def test_detection_without_answer_falls_back_to_english(fakes, monkeypatch):
    monkeypatch.setattr(audio, "detect_language", lambda s: (None, 0.0))
    result = service.transcribe_files(
        files("a.wav"), TranscriptionOptions(model="m")
    )
    assert result.transcripts[0].language == "en"


def test_models_are_loaded_once_per_batch(fakes, monkeypatch):
    languages = iter([("de", 1.0), ("de", 1.0), ("fr", 1.0)])
    monkeypatch.setattr(audio, "detect_language", lambda s: next(languages))
    result = service.transcribe_files(
        files("a.wav", "b.wav", "c.wav"), TranscriptionOptions(model="m")
    )
    assert len(result.transcripts) == 3
    assert fakes.loaded_align == ["de", "fr"]  # swapped only on change
    assert fakes.diarizers == 1


def test_undecodable_files_are_skipped_and_the_rest_continue(fakes):
    result = service.transcribe_files(
        files("broken.mp3", "b.wav"),
        TranscriptionOptions(model="m", language="en"),
    )
    assert [t.name for t in result.transcripts] == ["b.wav"]
    assert result.skipped[0].name == "broken.mp3"


def test_vad_transcribes_speech_with_time_offsets(fakes, monkeypatch):
    rate = audio.SAMPLE_RATE
    speech = [{"start": rate, "end": 2 * rate}, {"start": 5, "end": 5}]
    monkeypatch.setattr(audio, "vad_segments", lambda samples, pause: speech)
    result = service.transcribe_files(
        files("a.wav"),
        TranscriptionOptions(model="m", language="en", vad_max_pause=0.3),
    )
    segments = result.transcripts[0].segments
    assert [(s["start"], s["end"]) for s in segments] == [(1.0, 1.5)]
    assert fakes.transcribed == [(rate, "en")]
    assert result.notes == ["Skipped empty VAD segment 1 (5:5)."]


def test_progress_names_the_file_in_a_batch(fakes):
    updates = []
    service.transcribe_files(
        files("a.wav", "b.wav"),
        TranscriptionOptions(model="m", language="en"),
        on_progress=updates.append,
    )
    messages = [update.message for update in updates]
    assert "(2/2) b.wav: Aligning word-level timestamps..." in messages
    assert updates[-1].message == "Transcription complete."
    assert updates[-1].fraction == 1.0


def test_cancel_stops_before_the_next_file(fakes):
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(CancelledError):
        service.transcribe_files(
            files("a.wav"),
            TranscriptionOptions(model="m", language="en"),
            cancel=cancel,
        )
