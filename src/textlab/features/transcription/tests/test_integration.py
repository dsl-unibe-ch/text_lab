"""The real pipeline in a worker process, on a few seconds of generated audio.

Needs the Text Lab image with the model stores mounted, as
``scripts/test_on_node.sbatch`` does. The audio is a tone, so the transcript
itself is not checked, only that every step runs and the result comes back.
"""

import os

import numpy as np
import pytest

from textlab.common import storage
from textlab.common.config import get_settings
from textlab.features.transcription import audio, service
from textlab.features.transcription.models import TranscriptionOptions

pytestmark = [pytest.mark.container, pytest.mark.slow]

HF_HUB = os.path.join(os.environ.get("HF_HOME", "/opt/huggingface"), "hub")


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    monkeypatch.setenv("TEXT_LAB_WORKDIR", str(tmp_path / "work"))
    get_settings.cache_clear()
    storage.get_workspace.cache_clear()
    yield
    get_settings.cache_clear()
    storage.get_workspace.cache_clear()


@pytest.mark.skipif(
    not (os.path.isdir(HF_HUB) and os.listdir(HF_HUB)),
    reason="the Hugging Face model store is not mounted",
)
def test_a_short_recording_is_transcribed_in_a_worker(workspace):
    seconds = 3
    time = np.arange(audio.SAMPLE_RATE * seconds) / audio.SAMPLE_RATE
    tone = (0.1 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    updates = []
    with service.staged_uploads(
        [("tone.wav", audio.to_wav_bytes(tone))]
    ) as files:
        result = service.run_transcription(
            files,
            TranscriptionOptions(model="tiny", language="en", diarize=False),
            on_progress=updates.append,
        )
    (transcript,) = result.transcripts
    assert transcript.name == "tone.wav"
    assert transcript.duration_seconds == pytest.approx(seconds, abs=0.1)
    assert updates[-1].message == "Transcription complete."
