"""Data classes survive JSON, and models are chosen per language."""

import json
import os

import pytest

from textlab.common.config import MissingSettingError, get_settings
from textlab.features.transcription import whisper_models
from textlab.features.transcription.models import (
    SkippedFile,
    Transcript,
    TranscriptionOptions,
    TranscriptionResult,
)


@pytest.fixture
def settings(monkeypatch):
    get_settings.cache_clear()
    yield monkeypatch
    get_settings.cache_clear()


def test_options_round_trip_through_json():
    options = TranscriptionOptions(
        model="large-v3-turbo", language="ch_de", max_speakers=3
    )
    data = json.loads(json.dumps(options.to_dict()))
    assert TranscriptionOptions.from_dict(data) == options


def test_result_round_trips_through_json():
    result = TranscriptionResult(
        transcripts=[
            Transcript("a.wav", "de", 2.5, [{"start": 0.0, "text": "x"}], True)
        ],
        skipped=[SkippedFile("b.wav", "could not decode audio")],
        notes=["note"],
    )
    data = json.loads(json.dumps(result.to_dict()))
    assert TranscriptionResult.from_dict(data) == result


@pytest.mark.parametrize(
    "language,expected",
    [("ch_de", "de"), ("ch_de_flurin", "de"), ("fr", "fr")],
)
def test_alignment_language(language, expected):
    assert whisper_models.alignment_language(language) == expected


def test_default_model_for_ordinary_and_detected_languages():
    assert whisper_models.default_model("en") == "large-v3-turbo"
    assert whisper_models.default_model(None) == "large-v3-turbo"


def test_swiss_german_uses_the_custom_folder(settings, tmp_path):
    settings.setenv("TEXT_LAB_CUSTOM_WHISPER_DIR", str(tmp_path))
    assert whisper_models.default_model("ch_de") == str(
        tmp_path / "swhisper-large-1.1"
    )


def test_swiss_german_without_the_setting_names_it(settings):
    settings.delenv("TEXT_LAB_CUSTOM_WHISPER_DIR", raising=False)
    with pytest.raises(MissingSettingError, match="CUSTOM_WHISPER_DIR"):
        whisper_models.default_model("ch_de_flurin")


def test_converted_model_missing(tmp_path):
    missing = str(tmp_path / "flurin")
    assert whisper_models.converted_model_missing("ch_de_flurin", missing)
    os.mkdir(missing)
    assert not whisper_models.converted_model_missing("ch_de_flurin", missing)
    assert not whisper_models.converted_model_missing("de", "/nowhere")


def test_model_label_hides_storage_paths():
    assert whisper_models.model_label("/storage/x/swhisper") == "swhisper"
    assert whisper_models.model_label("large-v3-turbo") == "large-v3-turbo"
