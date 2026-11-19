"""The model configuration and the result model."""

import pytest

from textlab.features.handwriting import models


def test_the_configured_models_are_complete():
    configured = models.available_models()
    assert configured, "no handwriting_models in common/models.json"
    for model in configured:
        assert model.label and model.base_model and model.prompt
        assert models.get_model(model.key) == model
        assert models.HandwritingModel.from_dict(model.to_dict()) == model


def test_a_model_names_what_it_runs_before_a_run():
    plain = models.HandwritingModel("k", "L", "", "org/base", None, "Read.")
    assert plain.models_used() == {
        "line_detection": f"PaddleOCR {models.LINE_DETECTOR}",
        "recognition": "org/base",
    }


def test_an_unknown_model_is_refused():
    with pytest.raises(models.HandwritingError):
        models.get_model("no-such-model")


def test_an_incomplete_entry_is_refused():
    with pytest.raises(ValueError, match="base_model"):
        models.HandwritingModel.from_dict(
            {"key": "k", "label": "L", "prompt": "p"}
        )


def test_duplicate_keys_are_refused(monkeypatch):
    entry = {"key": "k", "label": "L", "base_model": "b", "prompt": "p"}
    monkeypatch.setattr(
        models,
        "load_model_config",
        lambda: {models.CONFIG_KEY: [entry, entry]},
    )
    with pytest.raises(ValueError, match="Duplicate"):
        models.available_models()


def test_a_document_reports_its_text_and_models():
    spec = models.HandwritingModel(
        "k", "Label", "", "org/base", "org/adapter", "Read."
    )
    page = models.PageTranscription(
        1, 100, 50, [models.Line((1, 2, 3, 4), "a"), models.Line((1, 5, 3, 8))]
    )
    document = models.DocumentTranscription("scan.png", spec, [page, page])
    assert document.text == "a\n\n\na\n"
    assert document.line_count == 4
    assert document.models_used() == {
        "line_detection": f"PaddleOCR {models.LINE_DETECTOR}",
        "recognition": "org/adapter (LoRA adapter on org/base)",
    }
    data = document.to_dict()
    assert data["pages"][0]["lines"][0] == {"box": [1, 2, 3, 4], "text": "a"}
