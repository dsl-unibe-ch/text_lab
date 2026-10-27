"""KG uses the shared model configuration rather than installed models."""

import json
from unittest.mock import Mock

import ollama
import pytest

from textlab.common import model_config
from textlab.features.knowledge_graph import service


@pytest.mark.parametrize(
    "gpu, expected",
    [
        ("NVIDIA RTX 4090", ["standard-model"]),
        ("NVIDIA A100-SXM4-80GB", ["large-model", "another-large-model"]),
        ("NVIDIA H100", ["large-model", "another-large-model"]),
        ("Unknown/CPU", ["standard-model"]),
    ],
)
def test_kg_uses_configured_models_for_the_session_gpu(
    monkeypatch, tmp_path, gpu, expected
):
    config = tmp_path / "models.json"
    config.write_text(
        json.dumps(
            {
                "high_memory_gpus": ["A100", "H100", "H200"],
                "small_models": ["standard-model"],
                "large_models": ["large-model", "another-large-model"],
            }
        )
    )
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", config)
    monkeypatch.setattr(
        model_config, "LOCAL_CONFIG_PATH", tmp_path / "local.json"
    )
    monkeypatch.setattr(service, "get_gpu_name", lambda: gpu)
    installed = Mock(
        side_effect=AssertionError("Do not list installed models")
    )
    monkeypatch.setattr(ollama, "list", installed)
    assert service.ollama_models() == expected
    installed.assert_not_called()


def test_kg_honours_local_model_configuration(monkeypatch, tmp_path):
    config = tmp_path / "models.json"
    config.write_text(json.dumps({"small_models": ["base-model"]}))
    local = tmp_path / "models.local.json"
    local.write_text(json.dumps({"small_models": ["local-model"]}))
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", config)
    monkeypatch.setattr(model_config, "LOCAL_CONFIG_PATH", local)
    monkeypatch.setattr(service, "get_gpu_name", lambda: "Unknown/CPU")
    assert service.ollama_models() == ["local-model"]


def test_kg_does_not_fall_back_to_installed_models_when_config_is_empty(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        model_config, "DEFAULT_CONFIG_PATH", tmp_path / "missing"
    )
    monkeypatch.setattr(model_config, "LOCAL_CONFIG_PATH", tmp_path / "local")
    monkeypatch.setattr(service, "get_gpu_name", lambda: "Unknown/CPU")
    assert service.ollama_models() == []
