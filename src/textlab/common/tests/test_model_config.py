"""The language models offered, by the session's GPU."""

from textlab.common import model_config


def test_the_models_follow_the_gpu(monkeypatch):
    monkeypatch.setattr(
        model_config, "session_gpu", lambda: "NVIDIA H100 80GB"
    )
    options = model_config.model_options()
    assert options.large and options.gpu == "NVIDIA H100 80GB"
    assert options.models
    monkeypatch.setattr(model_config, "session_gpu", lambda: "NVIDIA RTX 4090")
    assert not model_config.model_options().large
