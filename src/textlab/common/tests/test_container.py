"""The image's conda environments."""

from textlab.common import container


def test_env_python_defaults_to_the_image_layout(monkeypatch):
    monkeypatch.delenv("PADDLE_VL_BACKEND_PYTHON", raising=False)
    assert container.env_python(container.PADDLE_VL_ENV) == (
        "/opt/conda/envs/paddle_vl_backend/bin/python"
    )


def test_the_first_set_override_wins(monkeypatch):
    monkeypatch.delenv("FIRST", raising=False)
    monkeypatch.setenv("SECOND", "/custom/python")
    assert container.env_python("env", "FIRST", "SECOND") == "/custom/python"
    monkeypatch.setenv("FIRST", "")
    assert container.env_python("env", "FIRST", "SECOND") == "/custom/python"
