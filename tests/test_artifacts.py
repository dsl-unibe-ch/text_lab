"""Artifacts belong to one user, not to the deployed source tree.

They used to be written into ``src/mcp_artifacts`` at import time, chmod 0700.
Whoever opened the page first owned that directory, and every other user got
``PermissionError`` before the page could render.
"""

import conftest_path  # noqa: F401

import os
import pathlib
import stat

import pytest

from core import artifacts


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv(artifacts.ENV_OVERRIDE, raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    return monkeypatch


def test_the_override_wins_outright(clean_env, tmp_path):
    clean_env.setenv(artifacts.ENV_OVERRIDE, str(tmp_path / "elsewhere"))
    assert artifacts.artifacts_root() == tmp_path / "elsewhere"


def test_it_defaults_to_the_user_cache(clean_env, tmp_path):
    """Deterministic, and already bound into the container: no job-script
    change, and no silent relocation if the cluster's mounts change."""
    clean_env.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    assert artifacts.artifacts_root() == tmp_path / "cache" / "text_lab" / "mcp_artifacts"


def test_it_lands_under_home_when_xdg_is_unset(clean_env, tmp_path):
    clean_env.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    assert artifacts.artifacts_root() == (
        tmp_path / "home" / ".cache" / "text_lab" / "mcp_artifacts"
    )


def test_the_directory_is_created_private(clean_env, tmp_path):
    clean_env.setenv(artifacts.ENV_OVERRIDE, str(tmp_path / "root"))

    made = pathlib.Path(artifacts.ensure_artifacts_dir())
    images = pathlib.Path(artifacts.ensure_artifacts_dir("images"))

    assert made.is_dir() and images.is_dir()
    assert images.parent == made
    # Uploaded data is the user's own: nothing for group or other.
    for path in (made, images):
        assert stat.S_IMODE(path.stat().st_mode) == 0o700, path


def test_an_existing_loose_directory_is_tightened(clean_env, tmp_path):
    """A directory left behind by an earlier, more permissive version is
    re-secured rather than trusted."""
    root = tmp_path / "root"
    root.mkdir(mode=0o755)
    assert stat.S_IMODE(root.stat().st_mode) == 0o755

    clean_env.setenv(artifacts.ENV_OVERRIDE, str(root))
    artifacts.ensure_artifacts_dir()

    assert stat.S_IMODE(root.stat().st_mode) == 0o700


def test_it_is_idempotent(clean_env, tmp_path):
    clean_env.setenv(artifacts.ENV_OVERRIDE, str(tmp_path / "root"))
    first = artifacts.ensure_artifacts_dir("images")
    assert artifacts.ensure_artifacts_dir("images") == first


def test_nothing_is_written_into_the_source_tree(clean_env, tmp_path):
    """The whole point: the path must not land under the deployed src/."""
    clean_env.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    src_dir = pathlib.Path(artifacts.__file__).resolve().parent.parent
    assert src_dir not in artifacts.artifacts_root().parents
