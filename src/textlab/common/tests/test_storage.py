"""The workspace keeps temporary user data private and in one place.

It replaces the per-user ``~/.cache/text_lab/mcp_artifacts`` folder, which
was never cleaned up, and the OCR and Grobid folders in the home directory.
"""

import os
import stat
from pathlib import Path

import pytest

from textlab.common import storage
from textlab.common.config import get_settings


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Isolate the environment and the cached settings and workspace."""
    monkeypatch.delenv("TEXT_LAB_WORKDIR", raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    storage.tempfile.tempdir = None  # re-read TMPDIR
    get_settings.cache_clear()
    storage.get_workspace.cache_clear()
    yield monkeypatch
    storage.tempfile.tempdir = None
    get_settings.cache_clear()
    storage.get_workspace.cache_clear()


def mode(path):
    return stat.S_IMODE(path.stat().st_mode)


def test_the_workdir_setting_wins(env, tmp_path):
    env.setenv("TEXT_LAB_WORKDIR", str(tmp_path / "job"))
    assert storage.get_workspace().root == tmp_path / "job"


def test_without_a_workdir_it_falls_back_to_tmpdir(env, tmp_path):
    root = storage.get_workspace().root
    assert root == tmp_path / "tmp" / f"textlab-{os.getuid()}"


def test_directories_are_private_at_every_level(tmp_path):
    workspace = storage.Workspace(tmp_path / "deep" / "root")
    area = workspace.dir("chat", "session")
    assert area == tmp_path / "deep" / "root" / "chat" / "session"
    for path in (area, area.parent, workspace.root):
        assert mode(path) == 0o700, path


def test_an_existing_loose_directory_is_tightened(tmp_path):
    root = tmp_path / "root"
    root.mkdir(mode=0o755)
    storage.Workspace(root).dir("ocr")
    assert mode(root) == 0o700


def test_dir_is_idempotent(tmp_path):
    workspace = storage.Workspace(tmp_path / "root")
    assert workspace.dir("ocr") == workspace.dir("ocr")


def test_a_directory_owned_by_someone_else_is_refused(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    other_user = os.getuid() + 1
    monkeypatch.setattr(storage.os, "getuid", lambda: other_user)
    with pytest.raises(storage.WorkspaceError, match="another user"):
        storage.Workspace(root).dir()


def test_make_temp_dir_creates_unique_private_folders(tmp_path):
    workspace = storage.Workspace(tmp_path / "root")
    first = workspace.make_temp_dir("chat", prefix="chat-")
    second = workspace.make_temp_dir("chat", prefix="chat-")
    assert first != second
    assert first.parent == second.parent == workspace.dir("chat")
    assert first.name.startswith("chat-")
    assert mode(first) == 0o700


def test_temp_dir_is_removed_afterwards_even_on_error(tmp_path):
    workspace = storage.Workspace(tmp_path / "root")
    with pytest.raises(RuntimeError):
        with workspace.temp_dir("visualization") as path:
            (path / "upload.csv").write_text("a,b\n")
            seen = path
            raise RuntimeError("analysis failed")
    assert not seen.exists()


def test_temp_dir_removes_read_only_content(tmp_path):
    workspace = storage.Workspace(tmp_path / "root")
    with workspace.temp_dir("ocr") as path:
        locked = path / "cache" / "model"
        locked.mkdir(parents=True)
        (locked / "weights.bin").write_bytes(b"x")
        (locked / "weights.bin").chmod(0o400)
        locked.chmod(0o500)
        seen = path
    assert not seen.exists()


def test_remove_tree_ignores_a_missing_directory(tmp_path):
    storage.remove_tree(tmp_path / "gone")


def test_nothing_is_written_into_the_source_tree(env):
    src_dir = Path(storage.__file__).resolve().parents[2]
    root = storage.get_workspace().root
    assert src_dir not in root.parents
