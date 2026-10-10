"""Uploads and ZIP batches are staged in the workspace and removed again."""

import io
import zipfile

import pytest

from textlab.common import storage
from textlab.common.config import get_settings
from textlab.features.transcription import service


@pytest.fixture(autouse=True)
def workspace(monkeypatch, tmp_path):
    monkeypatch.setenv("TEXT_LAB_WORKDIR", str(tmp_path / "work"))
    get_settings.cache_clear()
    storage.get_workspace.cache_clear()
    yield tmp_path / "work" / service.WORKSPACE_AREA
    get_settings.cache_clear()
    storage.get_workspace.cache_clear()


def make_zip(members):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    buffer.seek(0)
    return buffer


def test_uploads_are_written_and_removed(workspace):
    uploads = [("talk.mp3", b"one"), ("../../etc/talk.mp3", b"two")]
    with service.staged_uploads(uploads) as files:
        paths = [f.path for f in files]
        assert [p.read_bytes() for p in paths] == [b"one", b"two"]
        assert [p.name for p in paths] == ["talk.mp3", "talk_2.mp3"]
        assert all(p.parent.parent == workspace for p in paths)
        assert [f.name for f in files] == ["talk.mp3", "../../etc/talk.mp3"]
    assert not any(p.exists() for p in paths)


def test_zip_keeps_only_audio_and_flattens_folders(workspace):
    archive = make_zip(
        {
            "a/interview.wav": b"1",
            "b/interview.WAV": b"2",
            "notes.txt": b"x",
            "__MACOSX/a/._interview.wav": b"meta",
            "c/talk.opus": b"3",
        }
    )
    with service.staged_zip(archive) as files:
        names = [f.name for f in files]
        assert names == ["interview.wav", "interview_2.WAV", "talk.opus"]
        assert files[1].path.read_bytes() == b"2"
        folder = files[0].path.parent
    assert not folder.exists()


def test_zip_without_audio_gives_no_files():
    with service.staged_zip(make_zip({"readme.txt": b"x"})) as files:
        assert files == []
