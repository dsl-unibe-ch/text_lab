"""Manual engine selection, with a fake engine in place of the real ones."""

import io
import json
import zipfile

import pytest

from textlab.common.progress import Progress
from textlab.common.storage import Workspace
from textlab.features.ocr import manual, service
from textlab.features.ocr.engines import base


class FakeEngine(base.Engine):
    """Reads two pages from any file, or one record like OlmOCR."""

    name = "Fake"

    def __init__(self, record=False):
        self.record = record
        self.prepared = []
        self.calls = []

    def prepare(self, options, on_progress):
        self.prepared.append(options)
        on_progress(Progress("Preparing..."))

    def recognize(self, input_path, work_dir, options, on_progress=None):
        self.calls.append((input_path.name, input_path.read_bytes(), options))
        work_dir.mkdir(parents=True)
        (work_dir / "page.png").write_bytes(b"image")
        if self.record:
            return base.EngineOutput(
                pages=[base.PageText(1, "whole", {"text": "whole"})],
                record='{"text": "whole"}',
            )
        return base.EngineOutput(
            pages=[
                base.PageText(1, f"one {input_path.stem}", [("box", "one")]),
                base.PageText(2, "two", None),
            ],
            previews=[base.Preview(b"png")] if options.previews else [],
        )


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    workspace = Workspace(tmp_path / "workspace")
    monkeypatch.setattr(manual, "get_workspace", lambda: workspace)
    return workspace


@pytest.fixture
def engine(monkeypatch):
    fake = FakeEngine()
    monkeypatch.setitem(manual.ENGINES, "Fake", fake)
    return fake


def names_in(zip_bytes):
    return sorted(zipfile.ZipFile(io.BytesIO(zip_bytes)).namelist())


def test_the_four_engines_are_offered_in_order():
    assert list(service.ENGINES) == [
        "EasyOCR",
        "PaddleOCR",
        "OlmOCR",
        "GLM-OCR",
    ]


def test_a_document_is_read_with_the_chosen_engine(workspace, engine):
    updates = []
    options = service.EngineOptions(language="de")
    run = service.recognize_with_engine(
        "../in/scan.pdf", b"%PDF", "Fake", options, on_progress=updates.append
    )
    assert engine.calls == [("scan.pdf", b"%PDF", options)]
    assert engine.prepared == [options]
    assert updates == [Progress("Preparing...")]
    assert run.engine == "Fake"
    assert run.text == "one scan\n\ntwo"
    assert (run.text_name, run.json_name) == ("scan.txt", "scan.json")
    assert json.loads(run.json) == [
        {"page": 1, "text": "one scan", "raw": [["box", "one"]]},
        {"page": 2, "text": "two", "raw": None},
    ]
    assert run.previews == [base.Preview(b"png")]
    assert names_in(run.zip_bytes) == [
        "page_0001.json",
        "page_0001.txt",
        "page_0002.json",
        "page_0002.txt",
        "scan.json",
        "scan.txt",
    ]
    assert list(workspace.dir("ocr").iterdir()) == []


def test_an_engine_record_is_kept_as_json_lines(workspace, monkeypatch):
    monkeypatch.setitem(manual.ENGINES, "Fake", FakeEngine(record=True))
    run = service.recognize_with_engine(
        "paper.pdf", b"%PDF", "Fake", service.EngineOptions()
    )
    assert run.text == "whole"
    assert (run.json_name, run.json) == ("paper.jsonl", '{"text": "whole"}\n')
    assert names_in(run.zip_bytes) == ["paper.jsonl", "paper.txt"]


def test_the_job_folder_is_removed_when_the_engine_fails(
    workspace, monkeypatch
):
    class Failing(FakeEngine):
        def recognize(self, *args, **kwargs):
            raise base.EngineError("broken", details="log")

    monkeypatch.setitem(manual.ENGINES, "Fake", Failing())
    with pytest.raises(service.EngineError, match="broken"):
        service.recognize_with_engine(
            "a.png", b"x", "Fake", service.EngineOptions()
        )
    assert list(workspace.dir("ocr").iterdir()) == []


def zip_of(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    buffer.seek(0)
    return buffer


def test_a_batch_mirrors_the_archive(workspace, engine):
    archive = zip_of(
        {
            "b.pdf": b"b",
            "scans/a.png": b"a",
            "notes.txt": b"skipped",
            "._b.pdf": b"macOS metadata",
        }
    )
    updates = []
    zip_bytes = service.recognize_batch_with_engine(
        archive, "Fake", service.EngineOptions(), on_progress=updates.append
    )
    assert [call[0] for call in engine.calls] == ["b.pdf", "a.png"]
    assert all(call[2].previews is False for call in engine.calls)
    names = names_in(zip_bytes)
    assert "b/b.txt" in names
    assert "b/page_0002.txt" in names
    assert "scans/a/a.json" in names
    assert not any("notes" in name or "png" in name for name in names)
    assert Progress("Processing (2/2): scans/a.png", 0.5) in updates
    assert updates[-1].fraction == 1.0
    assert list(workspace.dir("ocr").iterdir()) == []


def test_a_batch_without_documents_is_refused(workspace, engine):
    with pytest.raises(ValueError, match="No valid documents"):
        service.recognize_batch_with_engine(
            zip_of({"notes.txt": b"x"}), "Fake", service.EngineOptions()
        )
    assert engine.calls == []


def test_a_failing_file_is_named(workspace, monkeypatch):
    class Failing(FakeEngine):
        def recognize(self, input_path, *args, **kwargs):
            raise base.EngineError("backend failed", details="log")

    monkeypatch.setitem(manual.ENGINES, "Fake", Failing())
    with pytest.raises(service.EngineError, match="^scans/a.png: backend"):
        service.recognize_batch_with_engine(
            zip_of({"scans/a.png": b"a"}), "Fake", service.EngineOptions()
        )
    assert list(workspace.dir("ocr").iterdir()) == []
