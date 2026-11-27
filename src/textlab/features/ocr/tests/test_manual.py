"""Manual engine selection, with a fake engine in place of the real ones."""

import io
import json
import threading
import zipfile

import pytest
from PIL import Image

from textlab.common.progress import CancelledError, Progress
from textlab.common.storage import Workspace
from textlab.features.ocr import manual, service
from textlab.features.ocr.engines import base


def image_data(format_="PNG", size=(40, 20)):
    buffer = io.BytesIO()
    Image.new("RGB", size, "white").save(buffer, format=format_)
    return buffer.getvalue()


class FakeEngine(base.Engine):
    """Reads two pages from any file, or one record like OlmOCR."""

    name = "Fake"

    def __init__(self, record=False):
        self.record = record
        self.prepared = []
        self.calls = []
        self.cancels = []

    def prepare(self, options, on_progress):
        self.prepared.append(options)
        on_progress(Progress("Preparing..."))

    def recognize(
        self, input_path, work_dir, options, on_progress=None, cancel=None
    ):
        self.calls.append((input_path.name, input_path.read_bytes(), options))
        self.cancels.append(cancel)
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
    batch = service.recognize_batch_with_engine(
        archive, "Fake", service.EngineOptions(), on_progress=updates.append
    )
    assert [call[0] for call in engine.calls] == ["b.pdf", "a.png"]
    assert all(call[2].previews is False for call in engine.calls)
    assert [(item.name, item.pages) for item in batch.files] == [
        ("b.pdf", 2),
        ("scans/a.png", 2),
    ]
    assert batch.files[0].characters == len("one b\n\ntwo")
    assert batch.seconds >= sum(item.seconds for item in batch.files)
    names = names_in(batch.zip_bytes)
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


def test_a_run_can_be_cancelled_before_it_starts(workspace, engine):
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(CancelledError):
        service.recognize_with_engine(
            "a.png", b"x", "Fake", service.EngineOptions(), cancel=cancel
        )
    assert engine.calls == []
    assert list(workspace.dir("ocr").iterdir()) == []


def test_the_engine_is_given_the_cancel_event(workspace, engine):
    cancel = threading.Event()
    service.recognize_with_engine(
        "a.png", b"x", "Fake", service.EngineOptions(), cancel=cancel
    )
    assert engine.cancels == [cancel]


def test_a_batch_stops_between_files_and_is_not_blamed_on_one(
    workspace, monkeypatch
):
    cancel = threading.Event()

    class Cancelled(FakeEngine):
        def recognize(self, input_path, *args, **kwargs):
            self.calls.append(input_path.name)
            raise CancelledError("stopped")

    fake = Cancelled()
    monkeypatch.setitem(manual.ENGINES, "Fake", fake)
    archive = zip_of({"a.png": b"a", "b.png": b"b"})
    with pytest.raises(CancelledError, match="^stopped$"):
        service.recognize_batch_with_engine(
            archive, "Fake", service.EngineOptions(), cancel=cancel
        )
    assert fake.calls == ["a.png"]
    assert list(workspace.dir("ocr").iterdir()) == []


TABLE = (
    '<table onclick="alert(1)"><tr><th>a</th></tr><tr><td>1</td></tr></table>'
)


class TableEngine(FakeEngine):
    """Reads a page of text and a table, like GLM-OCR, with its image."""

    markdown = True

    def recognize(self, input_path, *args, **kwargs):
        return base.EngineOutput(
            pages=[
                base.PageText(
                    1,
                    f"# Prices\n{TABLE}",
                    image=image_data("TIFF"),
                )
            ]
        )


class LineEngine(FakeEngine):
    """Reads two outlined lines, like EasyOCR, with the page image."""

    def recognize(self, input_path, *args, **kwargs):
        lines = [
            base.TextLine("Hello", ((1, 2), (11, 2), (11, 6), (1, 6))),
            base.TextLine("there", ((1, 8), (11.25, 8), (11, 12), (1, 12))),
        ]
        return base.EngineOutput(
            pages=[
                base.PageText(
                    1, "Hello\nthere", lines=lines, image=image_data("JPEG")
                )
            ]
        )


def test_tables_are_written_as_csv(workspace, monkeypatch):
    monkeypatch.setitem(manual.ENGINES, "Fake", TableEngine())
    run = service.recognize_with_engine(
        "prices.png", b"x", "Fake", service.EngineOptions()
    )
    assert "page_0001_table_1.csv" in names_in(run.zip_bytes)


def test_a_run_is_saved_with_its_tables_and_page_image(
    workspace, monkeypatch, tmp_path
):
    monkeypatch.setitem(manual.ENGINES, "Fake", TableEngine())
    run = service.recognize_with_engine(
        "prices.tif", b"x", "Fake", service.EngineOptions()
    )
    saved = service.save_engine_run(run, "prices.tif", tmp_path)
    assert saved.markdown is True
    assert [(item.name, item.listed) for item in saved.files] == [
        ("prices_outputs.zip", True),
        ("prices.txt", True),
        ("prices.json", True),
        ("prices_page-0001.png", False),
        ("prices_page-0001_table-1.csv", False),
    ]
    (page,) = saved.pages
    assert (page.image, page.width, page.height) == (
        "prices_page-0001.png",
        40,
        20,
    )
    assert Image.open(tmp_path / page.image).format == "PNG"
    text, table = page.parts
    assert (text.kind, text.text) == ("text", "# Prices")
    assert table.kind == "table" and table.csv.endswith(".csv")
    assert "<td>1</td>" in table.html and "alert" not in table.html
    assert (tmp_path / table.csv).read_text().splitlines() == ["a", "1"]
    assert saved.tables == 1 and saved.lines == 0


def test_a_run_is_saved_with_its_outlined_lines(
    workspace, monkeypatch, tmp_path
):
    monkeypatch.setitem(manual.ENGINES, "Fake", LineEngine())
    run = service.recognize_with_engine(
        "photo.jpg", b"x", "Fake", service.EngineOptions()
    )
    saved = service.save_engine_run(run, "photo.jpg", tmp_path)
    (page,) = saved.pages
    assert page.image == "photo_page-0001.jpg"
    assert [(line.id, line.text) for line in page.lines] == [
        ("p1-l1", "Hello"),
        ("p1-l2", "there"),
    ]
    assert page.lines[1].points == "1.0,8.0 11.2,8.0 11.0,12.0 1.0,12.0"
    assert page.parts == ()
    assert saved.lines == 2 and saved.markdown is False


def test_an_engine_record_is_offered_as_json_lines(
    workspace, monkeypatch, tmp_path
):
    monkeypatch.setitem(manual.ENGINES, "Fake", FakeEngine(record=True))
    run = service.recognize_with_engine(
        "paper.pdf", b"%PDF", "Fake", service.EngineOptions()
    )
    saved = service.save_engine_run(run, "paper.pdf", tmp_path)
    record = saved.files[2]
    assert (record.name, record.label) == (
        "paper.jsonl",
        "Fake's record, JSON Lines",
    )
    assert saved.pages[0].image is None
