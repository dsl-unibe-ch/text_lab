"""The service with fake workers: steps, results, batch layout, cleanup."""

import io
import json
import threading
import zipfile

import pytest
from PIL import Image

from textlab.common.progress import CancelledError
from textlab.common.storage import Workspace
from textlab.features.handwriting import service
from textlab.features.handwriting.models import HandwritingModel

SPEC = HandwritingModel(
    "test-model", "Test model", "", "org/base", "org/adapter", "Read.", 10_000
)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    workspace = Workspace(tmp_path / "workspace")
    monkeypatch.setattr(service, "get_workspace", lambda: workspace)
    monkeypatch.setattr(service, "get_model", lambda key: SPEC)
    return workspace


@pytest.fixture
def workers(monkeypatch):
    """Fake workers: two lines per page; each line read as its crop name."""
    calls = {"detect": [], "recognize": [], "cancel": []}

    def detect(images, *, on_progress, cancel=None):
        calls["detect"].append(list(images))
        calls["cancel"].append(cancel)
        on_progress(service.Progress("Finding lines...", 1.0))
        two_lines = [
            [[10, 60], [300, 60], [300, 90], [10, 90]],
            [[10, 10], [300, 10], [300, 40], [10, 40]],
        ]
        return [two_lines for _ in images]

    def recognize(crops, spec, *, on_progress, cancel=None):
        calls["recognize"].append((list(crops), spec))
        calls["cancel"].append(cancel)
        for crop in crops:
            assert crop.exists()
        on_progress(service.Progress("Reading lines...", 1.0))
        return [crop.stem for crop in crops]

    monkeypatch.setattr(service, "detect_lines", detect)
    monkeypatch.setattr(service, "recognize_lines", recognize)
    return calls


def png(width=400, height=120):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def names_in(zip_bytes):
    return sorted(zipfile.ZipFile(io.BytesIO(zip_bytes)).namelist())


def test_a_document_is_detected_cropped_and_read(workspace, workers):
    updates = []
    run = service.transcribe_document(
        "../x/letter.png",
        png(),
        service.HandwritingOptions(model="test-model"),
        on_progress=updates.append,
    )
    page = run.document.pages[0]
    # Lines in the detector's order, which is kraken's reading order.
    assert [line.box for line in page.lines] == [
        (10, 60, 300, 90),
        (10, 10, 300, 40),
    ]
    assert run.text == "line-000000\nline-000001"
    assert run.text_name == "letter.txt"
    assert json.loads(run.json)["models_used"]["recognition"].startswith(
        "org/adapter"
    )
    assert len(run.previews) == 1
    assert workers["recognize"][0][1] == SPEC
    assert names_in(run.zip_bytes) == [
        "letter.json",
        "letter.txt",
        "models_used.txt",
        "page_xml/page_0001.xml",
    ]
    fractions = [u.fraction for u in updates if u.fraction is not None]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0
    assert not any(workspace.dir(service.WORKSPACE_AREA).iterdir())


def test_previews_are_optional(workspace, workers):
    options = service.HandwritingOptions(model="test-model", previews=False)
    run = service.transcribe_document("a.png", png(), options)
    assert run.previews == []


def test_a_page_without_lines_skips_recognition(
    workspace, workers, monkeypatch
):
    monkeypatch.setattr(
        service, "detect_lines", lambda images, on_progress, cancel: [[]]
    )
    run = service.transcribe_document(
        "blank.png", png(), service.HandwritingOptions(model="test-model")
    )
    assert run.document.line_count == 0
    assert workers["recognize"] == []


def test_a_batch_mirrors_the_archive(workspace, workers):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("box1/a.png", png())
        archive.writestr("box1/a.jpg", png())
        archive.writestr("b.png", png())
        archive.writestr("readme.txt", b"ignored")
    buffer.seek(0)
    result = service.transcribe_batch(
        buffer, service.HandwritingOptions(model="test-model")
    )
    assert (result.files, result.lines) == (3, 6)
    assert [(item.name, item.pages, item.lines) for item in result.items] == [
        ("b.png", 1, 2),
        ("box1/a.jpg", 1, 2),
        ("box1/a.png", 1, 2),
    ]
    # One worker run each for the whole batch.
    assert len(workers["detect"]) == 1 and len(workers["detect"][0]) == 3
    assert len(workers["recognize"]) == 1
    names = names_in(result.zip_bytes)
    assert "models_used.txt" in names
    for folder, stem in (("b", "b"), ("box1/a", "a"), ("box1/a_png", "a")):
        assert f"{folder}/{stem}.txt" in names, names
        assert f"{folder}/page_xml/page_0001.xml" in names
    text = zipfile.ZipFile(io.BytesIO(result.zip_bytes)).read("b/b.txt")
    # Crops are numbered across the batch, so b.png's lines come first.
    assert text.decode() == "line-000000\nline-000001"


def test_an_archive_without_documents_is_an_error(workspace, workers):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("notes.txt", b"x")
    buffer.seek(0)
    with pytest.raises(service.HandwritingError, match="no supported"):
        service.transcribe_batch(
            buffer, service.HandwritingOptions(model="test-model")
        )


def test_progress_of_a_step_is_mapped_onto_its_share():
    updates = []
    report = service._scaled(updates.append, 0.2, 0.6)
    report(service.Progress("half", 0.5))
    report(service.Progress("unknown", None))
    assert [(u.message, u.fraction) for u in updates] == [
        ("half", pytest.approx(0.4)),
        ("unknown", None),
    ]


def test_the_workers_are_given_the_cancel_event(workspace, workers):
    cancel = threading.Event()
    service.transcribe_document(
        "a.png",
        png(),
        service.HandwritingOptions(model="test-model"),
        cancel=cancel,
    )
    assert workers["cancel"] == [cancel, cancel]


def test_a_cancelled_run_stops_before_the_next_step(workspace, workers):
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(CancelledError):
        service.transcribe_document(
            "a.png",
            png(),
            service.HandwritingOptions(model="test-model"),
            cancel=cancel,
        )
    assert workers["detect"] == []
    assert not any(workspace.dir(service.WORKSPACE_AREA).iterdir())


def test_a_transcription_is_saved_with_its_page_images_and_lines(
    workspace, workers, tmp_path
):
    run = service.transcribe_document(
        "letter.tif",
        png(width=4000, height=1200),
        service.HandwritingOptions(model="test-model", previews=False),
    )
    out = tmp_path / "out"
    out.mkdir()
    saved = service.save_transcription(run, out)
    assert [(item.name, item.listed) for item in saved.files] == [
        ("letter_handwriting.zip", True),
        ("letter.txt", True),
        ("letter.json", True),
        ("letter_page-0001.jpg", False),
    ]
    (page,) = saved.pages
    # The image is scaled down for display; the boxes stay in the full
    # page's pixels, which the outlines are drawn in.
    assert Image.open(out / page.image).size == (1600, 480)
    assert (page.width, page.height) == (4000, 1200)
    assert [(line.id, line.number) for line in page.lines] == [
        ("p1-l1", 1),
        ("p1-l2", 2),
    ]
    assert page.lines[0].box == (10, 60, 290, 30)
    assert page.text == "line-000000\nline-000001"
    assert saved.lines == 2 and saved.model == "Test model"
    assert saved.models[0][0] == "Line detection"
    assert saved.models[1] == (
        "Text recognition",
        "org/adapter (LoRA adapter on org/base)",
    )


def test_models_are_named_for_people():
    assert service.cited_models({"recognition": "x", "other": "y"}) == (
        ("Text recognition", "x"),
        ("other", "y"),
    )
