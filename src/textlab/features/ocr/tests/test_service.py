"""The OCR service, with the recognition pipeline replaced by a fake."""

import io
import zipfile

import pytest

from textlab.common.storage import Workspace
from textlab.features.ocr import doc_ir, service


def make_document(name):
    region = doc_ir.Region(
        id="p1_r0",
        type=doc_ir.TEXT,
        bbox=[0, 0, 10, 10],
        reading_order=0,
        content={"text": f"Text of {name}."},
        source="native",
    )
    return doc_ir.Document(
        pages=[doc_ir.Page(page_number=1, regions=[region], source="native")],
        source_name=name,
    )


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    workspace = Workspace(tmp_path / "workspace")
    monkeypatch.setattr(service, "get_workspace", lambda: workspace)
    return workspace


@pytest.fixture
def pipeline(monkeypatch):
    """Record what the pipeline is asked to do, and report progress."""
    calls = []

    def process_document(input_path, workspace_dir, **kwargs):
        calls.append(
            {
                "name": input_path.name,
                "data": input_path.read_bytes(),
                "workspace": workspace_dir,
                **kwargs,
            }
        )
        kwargs["progress"](0.5, "Recognizing...")
        return make_document(input_path.name)

    monkeypatch.setattr(service, "process_document", process_document)
    return calls


def test_a_document_is_recognized_from_the_workspace(workspace, pipeline):
    updates = []
    options = service.OcrOptions(
        native_fast_lane=False, searchable_pdf=True, ocr_lang="deu"
    )
    result = service.recognize_document(
        "../uploads/scan.pdf", b"%PDF", options, on_progress=updates.append
    )
    call = pipeline[0]
    assert call["name"] == "scan.pdf"  # Only the last part of the name.
    assert call["data"] == b"%PDF"
    assert call["native_fast_lane"] is False
    assert call["searchable_pdf"] is True
    assert call["ocr_lang"] == "deu"
    assert workspace.dir("ocr") in call["workspace"].parents
    assert [(u.message, u.fraction) for u in updates] == [
        ("Recognizing...", 0.5)
    ]
    assert result.summary["n_pages"] == 1
    assert result.downloads.stem == "scan"
    assert result.downloads.text == b"Text of scan.pdf.\n"
    # The job folder is gone once the document is done.
    assert list(workspace.dir("ocr").iterdir()) == []


def test_downloads_are_refreshed_after_a_review():
    document = make_document("a.pdf")
    downloads = service.DocumentDownloads.build(document, "a")
    document.pages[0].regions[0].content["text"] = "Corrected."
    downloads.refresh_responses(document)
    assert b"Corrected." in downloads.json


def zip_of(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    buffer.seek(0)
    return buffer


def test_a_batch_mirrors_the_archive(workspace, pipeline):
    archive = zip_of(
        {
            "b.pdf": b"%PDF b",
            "scans/A.png": b"png",
            "notes.txt": b"not a document",
            "._b.pdf": b"macOS metadata",
        }
    )
    updates = []
    result = service.recognize_batch(
        archive, service.OcrOptions(), on_progress=updates.append
    )
    assert [call["name"] for call in pipeline] == ["b.pdf", "A.png"]
    assert result.file_count == 2
    assert result.survey is None
    names = zipfile.ZipFile(io.BytesIO(result.zip_bytes)).namelist()
    assert "b/document.md" in names
    assert "scans/A/document.md" in names
    assert not any("notes" in name for name in names)
    messages = [update.message for update in updates]
    assert "File 1 of 2 · b.pdf — Recognizing..." in messages
    assert updates[-1].fraction == 1.0
    assert list(workspace.dir("ocr").iterdir()) == []


def test_a_batch_without_documents_is_refused(workspace, pipeline):
    with pytest.raises(ValueError, match="No valid documents"):
        service.recognize_batch(
            zip_of({"notes.txt": b"x"}), service.OcrOptions()
        )
    assert pipeline == []
    assert list(workspace.dir("ocr").iterdir()) == []


def test_batch_files_share_one_worker_session(workspace, pipeline):
    service.recognize_batch(
        zip_of({"a.pdf": b"a", "b.pdf": b"b"}), service.OcrOptions()
    )
    sessions = {id(call["vl_session"]) for call in pipeline}
    assert len(sessions) == 1


def test_the_page_reads_input_types_from_the_service():
    assert ".pdf" in service.INPUT_EXTENSIONS
    assert all(ext.startswith(".") for ext in service.INPUT_EXTENSIONS)
