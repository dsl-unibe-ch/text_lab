"""The OCR service, with the recognition pipeline replaced by a fake."""

import base64
import io
import threading
import zipfile

import pytest

from textlab.common.progress import CancelledError
from textlab.common.storage import Workspace
from textlab.features.ocr import doc_ir, service
from textlab.features.ocr import pipeline as ocr_pipeline


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


def test_the_cancel_event_reaches_the_pipeline(workspace, pipeline):
    cancel = threading.Event()
    service.recognize_document(
        "scan.pdf", b"%PDF", service.OcrOptions(), cancel=cancel
    )
    service.recognize_batch(
        zip_of({"a.pdf": b"a"}), service.OcrOptions(), cancel=cancel
    )
    assert [call["cancel"] for call in pipeline] == [cancel, cancel]


def test_a_cancelled_batch_stops_before_the_next_file(
    workspace, pipeline, monkeypatch
):
    cancel = threading.Event()
    real = service.process_document

    def process_and_cancel(input_path, workspace_dir, **kwargs):
        cancel.set()
        return real(input_path, workspace_dir, **kwargs)

    monkeypatch.setattr(service, "process_document", process_and_cancel)
    with pytest.raises(CancelledError):
        service.recognize_batch(
            zip_of({"a.pdf": b"a", "b.pdf": b"b"}),
            service.OcrOptions(),
            cancel=cancel,
        )
    assert [call["name"] for call in pipeline] == ["a.pdf"]
    assert list(workspace.dir("ocr").iterdir()) == []


def test_a_cancelled_pipeline_stops_before_it_starts(tmp_path):
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(CancelledError):
        ocr_pipeline.process_document(
            tmp_path / "scan.pdf", tmp_path / "work", cancel=cancel
        )


def test_a_batch_gives_an_overview_of_each_file(workspace, pipeline):
    result = service.recognize_batch(
        zip_of({"b.pdf": b"b", "scans/a.png": b"png"}), service.OcrOptions()
    )
    assert [item.name for item in result.files] == ["b.pdf", "scans/a.png"]
    first = result.files[0]
    assert (first.pages, first.text_layer_pages) == (1, 1)
    assert first.region_counts == {"text": 1}


def png(width=40, height=20):
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def rich_document():
    """A page with an image, a title, a table, a figure and a check box."""
    regions = [
        doc_ir.Region(
            id="p1_r0",
            type=doc_ir.TITLE,
            bbox=[2, 2, 30, 6],
            reading_order=0,
            content={"text": "Report"},
        ),
        doc_ir.Region(
            id="p1_r1",
            type=doc_ir.TABLE,
            bbox=[2, 8, 38, 14],
            reading_order=1,
            content={
                "html": "<table><tr><td>a</td><td onclick='x()'>b</td></tr>"
                "<script>alert(1)</script></table>"
            },
        ),
        doc_ir.Region(
            id="p1_r2",
            type=doc_ir.FIGURE,
            bbox=[2, 15, 10, 19],
            reading_order=2,
            content={"text": "Figure 1"},
            asset={"b64": base64.b64encode(png(8, 4)).decode(), "ext": "png"},
            visual_description=doc_ir.VisualDescription(
                description="A small white square.", model="vision:7b"
            ),
        ),
        doc_ir.Region(
            id="p1_r3",
            type=doc_ir.CHECKBOX,
            bbox=[30, 15, 34, 19],
            reading_order=3,
            content={"text": "Yes"},
            markup={"state": "checked"},
        ),
    ]
    page = doc_ir.Page(
        page_number=1,
        regions=regions,
        image_b64=base64.b64encode(png()).decode(),
    )
    return doc_ir.Document(pages=[page], source_name="report.pdf")


def test_a_saved_document_has_its_files_and_what_to_show(tmp_path):
    document = rich_document()
    result = service.DocumentResult(
        document=document,
        summary=service.document_summary(document),
        downloads=service.DocumentDownloads.build(document, "report"),
    )
    saved = service.save_document(result, tmp_path)

    listed = [item.name for item in saved.files if item.listed]
    assert listed[0] == "report_bundle.zip"
    assert {"report.txt", "report.json", "report_tables.zip"} <= set(listed)
    for item in saved.files:
        assert item.path == tmp_path / item.name
        assert item.path.is_file()

    page = saved.pages[0]
    assert (page.width, page.height) == (40, 20)
    assert page.image == "report_page-0001.png"
    assert page.text_layer is False
    title, table, figure, mark = page.regions
    assert title.box == (2.0, 2.0, 28.0, 4.0)
    assert "<td>a</td>" in table.html
    assert "script" not in table.html and "onclick" not in table.html
    assert saved.tables[0].csv == "report_table-p1_r1.csv"
    assert figure.image == "report_p1_r2.png"
    assert figure.description == "A small white square."
    assert saved.figures == ((1, figure),)
    assert mark.mark == "checked"
    assert ("Figure descriptions", "vision:7b") in saved.models
