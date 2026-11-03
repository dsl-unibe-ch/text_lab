"""Uploaded PDFs are staged privately and preserve corpus filename identity."""

from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from textlab.common.storage import Workspace
from textlab.features.knowledge_graph import service

PDF = b"%PDF-1.7\nA small test paper."


def archive_bytes(members):
    data = BytesIO()
    with ZipFile(data, "w", compression=ZIP_DEFLATED) as archive:
        for name, content in members:
            archive.writestr(name, content)
    return data.getvalue()


@pytest.fixture(autouse=True)
def workspace(monkeypatch, tmp_path):
    workspace = Workspace(tmp_path / "workspace")
    monkeypatch.setattr(service, "get_workspace", lambda: workspace)
    return workspace


def test_single_pdf_is_temporary_and_keeps_its_name(workspace):
    with service.staged_papers("../Paper.PDF", PDF) as folder:
        assert folder.is_relative_to(workspace.root)
        assert [p.name for p in service.list_pdfs(folder)] == ["Paper.PDF"]
        assert (folder / "Paper.PDF").read_bytes() == PDF
    assert not folder.exists()


def test_zip_reads_nested_pdfs_and_ignores_other_files():
    data = archive_bytes(
        [
            ("collection/First.pdf", PDF),
            ("Second.PDF", PDF),
            ("notes.txt", b"Notes"),
            ("__MACOSX/._First.pdf", b"Mac metadata"),
        ]
    )
    with service.staged_papers("papers.zip", data) as folder:
        assert [p.name for p in service.list_pdfs(folder)] == [
            "First.pdf",
            "Second.PDF",
        ]
    assert not folder.exists()


@pytest.mark.parametrize(
    "members, message",
    [
        ([("a/Paper.pdf", PDF), ("b/Paper.pdf", PDF)], "unique filename"),
        ([("a/Paper.pdf", PDF), ("b/paper.PDF", PDF)], "unique filename"),
        ([("../outside.pdf", PDF)], "escapes"),
        ([("/outside.pdf", PDF)], "escapes"),
        ([("notes.txt", b"Notes")], "no PDF"),
        ([("fake.pdf", b"Not a PDF")], "PDF header"),
    ],
)
def test_invalid_zip_is_rejected_and_cleaned(workspace, members, message):
    with pytest.raises(ValueError, match=message):
        with service.staged_papers("papers.zip", archive_bytes(members)):
            pytest.fail("Invalid archive was accepted")
    assert not list(workspace.dir("knowledge_graph").iterdir())


@pytest.mark.parametrize(
    "name, data",
    [
        ("papers.zip", b"not a zip"),
        ("paper.pdf", b"not a PDF"),
        ("notes.txt", b"text"),
    ],
)
def test_invalid_upload_is_rejected(name, data):
    with pytest.raises(ValueError):
        with service.staged_papers(name, data):
            pytest.fail("Invalid upload was accepted")


def test_extractions_survive_cleanup_and_reupload_skips_grobid(tmp_path, tei):
    corpus = tmp_path / "saved_project_corpus"
    calls = []

    def parse(path):
        calls.append(path.name)
        return tei

    with service.staged_papers("Paper.pdf", PDF) as folder:
        first = service.build_corpus(
            service.list_pdfs(folder), corpus, parse=parse
        )
    assert not folder.exists()
    assert first.processed == 1
    assert (corpus / "P0001" / "metadata.json").is_file()
    with service.staged_papers("Paper.pdf", PDF) as folder:
        second = service.build_corpus(
            service.list_pdfs(folder), corpus, parse=parse
        )
    assert second.skipped == 1
    assert calls == ["Paper.pdf"]


def test_staging_is_cleaned_when_processing_raises():
    with pytest.raises(RuntimeError, match="Processing failed"):
        with service.staged_papers("Paper.pdf", PDF) as folder:
            raise RuntimeError("Processing failed")
    assert not folder.exists()


@pytest.mark.parametrize(
    "name", ["", " ", ".", "..", "../project", "/tmp/a", "a\\b"]
)
def test_project_name_cannot_escape_save_location(name):
    with pytest.raises(ValueError):
        service.project_folder_name(name)


def test_project_name_keeps_readable_spaces():
    assert service.project_folder_name(" Literature review ") == (
        "Literature review_project_corpus"
    )
