"""Exercise project reuse and graph persistence without cluster services."""

import json
from io import BytesIO
from unittest.mock import Mock
from zipfile import ZipFile

import pytest
from streamlit.testing.v1 import AppTest

from textlab.features.knowledge_graph import service
from textlab.features.knowledge_graph.corpus import build_corpus
from textlab.features.knowledge_graph.tests import conftest as samples
from textlab.features.knowledge_graph.tests.test_knowledge_graph_page import (
    PAGE,
)
from textlab.ui.streamlit import auth


@pytest.fixture
def records():
    return samples.records.__wrapped__()


@pytest.fixture
def tei():
    return samples.TEI


def write_project(folder, records, *, topics=True):
    folder.mkdir()
    lines = "".join(json.dumps(record) + "\n" for record in records)
    (folder / service.TABLE_JSONL).write_text(lines)
    if topics:
        (folder / service.TOPICS_JSONL).write_text(lines)
    return folder


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr(auth, "check_token", lambda: None)
    monkeypatch.setattr(service, "ollama_models", lambda: ["test-model"])
    return AppTest.from_file(str(PAGE), default_timeout=20)


def button(app, label):
    return next(item for item in app.button if item.label == label)


def test_saved_project_needs_neither_pdfs_nor_processing_services(
    app, monkeypatch, tmp_path, records
):
    corpus = write_project(tmp_path / "saved_project_corpus", records)
    grobid = Mock(side_effect=AssertionError("Grobid must not start"))
    models = Mock(side_effect=AssertionError("Models must not be queried"))
    monkeypatch.setattr(service, "ensure_grobid_server", grobid)
    monkeypatch.setattr(service, "ollama_models", models)
    app.session_state["kg_project"] = str(corpus)
    app.run()
    assert not app.exception
    assert not app.error
    assert (corpus / "full_corpus_graph.html").is_file()
    assert app.session_state["kg_graph"]["papers"] == 2
    grobid.assert_not_called()
    models.assert_not_called()


def test_graph_survives_reruns_and_view_changes(
    app, monkeypatch, tmp_path, records
):
    corpus = write_project(tmp_path / "saved_project_corpus", records)
    app.session_state["kg_project"] = str(corpus)
    app.run()
    original = app.session_state["kg_graph"]["html"]
    full_graph = Mock(wraps=service.build_full_corpus_graph)
    monkeypatch.setattr(service, "build_full_corpus_graph", full_graph)
    app.run()
    assert app.session_state["kg_graph"]["html"] == original
    full_graph.assert_not_called()
    assert all(item.label != "Update view" for item in app.button)
    assert app.multiselect[0].options == ["P0001 - a.pdf", "P0002 - b.pdf"]
    app.radio[0].set_value("Single paper").run()
    assert app.selectbox[0].options == ["P0001 - a.pdf", "P0002 - b.pdf"]
    assert not app.exception
    assert app.session_state["kg_graph"]["filename"] == "P0001_ego_graph.html"
    app.run()
    assert app.session_state["kg_graph"]["label"] == "Single paper"
    app.selectbox[0].set_value("P0002").run()
    assert not app.exception
    assert app.session_state["kg_graph"]["filename"] == "P0002_ego_graph.html"
    assert (corpus / "P0002_ego_graph.html").is_file()
    app.radio[0].set_value("Whole collection").run()
    assert (
        app.session_state["kg_graph"]["filename"] == "full_corpus_graph.html"
    )
    assert full_graph.call_count == 1
    app.multiselect[0].set_value(["P0002"]).run()
    assert app.session_state["kg_graph"]["papers"] == 1
    app.multiselect[0].set_value([]).run()
    assert any("Select at least one paper" in item.value for item in app.info)
    assert not app.get("iframe")


def test_open_another_project_resets_graph_and_paper_selection(
    app, tmp_path, records
):
    first = write_project(tmp_path / "first_project_corpus", records)
    second = write_project(tmp_path / "second_project_corpus", records[1:])
    app.session_state["kg_project"] = str(first)
    app.run()
    next(
        t for t in app.toggle if t.label == "Choose another project"
    ).set_value(True).run()
    app.radio[0].set_value("Open an existing project").run()
    app.text_input[0].set_value(str(second)).run()
    button(app, "Open project").click().run()
    assert not app.exception
    assert app.session_state["kg_project"] == str(second)
    assert app.session_state["kg_graph"]["papers"] == 1
    assert app.multiselect[0].options == ["P0002 - b.pdf"]


def test_partial_project_can_be_opened_without_grobid(
    app, monkeypatch, tmp_path, records
):
    corpus = write_project(
        tmp_path / "partial_project_corpus", records, topics=False
    )
    grobid = Mock(side_effect=AssertionError("Grobid must not start"))
    monkeypatch.setattr(service, "ensure_grobid_server", grobid)
    app.session_state["kg_project"] = str(corpus)
    app.run()
    assert not app.exception
    assert len(app.dataframe) == 1
    assert any("Re-extract topics" in item.value for item in app.info)
    grobid.assert_not_called()


def test_single_build_saves_papers_before_topic_failure_and_reuses_them(
    app, monkeypatch, tmp_path, tei
):
    papers = tmp_path / "papers"
    papers.mkdir()
    (papers / "a.pdf").write_bytes(b"test PDF")
    parser = Mock(return_value=tei)
    monkeypatch.setattr(service, "ensure_grobid_server", lambda: None)
    monkeypatch.setattr(
        service,
        "build_corpus",
        lambda pdfs, corpus, **kwargs: build_corpus(
            pdfs, corpus, parse=parser, **kwargs
        ),
    )
    monkeypatch.setattr(service, "ollama_client", lambda: object())
    from textlab.ui.streamlit.components import gpu

    monkeypatch.setattr(gpu, "free_gpu_for", lambda *args, **kwargs: None)
    extractor = Mock(side_effect=RuntimeError("Model unavailable"))
    monkeypatch.setattr(service, "extract_topics", extractor)
    app.run()
    next(r for r in app.radio if r.label == "Add papers").set_value(
        "Use a folder on this cluster"
    ).run()
    app.text_input[0].set_value(str(papers))
    app.text_input[1].set_value(str(tmp_path))
    button(app, "Build knowledge graph").click().run()
    assert not app.exception
    corpus = tmp_path / service.corpus_folder_name(papers)
    assert (corpus / "P0001" / "metadata.json").is_file()
    assert any("Model unavailable" in item.value for item in app.error)
    # Retry the original action: saved PDF extraction must not be repeated.
    app.run()
    next(
        t for t in app.toggle if t.label == "Choose another project"
    ).set_value(True).run()
    next(r for r in app.radio if r.label == "Add papers").set_value(
        "Use a folder on this cluster"
    ).run()
    app.text_input[0].set_value(str(papers))
    app.text_input[1].set_value(str(tmp_path))
    button(app, "Build knowledge graph").click().run()
    assert not app.exception
    assert parser.call_count == 1
    assert extractor.call_count == 2

    def complete_topics(corpus, *args, **kwargs):
        rows = service.read_records(corpus / service.TABLE_JSONL)
        for row in rows:
            row["topics"] = [
                {"label": "Graphs", "category": "Science", "confidence": 0.9}
            ]
        target = corpus / service.TOPICS_JSONL
        target.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return service.TopicSummary(len(rows), 0, target)

    extractor.side_effect = complete_topics
    app.run()
    next(
        t for t in app.toggle if t.label == "Show processing controls"
    ).set_value(True).run()
    button(app, "Re-extract topics").click().run()
    assert not app.exception
    assert not app.error
    assert (corpus / service.TOPICS_JSONL).is_file()
    assert (corpus / "full_corpus_graph.html").is_file()
    assert app.session_state["kg_graph"]["papers"] == 1
    assert parser.call_count == 1


@pytest.mark.parametrize("extension", ["pdf", "zip"])
def test_uploaded_project_builds_and_reopens_after_temporary_pdfs_are_removed(
    app, monkeypatch, tmp_path, tei, extension
):
    import streamlit as st

    from textlab.common.storage import Workspace
    from textlab.ui.streamlit.components import gpu

    pdf = b"%PDF-1.7\nTest paper"
    upload = BytesIO(pdf)
    if extension == "zip":
        upload = BytesIO()
        with ZipFile(upload, "w") as archive:
            archive.writestr("collection/Paper.pdf", pdf)
    upload.name = f"Paper.{extension}"
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: upload)
    workspace = Workspace(tmp_path / "private-workspace")
    monkeypatch.setattr(service, "get_workspace", lambda: workspace)
    monkeypatch.setattr(service, "ensure_grobid_server", lambda: None)
    monkeypatch.setattr(service, "ollama_client", lambda: object())
    monkeypatch.setattr(gpu, "free_gpu_for", lambda *args, **kwargs: None)
    staged_folders = []

    def parse(path):
        staged_folders.append(path.parent)
        return tei

    monkeypatch.setattr(
        service,
        "build_corpus",
        lambda pdfs, corpus, **kwargs: build_corpus(
            pdfs, corpus, parse=parse, **kwargs
        ),
    )

    def complete_topics(corpus, *args, **kwargs):
        rows = service.read_records(corpus / service.TABLE_JSONL)
        for row in rows:
            row["topics"] = []
        target = corpus / service.TOPICS_JSONL
        target.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return service.TopicSummary(len(rows), 0, target)

    monkeypatch.setattr(service, "extract_topics", complete_topics)
    app.run()
    assert not button(app, "Build knowledge graph").value
    assert button(app, "Build knowledge graph").disabled
    next(t for t in app.text_input if t.label == "Project name").set_value(
        "My literature"
    )
    next(t for t in app.text_input if t.label == "Save project in").set_value(
        str(tmp_path)
    )
    app.run()
    button(app, "Build knowledge graph").click().run()
    assert not app.exception
    assert not app.error
    corpus = tmp_path / "My literature_project_corpus"
    assert app.session_state["kg_project"] == str(corpus)
    assert (corpus / "P0001" / "metadata.json").is_file()
    assert staged_folders and not staged_folders[0].exists()
    assert app.multiselect[0].options == ["P0001 - Paper.pdf"]
    app.session_state["kg_graph"] = None
    app.run()
    assert not app.exception
    assert app.session_state["kg_graph"]["papers"] == 1
    assert (corpus / "full_corpus_graph.html").is_file()


def test_model_choices_reset_a_selection_removed_from_configuration(
    app, monkeypatch
):
    app.session_state["new_model"] = "old-installed-model"
    monkeypatch.setattr(
        service, "ollama_models", lambda: ["approved-a", "approved-b"]
    )
    app.run()
    model = next(
        s for s in app.selectbox if s.label == "Model for topic extraction"
    )
    assert model.options == ["approved-a", "approved-b"]
    assert model.value == "approved-a"
    model.set_value("approved-b").run()
    monkeypatch.setattr(service, "ollama_models", lambda: ["approved-c"])
    app.run()
    assert not app.exception
    model = next(
        s for s in app.selectbox if s.label == "Model for topic extraction"
    )
    assert model.value == "approved-c"


def test_gpustack_model_choice_does_not_require_session_models(
    app, monkeypatch
):
    app.run()
    session_models = Mock(
        side_effect=AssertionError("Session models not needed")
    )
    monkeypatch.setattr(service, "ollama_models", session_models)
    mode = next(r for r in app.radio if r.label == "Topic extraction mode")
    mode.set_value("University GPUStack service (API key required)").run()
    assert not app.exception
    assert any(t.label == "GPUStack API key" for t in app.text_input)
    session_models.assert_not_called()
