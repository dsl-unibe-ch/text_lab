"""Projects in one go: staging uploads from disk, building with topics,
reading a project back, views of its graph, and stopping the steps.

Grobid and the model are faked: Grobid returns the sample TEI, and the
model is the fake OpenAI client of ``test_topics``.
"""

import json
import threading
from functools import partial
from io import BytesIO
from zipfile import ZipFile

import pytest

from textlab.common.config import Settings
from textlab.common.progress import CancelledError, Progress
from textlab.features.knowledge_graph import (
    corpus,
    grobid,
    projects,
    service,
)
from textlab.features.knowledge_graph.tests.test_topics import (
    ANSWER,
    FakeClient,
)

PDF = b"%PDF-1.7\nA small test paper."


def zip_of(members):
    data = BytesIO()
    with ZipFile(data, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return data.getvalue()


@pytest.fixture
def papers(tmp_path):
    folder = tmp_path / "papers"
    folder.mkdir()
    for name in ("a.pdf", "b.pdf"):
        (folder / name).write_bytes(PDF)
    return folder


@pytest.fixture
def grobid_reads(monkeypatch, tei):
    """Grobid runs, and reads every PDF but those named in ``fail``."""
    fail = set()

    def parse(pdf):
        if pdf.name in fail:
            raise RuntimeError("Grobid returned status 500")
        return tei

    monkeypatch.setattr(projects, "ensure_grobid_server", lambda: None)
    monkeypatch.setattr(
        projects, "build_corpus", partial(corpus.build_corpus, parse=parse)
    )
    return fail


@pytest.fixture
def model(monkeypatch):
    """The session's model, installed, answering every paper."""
    client = FakeClient(*[ANSWER] * 10)
    monkeypatch.setattr(projects, "model_installed", lambda name: True)
    monkeypatch.setattr(projects, "ollama_client", lambda: client)
    return client


# --- staging uploads on disk ----------------------------------------------


def test_an_uploaded_pdf_is_staged_under_its_name(tmp_path):
    upload = tmp_path / "upload.bin"
    upload.write_bytes(PDF)
    staged = service.stage_papers(upload, tmp_path / "papers", "Paper.PDF")
    assert [path.name for path in staged] == ["Paper.PDF"]
    assert staged[0].read_bytes() == PDF


def test_a_zip_on_disk_gives_its_pdfs_without_subfolders(tmp_path):
    upload = tmp_path / "papers.zip"
    upload.write_bytes(
        zip_of(
            {
                "set/one.pdf": PDF,
                "set/deep/two.pdf": PDF,
                "set/notes.txt": b"not a paper",
                "__MACOSX/set/._one.pdf": b"metadata",
            }
        )
    )
    staged = service.stage_papers(upload, tmp_path / "papers")
    assert [path.name for path in staged] == ["one.pdf", "two.pdf"]


@pytest.mark.parametrize(
    "name,data,message",
    [
        ("paper.docx", PDF, "PDF document or ZIP"),
        ("paper.pdf", b"no header", "PDF header"),
        ("papers.zip", b"not a zip", "not valid"),
        ("papers.zip", zip_of({"a.pdf": PDF, "b/A.PDF": PDF}), "unique"),
        ("papers.zip", zip_of({"notes.txt": b"x"}), "no PDF papers"),
    ],
)
def test_bad_uploads_on_disk_are_refused(tmp_path, name, data, message):
    upload = tmp_path / name
    upload.write_bytes(data)
    with pytest.raises(ValueError, match=message):
        service.stage_papers(upload, tmp_path / "papers")


# --- building a project ---------------------------------------------------


def test_a_project_is_built_with_its_topics(papers, grobid_reads, model):
    folder = papers.parent / "review_project_corpus"
    updates = []
    summary = service.build_project(
        service.list_pdfs(papers),
        folder,
        service.TopicModel("small:8b"),
        on_progress=updates.append,
    )
    assert summary == service.BuildSummary(
        read=2, reused=0, failed=0, papers=2, topics=2, without_abstract=0
    )
    assert model.requests[0]["model"] == "small:8b"
    project = service.read_project(folder)
    assert project.name == "review"
    assert [p["paper_id"] for p in project.papers] == ["P0001", "P0002"]
    assert project.with_topics == 2 and not project.topics_stale
    assert any(u.message.startswith("Processing") for u in updates)


def test_failed_topics_keep_the_papers(papers, grobid_reads, monkeypatch):
    monkeypatch.setattr(projects, "model_installed", lambda name: True)
    monkeypatch.setattr(
        projects, "ollama_client", lambda: FakeClient(*[OSError("down")] * 9)
    )
    folder = papers.parent / "review_project_corpus"
    summary = service.build_project(
        service.list_pdfs(papers), folder, service.TopicModel("small:8b")
    )
    assert summary.papers == 2 and summary.topics is None
    assert "down" in summary.topic_error
    project = service.read_project(folder)
    assert len(project.papers) == 2 and project.records == []
    assert not (folder / "corpus_table.with_topics.tmp.jsonl").exists()


def test_a_build_reuses_papers_and_says_which_failed(
    papers, grobid_reads, model
):
    folder = papers.parent / "review_project_corpus"
    grobid_reads.add("b.pdf")
    first = service.build_project(
        service.list_pdfs(papers), folder, service.TopicModel("small:8b")
    )
    assert (first.read, first.failed, first.papers) == (1, 1, 1)
    grobid_reads.clear()
    second = service.build_project(
        service.list_pdfs(papers), folder, service.TopicModel("small:8b")
    )
    assert (second.read, second.reused, second.papers) == (1, 1, 2)


def test_a_build_without_a_readable_paper_fails(papers, grobid_reads, model):
    grobid_reads.update({"a.pdf", "b.pdf"})
    with pytest.raises(ValueError, match="No paper could be read"):
        service.build_project(
            service.list_pdfs(papers),
            papers.parent / "x_project_corpus",
            service.TopicModel("small:8b"),
        )


def test_a_stopped_build_keeps_what_it_read(papers, grobid_reads, model):
    folder = papers.parent / "review_project_corpus"
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(CancelledError):
        service.build_project(
            service.list_pdfs(papers),
            folder,
            service.TopicModel("small:8b"),
            cancel=cancel,
        )
    # The table is written, with the papers read until then: none here.
    assert (folder / service.TABLE_JSONL).exists()
    assert model.requests == []


def test_a_missing_session_model_is_downloaded_first(
    tmp_path, monkeypatch, model
):
    (tmp_path / service.TABLE_JSONL).write_text(
        json.dumps({"paper_id": "P0001", "title": "T", "abstract": "A"}) + "\n"
    )
    pulled = []
    monkeypatch.setattr(projects, "model_installed", lambda name: False)
    monkeypatch.setattr(
        projects,
        "pull_model",
        lambda name, on_progress, cancel: pulled.append(name),
    )
    summary = service.extract_topics_with(
        tmp_path, service.TopicModel("big:30b")
    )
    assert pulled == ["big:30b"] and summary.processed == 1


def test_gpustack_extracts_with_the_users_key(tmp_path, monkeypatch):
    (tmp_path / service.TABLE_JSONL).write_text(
        json.dumps({"paper_id": "P0001", "title": "T", "abstract": "A"}) + "\n"
    )
    client = FakeClient(ANSWER)
    keys = []

    def gpustack(key):
        keys.append(key)
        return client

    monkeypatch.setattr(projects, "gpustack_client", gpustack)
    monkeypatch.setattr(
        projects,
        "model_installed",
        lambda name: pytest.fail("GPUStack needs no session model"),
    )
    topic_model = service.TopicModel(service.GPUSTACK_MODEL, api_key="secret")
    assert topic_model.remote and "secret" not in repr(topic_model)
    service.extract_topics_with(tmp_path, topic_model)
    assert keys == ["secret"]
    assert client.requests[0]["model"] == service.GPUSTACK_MODEL


def test_stopped_topics_keep_the_ones_before(tmp_path):
    (tmp_path / service.TABLE_JSONL).write_text(
        json.dumps({"paper_id": "P0001", "title": "T", "abstract": "A"}) + "\n"
    )
    (tmp_path / service.TOPICS_JSONL).write_text("before\n")
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(CancelledError):
        service.extract_topics(tmp_path, FakeClient(), cancel=cancel)
    assert (tmp_path / service.TOPICS_JSONL).read_text() == "before\n"
    assert not (tmp_path / "corpus_table.with_topics.tmp.jsonl").exists()


# --- reading a project ----------------------------------------------------


def write_records(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def test_a_project_is_read_back(tmp_path, records):
    folder = tmp_path / "review_project_corpus"
    folder.mkdir()
    assert service.read_project(folder).papers == []
    write_records(folder / service.TABLE_JSONL, records)
    project = service.read_project(folder)
    assert len(project.papers) == 2 and project.records == []
    assert not project.topics_stale
    write_records(folder / service.TOPICS_JSONL, records[:1])
    project = service.read_project(folder)
    assert project.with_topics == 1 and project.topics_stale
    assert service.is_project(folder)
    assert service.is_project(tmp_path) is False
    assert service.project_name(folder) == "review"


def test_project_names_cannot_hold_a_path():
    assert service.project_folder_name(" Review ") == "Review_project_corpus"
    with pytest.raises(ValueError):
        service.project_folder_name("../escape")


# --- views of the graph ---------------------------------------------------


def test_a_view_of_the_whole_collection(records):
    view = service.draw_graph(records, service.GraphOptions())
    ids = {node["id"] for node in view.data["nodes"]}
    assert {"doi:10.1/abc", "P0002", "author:Grace Hopper"} <= ids
    assert "topic:Graph Drawing" in ids
    assert view.papers == 2 and view.nodes == len(ids)
    assert view.edges == len(view.data["edges"])
    assert view.filename == "full_corpus_graph.html"
    assert view.data["options"]["interaction"]["navigationButtons"]
    # Topics belong to their categories by dashed edges, as vis.js draws
    # them.
    assert any(
        edge.get("dashes") and edge["label"] == "belongs_to"
        for edge in view.data["edges"]
    )
    json.dumps(view.data)
    assert "vis-network" in view.html()


def test_a_view_of_some_papers_or_one(records):
    some = service.draw_graph(records, service.GraphOptions(papers=("P0002",)))
    assert some.papers == 1
    one = service.draw_graph(
        records,
        service.GraphOptions(
            paper_id="P0001", authors=False, cited_papers=True
        ),
    )
    assert one.filename == "P0001_ego_graph.html"
    labels = {node["label"] for node in one.data["nodes"]}
    assert "Ada Lovelace" not in labels and "P0002" in labels


@pytest.mark.parametrize(
    "options,message",
    [
        (service.GraphOptions(paper_id="P9999"), "no paper P9999"),
        (service.GraphOptions(papers=("P9999",)), "at least one paper"),
    ],
)
def test_a_view_of_nothing_is_refused(records, options, message):
    with pytest.raises(ValueError, match=message):
        service.draw_graph(records, options)


# --- Grobid ---------------------------------------------------------------


def test_grobid_says_why_it_cannot_run(monkeypatch):
    monkeypatch.setattr(grobid, "_port_open", lambda host, port: True)
    assert service.grobid_problem() == ""
    monkeypatch.setattr(grobid, "_port_open", lambda host, port: False)
    monkeypatch.setattr(grobid, "get_settings", lambda: Settings())
    assert "TEXT_LAB_GROBID_CONTAINER" in service.grobid_problem()


def test_progress_reaches_the_caller(papers, grobid_reads, model):
    updates = []
    service.build_project(
        service.list_pdfs(papers),
        papers.parent / "r_project_corpus",
        service.TopicModel("small:8b"),
        on_progress=updates.append,
    )
    assert Progress("Building corpus metadata table...", 1.0) in updates
