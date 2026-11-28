"""The Knowledge Graph page, with the reading and the model faked.

``service.build_project`` and ``service.extract_topics_with`` write a
sample project (two papers with topics) and can be held at a gate, to
test cancelling; the rest runs for real: staging uploads, checking the
paths, finding and reading projects, drawing views of the graph with
NetworkX and Pyvis, and the downloads.
"""

import json
import re
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from textlab.common.progress import CancelledError, Progress
from textlab.features.knowledge_graph import graphs, service
from textlab.ui.web import pages
from textlab.ui.web.jobs import JobState
from textlab.ui.web.tests.conftest import BASE, wait_for
from textlab.ui.web.tests.test_job_pages import htmx

MODELS = service.ModelOptions(
    ("small:8b", "big:30b"), "NVIDIA RTX 4090", False
)

PDF = b"%PDF-1.7\nA small test paper."

RECORDS = [
    {
        "paper_id": "P0001",
        "title": "Graphs of Papers",
        "authors": "Ada Lovelace; Grace Hopper",
        "DOI": "10.1/ABC",
        "publication_date": "2024-05-01",
        "n_citations": 1,
        "pdf_filename": "a.pdf",
        "cited_titles": "Second Paper",
        "citations": '[{"title": "Second Paper", "DOI": ""}]',
        "cited_authors": "Alan Turing",
        "topics": [
            {
                "category": "Computer Science",
                "label": "Graph Drawing",
                "confidence": 0.9,
                "rationale": "It draws graphs.",
            }
        ],
    },
    {
        "paper_id": "P0002",
        "title": "Second Paper",
        "authors": "Grace Hopper",
        "DOI": "",
        "publication_date": "2023",
        "n_citations": 0,
        "pdf_filename": "b.pdf",
        "cited_titles": "",
        "citations": "[]",
        "cited_authors": "",
        "topics": [
            {
                "category": "Computer Science",
                "label": "Citation Networks",
                "confidence": 0.4,
                "rationale": "It counts citations.",
            }
        ],
    },
]


def write_project(folder, *, topics=True):
    """Write a project of the two sample papers into a folder."""
    folder.mkdir(parents=True, exist_ok=True)
    plain = [{k: v for k, v in r.items() if k != "topics"} for r in RECORDS]
    (folder / service.TABLE_JSONL).write_text(
        "".join(json.dumps(r) + "\n" for r in plain)
    )
    (folder / service.TABLE_CSV).write_text("paper_id,title\nP0001,Graphs\n")
    if topics:
        (folder / service.TOPICS_JSONL).write_text(
            "".join(json.dumps(r) + "\n" for r in RECORDS)
        )
    return folder


@pytest.fixture
def kg(monkeypatch):
    """Fake the reading and the model; record what each job was given."""
    calls = SimpleNamespace(builds=[], topics=[], gate=None, topic_error="")

    def hold(cancel):
        if calls.gate is not None:
            while not calls.gate.wait(0.01):
                if cancel.is_set():
                    raise CancelledError("stopped")

    def build_project(pdfs, corpus, topic_model, *, on_progress, cancel):
        calls.builds.append(
            SimpleNamespace(
                pdfs=[pdf.name for pdf in pdfs],
                found=all(pdf.is_file() for pdf in pdfs),
                corpus=corpus,
                topic_model=topic_model,
            )
        )
        on_progress(Progress("Processing 1/2: a.pdf", 0.0))
        hold(cancel)
        write_project(corpus, topics=not calls.topic_error)
        return service.BuildSummary(
            read=2,
            reused=0,
            failed=1,
            papers=2,
            topics=None if calls.topic_error else 2,
            topic_error=calls.topic_error,
        )

    def extract_topics_with(corpus, topic_model, *, on_progress, cancel):
        calls.topics.append(SimpleNamespace(corpus=corpus, model=topic_model))
        hold(cancel)
        return service.TopicSummary(2, 1, corpus / service.TOPICS_JSONL)

    monkeypatch.setattr(service, "build_project", build_project)
    monkeypatch.setattr(service, "extract_topics_with", extract_topics_with)
    monkeypatch.setattr(service, "model_options", lambda: MODELS)
    monkeypatch.setattr(service, "server_ready", lambda attempts=1: True)
    monkeypatch.setattr(service, "gpustack_ready", lambda: False)
    monkeypatch.setattr(service, "grobid_problem", lambda: "")
    return calls


def build(client, **fields):
    files = fields.pop("files", None)
    return client.post(
        f"{BASE}/knowledge-graph/build",
        data=fields,
        files=files,
        headers=htmx(client),
    )


def finished(runner, response):
    assert response.status_code == 200, response.text[:500]
    job_id = re.search(r'id="job-([0-9a-f]+)"', response.text).group(1)
    job = runner.get(job_id)
    wait_for(lambda: not job.active)
    return job


def open_folder(client, folder):
    response = client.post(
        f"{BASE}/knowledge-graph/open",
        data={"folder": str(folder)},
        headers=htmx(client),
    )
    assert response.status_code == 204, response.text[:500]
    url = response.headers["HX-Redirect"]
    assert url.startswith(f"{BASE}/knowledge-graph/")
    return url


def card(client, job):
    return client.get(f"{BASE}/jobs/{job.id}").text


@pytest.fixture
def papers(tmp_path):
    folder = tmp_path / "papers"
    folder.mkdir()
    for name in ("a.pdf", "b.pdf"):
        (folder / name).write_bytes(PDF)
    return folder


# --- starting a project ---------------------------------------------------


def test_the_page_offers_new_and_saved_projects(client, kg):
    html = client.get(f"{BASE}/knowledge-graph").text
    assert 'data-switch="start"' in html
    assert f'hx-post="{BASE}/knowledge-graph/build"' in html
    assert f'hx-get="{BASE}/knowledge-graph/projects-in"' in html
    assert 'accept=".pdf,.zip"' in html
    assert f'hx-get="{BASE}/knowledge-graph/papers-folder"' in html
    assert "<option selected>small:8b</option><option>big:30b</option>" in html
    assert 'value="gpustack" disabled' in html and "(not set up)" in html
    assert "New papers cannot be read" not in html
    assert "Build the knowledge graph" in html
    assert "Your projects" not in html


def test_a_silent_grobid_is_announced(client, kg, monkeypatch):
    monkeypatch.setattr(service, "grobid_problem", lambda: "No apptainer")
    html = client.get(f"{BASE}/knowledge-graph").text
    assert "New papers cannot be read in this session" in html
    assert "No apptainer" in html


def test_a_folder_of_papers_is_counted(client, kg, papers, tmp_path):
    def check(path):
        return client.get(
            f"{BASE}/knowledge-graph/papers-folder", params={"papers": path}
        ).text

    assert "2 PDFs in this folder." in check(str(papers))
    (tmp_path / "empty").mkdir()
    assert "No PDFs directly in this folder" in check(str(tmp_path / "empty"))
    assert "starting with / or ~" in check("papers")
    assert "There is no folder" in check(str(tmp_path / "missing"))
    assert "nothing is copied" in check("")


def test_a_project_is_built_from_an_upload(
    client, runner, kg, released, tmp_path
):
    job = finished(
        runner,
        build(
            client,
            source="upload",
            name="Review",
            save_in=str(tmp_path),
            model="big:30b",
            files={"file": ("paper.pdf", PDF, "application/pdf")},
        ),
    )
    assert job.state is JobState.DONE, job.error
    (run,) = kg.builds
    assert run.pdfs == ["paper.pdf"] and run.found
    assert run.corpus == (tmp_path / "Review_project_corpus").resolve()
    assert run.topic_model == service.TopicModel("big:30b")
    assert released == ["llm:big:30b"]
    assert not job.inputs.exists()
    assert job.title == "Review"
    html = card(client, job)
    assert "2 read" in html and "1 not read" in html
    assert "Topics for 2 papers." in html
    assert "tried again when you add papers" in html
    project_id = job.result.project_id
    assert f'href="{BASE}/knowledge-graph/{project_id}"' in html
    assert "Open the project" in html
    page = client.get(f"{BASE}/knowledge-graph").text
    assert "Your projects" in page and "Review" in page


def test_a_project_is_built_from_a_folder(client, runner, kg, papers):
    job = finished(runner, build(client, source="folder", papers=str(papers)))
    assert job.state is JobState.DONE, job.error
    (run,) = kg.builds
    assert run.pdfs == ["a.pdf", "b.pdf"]
    # Without a name or a place, next to the papers, named after them.
    assert run.corpus == (papers.parent / "papers_project_corpus").resolve()
    assert run.topic_model == service.TopicModel("small:8b")


def test_gpustack_needs_a_key_and_no_gpu(
    client, runner, kg, released, papers, monkeypatch
):
    monkeypatch.setattr(service, "gpustack_ready", lambda: True)
    fields = {"source": "folder", "papers": str(papers)}
    refused = build(client, topics_from="gpustack", **fields)
    assert refused.status_code == 422
    assert "Give your GPUStack API key" in refused.text
    job = finished(
        runner, build(client, topics_from="gpustack", api_key="key", **fields)
    )
    assert job.state is JobState.DONE, job.error
    assert kg.builds[0].topic_model == service.TopicModel(
        service.GPUSTACK_MODEL, api_key="key"
    )
    assert job.gpu is None and released == []


@pytest.mark.parametrize(
    "fields,message",
    [
        ({"source": "upload", "save_in": "/tmp"}, "Name the project"),
        ({"name": "a/b"}, "without slashes"),
        ({"name": "Review", "save_in": "relative"}, "starting with / or ~"),
        (
            {"name": "Review", "save_in": "/no/such/place"},
            "There is no folder",
        ),
        ({"name": "Review", "model": "other:1b"}, "Choose one of the models"),
        ({"topics_from": "gpustack", "api_key": "k"}, "not set up"),
    ],
)
def test_bad_uploads_are_explained(client, kg, fields, message):
    response = build(
        client,
        **{"source": "upload", "save_in": "/tmp", **fields},
        files={"file": ("paper.pdf", PDF, "application/pdf")},
    )
    assert response.status_code == 422
    assert message in response.text
    assert kg.builds == []


def test_bad_folders_are_explained(client, kg, tmp_path, monkeypatch):
    (tmp_path / "empty").mkdir()
    response = build(client, source="folder", papers=str(tmp_path / "empty"))
    assert response.status_code == 422
    assert "There are no PDFs" in response.text
    response = build(client, source="upload")
    assert response.status_code == 422
    assert "Choose a file first" in response.text
    monkeypatch.setattr(service, "server_ready", lambda attempts=1: False)
    papers = tmp_path / "papers"
    papers.mkdir()
    (papers / "a.pdf").write_bytes(PDF)
    response = build(client, source="folder", papers=str(papers))
    assert response.status_code == 422
    assert "does not answer" in response.text
    assert kg.builds == []


def test_a_build_can_be_cancelled(client, runner, kg, papers):
    kg.gate = threading.Event()
    response = build(client, source="folder", papers=str(papers))
    job_id = re.search(r'id="job-([0-9a-f]+)"', response.text).group(1)
    job = runner.get(job_id)
    wait_for(lambda: kg.builds)
    client.post(f"{BASE}/jobs/{job.id}/cancel", headers=htmx(client))
    wait_for(lambda: not job.active)
    assert job.state is JobState.CANCELLED
    assert "The papers read until then are kept" in card(client, job)


def test_failed_topics_are_shown_with_the_saved_papers(
    client, runner, kg, papers
):
    kg.topic_error = "The model server is down"
    job = finished(runner, build(client, source="folder", papers=str(papers)))
    assert job.state is JobState.DONE
    html = card(client, job)
    assert "The topics could not be extracted" in html
    assert "The model server is down" in html
    page = client.get(f"{BASE}/knowledge-graph/{job.result.project_id}").text
    assert "The topics are not extracted yet" in page
    assert "2 papers" in page and "no topics yet" in page


# --- finding and opening saved projects -----------------------------------


def test_saved_projects_are_found_and_opened(client, kg, tmp_path):
    write_project(tmp_path / "a_project_corpus")
    (tmp_path / "b_project_corpus").mkdir()

    def find(location):
        return client.get(
            f"{BASE}/knowledge-graph/projects-in",
            params={"location": location},
        ).text

    html = find(str(tmp_path))
    assert "Papers and topics" in html and "No papers yet" in html
    assert html.count(f'hx-post="{BASE}/knowledge-graph/open"') == 2
    assert '"folder": "' in html
    # A project's own folder is found by itself.
    assert find(str(tmp_path / "a_project_corpus")).count("found__item") == 1
    assert "There is no folder" in find(str(tmp_path / "missing"))

    url = open_folder(client, tmp_path / "a_project_corpus")
    # Opening it again finds the same project.
    assert open_folder(client, tmp_path / "a_project_corpus") == url
    page = client.get(url)
    assert page.status_code == 200
    assert "<h1" in page.text and ">a</h1>" in page.text

    refused = client.post(
        f"{BASE}/knowledge-graph/open",
        data={"folder": str(tmp_path)},
        headers=htmx(client),
    )
    assert refused.status_code == 422
    assert "is not a Knowledge Graph project" in refused.text


def test_a_folder_without_projects_says_so(client, kg, tmp_path):
    (tmp_path / "other").mkdir()
    html = client.get(
        f"{BASE}/knowledge-graph/projects-in",
        params={"location": str(tmp_path / "other")},
    ).text
    assert "No projects here" in html


def test_unknown_projects_are_not_found(client, kg):
    assert client.get(f"{BASE}/knowledge-graph/0123456789ab").status_code == (
        404
    )
    assert client.get(f"{BASE}/knowledge-graph/not-an-id").status_code == 404


# --- a project's page -----------------------------------------------------


def test_the_project_shows_its_graph_papers_and_topics(client, kg, tmp_path):
    url = open_folder(
        client, write_project(tmp_path / "review_project_corpus")
    )
    html = client.get(url).text
    assert f'data-page="knowledge_graph:{url.rsplit("/", 1)[1]}"' in html
    assert "2 papers" in html and "topics for 2" in html
    assert 'role="tab"' in html and "Papers" in html and "Topics" in html
    key = re.search(r'/graphs/([0-9a-f]+)\.json"', html).group(1)
    assert "data-graph-focus" in html
    assert "vis-network-9.1.2/vis-network.min.js" in html
    assert "vis-network-9.1.2/vis-network.min.css" in html
    assert 'name="papers" value="P0002" checked' in html
    assert '<span data-check-count="papers">2 of 2</span>' in html
    assert "Graphs of Papers" in html and "Ada Lovelace; Grace Hopper" in html
    assert "Citation Networks" in html and "It counts citations." in html
    assert "0.90" in html
    assert "graph-key__mark--topic" in html

    data = client.get(f"{url}/graphs/{key}.json").json()
    ids = {node["id"] for node in data["nodes"]}
    assert {"doi:10.1/abc", "P0002", "topic:Graph Drawing"} <= ids
    assert data["edges"] and data["options"]["physics"]["enabled"]

    page = client.get(f"{url}/graphs/{key}.html")
    assert page.headers["content-type"].startswith("text/html")
    assert "full_corpus_graph.html" in page.headers["content-disposition"]
    assert "vis-network" in page.text

    table = client.get(f"{url}/papers.csv")
    assert table.text.startswith("paper_id,title")
    assert "review_papers.csv" in table.headers["content-disposition"]
    topics = client.get(f"{url}/topics.json").json()
    assert [r["paper_id"] for r in topics] == ["P0001", "P0002"]


def test_the_graph_options_draw_other_views(client, kg, tmp_path):
    url = open_folder(
        client, write_project(tmp_path / "review_project_corpus")
    )

    def view(**fields):
        response = client.post(
            f"{url}/graph", data=fields, headers=htmx(client)
        )
        assert response.status_code == 200
        return response.text

    one = view(view="paper", paper="P0001", show=["topics"])
    assert "data-graph-focus" not in one
    key = re.search(r'/graphs/([0-9a-f]+)\.json"', one).group(1)
    labels = {
        node["label"]
        for node in client.get(f"{url}/graphs/{key}.json").json()["nodes"]
    }
    assert "Graph Drawing" in labels and "Ada Lovelace" not in labels
    assert (
        "P0001_ego_graph.html"
        in client.get(f"{url}/graphs/{key}.html").headers[
            "content-disposition"
        ]
    )

    some = view(
        view="collection", papers=["P0002"], show=["authors"], confidence="0.5"
    )
    assert "<li>1 paper</li>" in some and "data-graph-focus" in some

    # The same options give the same view.
    again = view(
        view="collection", papers=["P0002"], show=["authors"], confidence="0.5"
    )
    assert re.findall(r"/graphs/(\w+)\.json", again) == re.findall(
        r"/graphs/(\w+)\.json", some
    )

    assert "Tick at least one paper" in view(view="collection")
    assert "Choose one of the project" in view(view="paper", paper="P9")
    assert "a number from 0 to 1" in view(papers=["P0001"], confidence="2")
    assert client.get(f"{url}/graphs/0123456789ab.json").status_code == 404


def test_project_tools_run_as_jobs_of_the_project(
    client, runner, kg, papers, tmp_path
):
    url = open_folder(
        client, write_project(tmp_path / "review_project_corpus")
    )
    project_id = url.rsplit("/", 1)[1]

    update = finished(
        runner,
        client.post(
            f"{url}/update",
            data={"source": "folder", "papers": str(papers)},
            headers=htmx(client),
        ),
    )
    assert update.page == f"knowledge_graph:{project_id}"
    assert update.title == "review: new papers"
    assert (
        kg.builds[0].corpus == (tmp_path / "review_project_corpus").resolve()
    )
    assert "Show the new graph" in card(client, update)

    topics = finished(
        runner,
        client.post(
            f"{url}/topics", data={"model": "big:30b"}, headers=htmx(client)
        ),
    )
    assert topics.title == "review: topics"
    assert kg.topics[0].model == service.TopicModel("big:30b")
    html = card(client, topics)
    assert "Topics for 2 papers; 1 without an abstract have none." in html

    page = client.get(url).text
    assert f'id="job-{update.id}"' in page and f'id="job-{topics.id}"' in page

    (tmp_path / "review_project_corpus" / service.TABLE_JSONL).unlink()
    rebuilt = client.post(f"{url}/table", headers=htmx(client))
    assert rebuilt.status_code == 204
    assert rebuilt.headers["HX-Redirect"] == url
    assert (tmp_path / "review_project_corpus" / service.TABLE_JSONL).exists()


def test_the_table_waits_for_the_projects_job(client, runner, kg, tmp_path):
    url = open_folder(
        client, write_project(tmp_path / "review_project_corpus")
    )
    kg.gate = threading.Event()
    response = client.post(
        f"{url}/topics", data={"model": "small:8b"}, headers=htmx(client)
    )
    job_id = re.search(r'id="job-([0-9a-f]+)"', response.text).group(1)
    wait_for(lambda: kg.topics)
    refused = client.post(f"{url}/table", headers=htmx(client))
    assert refused.status_code == 422
    assert "A job works on this project now" in refused.text
    kg.gate.set()
    job = runner.get(job_id)
    wait_for(lambda: not job.active)


def test_topics_need_the_papers_first(client, kg, tmp_path):
    folder = tmp_path / "new_project_corpus"
    folder.mkdir()
    url = open_folder(client, folder)
    html = client.get(url).text
    assert "This project has no papers yet" in html
    assert "Extract the topics again" not in html
    refused = client.post(
        f"{url}/topics", data={"model": "small:8b"}, headers=htmx(client)
    )
    assert refused.status_code == 422
    assert "has no papers yet" in refused.text


# --- shared -----------------------------------------------------------------


def test_job_links_go_to_the_project_page():
    assert (
        pages.page_path("knowledge_graph:3f2a9c") == "/knowledge-graph/3f2a9c"
    )
    assert pages.page_label("knowledge_graph:3f2a9c") == "Knowledge Graph"
    assert pages.page_path("knowledge_graph") == "/knowledge-graph"
    assert pages.page_path("chat:ab12") == "/chat/ab12"


def test_the_key_has_the_colours_the_graph_is_drawn_in():
    css = (
        Path(pages.__file__).parent.parent
        / "static"
        / "css"
        / "components.css"
    ).read_text()
    for name, colour in {
        "paper": graphs.PAPER_COLOR,
        "author": graphs.AUTHOR_COLOR,
        "category": graphs.CATEGORY_COLOR,
        "topic": graphs.TOPIC_COLOR,
        "cited": graphs.CITED_EXTERNAL_COLOR,
        "cited-author": graphs.CITED_AUTHOR_COLOR,
    }.items():
        assert f"--node-{name}: {colour.lower()};" in css, name
    assert graphs.CITED_IN_CORPUS_COLOR == graphs.CATEGORY_COLOR
