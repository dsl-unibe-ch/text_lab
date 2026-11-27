"""The Knowledge Graph page: graphs of a collection of research papers.

A project is a folder on the cluster (``<name>_project_corpus``) that keeps
what Grobid read from the papers and the topics a model found in them, so
it can be explored again in a later session. The page ``/knowledge-graph``
starts a project, from uploaded PDFs or a folder of them on the cluster,
or opens a saved one; a build is a job (Grobid reads the PDFs, then a
model extracts each paper's topics). A project's own page,
``/knowledge-graph/<id>``, shows its graph, papers and topics, and runs its
updates as jobs of that page (``knowledge_graph:<id>``).

The graph is drawn in the browser by vis-network (vendored as the version
Pyvis embeds in the downloads), from the nodes and edges that
``service.draw_graph`` gives; ``app.js`` loads it only then. The session's
projects are kept in ``app.state.kg_projects``; links name a project by a
random id, never by its path.
"""

from __future__ import annotations

import dataclasses
import os
import re
import secrets
import threading
from functools import partial
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)
from markupsafe import Markup
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import FormData, UploadFile

from textlab.common import gpu_manager
from textlab.common.progress import Progress
from textlab.features.knowledge_graph import service
from textlab.ui.web import uploads
from textlab.ui.web.auth import Checked, protected_router
from textlab.ui.web.jobs import (
    Job,
    JobContext,
    JobRunner,
    content_disposition,
    render_card,
    runner_of,
)
from textlab.ui.web.render import is_htmx, renderer, root

router = protected_router()

#: The page's id, which names its build jobs; a project's page is
#: ``knowledge_graph:<id>`` (:func:`stream_page`).
PAGE = "knowledge_graph"

#: The page's path below the base path.
PATH = "/knowledge-graph"

#: What can be uploaded: one PDF, or a ZIP of PDFs.
EXTENSIONS = (".pdf", ".zip")

#: How many times a job that uses the session's model checks that the
#: model server answers, half a second apart.
SERVER_ATTEMPTS = 4

#: Views of a project's graph kept for the browser to fetch, the newest.
GRAPHS_KEPT = 4

#: Most fields of the graph's options: a check box for each paper, and a
#: few more.
MAX_FIELDS = 10_000

#: What the cards of cancelled jobs say: the work kept is in the project.
CANCELLED_BUILD = (
    "Cancelled. The papers read until then are kept in the project, and "
    "reused when you add papers again."
)
CANCELLED_TOPICS = "Cancelled. The topics before are kept."

#: The templates of the jobs' results.
BUILT_TEMPLATE = "knowledge_graph/built.html"
TOPICS_TEMPLATE = "knowledge_graph/topics_done.html"

#: The node types a graph can show, by the form's values.
SHOWN = ("authors", "topics", "cited_papers", "cited_authors")

_PROJECT_ID = re.compile(r"[0-9a-f]{12}")


# ---------------------------------------------------------------------------
# The session's projects
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DrawnGraph:
    """A view of a project's graph, kept for the browser to fetch.

    Attributes:
        key: Random id, used in links.
        options: What it shows.
        stamp: The topics file's time and size when it was drawn, so a
            view of older topics is not reused.
        view: The view.
    """

    key: str
    options: service.GraphOptions
    stamp: tuple[int, int]
    view: service.GraphView


@dataclasses.dataclass(eq=False)
class OpenProject:
    """A project of the session: opened, or built here.

    Attributes:
        id: Random id, used in links.
        folder: The project folder.
        graphs: The views drawn last, oldest first.
        lock: Guards ``graphs``.
    """

    id: str
    folder: Path
    graphs: list[DrawnGraph] = dataclasses.field(default_factory=list)
    lock: threading.Lock = dataclasses.field(
        default_factory=threading.Lock, repr=False
    )

    @property
    def name(self) -> str:
        """The project's name: its folder's, without the suffix."""
        return service.project_name(self.folder)

    def drawn(self, key: str) -> DrawnGraph | None:
        """Return the view kept under a key, or ``None``."""
        with self.lock:
            return next(
                (item for item in self.graphs if item.key == key), None
            )


class ProjectStore:
    """The session's projects, newest first; one per folder."""

    def __init__(self) -> None:
        """Start without projects."""
        self.lock = threading.Lock()
        self._projects: dict[str, OpenProject] = {}

    def add(self, folder: Path) -> OpenProject:
        """Return the project of a folder, added if it is new.

        Args:
            folder: The project folder; it need not exist yet.

        Returns:
            The project, now the newest.
        """
        folder = folder.resolve()
        with self.lock:
            for key, project in list(self._projects.items()):
                if project.folder == folder:
                    # The newest come first.
                    del self._projects[key]
                    self._projects[key] = project
                    return project
            project = OpenProject(secrets.token_hex(6), folder)
            self._projects[project.id] = project
            return project

    def get(self, project_id: str) -> OpenProject | None:
        """Return a project by its id, or ``None``."""
        with self.lock:
            return self._projects.get(project_id)

    def all(self) -> list[OpenProject]:
        """Return the projects, the newest first."""
        with self.lock:
            return list(self._projects.values())[::-1]


_STORE_LOCK = threading.Lock()


def store_of(request: Request) -> ProjectStore:
    """Return the session's projects, created on first use."""
    state = request.app.state
    with _STORE_LOCK:
        store = getattr(state, "kg_projects", None)
        if store is None:
            store = state.kg_projects = ProjectStore()
    return store


def project_or_404(request: Request, project_id: str) -> OpenProject:
    """Return a project of the session, or answer 404."""
    project = None
    if _PROJECT_ID.fullmatch(project_id):
        project = store_of(request).get(project_id)
    if project is None:
        raise HTTPException(
            status_code=404,
            detail="This project is not open in this session. Open it again "
            "on the Knowledge Graph page.",
        )
    return project


def stream_page(project: OpenProject) -> str:
    """Return the page id of a project's page and of its jobs."""
    return f"{PAGE}:{project.id}"


def project_jobs(runner: JobRunner, project: OpenProject) -> list[Job]:
    """Return the jobs that work on a project, oldest first."""
    return [
        job
        for job in runner.jobs()
        if job.page_data.get("project") == project.id
    ]


# ---------------------------------------------------------------------------
# Routes: starting and opening projects
# ---------------------------------------------------------------------------


@router.get(PATH)
def page(request: Request) -> HTMLResponse:
    """Show the forms to start or open a project, and the builds.

    Args:
        request: The current request.

    Returns:
        The page.
    """
    jobs = runner_of(request).jobs(PAGE)
    return renderer(request).response(
        request,
        "knowledge_graph/page.html",
        page_id=PAGE,
        projects=store_of(request).all(),
        **form_context(),
        cards=Markup("".join(render_card(request, job) for job in jobs[::-1])),
    )


@router.get(PATH + "/papers-folder")
def papers_folder(request: Request, papers: str = "") -> HTMLResponse:
    """Say how many PDFs a folder on the cluster holds, as it is typed.

    Args:
        request: The current request.
        papers: The folder, as typed.

    Returns:
        A line about the folder; empty for an empty field.
    """
    problem, count = "", 0
    if papers.strip():
        try:
            count = len(service.list_pdfs(cluster_folder(papers, "folder")))
        except uploads.UploadError as error:
            problem = str(error)
    return renderer(request).response(
        request,
        "knowledge_graph/folder_check.html",
        asked=bool(papers.strip()),
        problem=problem,
        count=count,
    )


@router.post(PATH + "/build")
async def build_project(request: Request, session: Checked) -> HTMLResponse:
    """Start building a project, from uploaded papers or a folder.

    Args:
        request: The form: ``source`` (``upload`` or ``folder``), ``file``
            or ``papers``, ``name``, ``save_in`` and the topic model.
        session: The session, with its CSRF token checked.

    Returns:
        The build job's card.

    Raises:
        UploadError: If a choice is missing or not valid, or the model
            server does not answer.
    """
    form = await uploads.read_form(request, max_files=1, max_fields=20)
    topic_model = read_topic_model(form, service.model_options())
    upload, papers = read_papers(form)
    corpus = await run_in_threadpool(project_folder, form, papers)
    await check_server(topic_model)
    project = store_of(request).add(corpus)
    return await start_build(
        request, project, upload, papers, topic_model, page=PAGE
    )


@router.get(PATH + "/projects-in")
def projects_in(request: Request, location: str = "") -> HTMLResponse:
    """List the saved projects in a folder on the cluster.

    Args:
        request: The current request.
        location: A project folder, or a folder that holds projects.

    Returns:
        The projects found, each with a button that opens it, or why there
        are none.
    """
    problem, found = "", []
    try:
        folder = cluster_folder(location, "folder")
    except uploads.UploadError as error:
        problem = str(error)
    else:
        candidates = (
            [folder]
            if service.is_project(folder)
            else service.find_corpora(folder)
        )
        found = [
            (
                path,
                (path / service.TABLE_JSONL).is_file(),
                (path / service.TOPICS_JSONL).is_file(),
            )
            for path in candidates
        ]
    return renderer(request).response(
        request,
        "knowledge_graph/found.html",
        problem=problem,
        found=found,
        project_name=service.project_name,
    )


@router.post(PATH + "/open")
async def open_project(request: Request, session: Checked) -> Response:
    """Open a saved project: go to its page.

    Args:
        request: The form: ``folder``, the project folder.
        session: The session, with its CSRF token checked.

    Returns:
        A redirect to the project's page.

    Raises:
        UploadError: If the folder is not a project.
    """
    form = await uploads.read_form(request, max_files=0, max_fields=4)
    folder = cluster_folder(form.get("folder"), "project folder")
    if not service.is_project(folder):
        raise uploads.UploadError(
            f"{folder} is not a Knowledge Graph project: its name ends in "
            f"{service.CORPUS_SUFFIX}, or it holds {service.TABLE_JSONL}."
        )
    project = store_of(request).add(folder)
    return go_to(request, f"{PATH}/{project.id}")


# ---------------------------------------------------------------------------
# Routes: a project
# ---------------------------------------------------------------------------


@router.get(PATH + "/{project_id}")
def project_page(request: Request, project_id: str) -> HTMLResponse:
    """Show a project: its graph, papers and topics, and its tools.

    Args:
        request: The current request.
        project_id: The project's id.

    Returns:
        The page.
    """
    project = project_or_404(request, project_id)
    data = service.read_project(project.folder)
    drawn, problem = None, ""
    if data.records:
        try:
            drawn = graph_of(project, data, service.GraphOptions())
        except ValueError as error:
            problem = str(error)
    runner = runner_of(request)
    jobs = runner.jobs(stream_page(project))
    return renderer(request).response(
        request,
        "knowledge_graph/project.html",
        page_id=PAGE,
        stream_page=stream_page(project),
        project=project,
        data=data,
        drawn=drawn,
        problem=problem,
        defaults=service.GraphOptions(),
        **form_context(),
        cards=Markup("".join(render_card(request, job) for job in jobs[::-1])),
    )


@router.post(PATH + "/{project_id}/graph")
async def graph(
    request: Request, project_id: str, session: Checked
) -> HTMLResponse:
    """Draw the graph with the options chosen.

    Args:
        request: The form: ``view`` (``collection`` or ``paper``),
            ``paper``, the node types to ``show``, ``confidence`` and the
            ``papers`` shown.
        project_id: The project's id.
        session: The session, with its CSRF token checked.

    Returns:
        The graph's panel, or why there is no graph.
    """
    project = project_or_404(request, project_id)
    form = await uploads.read_form(request, max_files=0, max_fields=MAX_FIELDS)
    data = await run_in_threadpool(service.read_project, project.folder)
    drawn, problem = None, ""
    try:
        options = graph_options(form, data)
        drawn = await run_in_threadpool(graph_of, project, data, options)
    except ValueError as error:
        problem = str(error)
    return renderer(request).response(
        request,
        "knowledge_graph/graph.html",
        project=project,
        drawn=drawn,
        problem=problem,
    )


@router.get(PATH + "/{project_id}/graphs/{key}.json")
def graph_data(request: Request, project_id: str, key: str) -> JSONResponse:
    """Send a view's nodes, edges and options, for vis-network.

    Args:
        request: The current request.
        project_id: The project's id.
        key: The view's key.

    Returns:
        The view's data.
    """
    return JSONResponse(drawn_or_404(request, project_id, key).view.data)


@router.get(PATH + "/{project_id}/graphs/{key}.html")
def graph_page(request: Request, project_id: str, key: str) -> Response:
    """Send a view as a page that works offline.

    Args:
        request: The current request.
        project_id: The project's id.
        key: The view's key.

    Returns:
        The page, as an attachment.
    """
    view = drawn_or_404(request, project_id, key).view
    return Response(
        view.html(),
        media_type="text/html",
        headers={"Content-Disposition": content_disposition(view.filename)},
    )


@router.get(PATH + "/{project_id}/papers.csv")
def papers_table(request: Request, project_id: str) -> Response:
    """Send the project's paper table.

    Args:
        request: The current request.
        project_id: The project's id.

    Returns:
        The table, as an attachment.
    """
    project = project_or_404(request, project_id)
    path = project.folder / service.TABLE_CSV
    if not path.is_file():
        raise HTTPException(
            status_code=404, detail="The project has no papers yet."
        )
    return FileResponse(
        path,
        media_type="text/csv",
        headers={
            "Content-Disposition": content_disposition(
                f"{project.name}_papers.csv"
            )
        },
    )


@router.get(PATH + "/{project_id}/topics.json")
def topics_table(request: Request, project_id: str) -> Response:
    """Send the papers with their topics, as one JSON document.

    Args:
        request: The current request.
        project_id: The project's id.

    Returns:
        The document, as an attachment.
    """
    project = project_or_404(request, project_id)
    if not (project.folder / service.TOPICS_JSONL).is_file():
        raise HTTPException(
            status_code=404, detail="The project has no topics yet."
        )
    return Response(
        service.topics_json(project.folder),
        media_type="application/json",
        headers={
            "Content-Disposition": content_disposition(
                f"{project.name}_papers_with_topics.json"
            )
        },
    )


@router.post(PATH + "/{project_id}/update")
async def update_project(
    request: Request, project_id: str, session: Checked
) -> HTMLResponse:
    """Add papers to a project, or read the ones that failed again.

    Args:
        request: The form: ``source``, ``file`` or ``papers``, and the
            topic model.
        project_id: The project's id.
        session: The session, with its CSRF token checked.

    Returns:
        The job's card.
    """
    project = project_or_404(request, project_id)
    form = await uploads.read_form(request, max_files=1, max_fields=20)
    topic_model = read_topic_model(form, service.model_options())
    upload, papers = read_papers(form)
    await check_server(topic_model)
    return await start_build(
        request,
        project,
        upload,
        papers,
        topic_model,
        page=stream_page(project),
        title=f"{project.name}: new papers",
    )


@router.post(PATH + "/{project_id}/topics")
async def topics_again(
    request: Request, project_id: str, session: Checked
) -> HTMLResponse:
    """Extract every paper's topics again, with the model chosen.

    Args:
        request: The form: the topic model.
        project_id: The project's id.
        session: The session, with its CSRF token checked.

    Returns:
        The job's card.

    Raises:
        UploadError: If the project has no papers yet.
    """
    project = project_or_404(request, project_id)
    form = await uploads.read_form(request, max_files=0, max_fields=20)
    topic_model = read_topic_model(form, service.model_options())
    if not (project.folder / service.TABLE_JSONL).is_file():
        raise uploads.UploadError(
            "The project has no papers yet: add them first."
        )
    await check_server(topic_model)
    runner = runner_of(request)
    job = create_job(
        runner,
        project,
        topic_model,
        title=f"{project.name}: topics",
        page=stream_page(project),
        result_template=TOPICS_TEMPLATE,
        cancelled=CANCELLED_TOPICS,
    )
    runner.submit(
        job, partial(extract, project.id, project.folder, topic_model)
    )
    return HTMLResponse(render_card(request, job))


@router.post(PATH + "/{project_id}/table")
def rebuild_table(
    request: Request, project_id: str, session: Checked
) -> Response:
    """Rebuild the paper table from the papers read, then show the project.

    Args:
        request: The current request.
        project_id: The project's id.
        session: The session, with its CSRF token checked.

    Returns:
        A redirect to the project's page.

    Raises:
        UploadError: If a job works on the project now.
    """
    project = project_or_404(request, project_id)
    if any(job.active for job in project_jobs(runner_of(request), project)):
        raise uploads.UploadError(
            "A job works on this project now: rebuild the table when it "
            "has finished."
        )
    service.build_corpus_table(project.folder)
    return go_to(request, f"{PATH}/{project.id}")


# ---------------------------------------------------------------------------
# Forms
# ---------------------------------------------------------------------------


def form_context() -> dict[str, object]:
    """Return what the forms that start jobs show."""
    return {
        "home": str(Path.home()),
        "accept": ",".join(EXTENSIONS),
        "formats": uploads.format_names(EXTENSIONS),
        "options": service.model_options(),
        "ready": service.server_ready(),
        "gpustack": service.gpustack_ready(),
        "gpustack_model": service.GPUSTACK_MODEL,
        "grobid": service.grobid_problem(),
    }


def cluster_folder(
    value: object, what: str, *, writable: bool = False
) -> Path:
    """Check a folder on the cluster that a form names.

    The server runs as the session's owner, so it reads and writes what
    they can.

    Args:
        value: The path as typed; ``~`` stands for the home folder.
        what: What the folder is, for messages: ``"papers folder"``.
        writable: Whether a project is to be saved in it.

    Returns:
        The folder.

    Raises:
        UploadError: If the path is empty or not absolute, names no
            folder, or one that cannot be read (or written).
    """
    text = str(value or "").strip()
    if not text:
        raise uploads.UploadError(f"Give the full path of the {what}.")
    path = Path(text).expanduser()
    if not path.is_absolute():
        raise uploads.UploadError(
            f"Give the full path of the {what}, starting with / or ~."
        )
    if not path.is_dir():
        raise uploads.UploadError(f"There is no folder {text} on the cluster.")
    access = os.R_OK | os.X_OK | (os.W_OK if writable else 0)
    if not os.access(path, access):
        verb = "save into" if writable else "read"
        raise uploads.UploadError(f"You cannot {verb} {text}.")
    return path


def read_papers(form: FormData) -> tuple[UploadFile | None, Path | None]:
    """Read where the papers come from: an upload, or a folder.

    Args:
        form: The form: ``source``, and ``file`` or ``papers``.

    Returns:
        ``(upload, None)`` or ``(None, folder)``.

    Raises:
        UploadError: If the upload or the folder is missing or not valid.
    """
    if str(form.get("source") or "upload") == "folder":
        folder = cluster_folder(form.get("papers"), "papers folder")
        if not service.list_pdfs(folder):
            raise uploads.UploadError(
                f"There are no PDFs in {folder}: only the PDFs directly in "
                "the folder are read, not those in its subfolders."
            )
        return None, folder
    return uploads.single_file(form, "file", suffixes=EXTENSIONS), None


def project_folder(form: FormData, papers: Path | None) -> Path:
    """Return the folder of a new project, from its name and location.

    Uploads need a name; a papers folder names the project after itself
    unless another name is given. Projects are saved where the form says,
    else next to the papers folder, or in the home folder for uploads.

    Args:
        form: The form: ``name`` and ``save_in``.
        papers: The papers folder, or ``None`` for an upload.

    Returns:
        The project folder; it may exist, and is then updated.

    Raises:
        UploadError: If the name or the location is not valid.
    """
    name = str(form.get("name") or "").strip()
    if not name and papers is None:
        raise uploads.UploadError(
            "Name the project: its folder is named after it."
        )
    try:
        folder_name = (
            service.project_folder_name(name)
            if name
            else service.corpus_folder_name(papers)
        )
    except ValueError as error:
        raise uploads.UploadError(str(error)) from error
    place = str(form.get("save_in") or "").strip()
    if not place:
        place = str(papers.parent if papers is not None else Path.home())
    corpus = (
        cluster_folder(place, "folder to save in", writable=True) / folder_name
    )
    if corpus.exists() and not corpus.is_dir():
        raise uploads.UploadError(
            f"{corpus} is a file: give the project another name."
        )
    return corpus


def read_topic_model(
    form: FormData, options: service.ModelOptions
) -> service.TopicModel:
    """Read the model that extracts the topics from the form.

    Args:
        form: The form: ``topics_from`` (``session`` or ``gpustack``),
            ``model`` or ``api_key``.
        options: The session's models.

    Returns:
        The model.

    Raises:
        UploadError: If the model is not offered, or GPUStack is not set
            up or has no API key.
    """
    if str(form.get("topics_from") or "session") == "gpustack":
        if not service.gpustack_ready():
            raise uploads.UploadError(
                "The University's GPUStack service is not set up on this "
                "site: use a model of this session."
            )
        key = str(form.get("api_key") or "").strip()
        if not key:
            raise uploads.UploadError(
                "Give your GPUStack API key, or use a model of this session."
            )
        return service.TopicModel(service.GPUSTACK_MODEL, api_key=key)
    if not options.models:
        raise uploads.UploadError(
            "No language models are set up for this session. Please tell "
            "support, or use the University's GPUStack service."
        )
    model = str(form.get("model") or options.models[0])
    if model not in options.models:
        raise uploads.UploadError("Choose one of the models offered.")
    return service.TopicModel(model)


async def check_server(topic_model: service.TopicModel) -> None:
    """Refuse a job for the session's model while its server is silent."""
    if topic_model.remote:
        return
    if not await run_in_threadpool(service.server_ready, SERVER_ATTEMPTS):
        raise uploads.UploadError(
            "The language model server (Ollama) does not answer. Try again "
            "in a minute; if it keeps failing, send ollama.log from the "
            "session's folder to support."
        )


def graph_options(
    form: FormData, data: service.Project
) -> service.GraphOptions:
    """Read what the graph shows from its form.

    Args:
        form: The form: ``view``, ``paper``, ``show``, ``confidence``
            and ``papers``.
        data: The project.

    Returns:
        The options.

    Raises:
        ValueError: If a paper is unknown, the confidence is not a number
            from 0 to 1, or no paper is ticked.
    """
    shown = {str(value) for value in form.getlist("show")}
    kinds = {
        "authors": "authors" in shown,
        "topics": "topics" in shown,
        "cited_papers": "cited_papers" in shown,
        "cited_authors": "cited_authors" in shown,
    }
    ids = [str(record.get("paper_id")) for record in data.records]
    if str(form.get("view") or "collection") == "paper":
        paper_id = str(form.get("paper") or "")
        if paper_id not in ids:
            raise ValueError("Choose one of the project's papers.")
        return service.GraphOptions(paper_id=paper_id, **kinds)
    try:
        confidence = float(str(form.get("confidence") or "0"))
    except ValueError:
        confidence = -1.0
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("The confidence is a number from 0 to 1.")
    ticked = {str(value) for value in form.getlist("papers")}
    if not ticked:
        raise ValueError("Tick at least one paper to show.")
    if not ticked <= set(ids):
        raise ValueError("Choose the papers from the project's list.")
    papers = () if ticked == set(ids) else tuple(i for i in ids if i in ticked)
    return service.GraphOptions(
        min_confidence=confidence, papers=papers, **kinds
    )


# ---------------------------------------------------------------------------
# Jobs and views
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Built:
    """A finished build or update of a project, as its card shows it.

    Attributes:
        project_id: The project's id in the session.
        summary: The counts.
    """

    project_id: str
    summary: service.BuildSummary


@dataclasses.dataclass(frozen=True)
class Extracted:
    """Topics extracted again, as the card shows it.

    Attributes:
        project_id: The project's id in the session.
        summary: The counts.
    """

    project_id: str
    summary: service.TopicSummary


def create_job(
    runner: JobRunner,
    project: OpenProject,
    topic_model: service.TopicModel,
    *,
    title: str,
    page: str,
    result_template: str,
    cancelled: str,
) -> Job:
    """Create a job that works on a project.

    A job with the session's model takes the GPU for it; one with
    GPUStack's does not use the GPU.
    """
    local = not topic_model.remote
    return runner.create(
        title=title,
        page=page,
        gpu=gpu_manager.LLM if local else None,
        ollama_model=topic_model.model if local else None,
        result_template=result_template,
        page_data={"project": project.id, "cancelled": cancelled},
    )


async def start_build(
    request: Request,
    project: OpenProject,
    upload: UploadFile | None,
    papers: Path | None,
    topic_model: service.TopicModel,
    *,
    page: str,
    title: str = "",
) -> HTMLResponse:
    """Create and submit a job that reads papers into a project.

    Returns:
        The job's card.
    """
    runner = runner_of(request)
    job = create_job(
        runner,
        project,
        topic_model,
        title=title or project.name,
        page=page,
        result_template=BUILT_TEMPLATE,
        cancelled=CANCELLED_BUILD,
    )
    path = None
    if upload is not None:
        try:
            path = await run_in_threadpool(uploads.save, upload, job.inputs)
        except BaseException:
            runner.discard(job)
            raise
    runner.submit(
        job,
        partial(build, project.id, project.folder, path, papers, topic_model),
    )
    return HTMLResponse(render_card(request, job))


def build(
    project_id: str,
    corpus: Path,
    upload: Path | None,
    papers: Path | None,
    topic_model: service.TopicModel,
    context: JobContext,
) -> Built:
    """Read papers into a project and extract the topics; a job's work.

    Args:
        project_id: The project's id in the session.
        corpus: The project folder.
        upload: The uploaded PDF or ZIP, in the job's inputs, or ``None``.
        papers: The papers folder on the cluster, without an upload.
        topic_model: The model for the topics.
        context: The job's context.

    Returns:
        The counts.
    """
    if upload is not None:
        context.report(Progress("Unpacking the papers"))
        pdfs = service.stage_papers(upload, context.inputs / "papers")
    else:
        pdfs = service.list_pdfs(papers)
    summary = service.build_project(
        pdfs,
        corpus,
        topic_model,
        on_progress=context.report,
        cancel=context.cancel,
    )
    return Built(project_id, summary)


def extract(
    project_id: str,
    corpus: Path,
    topic_model: service.TopicModel,
    context: JobContext,
) -> Extracted:
    """Extract every paper's topics again; a job's work."""
    summary = service.extract_topics_with(
        corpus, topic_model, on_progress=context.report, cancel=context.cancel
    )
    return Extracted(project_id, summary)


def topics_stamp(folder: Path) -> tuple[int, int]:
    """Return the topics file's time and size, or zeros without one."""
    try:
        stat = (folder / service.TOPICS_JSONL).stat()
    except OSError:
        return 0, 0
    return stat.st_mtime_ns, stat.st_size


def graph_of(
    project: OpenProject, data: service.Project, options: service.GraphOptions
) -> DrawnGraph:
    """Return a view of a project's graph: one kept, or drawn now.

    Raises:
        ValueError: If the options name no paper to show.
    """
    stamp = topics_stamp(project.folder)
    with project.lock:
        for drawn in project.graphs:
            if drawn.options == options and drawn.stamp == stamp:
                return drawn
    drawn = DrawnGraph(
        secrets.token_hex(6),
        options,
        stamp,
        service.draw_graph(data.records, options),
    )
    with project.lock:
        project.graphs.append(drawn)
        del project.graphs[:-GRAPHS_KEPT]
    return drawn


def drawn_or_404(request: Request, project_id: str, key: str) -> DrawnGraph:
    """Return a kept view of a project's graph, or answer 404."""
    drawn = project_or_404(request, project_id).drawn(key)
    if drawn is None:
        raise HTTPException(
            status_code=404,
            detail="This view of the graph is no longer kept. Reload the "
            "page.",
        )
    return drawn


def go_to(request: Request, path: str) -> Response:
    """Send the browser to a page: by htmx's header, or a redirect."""
    url = f"{root(request)}{path}"
    if is_htmx(request):
        return Response(status_code=204, headers={"HX-Redirect": url})
    return RedirectResponse(url, status_code=303)
