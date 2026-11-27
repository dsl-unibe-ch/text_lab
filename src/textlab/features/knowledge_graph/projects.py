"""Projects: a corpus built with its topics in one go, and read back.

A project is a corpus folder (:mod:`.corpus`) in a place the user chooses,
named ``<name>_project_corpus``. :func:`build_project` runs what building
or updating one takes, in the caller's thread: it starts Grobid, reads the
PDFs not read yet, and extracts every paper's topics. :func:`read_project`
reads a project's papers and topics back for an interface.
"""

from __future__ import annotations

import dataclasses
import threading
from pathlib import Path
from typing import Any

from textlab.common.config import MissingSettingError, get_settings
from textlab.common.ollama import model_installed, pull_model
from textlab.common.progress import (
    CancelledError,
    ProgressCallback,
    no_progress,
)
from textlab.features.knowledge_graph.corpus import (
    CORPUS_SUFFIX,
    TABLE_JSONL,
    TOPICS_JSONL,
    build_corpus,
    read_records,
)
from textlab.features.knowledge_graph.grobid import ensure_grobid_server
from textlab.features.knowledge_graph.models import TopicSummary
from textlab.features.knowledge_graph.topics import (
    extract_topics,
    gpustack_client,
    ollama_client,
)


@dataclasses.dataclass(frozen=True)
class TopicModel:
    """The model that extracts the topics, and where it runs.

    Attributes:
        model: The model's name.
        api_key: The user's API key for the University's GPUStack service;
            empty for the session's own Ollama server.
    """

    model: str
    api_key: str = dataclasses.field(default="", repr=False)

    @property
    def remote(self) -> bool:
        """Whether the model runs on GPUStack, outside the session."""
        return bool(self.api_key)


@dataclasses.dataclass(frozen=True)
class BuildSummary:
    """The outcome of building or updating a project.

    Attributes:
        read: Papers read in this run.
        reused: Papers read before, which were kept.
        failed: Papers Grobid could not read; they are tried again next
            time.
        papers: Papers in the project now.
        topics: Papers given topics, if topic extraction ran to its end.
        without_abstract: Papers without an abstract, which get no topics.
        topic_error: Why topic extraction failed, if it did; the papers
            read are saved all the same.
    """

    read: int
    reused: int
    failed: int
    papers: int
    topics: int | None = None
    without_abstract: int = 0
    topic_error: str = ""


@dataclasses.dataclass(frozen=True)
class Project:
    """A project's papers and topics, as saved in its folder.

    Attributes:
        folder: The corpus folder.
        papers: The rows of its paper table (``corpus_table.jsonl``);
            empty until a build has read papers.
        records: The rows with their topics
            (``corpus_table.with_topics.jsonl``); empty until topics are
            extracted.
    """

    folder: Path
    papers: list[dict[str, Any]]
    records: list[dict[str, Any]]

    @property
    def name(self) -> str:
        """The project's name: its folder's, without the suffix."""
        return project_name(self.folder)

    @property
    def with_topics(self) -> int:
        """How many papers have topics."""
        return sum(1 for record in self.records if record.get("topics"))

    @property
    def topics_stale(self) -> bool:
        """Whether the topics are of another set of papers than the table.

        After papers are added, until topics are extracted again.
        """
        if not self.records:
            return False
        ids = {paper.get("paper_id") for paper in self.papers}
        return ids != {record.get("paper_id") for record in self.records}


def build_project(
    pdfs: list[Path],
    corpus: Path,
    topic_model: TopicModel,
    *,
    on_progress: ProgressCallback = no_progress,
    cancel: threading.Event | None = None,
) -> BuildSummary:
    """Build or update a project: read new PDFs, then extract the topics.

    PDFs read before (by name) are kept, and failed ones tried again. If
    topic extraction fails, the papers read stay saved, and the summary
    says why; the topics before stay too.

    Args:
        pdfs: The papers.
        corpus: The project folder, created if needed.
        topic_model: The model for the topics.
        on_progress: Receives each paper's step, and the model's download.
        cancel: Set to stop before the next paper.

    Returns:
        The counts.

    Raises:
        GrobidError: If Grobid cannot be started.
        ValueError: If the project has no paper after reading.
        CancelledError: If ``cancel`` was set; the papers read until then
            are kept.
    """
    ensure_grobid_server()
    summary = build_corpus(
        pdfs, corpus, on_progress=on_progress, cancel=cancel
    )
    papers = len(summary.table)
    if not papers:
        raise ValueError(
            "No paper could be read. Check that the files are PDFs with "
            "text; Grobid tries the ones it could not read again next time."
        )
    counts = {
        "read": summary.processed,
        "reused": summary.skipped,
        "failed": summary.errors,
        "papers": papers,
    }
    try:
        topics = extract_topics_with(
            corpus, topic_model, on_progress=on_progress, cancel=cancel
        )
    except CancelledError:
        raise
    except Exception as exc:
        return BuildSummary(
            **counts, topic_error=str(exc) or type(exc).__name__
        )
    return BuildSummary(
        **counts, topics=topics.processed, without_abstract=topics.skipped
    )


def extract_topics_with(
    corpus: Path,
    topic_model: TopicModel,
    *,
    on_progress: ProgressCallback = no_progress,
    cancel: threading.Event | None = None,
) -> TopicSummary:
    """Extract every paper's topics with a model, wherever it runs.

    The session's Ollama server downloads the model first if it does not
    have it.

    Args:
        corpus: The project folder, with its paper table.
        topic_model: The model.
        on_progress: Receives each paper's step, and the download.
        cancel: Set to stop before the next paper.

    Returns:
        The counts.

    Raises:
        FileNotFoundError: If the project has no paper table.
        RuntimeError: If the model cannot be reached.
        CancelledError: If ``cancel`` was set.
    """
    if topic_model.remote:
        client = gpustack_client(topic_model.api_key)
    else:
        if not model_installed(topic_model.model):
            pull_model(
                topic_model.model, on_progress=on_progress, cancel=cancel
            )
        client = ollama_client()
    return extract_topics(
        corpus,
        client,
        topic_model.model,
        on_progress=on_progress,
        cancel=cancel,
    )


def read_project(folder: Path) -> Project:
    """Read a project's papers and topics.

    Args:
        folder: The project folder.

    Returns:
        The project; without papers or topics while they are missing.
    """
    table, topics = folder / TABLE_JSONL, folder / TOPICS_JSONL
    return Project(
        folder=folder,
        papers=read_records(table) if table.is_file() else [],
        records=read_records(topics) if topics.is_file() else [],
    )


def is_project(folder: Path) -> bool:
    """Return whether a folder is a project: named so, or with a table."""
    return folder.is_dir() and (
        folder.name.endswith(CORPUS_SUFFIX) or (folder / TABLE_JSONL).is_file()
    )


def project_name(folder: Path) -> str:
    """Return a project's name: its folder's, without the suffix."""
    return folder.name.removesuffix(CORPUS_SUFFIX)


def project_folder_name(name: str) -> str:
    """Return a corpus folder name from a user-supplied project name.

    Args:
        name: A single folder name, without a path.

    Returns:
        The name with the corpus suffix appended.

    Raises:
        ValueError: If the name is empty or contains path components.
    """
    name = name.strip()
    if not name or name in {".", ".."} or any(c in name for c in "/\\\0"):
        raise ValueError("Enter a project name without slashes or a path.")
    return f"{name}{CORPUS_SUFFIX}"


def gpustack_ready() -> bool:
    """Return whether the site set up the University's GPUStack service."""
    try:
        get_settings().require("gpustack_url")
    except MissingSettingError:
        return False
    return True
