"""Graphs of a paper collection: the Knowledge Graph feature's API.

Interfaces call this module. A project is a corpus folder that the steps
read and write, so work can resume in a later session:
:func:`build_project` reads the PDFs of a folder with Grobid and extracts
each paper's topics with an LLM (or, step by step, :func:`build_corpus`
and :func:`extract_topics`); :func:`read_project` reads a project back;
:func:`draw_graph` gives a view of it to draw (or, for the Streamlit page,
:func:`build_paper_ego_graph` and :func:`build_full_corpus_graph`).
Uploaded PDFs and ZIPs are staged with :func:`stage_papers`.

The work happens outside the app process (Grobid and the model run in
servers of their own), so the steps run in the caller's process. NetworkX,
Pyvis and the OpenAI client are imported only when they are used.
"""

from __future__ import annotations

from textlab.common.gpu_manager import get_gpu_name
from textlab.common.model_config import (
    ModelOptions,
    get_available_models,
    model_options,
)
from textlab.common.ollama import check_ollama_server
from textlab.features.knowledge_graph.corpus import (
    CORPUS_SUFFIX,
    TABLE_CSV,
    TABLE_JSONL,
    TOPICS_JSONL,
    build_corpus,
    build_corpus_table,
    corpus_folder_name,
    find_corpora,
    list_pdfs,
    read_records,
    topics_json,
)
from textlab.features.knowledge_graph.graphs import (
    GraphOptions,
    GraphView,
    build_full_corpus_graph,
    build_paper_ego_graph,
    count_nodes,
    draw_graph,
    save_graph_html,
)
from textlab.features.knowledge_graph.grobid import (
    GrobidError,
    ensure_grobid_server,
    grobid_problem,
)
from textlab.features.knowledge_graph.models import CorpusSummary, TopicSummary
from textlab.features.knowledge_graph.projects import (
    BuildSummary,
    Project,
    TopicModel,
    build_project,
    extract_topics_with,
    gpustack_ready,
    is_project,
    project_folder_name,
    project_name,
    read_project,
)
from textlab.features.knowledge_graph.staging import (
    stage_papers,
    staged_papers,
)
from textlab.features.knowledge_graph.topics import (
    GPUSTACK_MODEL,
    extract_topics,
    gpustack_client,
    ollama_client,
)

__all__ = [
    "BuildSummary",
    "CorpusSummary",
    "GraphOptions",
    "GraphView",
    "GrobidError",
    "ModelOptions",
    "Project",
    "TopicModel",
    "TopicSummary",
    "build_corpus",
    "build_corpus_table",
    "build_full_corpus_graph",
    "build_paper_ego_graph",
    "build_project",
    "corpus_folder_name",
    "count_nodes",
    "draw_graph",
    "ensure_grobid_server",
    "extract_topics",
    "extract_topics_with",
    "find_corpora",
    "gpustack_client",
    "gpustack_ready",
    "grobid_problem",
    "is_project",
    "list_pdfs",
    "model_options",
    "ollama_client",
    "ollama_models",
    "project_folder_name",
    "project_name",
    "read_project",
    "read_records",
    "save_graph_html",
    "server_ready",
    "stage_papers",
    "staged_papers",
    "topics_json",
    # Re-exported for interfaces
    "CORPUS_SUFFIX",
    "GPUSTACK_MODEL",
    "TABLE_CSV",
    "TABLE_JSONL",
    "TOPICS_JSONL",
]


def ollama_models() -> list[str]:
    """Return the configured Ollama model choices for the session's GPU.

    Uses the shared models.json configuration and models.local.json
    overrides, following the same GPU-based selection as the other pages.
    Installed Ollama models do not define the choices offered here.

    Returns:
        Model names in their configured order for this session's GPU.
    """
    return get_available_models(get_gpu_name())


def server_ready(attempts: int = 1) -> bool:
    """Return whether the session's Ollama server accepts connections.

    Args:
        attempts: How many times to try, half a second apart; the server
            may still be starting when a session begins.

    Returns:
        True if it does.
    """
    return check_ollama_server(attempts=attempts)
