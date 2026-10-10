"""Graphs of a paper collection: the Knowledge Graph feature's API.

Interfaces call this module, in three steps: :func:`build_corpus` parses
the PDFs of a folder with Grobid into a corpus folder, :func:`extract_topics`
adds topics to each paper with an LLM, and :func:`build_paper_ego_graph` and
:func:`build_full_corpus_graph` draw the corpus. Every step reads and writes
the corpus folder, so it can be resumed in a later session.

The work happens outside the app process (Grobid and the model run in
servers of their own), so the steps run in the caller's process. NetworkX,
Pyvis and the OpenAI client are imported only when they are used.
"""

from __future__ import annotations

import ollama

from textlab.common.ollama import extract_model_name
from textlab.features.knowledge_graph.corpus import (
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
    build_full_corpus_graph,
    build_paper_ego_graph,
    count_nodes,
    save_graph_html,
)
from textlab.features.knowledge_graph.grobid import (
    GrobidError,
    ensure_grobid_server,
)
from textlab.features.knowledge_graph.models import CorpusSummary, TopicSummary
from textlab.features.knowledge_graph.topics import (
    GPUSTACK_MODEL,
    extract_topics,
    gpustack_client,
    ollama_client,
)

__all__ = [
    "CorpusSummary",
    "GrobidError",
    "TopicSummary",
    "build_corpus",
    "build_corpus_table",
    "build_full_corpus_graph",
    "build_paper_ego_graph",
    "corpus_folder_name",
    "count_nodes",
    "ensure_grobid_server",
    "extract_topics",
    "find_corpora",
    "gpustack_client",
    "list_pdfs",
    "ollama_client",
    "ollama_models",
    "read_records",
    "save_graph_html",
    "topics_json",
    # Re-exported for interfaces
    "GPUSTACK_MODEL",
    "TABLE_CSV",
    "TABLE_JSONL",
    "TOPICS_JSONL",
]


def ollama_models() -> list[str]:
    """Return the models on the session's Ollama server, as it lists them.

    Returns:
        The model names.

    Raises:
        Exception: If the server cannot be reached (the client's error).
    """
    response = ollama.list()
    if hasattr(response, "models"):
        entries = response.models
    else:
        entries = response.get("models", [])
    return [name for name in map(extract_model_name, entries) if name]
