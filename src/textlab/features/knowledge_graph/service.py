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

from collections.abc import Iterator
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

from textlab.common.gpu_manager import get_gpu_name
from textlab.common.model_config import get_available_models
from textlab.common.storage import get_workspace
from textlab.common.upload_safety import safe_upload_name, safe_zip_members
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
    "project_folder_name",
    "staged_papers",
    "save_graph_html",
    "topics_json",
    # Re-exported for interfaces
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
    return f"{name}_project_corpus"


def _write_pdf(path: Path, data: bytes) -> None:
    """Write a PDF after checking its header."""
    if b"%PDF-" not in data[:1024]:
        raise ValueError(f"{path.name} does not contain a PDF header.")
    path.write_bytes(data)


@contextmanager
def staged_papers(name: str, data: bytes) -> Iterator[Path]:
    """Stage a PDF or ZIP in a private folder removed after processing.

    ZIP subfolders are flattened, preserving PDF basenames for corpus reuse.
    Duplicate basenames are refused to avoid overwriting or skipping papers.
    Archive paths and expanded sizes use the shared upload safeguards.

    Args:
        name: The browser-supplied filename.
        data: The uploaded PDF or ZIP bytes.

    Yields:
        A temporary folder containing the uploaded PDF papers.

    Raises:
        ValueError: If the input is invalid, unsafe, or contains no PDFs.
    """
    name = safe_upload_name(name)
    extension = Path(name).suffix.lower()
    if extension not in {".pdf", ".zip"}:
        raise ValueError("Upload a PDF document or ZIP archive of PDFs.")
    with get_workspace().temp_dir("knowledge_graph", prefix="upload-") as root:
        if extension == ".pdf":
            _write_pdf(root / name, data)
        else:
            try:
                with ZipFile(BytesIO(data)) as archive:
                    members = safe_zip_members(
                        archive, allowed_extensions={".pdf"}
                    )
                    seen: set[str] = set()
                    for member in members:
                        relative = PurePosixPath(
                            member.filename.replace("\\", "/")
                        )
                        if (
                            "__MACOSX" in relative.parts
                            or relative.name.startswith("._")
                        ):
                            continue
                        filename = safe_upload_name(member.filename)
                        if filename.casefold() in seen:
                            raise ValueError(
                                f"More than one PDF is named {filename}. "
                                "Give each paper a unique filename."
                            )
                        seen.add(filename.casefold())
                        _write_pdf(root / filename, archive.read(member))
                    if not seen:
                        raise ValueError(
                            "The ZIP archive contains no PDF papers."
                        )
            except BadZipFile as exc:
                raise ValueError(
                    "The uploaded ZIP archive is not valid."
                ) from exc
        yield root
