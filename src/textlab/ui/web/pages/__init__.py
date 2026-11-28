"""The web UI's pages: the navigation, and one module per page.

Each page module has a ``router``; :func:`routers` collects them. Rules for
page modules, checked by ``tests/test_rules.py``:

- From the features, import only their ``service`` modules, and from those
  only the names they export in ``__all__``.
- Keep processing out of routes and templates: what a page needs beyond
  calling the service goes into the service.
"""

from __future__ import annotations

import dataclasses

from fastapi import APIRouter


@dataclasses.dataclass(frozen=True)
class Tool:
    """A tool in the navigation and on the Home page.

    Attributes:
        id: The page id, which also names the page's jobs.
        label: The name in the navigation.
        path: The page's path below the base path.
        group: What the tool works on, which groups the navigation:
            ``"Audio"``, ``"Documents"``, ``"Text"`` or ``"Data"``.
        summary: What the tool does, in one sentence.
    """

    id: str
    label: str
    path: str
    group: str
    summary: str

    @property
    def number(self) -> str:
        """The tool's number in the navigation, such as ``"01"``."""
        return f"{TOOLS.index(self) + 1:02d}"


#: The tools, in the order of the navigation, grouped by what they work on.
TOOLS: tuple[Tool, ...] = (
    Tool(
        "transcription",
        "Transcription",
        "/transcription",
        "Audio",
        "Turn recordings into text, with timestamps and the speakers told "
        "apart.",
    ),
    Tool(
        "meeting_notes",
        "Meeting Notes",
        "/meeting-notes",
        "Audio",
        "Transcribe a meeting and get a summary that cites the recording.",
    ),
    Tool(
        "ocr",
        "OCR",
        "/ocr",
        "Documents",
        "Read the text in scans, photos and PDFs, historic handwriting "
        "included.",
    ),
    Tool(
        "knowledge_graph",
        "Knowledge Graph",
        "/knowledge-graph",
        "Documents",
        "Map the topics of a set of research papers and how they connect.",
    ),
    Tool(
        "translation",
        "Translation",
        "/translation",
        "Text",
        "Translate text and documents between more than 200 languages.",
    ),
    Tool(
        "chat",
        "Chat",
        "/chat",
        "Text",
        "Ask a language model questions, about your own documents too.",
    ),
    Tool(
        "topic_modeling",
        "Topic Modeling",
        "/topic-modeling",
        "Text",
        "Find the themes that run through a large set of texts.",
    ),
    Tool(
        "visualize_data",
        "Visualize Data",
        "/visualize-data",
        "Data",
        "Describe the chart you need and get it drawn from your data.",
    ),
)

#: Paths of pages that are not tools, by page id.
OTHER_PAGES = {"home": "/"}


def tool_groups() -> list[tuple[str, list[Tool]]]:
    """Return the tools by group, in the order of :data:`TOOLS`.

    Returns:
        ``(group, tools)`` pairs.
    """
    groups: dict[str, list[Tool]] = {}
    for tool in TOOLS:
        groups.setdefault(tool.group, []).append(tool)
    return list(groups.items())


def page_path(page_id: str) -> str:
    """Return the path of a page below the base path.

    A tool's page of one item has the id ``<tool>:<key>`` and the path
    ``<tool's path>/<key>``, such as a Knowledge Graph project's page.

    Args:
        page_id: The page's id, such as ``"transcription"`` or
            ``"knowledge_graph:3f2a9c"``.

    Returns:
        The path; ``"/"`` for an unknown id.
    """
    tool_id, _, key = page_id.partition(":")
    for tool in TOOLS:
        if tool.id == tool_id:
            return f"{tool.path}/{key}" if key else tool.path
    return OTHER_PAGES.get(page_id, "/")


def page_label(page_id: str) -> str:
    """Return the name of a page, for lists of jobs.

    Args:
        page_id: The page's id; a tool's page of one item is named after
            the tool.

    Returns:
        The tool's label, or the id itself for other pages.
    """
    tool_id = page_id.partition(":")[0]
    for tool in TOOLS:
        if tool.id == tool_id:
            return tool.label
    return page_id.replace("_", " ").capitalize()


def routers() -> list[APIRouter]:
    """Return the routers of all pages.

    Returns:
        The routers.
    """
    from textlab.ui.web.pages import (
        chat,
        home,
        knowledge_graph,
        meeting_notes,
        ocr,
        ocr_handwriting,
        ocr_huggingface,
        topic_modeling,
        transcription,
        translation,
        visualize_data,
    )

    return [
        home.router,
        transcription.router,
        meeting_notes.router,
        ocr.router,
        ocr_handwriting.router,
        ocr_huggingface.router,
        knowledge_graph.router,
        translation.router,
        chat.router,
        topic_modeling.router,
        visualize_data.router,
    ]
