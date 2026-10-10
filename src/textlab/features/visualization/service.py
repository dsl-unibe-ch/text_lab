"""Analyzing a table with LLM agents: the Visualization feature's API.

Interfaces call this module. :func:`read_preview` and
:func:`column_profile` give a first look at an upload;
:func:`start_analysis` runs the agents in the background and returns an
:class:`AnalysisRun` to poll; :func:`results_zip` packages the result. Chat
uses the same functions for its data questions.

The agents (:mod:`.viz_agent`) and their tools (:mod:`.mcp_server`) are
imported only when a run starts.
"""

from __future__ import annotations

import os

from textlab.features.visualization.models import (
    AnalysisRequest,
    AnalysisResult,
    Artifact,
)
from textlab.features.visualization.preview import (
    column_profile,
    read_preview,
)
from textlab.features.visualization.reports import (
    R_NOT_AVAILABLE,
    build_html_report,
    figure,
    results_zip,
)
from textlab.features.visualization.runs import (
    ANALYSIS_TIMEOUT_SECONDS,
    CANCELLED,
    DONE,
    ERROR,
    RUNNING,
    TIMEOUT,
    AnalysisRun,
    start_analysis,
)
from textlab.features.visualization.viz_config import (
    DEFAULT_PROMPT,
    MAX_ROWS,
)
from textlab.features.visualization.viz_config import (
    get_tool_label as tool_label,
)
from textlab.features.visualization.viz_utils import save_data_file

__all__ = [
    "INPUT_TYPES",
    "AnalysisRequest",
    "AnalysisResult",
    "AnalysisRun",
    "Artifact",
    "build_html_report",
    "column_profile",
    "figure",
    "is_table",
    "prepare_data_file",
    "read_preview",
    "results_zip",
    "start_analysis",
    "tool_label",
    # Re-exported for interfaces
    "ANALYSIS_TIMEOUT_SECONDS",
    "CANCELLED",
    "DEFAULT_PROMPT",
    "DONE",
    "ERROR",
    "MAX_ROWS",
    "RUNNING",
    "R_NOT_AVAILABLE",
    "TIMEOUT",
]

#: File types the agents read.
INPUT_TYPES = ("csv", "tsv", "xls", "xlsx", "json")


def prepare_data_file(name: str, data: bytes, folder: str) -> tuple[str, str]:
    """Save an uploaded table for repeated analyses, and summarize it.

    Args:
        name: The uploaded file's name.
        data: Its content.
        folder: Where to save it, such as a conversation's folder in the
            workspace.

    Returns:
        The saved file's path, and a summary of its columns for a model's
        prompt (or a note that it could not be summarized).
    """
    from textlab.features.visualization.plot_data import (
        get_all_columns_summary_impl,
    )

    path = save_data_file(data, name, folder)
    try:
        summary = get_all_columns_summary_impl(path)
    except Exception as exc:
        summary = f"[Could not summarise dataset: {exc}]"
    return path, summary


def is_table(name: str) -> bool:
    """Return True if a file name has one of :data:`INPUT_TYPES`."""
    return os.path.splitext(name)[1].lower().lstrip(".") in INPUT_TYPES
