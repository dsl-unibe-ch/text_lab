"""Requests and results of a data analysis.

The agent's own types (plans, tool reports, raw artifacts) are in
:mod:`.viz_config`; these are what interfaces send and receive.
"""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class AnalysisRequest:
    """What to analyze and how.

    Attributes:
        model: The Ollama model of the supervisor and its agents.
        prompt: The user's request; empty for the default exploratory
            analysis (``viz_config.DEFAULT_PROMPT``).
        columns: Columns to focus on; empty for all columns.
        include_r_code: Also give R code next to the Python code.
        file_name: The uploaded file's name, for the report.
    """

    model: str
    prompt: str = ""
    columns: tuple[str, ...] = ()
    include_r_code: bool = False
    file_name: str = ""


@dataclasses.dataclass
class Artifact:
    """One chart of an analysis.

    Attributes:
        filename: The chart's file name: ``.json`` for an interactive
            Plotly chart, an image name for a static one.
        data: The file's content.
        code: Python code that reproduces the chart.
        r_code: R code that reproduces it, if R code was requested and the
            chart has an R version.
        tool_name: The tool that drew it (see ``viz_config.get_tool_label``).
        figure_json: The Plotly figure as JSON, for interactive charts.
    """

    filename: str
    data: bytes
    code: str = ""
    r_code: str = ""
    tool_name: str = ""
    figure_json: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the chart as the dictionary chat messages store."""
        return {
            "filename": self.filename,
            "bytes": self.data,
            "code": self.code,
            "r_code": self.r_code,
            "tool_name": self.tool_name,
            "fig_json": self.figure_json,
        }


@dataclasses.dataclass
class AnalysisResult:
    """The outcome of a finished analysis.

    Attributes:
        run_id: Identifies the run in file names and the report.
        summary: The supervisor's summary, in Markdown.
        artifacts: The charts.
        stats: Statistical results: dictionaries with ``title``,
            ``result`` (Markdown), ``code`` and ``r_code``.
    """

    run_id: str
    summary: str
    artifacts: list[Artifact]
    stats: list[dict[str, Any]]

    def to_payload(self) -> dict[str, Any]:
        """Return the result as the ``analysis`` entry of a chat message."""
        return {
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "stats": self.stats,
            "run_id": self.run_id,
        }
