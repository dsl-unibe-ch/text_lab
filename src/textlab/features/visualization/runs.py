"""Running an analysis in the background, so a page stays responsive.

:func:`start_analysis` starts the agent (:func:`.viz_agent.run_analysis`)
in a daemon thread and returns an :class:`AnalysisRun` that an interface
polls: its status, its activity log and, when it is done, its result. The
heavy work happens outside this process anyway, in the Ollama server and in
the agent's MCP server process.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import threading
import uuid
from typing import Any

import ollama

from textlab.common.ollama import model_installed
from textlab.common.storage import get_workspace
from textlab.features.visualization.models import (
    AnalysisRequest,
    AnalysisResult,
    Artifact,
)
from textlab.features.visualization.viz_config import DEFAULT_PROMPT
from textlab.features.visualization.viz_utils import (
    get_fast_data_preview,
    save_data_file,
)

#: Seconds before a run is stopped and reported as timed out.
ANALYSIS_TIMEOUT_SECONDS = 600

#: Workspace area for uploaded tables and their charts.
WORKSPACE_AREA = "visualization"

RUNNING = "running"
DONE = "done"
CANCELLED = "cancelled"
TIMEOUT = "timeout"
ERROR = "error"


class AnalysisRun:
    """An analysis running in a background thread.

    Attributes:
        run_id: Identifies the run in file names and the report.
        status: One of ``"running"``, ``"done"``, ``"cancelled"``,
            ``"timeout"`` or ``"error"``.
        logs: The agents' activity, as ``(level, message)`` pairs; level is
            ``"info"``, ``"warning"`` or ``"error"``. Appended to while the
            run is going.
        result: The result, once the status is ``"done"``.
        error: What went wrong, once the status is ``"error"``.
    """

    def __init__(self, run_id: str):
        """Create a run that has not started yet."""
        self.run_id = run_id
        self.status = RUNNING
        self.logs: list[tuple[str, str]] = []
        self.result: AnalysisResult | None = None
        self.error: str | None = None
        self._cancel = threading.Event()

    @property
    def running(self) -> bool:
        """Whether the run has not finished yet."""
        return self.status == RUNNING

    def cancel(self) -> None:
        """Ask the agents to stop after their current step."""
        self._cancel.set()

    def log(self, level: str, message: str) -> None:
        """Add an entry to the activity log."""
        self.logs.append((level, message))

    def _fail(self, message: str) -> None:
        self.error = message
        self.status = ERROR


def start_analysis(
    request: AnalysisRequest,
    *,
    data: bytes | None = None,
    data_path: str | None = None,
    run_prefix: str = "ds",
    timeout: float = ANALYSIS_TIMEOUT_SECONDS,
) -> AnalysisRun:
    """Start an analysis of a table in a background thread.

    Pass the table either as ``data``, which is written to a folder in the
    workspace that is removed when the run ends, or as ``data_path``, a file
    that stays (Chat keeps one per conversation).

    Args:
        request: What to analyze and how.
        data: The uploaded file's content; its type comes from
            ``request.file_name``.
        data_path: A table file already on disk.
        run_prefix: Start of the run ID, which names the downloads.
        timeout: Seconds before the run is stopped.

    Returns:
        The run, already going.
    """
    if (data is None) == (data_path is None):
        raise ValueError("Pass either data or data_path.")
    run = AnalysisRun(f"{run_prefix}-{uuid.uuid4().hex[:8]}")
    threading.Thread(
        target=_execute,
        args=(run, request, data, data_path, timeout),
        daemon=True,
    ).start()
    return run


def analysis_messages(
    request: AnalysisRequest, columns: list[str]
) -> list[dict[str, str]]:
    """Return the request as the conversation the supervisor receives.

    Args:
        request: What to analyze.
        columns: The table's columns; selected columns not among them are
            ignored.

    Returns:
        One user message. The agent adds the dataset summary itself.
    """
    content = f"User Request: {request.prompt.strip() or DEFAULT_PROMPT}"
    selected = [column for column in request.columns if column in columns]
    if selected:
        content += (
            "\n\nColumn Selection: Focus ONLY on these columns chosen by the "
            f"user: {', '.join(selected)}"
        )
    return [{"role": "user", "content": content}]


def read_artifacts(
    plots: list[dict[str, Any]], log: Any = None
) -> list[Artifact]:
    """Read the chart files an analysis wrote.

    Args:
        plots: The agent's plot records (``path``, ``code``, ``r_code``,
            ``tool_name``).
        log: Receives ``(level, message)`` for charts that are missing or
            cannot be read.

    Returns:
        The charts, in order; unreadable ones are left out.
    """
    import plotly.io as pio

    artifacts = []
    for item in plots:
        path = item.get("path", "")
        if not path or not os.path.exists(path):
            if log is not None:
                log("warning", f"Could not find plot at: {path}")
            continue
        filename = os.path.basename(path)
        with open(path, "rb") as file:
            data = file.read()
        figure_json = None
        if filename.endswith(".json"):
            try:
                figure_json = data.decode("utf-8")
                pio.from_json(figure_json)
            except Exception:
                if log is not None:
                    log("warning", f"Failed to parse plot {filename}")
                continue
        artifacts.append(
            Artifact(
                filename=filename,
                data=data,
                code=item.get("code", ""),
                r_code=item.get("r_code", ""),
                tool_name=item.get("tool_name", ""),
                figure_json=figure_json,
            )
        )
    return artifacts


def _execute(
    run: AnalysisRun,
    request: AnalysisRequest,
    data: bytes | None,
    data_path: str | None,
    timeout: float,
) -> None:
    """Run the analysis; the body of the background thread."""
    from textlab.features.visualization.viz_agent import run_analysis

    try:
        if not model_installed(request.model):
            run.log("info", f"Pulling model '{request.model}'...")
            ollama.pull(request.model)
    except Exception as exc:
        run._fail(f"Failed to pull model '{request.model}': {exc}")
        return

    try:
        with contextlib.ExitStack() as stack:
            if data_path is None:
                run_dir = stack.enter_context(
                    get_workspace().temp_dir(WORKSPACE_AREA, prefix="run-")
                )
                data_path = save_data_file(
                    data, request.file_name, str(run_dir)
                )
            preview = get_fast_data_preview(
                data_path, os.path.basename(data_path)
            )
            if preview is None:
                run._fail("Failed to generate a data preview.")
                return
            messages = analysis_messages(request, list(preview.columns))

            run.log("info", "Starting Supervisor Agent...")
            try:
                outcome = asyncio.run(
                    asyncio.wait_for(
                        run_analysis(
                            messages,
                            data_path,
                            request.model,
                            log_callback=run.log,
                            cancel_event=run._cancel,
                            include_r_code=request.include_r_code,
                        ),
                        timeout=timeout,
                    )
                )
            except TimeoutError:
                run.status = TIMEOUT
                return

            # Read the charts before their folder is removed.
            run.result = AnalysisResult(
                run_id=run.run_id,
                summary=outcome.get("summary", ""),
                artifacts=read_artifacts(outcome.get("plots", []), run.log),
                stats=outcome.get("stats", []),
            )
            run.status = CANCELLED if run._cancel.is_set() else DONE
    except Exception as exc:
        run._fail(str(exc))
