"""
Agentic Multi-Agent System (MAS) for the AI Visualization Engine.

Implements a plan-execute-summarise Supervisor-Worker pattern that keeps the
number of sequential LLM calls low for small models:

1. Plan: the supervisor model makes a single ``plan_tasks`` call.
2. Execute: delegated workers run concurrently, each with its own MCP session
   and a narrow tool set. A worker only loops to retry failed tool calls:
   plot workers stop as soon as their plots succeed, and stats workers get one
   tool-free turn to interpret their results.
3. Summarise: one tool-free supervisor call writes the final report. When no
   statistics were produced the report is built from a template instead.
"""

import asyncio
import copy
import difflib
import json
import re
import threading
import traceback
from typing import Any, Callable, TypedDict

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

from core.chat_engine import chat_no_think, message_text
from core.visualization.plot_data import get_all_columns_summary_impl
from core.visualization.viz_config import (
    AGENT_OPTIONS,
    AGENT_TOOLS,
    INTERACTIVE_PROMPT,
    MAX_ROWS,
    STATIC_PROMPT,
    STATS_PROMPT,
    SUMMARY_PROMPT,
    SUPERVISOR_PROMPT,
    PlotArtifact,
    StatsArtifact,
    VizAnalysisResult,
    get_tool_label,
)
from core.visualization.viz_utils import was_last_load_truncated

WORKER_PROMPTS = {
    "interactive": INTERACTIVE_PROMPT,
    "static": STATIC_PROMPT,
    "stats": STATS_PROMPT,
}

PLOT_ROLES = frozenset({"interactive", "static"})
PLOT_TOOLS = frozenset(AGENT_TOOLS["interactive"] + AGENT_TOOLS["static"])
STATS_TOOLS = frozenset(AGENT_TOOLS["stats"])

# Tool arguments supplied by the agent itself. They are hidden from the model's
# tool schemas so small models do not waste tokens inventing file paths.
INJECTED_ARGS = ("data_file_path",)

WORKER_MAX_ITERATIONS = 6
LOG_SNIPPET_CHARS = 300

# Minimum difflib similarity for mapping a misspelled tool name onto an allowed
# tool, e.g. 'run_rank_target_correlations' -> 'rank_target_correlations'.
TOOL_NAME_CUTOFF = 0.7

INTERPRET_INSTRUCTION = (
    "Write a short plain-English interpretation of the statistical results "
    "above. Do not call any tools."
)

# One field per specialist: small models often emit only a single tool call per
# response, so separate per-agent delegation calls silently dropped specialists.
# Requiring every field makes the model decide on each specialist explicitly.
PLAN_TASKS_TOOL = {
    "type": "function",
    "function": {
        "name": "plan_tasks",
        "description": (
            "Assign the work to the specialist agents in one call. Write an "
            "instruction for every specialist the request needs and leave the "
            "others as an empty string."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "interactive": {
                    "type": "string",
                    "description": "Instruction for the interactive Plotly agent, or empty.",
                },
                "static": {
                    "type": "string",
                    "description": "Instruction for the static Matplotlib/Seaborn agent, or empty.",
                },
                "stats": {
                    "type": "string",
                    "description": "Instruction for the statistical tests agent, or empty.",
                },
            },
            "required": ["interactive", "static", "stats"],
        },
    },
}

# Placeholder values small models write into fields that should stay empty.
EMPTY_INSTRUCTIONS = frozenset({"", "none", "null", "n/a", "na", "-", "not needed"})


class WorkerReport(TypedDict):
    """Structured outcome of one worker run, consumed by the summary step."""

    role: str
    instruction: str
    text: str           # free-text reply or interpretation from the worker model
    plots: list[str]    # labels of plots generated successfully
    stats: list[str]    # statistical result tables (without code)
    errors: list[str]   # errors still unresolved when the worker stopped
    completed: bool     # False when the worker hit its iteration limit or crashed


LogFn = Callable[[str, str], None]


# =========================================================================
# HELPERS
# =========================================================================

def _unwrap_exception_group(exc: BaseException) -> str:
    """Recursively unwrap ExceptionGroup / BaseExceptionGroup to reveal the
    actual root cause.  The MCP ``stdio_client`` uses ``anyio.TaskGroup``
    internally, and when the subprocess fails, the real error gets buried
    inside an ``ExceptionGroup`` whose ``str()`` only shows
    *"unhandled errors in a TaskGroup (1 sub-exception)"*."""
    parts: list[str] = []
    if hasattr(exc, "exceptions"):
        for sub in exc.exceptions:
            parts.append(_unwrap_exception_group(sub))
    else:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        parts.append(f"{type(exc).__name__}: {exc}\n{tb}")
    return "\n".join(parts) if parts else f"{type(exc).__name__}: {exc}"


def _snippet(text: str, limit: int = LOG_SNIPPET_CHARS) -> str:
    """Collapse whitespace and truncate text for single-line log messages."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def _new_report(role: str, instruction: str) -> WorkerReport:
    """Create an empty worker report."""
    return {
        "role": role,
        "instruction": instruction,
        "text": "",
        "plots": [],
        "stats": [],
        "errors": [],
        "completed": False,
    }


def _tool_message(tool_name: str, content: str) -> dict[str, Any]:
    """Build a ``tool`` role message for the Ollama chat history."""
    return {"role": "tool", "tool_name": tool_name, "content": content}


def _parse_tool_call(tool_call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return ``(name, arguments)`` from a tool call, tolerating JSON-string args."""
    function = tool_call.get("function", {})
    args = function.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return function.get("name", ""), dict(args)


def _resolve_tool_name(name: str, allowed: list[str]) -> str | None:
    """Map a requested tool name onto an allowed tool, tolerating near misses.

    Small models often copy a naming pattern from sibling tools (e.g. adding
    ``run_`` because most stats tools start with it) and then repeat the same
    wrong name on every retry. Accepting a single close match avoids burning
    the worker's iterations on a spelling mistake.

    Returns:
        The allowed tool name, or None when there is no sufficiently close match.
    """
    if name in allowed:
        return name
    matches = difflib.get_close_matches(name, allowed, n=1, cutoff=TOOL_NAME_CUTOFF)
    return matches[0] if matches else None


def _strip_injected_args(schema: dict[str, Any] | None) -> dict[str, Any]:
    """Return a copy of a tool's JSON schema without agent-injected arguments."""
    clean = copy.deepcopy(schema) if schema else {"type": "object", "properties": {}}
    properties = clean.get("properties", {})
    for name in INJECTED_ARGS:
        properties.pop(name, None)
    if "required" in clean:
        clean["required"] = [r for r in clean["required"] if r not in INJECTED_ARGS]
    return clean


def _extract_stats_code(tool_output: str) -> tuple[str, str]:
    """
    Splits a stats tool output into (result_text, code_snippet).
    Stats tools embed a ```python ... ``` block at the end of their output.
    Returns the markdown table/summary and the code block separately.
    """
    match = re.search(r"```python\n(.*?)```", tool_output, re.DOTALL)
    if match:
        code = match.group(1).strip()
        result_text = tool_output[:match.start()].strip()
    else:
        code = ""
        result_text = tool_output.strip()
    return result_text, code


async def _chat(
    model_name: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run a blocking Ollama chat call in a worker thread.

    ``ollama.chat`` is synchronous. Calling it directly inside a coroutine
    blocks the event loop, which serialises "concurrent" workers and stops the
    caller's timeout and cancellation from taking effect during a call.
    """
    return await asyncio.to_thread(
        chat_no_think,
        model=model_name,
        messages=messages,
        tools=tools,
        options=AGENT_OPTIONS,
    )


# =========================================================================
# MCP TOOL ACCESS
# =========================================================================

async def _get_mcp_tools(
    session: ClientSession, allowed_names: list[str] | None = None
) -> list[dict[str, Any]]:
    """List the MCP server's tools in Ollama format, filtered to ``allowed_names``.

    Agent-injected arguments (see ``INJECTED_ARGS``) are removed from each
    schema because the agent always supplies them itself.
    """
    tool_list_response = await session.list_tools()
    ollama_tools: list[dict[str, Any]] = []

    for tool in tool_list_response.tools:
        if allowed_names is not None and tool.name not in allowed_names:
            continue
        ollama_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": _strip_injected_args(tool.inputSchema),
                },
            }
        )

    return ollama_tools


async def _call_tool(
    session: ClientSession, tool_name: str, tool_args: dict[str, Any]
) -> tuple[bool, str]:
    """Call an MCP tool and return ``(succeeded, text_output)``.

    A call fails when MCP flags it as an error (e.g. invalid arguments) or when
    the tool's own output starts with ``"Error"``, the convention all tool
    implementations follow.
    """
    result = await session.call_tool(tool_name, arguments=tool_args)
    output = "\n".join(
        part.text for part in result.content if isinstance(part, types.TextContent)
    )
    failed = bool(result.isError) or output.strip().startswith("Error")
    return not failed, output


def _record_plot(
    global_plots: list[PlotArtifact], tool_name: str, output: str
) -> bool:
    """Store a plot artifact from a ``"path|||code"`` tool output.

    Returns False if the output is malformed. A plot whose path is already
    recorded (the model re-ran the same plot) replaces the earlier entry, since
    the file on disk was overwritten anyway.
    """
    if "|||" not in output:
        return False
    path_part, code_part = output.split("|||", 1)
    artifact: PlotArtifact = {
        "path": path_part.strip(),
        "code": code_part.strip(),
        "tool_name": tool_name,
    }
    for index, existing in enumerate(global_plots):
        if existing["path"] == artifact["path"]:
            global_plots[index] = artifact
            return True
    global_plots.append(artifact)
    return True


# =========================================================================
# WORKERS
# =========================================================================

async def _run_worker_agent(
    agent_role: str,
    task_instruction: str,
    data_file_path: str,
    schema: str,
    model_name: str,
    mcp_server_script: str,
    global_plots: list[PlotArtifact],
    global_stats: list[StatsArtifact],
    log: LogFn,
    cancel_event: threading.Event | None = None,
) -> WorkerReport:
    """Run one specialist worker inside its own MCP stdio session.

    Raises:
        RuntimeError: If the MCP server subprocess cannot be started or dies.
    """
    server_params = StdioServerParameters(
        command="python3",
        args=[mcp_server_script],
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await _run_worker_loop(
                    session=session,
                    agent_role=agent_role,
                    task_instruction=task_instruction,
                    data_file_path=data_file_path,
                    schema=schema,
                    model_name=model_name,
                    global_plots=global_plots,
                    global_stats=global_stats,
                    log=log,
                    cancel_event=cancel_event,
                )
    except BaseException as exc:
        if isinstance(exc, asyncio.CancelledError):
            raise
        detail = _unwrap_exception_group(exc)
        log("error", f"Worker '{agent_role}' MCP session failed:\n{detail}")
        raise RuntimeError(
            f"Worker '{agent_role}' MCP session failed: {detail}"
        ) from exc


async def _run_worker_loop(
    session: ClientSession,
    agent_role: str,
    task_instruction: str,
    data_file_path: str,
    schema: str,
    model_name: str,
    global_plots: list[PlotArtifact],
    global_stats: list[StatsArtifact],
    log: LogFn,
    cancel_event: threading.Event | None,
) -> WorkerReport:
    """Drive one worker model until its tool calls succeed or it gives up.

    Each round is one LLM call plus the tool calls it requests. The loop only
    continues while tool calls keep failing, so a fully successful round ends
    the worker:

    * plot workers return immediately (no extra "I am done" LLM turn);
    * stats workers get one tool-free turn to interpret the numbers.

    When the iteration limit is reached, everything that did succeed is still
    returned together with the unresolved errors.
    """
    report = _new_report(agent_role, task_instruction)
    allowed_tools = AGENT_TOOLS.get(agent_role, [])
    tools = await _get_mcp_tools(session, allowed_names=allowed_tools)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": WORKER_PROMPTS[agent_role]},
        {
            "role": "user",
            "content": (
                f"Dataset schema (already loaded — use your tools to analyze it):\n{schema}\n\n"
                f"Task: {task_instruction}"
            ),
        },
    ]

    log("info", f"Supervisor delegated task to '{agent_role}' agent.")

    for _ in range(WORKER_MAX_ITERATIONS):
        if cancel_event and cancel_event.is_set():
            log("warning", f"Worker '{agent_role}' cancelled by user.")
            report["errors"].append("Cancelled by user.")
            return report

        try:
            response = await _chat(model_name, messages, tools)
        except Exception as exc:
            log("error", f"Worker '{agent_role}' failed to communicate with Ollama: {exc}")
            report["errors"].append(f"Model call failed: {exc}")
            return report
        messages.append(response["message"])

        tool_calls = response["message"].get("tool_calls")
        if not tool_calls:
            report["text"] = message_text(response["message"])
            report["completed"] = True
            log("info", f"Worker '{agent_role}' finished task successfully.")
            return report

        round_errors: list[str] = []
        round_stats: list[str] = []
        round_plots: list[str] = []

        for tool_call in tool_calls:
            requested_name, tool_args = _parse_tool_call(tool_call)
            tool_name = _resolve_tool_name(requested_name, allowed_tools)

            if tool_name is None:
                error = (
                    f"Error: tool '{requested_name}' is not available. "
                    f"Use one of: {', '.join(allowed_tools)}."
                )
                log(
                    "warning",
                    f"Worker '{agent_role}' requested unknown tool '{requested_name}'. "
                    "Retrying...",
                )
                round_errors.append(error)
                messages.append(_tool_message(requested_name, error))
                continue
            if tool_name != requested_name:
                log(
                    "info",
                    f"Worker '{agent_role}' called '{requested_name}'; using '{tool_name}'.",
                )

            for name in INJECTED_ARGS:
                tool_args.pop(name, None)
            tool_args["data_file_path"] = data_file_path

            try:
                succeeded, output = await _call_tool(session, tool_name, tool_args)
            except Exception as exc:
                succeeded, output = False, f"Error: tool '{tool_name}' crashed: {exc}"

            if succeeded and tool_name in PLOT_TOOLS:
                succeeded = _record_plot(global_plots, tool_name, output)
                if not succeeded:
                    output = "Error: the plot tool did not return a plot file."

            if not succeeded:
                log(
                    "warning",
                    f"Worker '{agent_role}' tool '{tool_name}' failed: "
                    f"{_snippet(output)} Retrying...",
                )
                round_errors.append(f"{get_tool_label(tool_name)}: {output.strip()}")
                messages.append(_tool_message(
                    tool_name,
                    f"Execution Error: {output.strip()}\n"
                    "Please correct your code or parameters and try again.",
                ))
                continue

            if tool_name in PLOT_TOOLS:
                title = str(tool_args.get("title") or "").strip()
                label = get_tool_label(tool_name)
                round_plots.append(f"{label}: {title}" if title else label)
                # Generic message — never expose internal file paths to the model.
                messages.append(_tool_message(
                    tool_name,
                    "Plot generated successfully. It will be displayed to the user. "
                    "Do not generate it again.",
                ))
            else:
                if tool_name in STATS_TOOLS:
                    result_text, code_snippet = _extract_stats_code(output)
                    # A retry round may repeat a test that already succeeded.
                    if all(s["result"] != result_text for s in global_stats):
                        global_stats.append({
                            "title": get_tool_label(tool_name),
                            "result": result_text,
                            "code": code_snippet,
                        })
                    round_stats.append(result_text)
                messages.append(_tool_message(tool_name, output))

        report["plots"].extend(round_plots)
        report["stats"].extend(r for r in round_stats if r not in report["stats"])
        previous_errors, report["errors"] = report["errors"], round_errors

        if round_errors:
            # The model ignored the error feedback and repeated the exact same
            # failing calls; further rounds would only repeat them again.
            if round_errors == previous_errors and not (round_plots or round_stats):
                log(
                    "error",
                    f"Worker '{agent_role}' repeated the same failing call; stopping.",
                )
                return report
            continue

        if agent_role in PLOT_ROLES and round_plots:
            report["completed"] = True
            log("info", f"Worker '{agent_role}' finished task successfully.")
            return report

        if agent_role == "stats" and round_stats:
            report["text"] = await _interpret_stats(model_name, messages, agent_role, log)
            report["completed"] = True
            log("info", f"Worker '{agent_role}' finished task successfully.")
            return report

    if report["plots"] or report["stats"]:
        log(
            "warning",
            f"Worker '{agent_role}' reached the iteration limit; returning partial results.",
        )
    else:
        log("error", f"Worker '{agent_role}' reached the iteration limit without any results.")
    return report


async def _interpret_stats(
    model_name: str, messages: list[dict[str, Any]], agent_role: str, log: LogFn
) -> str:
    """Ask the stats worker for a tool-free interpretation of its results.

    Returns an empty string on failure; the raw result tables still reach the
    summary step through the worker report.
    """
    messages.append({"role": "user", "content": INTERPRET_INSTRUCTION})
    try:
        response = await _chat(model_name, messages)
    except Exception as exc:
        log("warning", f"Worker '{agent_role}' could not interpret its results: {exc}")
        return ""
    return message_text(response["message"])


# =========================================================================
# SUPERVISOR
# =========================================================================

async def _plan_tasks(
    messages: list[dict[str, Any]], model_name: str, log: LogFn
) -> tuple[list[tuple[str, str]], str]:
    """Make the single supervisor planning call.

    Every tool call's arguments are read regardless of the tool name the model
    used, so a misspelled ``plan_tasks`` still yields its tasks.

    Returns:
        ``(tasks, direct_reply)`` where ``tasks`` is a de-duplicated list of
        ``(agent_role, task_instruction)`` pairs in role order. ``direct_reply``
        holds the supervisor's text when it answered without delegating.
    """
    supervisor_messages = [{"role": "system", "content": SUPERVISOR_PROMPT}] + messages
    response = await _chat(model_name, supervisor_messages, [PLAN_TASKS_TOOL])
    message = response["message"]

    tool_calls = message.get("tool_calls")
    if not tool_calls:
        return [], message_text(message)

    tasks: list[tuple[str, str]] = []
    for tool_call in tool_calls:
        _, args = _parse_tool_call(tool_call)
        for key in args:
            if key not in WORKER_PROMPTS:
                log("warning", f"Supervisor assigned work to unknown agent '{key}'; skipping it.")
        for role in WORKER_PROMPTS:
            instruction = str(args.get(role) or "").strip()
            if instruction.lower().rstrip(".") in EMPTY_INSTRUCTIONS:
                continue
            if (role, instruction) not in tasks:
                tasks.append((role, instruction))

    return tasks, message_text(message)


async def _run_workers(
    tasks: list[tuple[str, str]],
    data_file_path: str,
    schema: str,
    model_name: str,
    mcp_server_script: str,
    global_plots: list[PlotArtifact],
    global_stats: list[StatsArtifact],
    log: LogFn,
    cancel_event: threading.Event | None,
) -> list[WorkerReport]:
    """Run all delegated workers concurrently and collect their reports.

    A worker that raises is converted into a failed report so one broken
    worker never discards the results of the others.
    """
    outputs = await asyncio.gather(
        *(
            _run_worker_agent(
                agent_role=role,
                task_instruction=instruction,
                data_file_path=data_file_path,
                schema=schema,
                model_name=model_name,
                mcp_server_script=mcp_server_script,
                global_plots=global_plots,
                global_stats=global_stats,
                log=log,
                cancel_event=cancel_event,
            )
            for role, instruction in tasks
        ),
        return_exceptions=True,
    )

    reports: list[WorkerReport] = []
    for (role, instruction), output in zip(tasks, outputs):
        if isinstance(output, asyncio.CancelledError):
            raise output
        if isinstance(output, BaseException):
            report = _new_report(role, instruction)
            report["errors"].append(str(output))
            reports.append(report)
        else:
            reports.append(output)
    return reports


def _format_report(report: WorkerReport) -> str:
    """Render one worker report as plain text for the summary prompt."""
    lines = [f"### Task: {report['instruction']}"]
    if report["plots"]:
        lines.append("Visualisations generated: " + "; ".join(report["plots"]))
    if report["stats"]:
        lines.append("Statistical results:")
        lines.extend(report["stats"])
    if report["text"]:
        lines.append(f"Specialist notes:\n{report['text']}")
    if report["errors"]:
        lines.append("Could not be completed:")
        lines.extend(f"- {_snippet(error)}" for error in report["errors"])
    return "\n".join(lines)


def _template_summary(reports: list[WorkerReport], truncated: bool) -> str:
    """Build the final summary without an LLM call (used when there are no stats)."""
    plots = [label for report in reports for label in report["plots"]]
    notes = [report["text"].strip() for report in reports if report["text"].strip()]
    errors = [error for report in reports for error in report["errors"]]

    parts: list[str] = []
    if plots:
        parts.append("The following visualisations were generated and are shown below:")
        parts.append("\n".join(f"- {label}" for label in plots))
    if notes:
        parts.append("\n\n".join(notes))
    if errors:
        parts.append("Some steps could not be completed:")
        parts.append("\n".join(f"- {_snippet(error)}" for error in errors))
    if not parts:
        parts.append(
            "No results were produced. Try rephrasing the request or naming the "
            "columns to analyse."
        )
    if truncated:
        parts.append(f"Note: the dataset was truncated to the first {MAX_ROWS:,} rows.")
    return "\n\n".join(parts)


async def _summarise(
    messages: list[dict[str, Any]],
    reports: list[WorkerReport],
    model_name: str,
    truncated: bool,
    log: LogFn,
) -> str:
    """Produce the final user-facing summary.

    Plot-only runs use a template: the model cannot see the rendered plots, so
    an LLM summary would only restate the plot titles. When statistics exist,
    one tool-free call turns the numbers into a readable report.
    """
    if not any(report["stats"] for report in reports):
        return _template_summary(reports, truncated)

    request = "\n\n".join(
        str(m.get("content", "")) for m in messages if m.get("role") == "user"
    )
    results = "\n\n".join(_format_report(report) for report in reports)
    if truncated:
        results += f"\n\nNote: the dataset was truncated to the first {MAX_ROWS:,} rows."

    summary_messages = [
        {"role": "system", "content": SUMMARY_PROMPT},
        {"role": "user", "content": f"{request}\n\n## Specialist results\n\n{results}"},
    ]
    try:
        response = await _chat(model_name, summary_messages)
        summary = message_text(response["message"])
    except Exception as exc:
        log("warning", f"Supervisor could not write the summary: {exc}")
        summary = ""
    return summary or _template_summary(reports, truncated)


async def run_analysis(
    messages: list[dict[str, Any]],
    data_file_path: str,
    model_name: str,
    mcp_server_script: str,
    log_callback: LogFn | None = None,
    cancel_event: threading.Event | None = None,
) -> VizAnalysisResult:
    """Run a full plan-execute-summarise analysis of the user's request.

    Args:
        messages: The user turn(s) describing the request and a data preview.
        data_file_path: Path to the uploaded dataset.
        model_name: The Ollama model used by the supervisor and all workers.
        mcp_server_script: Path to the MCP server script launched per worker.
        log_callback: Optional callback receiving ``(level, message)`` logs.
        cancel_event: Optional event; when set, workers stop between rounds.

    Returns:
        The summary, plot and stats artifacts, and the run logs.
    """
    plot_results: list[PlotArtifact] = []
    stats_results: list[StatsArtifact] = []
    logs: list[tuple[str, str]] = []

    def _log(level: str, msg: str) -> None:
        logs.append((level, msg))
        if log_callback:
            log_callback(level, msg)

    summary = ""
    try:
        tasks, direct_reply = await _plan_tasks(messages, model_name, _log)

        if not tasks:
            _log("info", "Supervisor answered without delegating any tasks.")
            summary = direct_reply
        elif cancel_event and cancel_event.is_set():
            _log("warning", "Analysis cancelled by user.")
        else:
            _log("info", f"Supervisor planned {len(tasks)} task(s).")
            # Computed once here so concurrent workers do not each re-read the data.
            schema = await asyncio.to_thread(get_all_columns_summary_impl, data_file_path)
            truncated = was_last_load_truncated(data_file_path)

            reports = await _run_workers(
                tasks, data_file_path, schema, model_name, mcp_server_script,
                plot_results, stats_results, _log, cancel_event,
            )
            summary = await _summarise(messages, reports, model_name, truncated, _log)
            _log("info", "Supervisor synthesized the final summary.")
    except Exception as exc:
        _log("error", f"Fatal error in MAS session: {exc}\n{traceback.format_exc()}")

    if not summary:
        summary = "Analysis complete. Please review the generated visualizations below."

    final_logs: list[tuple[Any, str]] = [
        (level if level in ("info", "warning", "error") else "info", msg)
        for level, msg in logs
    ]

    return {
        "summary": summary,
        "plots": plot_results,
        "stats": stats_results,
        "logs": final_logs,
    }
