"""
Agentic Multi-Agent System (MAS) for the AI Visualization Engine.

Implements a plan-execute-summarise Supervisor-Worker pattern that keeps the
number of sequential LLM calls low for small models:

1. Plan: the supervisor model returns one JSON plan (structured output).
2. Execute: delegated workers run concurrently with narrow tool sets, sharing
   one MCP server process for the whole analysis. A worker only loops to retry failed tool calls:
   plot workers stop as soon as their plots succeed, and stats workers get one
   tool-free turn to interpret their results.
3. Summarise: one tool-free supervisor call writes the final report. When no
   statistics were produced the report is built from a template instead.
"""

import asyncio
import contextlib
import copy
import difflib
import json
import re
import sys
import threading
import traceback
from datetime import timedelta
from typing import Any, AsyncIterator, Callable, TypedDict

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import get_default_environment, stdio_client

from core.chat_engine import chat_no_think, message_text
from core.visualization import r_code
from core.visualization.plot_data import get_all_columns_summary_impl
from core.visualization.viz_config import (
    AGENT_OPTIONS,
    AGENT_REQUEST_TIMEOUT,
    AGENT_TOOLS,
    INTERACTIVE_PROMPT,
    MAX_ROWS,
    STATIC_PROMPT,
    STATS_PROMPT,
    SUMMARY_PROMPT,
    SUPERVISOR_PROMPT,
    TOOL_CALL_TIMEOUT,
    PlotArtifact,
    StatsArtifact,
    VizAnalysisResult,
    get_tool_label,
)
from core.visualization.viz_utils import load_data_safely, was_last_load_truncated

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

# Caps on tool output sent back to the model. Full results still reach the UI.
MAX_TOOL_OUTPUT_CHARS = 4000
MAX_TOOL_ERROR_CHARS = 1000

# Tool parameters whose value is one column name, or a list of column names.
# Their schemas list the dataset's real columns as allowed values (``enum``),
# so models pick existing names instead of guessing (e.g. 'Radius_mean').
COLUMN_PARAMS = frozenset({
    "column", "x_column", "y_column", "color_column", "hue_column",
    "text_column", "target_col", "group_col",
})
COLUMN_LIST_PARAMS = frozenset({"predictor_cols"})
# Above this many columns the enums are left out to keep tool schemas short.
MAX_ENUM_COLUMNS = 150

# Minimum difflib similarity for mapping a misspelled tool name onto an allowed
# tool, e.g. 'run_rank_target_correlations' -> 'rank_target_correlations'.
TOOL_NAME_CUTOFF = 0.7

INTERPRET_INSTRUCTION = (
    "Write a short plain-English interpretation of the statistical results "
    "above. Do not call any tools."
)

# The supervisor returns its plan as JSON via Ollama structured output instead
# of a tool call. Decoding is constrained to this schema, so no model-specific
# tool-call parsing is involved: Qwen's XML tool calls failed to parse on long
# multi-field plans (HTTP 500), which also left Ollama stuck afterwards.
# One field per specialist makes the model decide on each of them explicitly.
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "interactive": {"type": "string"},
        "static": {"type": "string"},
        "stats": {"type": "string"},
        "reply": {"type": "string"},
    },
    "required": ["interactive", "static", "stats", "reply"],
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


def _clip(text: str, limit: int) -> str:
    """Truncate multi-line text to ``limit`` characters, keeping line breaks."""
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "\n...[truncated]"


# Chat-template control tokens (e.g. "<|eot|>", "<|im_end|>") that some models
# leak into their reply text when the server's template does not match them.
_LEAKED_TOKEN_RE = re.compile(r"\s*<\|[A-Za-z0-9_]{1,40}\|>\s*$")


def _model_text(message: dict[str, Any]) -> str:
    """Return a reply's text with leaked trailing control tokens removed."""
    text = message_text(message)
    while True:
        cleaned = _LEAKED_TOKEN_RE.sub("", text)
        if cleaned == text:
            return text.strip()
        text = cleaned


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse the first JSON object in ``text``, ignoring anything around it.

    Even with structured output some models wrap the JSON in a Markdown code
    fence, add a sentence, or leak an end-of-turn token after it, which makes
    a strict ``json.loads`` fail on an otherwise valid plan.

    Returns:
        The parsed object, or None if no JSON object can be decoded.
    """
    start = text.find("{")
    if start == -1:
        return None
    try:
        parsed, _ = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


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


def _set_string_enum(prop: dict[str, Any], allowed: list[str]) -> None:
    """Attach ``allowed`` as the enum of a string property.

    Handles both plain ``{"type": "string"}`` properties and optional ones,
    which FastMCP writes as ``{"anyOf": [{"type": "string"}, {"type": "null"}]}``.
    """
    for branch in prop.get("anyOf") or [prop]:
        if branch.get("type") == "string":
            branch["enum"] = allowed


def _add_column_enums(schema: dict[str, Any], columns: list[str]) -> None:
    """List the dataset's columns as allowed values of every column parameter.

    Optional parameters also allow ``""``, which the tools treat as "not set".
    Modifies ``schema`` in place; callers pass a copy.
    """
    required = set(schema.get("required", []))
    for name, prop in schema.get("properties", {}).items():
        if name in COLUMN_PARAMS:
            allowed = list(columns) if name in required else list(columns) + [""]
            _set_string_enum(prop, allowed)
        elif name in COLUMN_LIST_PARAMS and isinstance(prop.get("items"), dict):
            _set_string_enum(prop["items"], list(columns))


def _match_column(value: str, columns: list[str]) -> str | None:
    """Return the column ``value`` refers to, tolerating case and whitespace slips.

    Only an exact match or a single unambiguous case-insensitive match is
    accepted; anything else returns None and the tool reports the error.
    """
    if value in columns:
        return value
    key = " ".join(value.split()).casefold()
    matches = [c for c in columns if " ".join(c.split()).casefold() == key]
    return matches[0] if len(matches) == 1 else None


def _fix_column_args(
    tool_args: dict[str, Any], columns: list[str]
) -> list[tuple[str, str]]:
    """Correct near-miss column names in ``tool_args`` in place.

    Returns:
        The ``(requested, corrected)`` pairs that were changed, for logging.
    """
    fixes: list[tuple[str, str]] = []

    def fix(value: Any) -> Any:
        if not isinstance(value, str) or not value or value in columns:
            return value
        match = _match_column(value, columns)
        if match is None:
            return value
        fixes.append((value, match))
        return match

    for name in COLUMN_PARAMS & tool_args.keys():
        tool_args[name] = fix(tool_args[name])
    for name in COLUMN_LIST_PARAMS & tool_args.keys():
        if isinstance(tool_args[name], list):
            tool_args[name] = [fix(v) for v in tool_args[name]]
    return fixes


def _dataset_columns(data_file_path: str) -> list[str]:
    """Return the dataset's column names, or an empty list if it cannot be read."""
    try:
        return [str(c) for c in load_data_safely(data_file_path).columns]
    except Exception:
        return []


_R_BLOCK_RE = re.compile(r"\n*```r\n(.*?)```", re.DOTALL)


def _split_r_block(tool_output: str) -> tuple[str, str]:
    """Split off the optional trailing ```r block of a stats tool output.

    Returns:
        ``(output_without_r, r_code)``; ``r_code`` is empty when there is none.
        The model only ever receives the output without the R block.
    """
    match = _R_BLOCK_RE.search(tool_output)
    if not match:
        return tool_output, ""
    without_r = tool_output[:match.start()] + tool_output[match.end():]
    return without_r.rstrip(), match.group(1).strip()


def _extract_stats_code(tool_output: str) -> tuple[str, str, str]:
    """
    Splits a stats tool output into (result_text, python_code, r_code).
    Stats tools embed a ```python ... ``` block, and when R code was requested
    a ```r ... ``` block, after the result. Missing parts are returned as "".
    """
    tool_output, r_snippet = _split_r_block(tool_output)
    match = re.search(r"```python\n(.*?)```", tool_output, re.DOTALL)
    if match:
        code = match.group(1).strip()
        result_text = tool_output[:match.start()].strip()
    else:
        code = ""
        result_text = tool_output.strip()
    return result_text, code, r_snippet


def _describe_model_error(exc: Exception) -> str:
    """Turn an Ollama call failure into a short, user-facing explanation.

    Ollama answers HTTP 500 when it cannot parse the model's tool call (seen
    with Qwen models, whose XML tool-call format breaks easily). Retrying is
    not useful: the next request to Ollama was observed to hang afterwards.
    """
    detail = _snippet(str(exc))
    if "tool call" in detail.lower() or "xml syntax" in detail.lower():
        return (
            f"The model produced a tool call Ollama could not parse ({detail}). "
            "This is a known issue with some models; please try another model."
        )
    if "timed out" in detail.lower() or "timeout" in type(exc).__name__.lower():
        return (
            f"The model did not answer within {AGENT_REQUEST_TIMEOUT:.0f} seconds. "
            "Please try again or choose another model."
        )
    return f"Model call failed: {detail}"


async def _chat(
    model_name: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    json_schema: dict[str, Any] | None = None,
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
        timeout=AGENT_REQUEST_TIMEOUT,
        json_schema=json_schema,
    )


# =========================================================================
# MCP TOOL ACCESS
# =========================================================================

async def _get_mcp_tools(
    session: ClientSession,
    allowed_names: list[str] | None = None,
    columns: list[str] | None = None,
) -> list[dict[str, Any]]:
    """List the MCP server's tools in Ollama format, filtered to ``allowed_names``.

    Agent-injected arguments (see ``INJECTED_ARGS``) are removed from each
    schema because the agent always supplies them itself. When ``columns`` is
    given (and not longer than ``MAX_ENUM_COLUMNS``), every column parameter
    lists them as its allowed values.
    """
    use_enums = bool(columns) and len(columns) <= MAX_ENUM_COLUMNS
    tool_list_response = await session.list_tools()
    ollama_tools: list[dict[str, Any]] = []

    for tool in tool_list_response.tools:
        if allowed_names is not None and tool.name not in allowed_names:
            continue
        parameters = _strip_injected_args(tool.inputSchema)
        if use_enums:
            _add_column_enums(parameters, columns)
        ollama_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": parameters,
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

    Raises:
        mcp.shared.exceptions.McpError: If the server does not answer within
            ``TOOL_CALL_TIMEOUT`` seconds. Model-written code has its own,
            shorter limit inside the server; this is the safety net for any
            tool that hangs.
    """
    result = await session.call_tool(
        tool_name,
        arguments=tool_args,
        read_timeout_seconds=timedelta(seconds=TOOL_CALL_TIMEOUT),
    )
    output = "\n".join(
        part.text for part in result.content if isinstance(part, types.TextContent)
    )
    failed = bool(result.isError) or output.strip().startswith("Error")
    return not failed, output


def _record_plot(
    global_plots: list[PlotArtifact], tool_name: str, output: str
) -> bool:
    """Store a plot artifact from a ``"path|||code"`` or ``"path|||code|||r"`` output.

    Returns False if the output is malformed. Plot files never overwrite each
    other, so distinct plots are all kept. Only an exact repeat (same tool and
    same generated code, e.g. the model re-ran a plot that already succeeded)
    replaces the earlier entry instead of showing the same plot twice.
    """
    if "|||" not in output:
        return False
    path_part, code_part, *r_part = output.split("|||", 2)
    artifact: PlotArtifact = {
        "path": path_part.strip(),
        "code": code_part.strip(),
        "r_code": r_part[0].strip() if r_part else "",
        "tool_name": tool_name,
    }
    for index, existing in enumerate(global_plots):
        if (existing["tool_name"], existing["code"]) == (tool_name, artifact["code"]):
            global_plots[index] = artifact
            return True
    global_plots.append(artifact)
    return True


# =========================================================================
# WORKERS
# =========================================================================

@contextlib.asynccontextmanager
async def _mcp_session(
    mcp_server_script: str, include_r_code: bool = False
) -> AsyncIterator[ClientSession]:
    """Start one MCP server subprocess and yield an initialised client session.

    The server is shared by all workers of an analysis. Starting it means
    importing the plotting/statistics stack and later parsing the dataset, so
    doing this once instead of once per worker saves several seconds per run
    and lets the server's dataset cache serve every worker. Concurrent workers
    multiplex their requests over the one session; the server runs the tool
    calls one at a time, which costs little because they are short compared
    to model calls.

    ``sys.executable`` guarantees the server runs in the same Python
    environment as the app, whatever ``python3`` resolves to on ``PATH``.

    With ``include_r_code`` the server's tools also return R equivalents of
    their Python snippets (see ``r_code.py``).
    """
    env = get_default_environment()
    if include_r_code:
        env[r_code.R_CODE_ENV] = "1"
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[mcp_server_script],
        env=env,
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def _run_worker_agent(
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
    # Cached in-process by load_data_safely, so this does not re-read the file.
    columns = await asyncio.to_thread(_dataset_columns, data_file_path)
    tools = await _get_mcp_tools(session, allowed_names=allowed_tools, columns=columns)

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

    for round_index in range(WORKER_MAX_ITERATIONS):
        if cancel_event and cancel_event.is_set():
            log("warning", f"Worker '{agent_role}' cancelled by user.")
            report["errors"].append("Cancelled by user.")
            return report

        try:
            response = await _chat(model_name, messages, tools)
        except Exception as exc:
            reason = _describe_model_error(exc)
            log("error", f"Worker '{agent_role}' stopped: {reason}")
            report["errors"].append(reason)
            return report
        messages.append(response["message"])

        tool_calls = response["message"].get("tool_calls")
        if not tool_calls:
            report["text"] = _model_text(response["message"])
            report["completed"] = True
            log("info", f"Worker '{agent_role}' finished task successfully.")
            return report

        log("info", _describe_round(agent_role, tool_calls, allowed_tools, round_index))

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

            for requested, corrected in _fix_column_args(tool_args, columns):
                log(
                    "info",
                    f"Worker '{agent_role}' used column '{requested}'; using '{corrected}'.",
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
                    f"Execution Error: {_clip(output, MAX_TOOL_ERROR_CHARS)}\n"
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
                    result_text, code_snippet, r_snippet = _extract_stats_code(output)
                    # A retry round may repeat a test that already succeeded.
                    if all(s["result"] != result_text for s in global_stats):
                        global_stats.append({
                            "title": get_tool_label(tool_name),
                            "result": result_text,
                            "code": code_snippet,
                            "r_code": r_snippet,
                        })
                    round_stats.append(result_text)
                model_output, _ = _split_r_block(output)
                messages.append(_tool_message(
                    tool_name, _clip(model_output, MAX_TOOL_OUTPUT_CHARS)
                ))

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
            log("info", f"Worker '{agent_role}' is interpreting the results.")
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


def _describe_round(
    agent_role: str,
    tool_calls: list[dict[str, Any]],
    allowed_tools: list[str],
    round_index: int,
) -> str:
    """Summarise a worker's planned tool calls for the activity log.

    Shows how many tools the agent runs and which ones, so users can see why
    some agents take longer than others. Rounds after the first are retries
    of tool calls that failed.
    """
    labels: list[str] = []
    for tool_call in tool_calls:
        requested_name, _ = _parse_tool_call(tool_call)
        tool_name = _resolve_tool_name(requested_name, allowed_tools) or requested_name
        labels.append(get_tool_label(tool_name))

    count = len(labels)
    noun = "tool" if count == 1 else "tools"
    action = "is running" if round_index == 0 else "is retrying"
    return f"Worker '{agent_role}' {action} {count} {noun}: {', '.join(labels)}."


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
    return _model_text(response["message"])


# =========================================================================
# SUPERVISOR
# =========================================================================

async def _plan_tasks(
    messages: list[dict[str, Any]], schema: str, model_name: str, log: LogFn
) -> tuple[list[tuple[str, str]], str]:
    """Make the single supervisor planning call and read its JSON plan.

    The supervisor gets the compact dataset summary (column names grouped by
    type, sample categories) rather than raw rows: it needs names and types to
    plan, and raw values on wide datasets only cost tokens and tempt small
    models into copying every column into their instructions.

    Returns:
        ``(tasks, direct_reply)`` where ``tasks`` is a de-duplicated list of
        ``(agent_role, task_instruction)`` pairs in role order. ``direct_reply``
        holds the supervisor's answer when no specialist is needed.
    """
    system_prompt = f"{SUPERVISOR_PROMPT}\nDataset summary:\n{schema}\n"
    supervisor_messages = [{"role": "system", "content": system_prompt}] + messages
    response = await _chat(model_name, supervisor_messages, json_schema=PLAN_SCHEMA)
    content = _model_text(response["message"])

    plan = _extract_json_object(content)
    if plan is None:
        log("warning", "Supervisor did not return a valid plan; showing its reply instead.")
        return [], content

    tasks: list[tuple[str, str]] = []
    for role in WORKER_PROMPTS:
        instruction = str(plan.get(role) or "").strip()
        if instruction.lower().rstrip(".") in EMPTY_INSTRUCTIONS:
            continue
        if (role, instruction) not in tasks:
            tasks.append((role, instruction))

    if not tasks:
        log("info", "Supervisor answered without delegating any tasks.")
    return tasks, str(plan.get("reply") or "").strip()


async def _run_workers(
    tasks: list[tuple[str, str]],
    session: ClientSession,
    data_file_path: str,
    schema: str,
    model_name: str,
    global_plots: list[PlotArtifact],
    global_stats: list[StatsArtifact],
    log: LogFn,
    cancel_event: threading.Event | None,
) -> list[WorkerReport]:
    """Run all delegated workers concurrently on one MCP session.

    A worker that raises is converted into a failed report so one broken
    worker never discards the results of the others.
    """
    outputs = await asyncio.gather(
        *(
            _run_worker_agent(
                session=session,
                agent_role=role,
                task_instruction=instruction,
                data_file_path=data_file_path,
                schema=schema,
                model_name=model_name,
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


async def _run_workers_on_shared_server(
    tasks: list[tuple[str, str]],
    mcp_server_script: str,
    data_file_path: str,
    schema: str,
    model_name: str,
    global_plots: list[PlotArtifact],
    global_stats: list[StatsArtifact],
    log: LogFn,
    cancel_event: threading.Event | None,
    include_r_code: bool = False,
) -> list[WorkerReport]:
    """Start the MCP server once, run every worker on it, then shut it down.

    If the server cannot start, or dies so that closing the session fails,
    the error is logged. Reports that were already completed are kept;
    otherwise each task gets a failed report, so the run still ends with a
    readable summary instead of a crash.
    """
    reports: list[WorkerReport] | None = None
    try:
        async with _mcp_session(mcp_server_script, include_r_code) as session:
            reports = await _run_workers(
                tasks, session, data_file_path, schema, model_name,
                global_plots, global_stats, log, cancel_event,
            )
    except BaseException as exc:
        if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
            raise
        log("error", f"MCP server session failed:\n{_unwrap_exception_group(exc)}")

    if reports is None:
        reports = []
        for role, instruction in tasks:
            report = _new_report(role, instruction)
            report["errors"].append("The analysis tools could not be started.")
            reports.append(report)
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
        summary = _model_text(response["message"])
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
    include_r_code: bool = False,
) -> VizAnalysisResult:
    """Run a full plan-execute-summarise analysis of the user's request.

    Args:
        messages: The user turn(s) describing the request and a data preview.
        data_file_path: Path to the uploaded dataset.
        model_name: The Ollama model used by the supervisor and all workers.
        mcp_server_script: Path to the MCP server script, started once per run.
        log_callback: Optional callback receiving ``(level, message)`` logs.
        cancel_event: Optional event; when set, workers stop between rounds.
        include_r_code: Also produce R equivalents of the Python snippets
            (Visualize Data option; the models never see the R code).

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
        # Computed once, before planning: the supervisor plans from it and the
        # concurrent workers reuse it instead of each re-reading the data.
        schema = await asyncio.to_thread(get_all_columns_summary_impl, data_file_path)
        truncated = was_last_load_truncated(data_file_path)

        try:
            tasks, direct_reply = await _plan_tasks(messages, schema, model_name, _log)
        except Exception as exc:
            reason = _describe_model_error(exc)
            _log("error", f"Supervisor stopped: {reason}")
            tasks, direct_reply = [], f"The analysis could not be started. {reason}"

        if not tasks:
            summary = direct_reply
        elif cancel_event and cancel_event.is_set():
            _log("warning", "Analysis cancelled by user.")
        else:
            _log("info", f"Supervisor planned {len(tasks)} task(s).")
            reports = await _run_workers_on_shared_server(
                tasks, mcp_server_script, data_file_path, schema, model_name,
                plot_results, stats_results, _log, cancel_event, include_r_code,
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
