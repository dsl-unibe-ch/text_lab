"""Chatting with a local model about the user's files: the Chat API.

Interfaces call this module. :func:`respond` replies to a message: with a
table attached and a question that needs charts or statistics
(:func:`decide_tool_use`), the Visualization feature's agents answer
(:func:`start_data_analysis`); otherwise the chat model does
(:func:`reply`), with the attached documents' text as context
(:func:`read_document`, :func:`document_context`). Conversations are
exported with :func:`format_chat_history` and
:func:`format_chat_history_html`.

Messages are dictionaries with ``role`` and ``content``; an assistant turn
answered by the agents also has an ``analysis`` entry
(``visualization.models.AnalysisResult.to_payload``).
"""

from __future__ import annotations

import dataclasses
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

from textlab.common.model_config import ModelOptions, model_options
from textlab.common.ollama import (
    check_ollama_server,
    chunk_text,
    estimate_tokens,
    is_model_loaded,
    model_installed,
    pull_model,
)
from textlab.common.progress import (
    CancelledError,
    Progress,
    ProgressCallback,
    no_progress,
)
from textlab.common.storage import get_workspace, remove_tree
from textlab.features.chat.context_window import (
    HISTORY_SHARE,
    ContextUsage,
    chat_context,
    context_usage,
    group_by_budget,
    message_tokens,
    text_budget,
    trim_history,
)
from textlab.features.chat.documents import (
    DOCUMENT_TYPES,
    MAX_FILE_MB,
    MAX_FILES,
    Document,
    document_context,
    read_document,
    read_documents,
)
from textlab.features.chat.exports import (
    format_chat_history,
    format_chat_history_html,
    has_analysis_plots,
)
from textlab.features.chat.generation import (
    Piece,
    get_chunk_answer,
    get_combined_answer,
    stream_reply,
    stream_synthesis,
)
from textlab.features.chat.router import decide_tool_use
from textlab.features.visualization import service as visualization

__all__ = [
    "ANALYSIS_TIMEOUT_SECONDS",
    "DOCUMENT_TYPES",
    "MAX_FILES",
    "MAX_FILE_MB",
    "AnalysisError",
    "ContextUsage",
    "DataTable",
    "Document",
    "Draft",
    "ModelOptions",
    "Piece",
    "Reply",
    "answer",
    "context_usage",
    "conversation_folder",
    "decide_tool_use",
    "discard_conversation_folder",
    "discard_table",
    "document_context",
    "format_chat_history",
    "format_chat_history_html",
    "has_analysis_plots",
    "is_model_loaded",
    "is_table",
    "keep_table",
    "model_installed",
    "model_options",
    "plain_messages",
    "prepare_data_file",
    "pull_model",
    "read_document",
    "read_documents",
    "render_figure",
    "reply",
    "respond",
    "server_ready",
    "start_data_analysis",
    "tool_label",
]

#: Workspace area for conversation folders.
WORKSPACE_AREA = "chat"

#: Seconds before a data analysis is stopped.
ANALYSIS_TIMEOUT_SECONDS = visualization.ANALYSIS_TIMEOUT_SECONDS

#: Seconds between two looks at a running data analysis.
ANALYSIS_POLL = 0.5

# Re-exported for interfaces
is_table = visualization.is_table
prepare_data_file = visualization.prepare_data_file
tool_label = visualization.tool_label


def conversation_folder() -> str:
    """Create a private folder for a conversation's table and charts.

    Returns:
        The folder, in the workspace (area ``chat``); remove it with
        :func:`discard_conversation_folder` when the conversation ends.
    """
    return str(get_workspace().make_temp_dir(WORKSPACE_AREA, "conversation-"))


def discard_conversation_folder(folder: str) -> None:
    """Remove a conversation's folder and everything in it."""
    remove_tree(Path(folder))


def plain_messages(messages: Sequence[dict[str, Any]]) -> list[dict]:
    """Return messages with only ``role`` and ``content``, for the model."""
    return [{"role": m["role"], "content": m["content"]} for m in messages]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def server_ready(attempts: int = 1) -> bool:
    """Return whether the session's Ollama server accepts connections.

    Args:
        attempts: How many times to try, half a second apart; the server
            may still be starting when a session begins.

    Returns:
        True if it does.
    """
    return check_ollama_server(attempts=attempts)


# ---------------------------------------------------------------------------
# Answering with the chat model
# ---------------------------------------------------------------------------


def reply(
    model: str,
    history: Sequence[dict[str, Any]],
    question: str,
    context: str = "",
    *,
    on_progress: ProgressCallback = no_progress,
    cancel: threading.Event | None = None,
) -> Iterator[Piece]:
    """Reply to a message with the chat model, with attachments as context.

    Context that fits the model's context window is put in front of the
    question; earlier turns that no longer fit are left out, oldest first.
    Longer context is split into parts that are answered one by one
    (reported to ``on_progress``); the partial answers are then combined
    into the streamed answer.

    Args:
        model: The Ollama model.
        history: The conversation before this message.
        question: The user's message.
        context: The attached documents' text (:func:`document_context`).
        on_progress: Receives an update per part of a long context.
        cancel: Set to stop the reply.

    Returns:
        The reply, streamed in pieces of thinking and of answer. Model
        errors (``ollama.ResponseError``) and
        :class:`~textlab.common.progress.CancelledError` are raised while
        it is consumed.
    """
    history = plain_messages(history)
    prompt = f"{context}\n\nUser Question: {question}" if context else question
    room = text_budget() - estimate_tokens(prompt)
    if context and room < 0:
        return _reply_in_parts(
            model, history, question, context, on_progress, cancel
        )

    messages = [
        *trim_history(history, room),
        {"role": "user", "content": prompt},
    ]
    return stream_reply(model, messages, chat_context(), cancel)


def answer(
    model: str,
    history: Sequence[dict[str, Any]],
    question: str,
    context: str = "",
    *,
    on_progress: ProgressCallback = no_progress,
) -> Iterator[str]:
    """Answer a message, with attached documents as context.

    The text of :func:`reply`, without the model's thinking.

    Args:
        model: The Ollama model.
        history: The conversation before this message.
        question: The user's message.
        context: The attached documents' text (:func:`read_documents`).
        on_progress: Receives an update per part of a long context.

    Returns:
        The answer, streamed in pieces. Model errors are raised while it is
        consumed (``ollama.ResponseError``).
    """
    pieces = reply(model, history, question, context, on_progress=on_progress)
    return (piece.text for piece in pieces if not piece.thinking)


def _reply_in_parts(
    model: str,
    history: list[dict[str, Any]],
    question: str,
    context: str,
    on_progress: ProgressCallback,
    cancel: threading.Event | None,
) -> Iterator[Piece]:
    """Reply from a context too long for one request, part by part.

    Each part, with the question and the most recent turns, fills one
    request. Partial answers that do not fit in one request together are
    combined in groups first.
    """
    budget = text_budget()
    history = trim_history(history, int(budget * HISTORY_SHARE))
    fixed_tokens = message_tokens(history) + estimate_tokens(question)
    num_ctx = chat_context()

    # A very long question leaves little room; parts then stay usable but
    # may be cut by Ollama.
    chunk_tokens = max(budget - fixed_tokens, budget // 4)
    chunks = chunk_text(context, chunk_size_tokens=chunk_tokens)
    total_tokens = estimate_tokens(context)
    partial_answers = []
    for index, chunk in enumerate(chunks, 1):
        on_progress(
            Progress(
                f"Analyzing document part {index} of {len(chunks)} "
                f"(~{total_tokens:,} tokens total)...",
                (index - 1) / len(chunks),
            )
        )
        partial_answers.append(
            get_chunk_answer(
                model,
                chunk,
                index,
                len(chunks),
                question,
                history,
                num_ctx,
                cancel=cancel,
            )
        )

    partial_answers = _combine_until_fitting(
        model, partial_answers, question, history, on_progress, cancel
    )
    on_progress(
        Progress(f"Synthesizing responses from {len(chunks)} chunks...")
    )
    yield from stream_synthesis(
        model, partial_answers, question, history, num_ctx, cancel=cancel
    )


def _combine_until_fitting(
    model: str,
    partial_answers: list[str],
    question: str,
    history: list[dict[str, Any]],
    on_progress: ProgressCallback,
    cancel: threading.Event | None = None,
) -> list[str]:
    """Combine consecutive partial answers until all fit in one request.

    Only needed for very long documents, whose partial answers do not fit
    in the context window together.

    Returns:
        The answers, combined as far as needed.
    """
    budget = (
        text_budget() - message_tokens(history) - estimate_tokens(question)
    )
    answers = list(partial_answers)
    while sum(estimate_tokens(answer) for answer in answers) > budget:
        groups = group_by_budget(answers, budget)
        if len(groups) == len(answers):
            break  # no two answers fit together; combining cannot help
        on_progress(Progress(f"Combining {len(answers)} partial answers..."))
        answers = [
            group[0]
            if len(group) == 1
            else get_combined_answer(
                model, group, question, history, chat_context(), cancel=cancel
            )
            for group in groups
        ]
    return answers


# ---------------------------------------------------------------------------
# Attached tables and the data agents
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DataTable:
    """An attached table, kept on disk for the data agents.

    Attributes:
        name: The uploaded file's name.
        path: The saved file, in the conversation's folder.
        schema: A summary of its columns, for the model's prompt.
    """

    name: str
    path: str
    schema: str


def keep_table(name: str, data: bytes, folder: str) -> DataTable:
    """Save an attached table for the data agents, and summarize it.

    Each table gets a folder of its own in the conversation's folder: the
    file keeps a fixed name there, and the agents write their charts next
    to it.

    Args:
        name: The uploaded file's name (:func:`is_table` is true for it).
        data: Its content.
        folder: The conversation's folder (:func:`conversation_folder`).

    Returns:
        The table.
    """
    own = tempfile.mkdtemp(prefix="table-", dir=folder)
    path, schema = prepare_data_file(name, data, own)
    return DataTable(name=name, path=path, schema=schema)


def discard_table(table: DataTable) -> None:
    """Remove a kept table with the charts made from it."""
    remove_tree(Path(table.path).parent)


def start_data_analysis(
    instruction: str, data_path: str, model: str
) -> visualization.AnalysisRun:
    """Answer a data question with the Visualization feature's agents.

    Args:
        instruction: What to analyze, from :func:`decide_tool_use`.
        data_path: The table, saved with :func:`prepare_data_file`.
        model: The Ollama model.

    Returns:
        The run, going in the background; poll it, and put its result's
        ``to_payload()`` in the assistant's message.
    """
    request = visualization.AnalysisRequest(model=model, prompt=instruction)
    return visualization.start_analysis(
        request, data_path=data_path, run_prefix="chat"
    )


def render_figure(artifact: dict[str, Any]) -> Any:
    """Return the Plotly figure of a chart in a chat message, or ``None``."""
    figure_json = artifact.get("fig_json")
    if not figure_json:
        return None
    import plotly.io as pio

    return pio.from_json(figure_json)


# ---------------------------------------------------------------------------
# Replying to a message
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Draft:
    """A reply while it is being written, for an interface to show.

    Attributes:
        thinking: The model's thinking so far.
        text: The answer so far.
        activity: The data agents' activity so far, as ``(level,
            message)`` pairs, when they answer.
    """

    thinking: str = ""
    text: str = ""
    activity: tuple[tuple[str, str], ...] = ()


@dataclasses.dataclass(frozen=True)
class Reply:
    """The finished reply to one message.

    Attributes:
        model: The model that wrote it.
        text: The answer, in Markdown; for a data analysis, the agents'
            summary.
        thinking: The model's thinking before it answered, if it showed
            any.
        analysis: The data agents' charts and statistics
            (``AnalysisResult.to_payload()``), when they answered.
        activity: The data agents' activity, as ``(level, message)`` pairs.
        stopped: Whether the user stopped the reply; the text is what was
            written until then.
        seconds: How long the reply took.
        thinking_seconds: How long the model thought before answering.
    """

    model: str
    text: str
    thinking: str = ""
    analysis: dict[str, Any] | None = None
    activity: tuple[tuple[str, str], ...] = ()
    stopped: bool = False
    seconds: float = 0.0
    thinking_seconds: float = 0.0

    def message(self) -> dict[str, Any]:
        """Return the reply as the assistant's message in a conversation."""
        message: dict[str, Any] = {"role": "assistant", "content": self.text}
        if self.analysis:
            message["analysis"] = self.analysis
        return message


class AnalysisError(RuntimeError):
    """The data agents could not answer; the message is for users."""


def respond(
    model: str,
    history: Sequence[dict[str, Any]],
    question: str,
    *,
    context: str = "",
    table: DataTable | None = None,
    on_progress: ProgressCallback = no_progress,
    on_draft: Callable[[Draft], None] | None = None,
    cancel: threading.Event | None = None,
) -> Reply:
    """Reply to a message: with the data agents, or with the chat model.

    The model is downloaded first if the server does not have it. With a
    table attached, the model decides whether the message needs charts or
    statistics (:func:`decide_tool_use`); if it does, the Visualization
    feature's agents answer. Otherwise the chat model answers
    (:func:`reply`), with the attachments' text as context.

    Args:
        model: The Ollama model.
        history: The conversation before this message.
        question: The user's message.
        context: The attached documents' text (:func:`document_context`).
        table: The attached table the agents may analyze, if any.
        on_progress: Receives what happens while nothing is written yet:
            downloading or loading the model, reading long documents in
            parts, the agents' steps.
        on_draft: Receives the reply so far, after every piece.
        cancel: Set to stop. What was written until then is kept, and the
            reply is returned with ``stopped`` set.

    Returns:
        The reply.

    Raises:
        AnalysisError: If the data agents fail or take too long.
        ollama.ResponseError: If the model server refuses the request.
    """
    started = time.monotonic()
    _pull_if_missing(model, on_progress=on_progress)
    if table is not None and not _stopped(cancel):
        on_progress(Progress("Deciding whether to use the data tools"))
        use_tools, instruction = decide_tool_use(
            model,
            question,
            table.schema,
            chat_history=plain_messages(history),
        )
        if use_tools and not _stopped(cancel):
            return _analyze(
                model,
                instruction or question,
                table,
                started=started,
                on_progress=on_progress,
                on_draft=on_draft,
                cancel=cancel,
            )
    if _stopped(cancel):
        return Reply(
            model, "", stopped=True, seconds=time.monotonic() - started
        )

    if is_model_loaded(model):
        on_progress(Progress("Waiting for the model"))
    else:
        on_progress(
            Progress(
                f"Loading {model} into the GPU's memory; the first answer "
                "of a model takes a minute or two"
            )
        )
    thinking, text = "", ""
    thinking_started = thinking_ended = None
    stopped = False
    try:
        for piece in reply(
            model,
            history,
            question,
            context,
            on_progress=on_progress,
            cancel=cancel,
        ):
            if piece.thinking:
                thinking_started = thinking_started or time.monotonic()
                thinking += piece.text
            else:
                if thinking_started and thinking_ended is None:
                    thinking_ended = time.monotonic()
                text += piece.text
            if on_draft is not None:
                on_draft(Draft(thinking=thinking, text=text))
    except CancelledError:
        stopped = True
    ended = time.monotonic()
    thought = (
        (thinking_ended or ended) - thinking_started
        if thinking_started
        else 0.0
    )
    return Reply(
        model,
        text.strip(),
        thinking=thinking.strip(),
        stopped=stopped,
        seconds=ended - started,
        thinking_seconds=thought,
    )


def _pull_if_missing(
    model: str, *, on_progress: ProgressCallback = no_progress
) -> None:
    """Download a model if the Ollama server does not have it.

    Args:
        model: The model.
        on_progress: Receives the download's progress.

    Raises:
        ollama.ResponseError: If the download fails.
    """
    if not model_installed(model):
        pull_model(model, on_progress=on_progress)


def _stopped(cancel: threading.Event | None) -> bool:
    """Return whether ``cancel`` is set."""
    return cancel is not None and cancel.is_set()


def _analyze(
    model: str,
    instruction: str,
    table: DataTable,
    *,
    started: float,
    on_progress: ProgressCallback,
    on_draft: Callable[[Draft], None] | None,
    cancel: threading.Event | None,
) -> Reply:
    """Answer with the data agents, following their run until it ends."""
    run = start_data_analysis(instruction, table.path, model)
    on_progress(Progress("The data agents are starting"))
    seen = 0
    stopping = False
    while run.running:
        if _stopped(cancel) and not stopping:
            run.cancel()
            stopping = True
            on_progress(Progress("Stopping after the agents' current step"))
        activity = tuple(run.logs)
        if len(activity) != seen:
            seen = len(activity)
            if on_draft is not None:
                on_draft(Draft(activity=activity))
            if not stopping:
                on_progress(Progress(activity[-1][1]))
        time.sleep(ANALYSIS_POLL)

    activity = tuple(run.logs)
    seconds = time.monotonic() - started
    if run.status == visualization.TIMEOUT:
        minutes = ANALYSIS_TIMEOUT_SECONDS // 60
        raise AnalysisError(
            f"The data analysis took longer than {minutes} minutes and was "
            "stopped. Ask for less at once, or use a smaller table."
        )
    if run.status == visualization.ERROR:
        raise AnalysisError(f"The data analysis failed: {run.error or ''}")
    result = run.result
    stopped = stopping or run.status == visualization.CANCELLED
    if result is None:
        return Reply(
            model, "", activity=activity, stopped=stopped, seconds=seconds
        )
    return Reply(
        model,
        result.summary or "Analysis complete.",
        analysis=result.to_payload(),
        activity=activity,
        stopped=stopped,
        seconds=seconds,
    )
