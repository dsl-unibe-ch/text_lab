"""Chatting with a local model about the user's files: the Chat API.

Interfaces call this module. A message is answered by the chat model
(:func:`answer`), with any attached documents as context
(:func:`read_documents`). With a table attached, :func:`decide_tool_use`
asks the model whether the message needs charts or statistics; if so,
:func:`start_data_analysis` runs the Visualization feature's agents on the
table instead. Conversations are exported with :func:`format_chat_history`
and :func:`format_chat_history_html`.

Messages are dictionaries with ``role`` and ``content``; an assistant turn
answered by the agents also has an ``analysis`` entry
(``visualization.models.AnalysisResult.to_payload``).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import ollama

from textlab.common.ollama import (
    MAX_CONTEXT_TOKENS,
    chunk_text,
    estimate_tokens,
    is_model_loaded,
    model_installed,
)
from textlab.common.progress import Progress, ProgressCallback, no_progress
from textlab.common.storage import get_workspace, remove_tree
from textlab.features.chat.documents import read_documents
from textlab.features.chat.exports import (
    format_chat_history,
    format_chat_history_html,
    has_analysis_plots,
)
from textlab.features.chat.generation import (
    get_chunk_answer,
    get_response_generator,
    get_synthesis_generator,
)
from textlab.features.chat.router import decide_tool_use
from textlab.features.visualization import service as visualization

__all__ = [
    "ANALYSIS_TIMEOUT_SECONDS",
    "answer",
    "conversation_folder",
    "decide_tool_use",
    "discard_conversation_folder",
    "format_chat_history",
    "format_chat_history_html",
    "has_analysis_plots",
    "is_model_loaded",
    "is_table",
    "model_installed",
    "plain_messages",
    "prepare_data_file",
    "pull_model",
    "read_documents",
    "render_figure",
    "start_data_analysis",
    "tool_label",
]

#: Workspace area for conversation folders.
WORKSPACE_AREA = "chat"

#: Seconds before a data analysis is stopped.
ANALYSIS_TIMEOUT_SECONDS = visualization.ANALYSIS_TIMEOUT_SECONDS

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


def answer(
    model: str,
    history: Sequence[dict[str, Any]],
    question: str,
    context: str = "",
    *,
    on_progress: ProgressCallback = no_progress,
) -> Iterator[str]:
    """Answer a message, with attached documents as context.

    Context that fits the model is put in front of the question. Longer
    context is split into parts that are answered one by one (reported to
    ``on_progress``); the partial answers are then combined into the
    streamed answer.

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
    history = plain_messages(history)
    if context and estimate_tokens(context) > MAX_CONTEXT_TOKENS:
        chunks = chunk_text(context)
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
                    model, chunk, index, len(chunks), question, history
                )
            )
        on_progress(
            Progress(f"Synthesizing responses from {len(chunks)} chunks...")
        )
        return get_synthesis_generator(
            model, partial_answers, question, history
        )

    prompt = f"{context}\n\nUser Question: {question}" if context else question
    return get_response_generator(
        model, [*history, {"role": "user", "content": prompt}]
    )


def pull_model(model: str) -> None:
    """Download a model to the Ollama server.

    Raises:
        ollama.ResponseError: If the download fails.
    """
    ollama.pull(model=model)


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
