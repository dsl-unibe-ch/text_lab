"""Answering a chat message with an Ollama model, streamed.

A reply streams as :class:`Piece` objects: the model's thinking (reasoning
models think before they answer) and the answer's text, so an interface
can show the two apart. A long attachment that does not fit the model's
context is answered part by part, and the partial answers are combined in
a final, streamed answer.

Every request passes ``num_ctx`` (see :mod:`.context_window`): without it,
Ollama uses its small default window and silently cuts longer requests.

Every function takes a ``cancel`` event. Once it is set, the reply stops
within :data:`CANCEL_POLL` seconds, even while the model is still loading,
and :class:`~textlab.common.progress.CancelledError` is raised; the request
to the model is closed as soon as its next piece arrives, which makes
Ollama stop writing.
"""

from __future__ import annotations

import dataclasses
import queue
import threading
from collections.abc import Iterable, Iterator
from typing import Any

import ollama

from textlab.common.ollama import message_text, message_thinking
from textlab.common.progress import CancelledError

# --- System Prompt ---
SYSTEM_PROMPT = (
    "You are a helpful, knowledgeable, and concise AI assistant. "
    "Answer questions accurately and honestly. "
    "If you are unsure about something, say so rather than guessing. "
    "When analysing documents or data provided by the user, focus on the "
    "content given "
    "and clearly indicate when you are drawing on your own knowledge versus "
    "the provided material."
)

#: Seconds between two looks at the ``cancel`` event while waiting for the
#: model.
CANCEL_POLL = 0.25


@dataclasses.dataclass(frozen=True)
class Piece:
    """A piece of a streamed reply.

    Attributes:
        text: The text of the piece.
        thinking: Whether it is part of the model's thinking rather than
            of the answer.
    """

    text: str
    thinking: bool = False


def _with_system(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return the messages after the system prompt."""
    return [{"role": "system", "content": SYSTEM_PROMPT}, *messages]


def _until_cancelled(
    stream: Iterable[Any], cancel: threading.Event | None
) -> Iterator[Any]:
    """Yield a stream's items until it ends or ``cancel`` is set.

    The stream is read in a thread of its own, so a cancel takes effect
    while the stream waits for its next item, such as while the model
    loads. That thread closes the stream when its next item arrives.

    Args:
        stream: The stream, such as the chunks of a streamed chat request.
        cancel: Set to stop; ``None`` reads the stream as it is.

    Yields:
        The stream's items.

    Raises:
        CancelledError: Once ``cancel`` is set.
        Exception: Whatever reading the stream raised.
    """
    if cancel is None:
        yield from stream
        return
    items: queue.Queue[tuple[str, Any]] = queue.Queue()
    stop = threading.Event()

    def pump() -> None:
        iterator = iter(stream)
        try:
            for item in iterator:
                if stop.is_set():
                    break
                items.put(("item", item))
        except Exception as error:
            items.put(("error", error))
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
            items.put(("end", None))

    threading.Thread(
        target=pump, name="textlab-chat-stream", daemon=True
    ).start()
    try:
        while True:
            try:
                kind, value = items.get(timeout=CANCEL_POLL)
            except queue.Empty:
                if cancel.is_set():
                    raise CancelledError("The reply was stopped.") from None
                continue
            if kind == "end":
                return
            if kind == "error":
                raise value
            yield value
            if cancel.is_set():
                raise CancelledError("The reply was stopped.")
    finally:
        stop.set()


def _pieces(
    model_name: str,
    messages: list[dict[str, str]],
    num_ctx: int,
    cancel: threading.Event | None = None,
) -> Iterator[Piece]:
    """Stream a reply's thinking and text from the Ollama chat endpoint."""
    stream = ollama.chat(
        model=model_name,
        messages=_with_system(messages),
        stream=True,
        options={"num_ctx": num_ctx},
    )
    for chunk in _until_cancelled(stream, cancel):
        message = (
            chunk["message"] if isinstance(chunk, dict) else chunk.message
        )
        thinking = message_thinking(message)
        if thinking:
            yield Piece(thinking, thinking=True)
        text = message_text(message)
        if text:
            yield Piece(text)


def _complete(
    model_name: str,
    messages: list[dict[str, str]],
    num_ctx: int,
    cancel: threading.Event | None = None,
) -> str:
    """Return a whole reply's text, without its thinking."""
    return "".join(
        piece.text
        for piece in _pieces(model_name, messages, num_ctx, cancel)
        if not piece.thinking
    )


def stream_reply(
    model_name: str,
    messages: list[dict[str, str]],
    num_ctx: int,
    cancel: threading.Event | None = None,
) -> Iterator[Piece]:
    """Stream a reply from the Ollama chat endpoint.

    Args:
        model_name (str): The language model to use.
        messages (list): The conversation history.
        num_ctx (int): Context window in tokens; it must hold the whole
            request and the answer, or Ollama drops part of the request.
        cancel: Set to stop the reply.

    Yields:
        Piece: The thinking and the text, in pieces as the model writes.

    Raises:
        CancelledError: Once ``cancel`` is set.
    """
    yield from _pieces(model_name, messages, num_ctx, cancel)


def get_chunk_answer(
    model_name: str,
    chunk: str,
    chunk_index: int,
    total_chunks: int,
    user_question: str,
    chat_history: list[dict[str, str]],
    num_ctx: int,
    cancel: threading.Event | None = None,
) -> str:
    """Answer a question from one part of a document (not streamed).

    Args:
        model_name (str): The language model to use.
        chunk (str): The document fragment for this call.
        chunk_index (int): 1-based index of this chunk.
        total_chunks (int): Total number of chunks.
        user_question (str): The user's original question.
        chat_history (list): Conversation history *excluding* the current user
            turn.
        num_ctx (int): Context window in tokens.
        cancel: Set to stop.

    Returns:
        str: The model's partial answer for this chunk.

    Raises:
        CancelledError: Once ``cancel`` is set.
    """
    chunk_prompt = (
        f"You are analyzing part {chunk_index} of {total_chunks} of a "
        "document.\n\n"
        f"--- Document Part {chunk_index}/{total_chunks} ---\n{chunk}\n"
        f"--- End of Part {chunk_index}/{total_chunks} ---\n\n"
        "Based only on the content above, provide a partial answer "
        f"to:\n{user_question}"
    )
    messages = chat_history + [{"role": "user", "content": chunk_prompt}]
    return _complete(model_name, messages, num_ctx, cancel)


def _synthesis_messages(
    partial_answers: list[str],
    user_question: str,
    chat_history: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Return the request that combines partial answers into one."""
    parts = "\n\n".join(
        f"--- Answer from Part {i + 1} ---\n{ans}"
        for i, ans in enumerate(partial_answers)
    )
    synthesis_prompt = (
        f"A long document was split into {len(partial_answers)} parts. "
        f'Each part was analyzed separately to answer: "{user_question}"\n\n'
        f"{parts}\n\n"
        f"--- End of Partial Answers ---\n\n"
        "Now synthesize all of the above into one comprehensive, "
        "well-structured final answer."
    )
    return chat_history + [{"role": "user", "content": synthesis_prompt}]


def get_combined_answer(
    model_name: str,
    partial_answers: list[str],
    user_question: str,
    chat_history: list[dict[str, str]],
    num_ctx: int,
    cancel: threading.Event | None = None,
) -> str:
    """Combine some partial answers into one (not streamed).

    Used when the partial answers of a very long document do not fit in
    one request together: consecutive answers are combined first.

    Args:
        model_name (str): The language model to use.
        partial_answers (list): Consecutive partial answers.
        user_question (str): The user's original question.
        chat_history (list): Conversation history *excluding* the current user
            turn.
        num_ctx (int): Context window in tokens.
        cancel: Set to stop.

    Returns:
        str: The combined answer.

    Raises:
        CancelledError: Once ``cancel`` is set.
    """
    messages = _synthesis_messages(
        partial_answers, user_question, chat_history
    )
    return _complete(model_name, messages, num_ctx, cancel)


def stream_synthesis(
    model_name: str,
    partial_answers: list[str],
    user_question: str,
    chat_history: list[dict[str, str]],
    num_ctx: int,
    cancel: threading.Event | None = None,
) -> Iterator[Piece]:
    """Stream one answer that combines the answers of every part.

    Args:
        model_name (str): The language model to use.
        partial_answers (list): Collected answers from each chunk.
        user_question (str): The user's original question.
        chat_history (list): Conversation history *excluding* the current user
            turn.
        num_ctx (int): Context window in tokens.
        cancel: Set to stop.

    Yields:
        Piece: The thinking and the text of the synthesized answer.

    Raises:
        CancelledError: Once ``cancel`` is set.
    """
    messages = _synthesis_messages(
        partial_answers, user_question, chat_history
    )
    yield from _pieces(model_name, messages, num_ctx, cancel)
