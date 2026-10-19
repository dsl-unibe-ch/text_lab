"""Answering a chat message with an Ollama model, streamed.

A long attachment that does not fit the model's context is answered part by
part, and the partial answers are combined in a final, streamed answer.

Every request passes ``num_ctx`` (see :mod:`.context_window`): without it,
Ollama uses its small default window and silently cuts longer requests.
"""

from __future__ import annotations

from collections.abc import Generator

import ollama

from textlab.common.ollama import message_text

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


def _with_system(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return the messages after the system prompt."""
    return [{"role": "system", "content": SYSTEM_PROMPT}, *messages]


def _stream(
    model_name: str, messages: list[dict[str, str]], num_ctx: int
) -> Generator[str, None, None]:
    """Stream a reply's text from the Ollama chat endpoint."""
    stream = ollama.chat(
        model=model_name,
        messages=_with_system(messages),
        stream=True,
        options={"num_ctx": num_ctx},
    )
    for chunk in stream:
        text = message_text(
            chunk["message"] if isinstance(chunk, dict) else chunk.message
        )
        if text:
            yield text


def _complete(
    model_name: str, messages: list[dict[str, str]], num_ctx: int
) -> str:
    """Return a whole reply from the Ollama chat endpoint (not streamed)."""
    response = ollama.chat(
        model=model_name,
        messages=_with_system(messages),
        stream=False,
        options={"num_ctx": num_ctx},
    )
    return message_text(
        response["message"] if isinstance(response, dict) else response.message
    )


def get_response_generator(
    model_name: str, messages: list[dict[str, str]], num_ctx: int
) -> Generator[str, None, None]:
    """Stream responses from the Ollama chat endpoint.

    Args:
        model_name (str): The language model to use.
        messages (list): The conversation history.
        num_ctx (int): Context window in tokens; it must hold the whole
            request and the answer, or Ollama drops part of the request.

    Yields:
        str: Incremental text chunks from the language model.
    """
    yield from _stream(model_name, messages, num_ctx)


def get_chunk_answer(
    model_name: str,
    chunk: str,
    chunk_index: int,
    total_chunks: int,
    user_question: str,
    chat_history: list[dict[str, str]],
    num_ctx: int,
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

    Returns:
        str: The model's partial answer for this chunk.
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
    return _complete(model_name, messages, num_ctx)


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

    Returns:
        str: The combined answer.
    """
    messages = _synthesis_messages(
        partial_answers, user_question, chat_history
    )
    return _complete(model_name, messages, num_ctx)


def get_synthesis_generator(
    model_name: str,
    partial_answers: list[str],
    user_question: str,
    chat_history: list[dict[str, str]],
    num_ctx: int,
) -> Generator[str, None, None]:
    """Stream one answer that combines the answers of every part.

    Args:
        model_name (str): The language model to use.
        partial_answers (list): Collected answers from each chunk.
        user_question (str): The user's original question.
        chat_history (list): Conversation history *excluding* the current user
            turn.
        num_ctx (int): Context window in tokens.

    Yields:
        str: Incremental text chunks of the synthesized answer.
    """
    messages = _synthesis_messages(
        partial_answers, user_question, chat_history
    )
    yield from _stream(model_name, messages, num_ctx)
