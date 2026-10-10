"""Answering a chat message with an Ollama model, streamed.

A long attachment that does not fit the model's context is answered part by
part, and the partial answers are combined in a final, streamed answer.
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


def get_response_generator(
    model_name: str, messages: list[dict[str, str]]
) -> Generator[str, None, None]:
    """Stream responses from the Ollama chat endpoint.

    Args:
        model_name (str): The language model to use.
        messages (list): The conversation history.

    Yields:
        str: Incremental text chunks from the language model.
    """
    system_message = {"role": "system", "content": SYSTEM_PROMPT}
    stream = ollama.chat(
        model=model_name, messages=[system_message] + messages, stream=True
    )
    for chunk in stream:
        text = message_text(
            chunk["message"] if isinstance(chunk, dict) else chunk.message
        )
        if text:
            yield text


def get_chunk_answer(
    model_name: str,
    chunk: str,
    chunk_index: int,
    total_chunks: int,
    user_question: str,
    chat_history: list[dict[str, str]],
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
    system_message = {"role": "system", "content": SYSTEM_PROMPT}
    messages = (
        [system_message]
        + chat_history
        + [{"role": "user", "content": chunk_prompt}]
    )
    response = ollama.chat(model=model_name, messages=messages, stream=False)
    return message_text(
        response["message"] if isinstance(response, dict) else response.message
    )


def get_synthesis_generator(
    model_name: str,
    partial_answers: list[str],
    user_question: str,
    chat_history: list[dict[str, str]],
) -> Generator[str, None, None]:
    """Stream one answer that combines the answers of every part.

    Args:
        model_name (str): The language model to use.
        partial_answers (list): Collected answers from each chunk.
        user_question (str): The user's original question.
        chat_history (list): Conversation history *excluding* the current user
            turn.

    Yields:
        str: Incremental text chunks of the synthesized answer.
    """
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
    system_message = {"role": "system", "content": SYSTEM_PROMPT}
    messages = (
        [system_message]
        + chat_history
        + [{"role": "user", "content": synthesis_prompt}]
    )
    stream = ollama.chat(model=model_name, messages=messages, stream=True)
    for chunk in stream:
        text = message_text(
            chunk["message"] if isinstance(chunk, dict) else chunk.message
        )
        if text:
            yield text
