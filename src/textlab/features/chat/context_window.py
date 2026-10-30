"""Fitting chat requests into the model's context window.

Ollama answers from a context window of ``num_ctx`` tokens. Without the
option it uses its own default, often only a few thousand tokens, and a
request that does not fit is cut silently: Ollama keeps the end of it, so a
model asked to summarize a long document would see only its last pages.

Every chat request therefore sets ``num_ctx`` to :func:`chat_context`, and
callers keep requests within :func:`text_budget`: earlier turns are dropped
oldest first (:func:`trim_history`) and documents that are too long are
split. All chat requests use the same window, because Ollama reloads the
model whenever ``num_ctx`` changes.

:func:`context_usage` reports how full the window is, so that interfaces
can show it and suggest a new chat before the conversation gets too long.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from textlab.common.ollama import estimate_tokens

#: Context window (tokens) of every chat request, unless
#: ``TEXTLAB_CHAT_NUM_CTX`` sets another.
DEFAULT_CHAT_CONTEXT = 32768
#: Smallest window accepted from ``TEXTLAB_CHAT_NUM_CTX``.
MIN_CHAT_CONTEXT = 8192
#: Tokens kept free for the model's answer.
RESPONSE_TOKENS = 4096
#: Token estimates assume four characters per token, which undercounts many
#: languages other than English, so estimates are multiplied by this factor.
ESTIMATE_MARGIN = 1.3
#: Tokens of instructions around the text (system prompt, part headers).
PROMPT_OVERHEAD_TOKENS = 1000
#: Share of the budget earlier turns may use while a long document is
#: answered part by part; the rest is left for the document.
HISTORY_SHARE = 0.25
#: Share of the budget from which the window counts as nearly full. Models
#: answer less reliably as the window fills, well before it is full.
NEARLY_FULL_SHARE = 0.75


def chat_context() -> int:
    """Return the context window, in tokens, of every chat request.

    Returns:
        ``TEXTLAB_CHAT_NUM_CTX`` if set to a number, otherwise
        :data:`DEFAULT_CHAT_CONTEXT`; at least :data:`MIN_CHAT_CONTEXT`.
    """
    raw = os.environ.get("TEXTLAB_CHAT_NUM_CTX", "").strip()
    try:
        value = int(raw) if raw else DEFAULT_CHAT_CONTEXT
    except ValueError:
        value = DEFAULT_CHAT_CONTEXT
    return max(value, MIN_CHAT_CONTEXT)


def text_budget() -> int:
    """Return how many estimated tokens of text one request can hold.

    The text is everything the caller sends: earlier turns, documents and
    the question. Room for the answer, the estimate margin and the prompt
    overhead are already taken off.
    """
    usable = (chat_context() - RESPONSE_TOKENS) / ESTIMATE_MARGIN
    return int(usable) - PROMPT_OVERHEAD_TOKENS


def message_tokens(messages: Sequence[dict[str, Any]]) -> int:
    """Return the estimated tokens of the messages' contents."""
    return sum(estimate_tokens(m.get("content") or "") for m in messages)


def trim_history(
    history: Sequence[dict[str, Any]], budget: int
) -> list[dict[str, Any]]:
    """Keep the most recent messages that fit a token budget.

    Args:
        history: The conversation, oldest first.
        budget: Token budget for the kept messages.

    Returns:
        The newest messages whose total fits ``budget``, oldest first;
        empty if not even the last message fits.
    """
    kept: list[dict[str, Any]] = []
    used = 0
    for message in reversed(history):
        used += message_tokens([message])
        if used > budget:
            break
        kept.append(message)
    kept.reverse()
    return kept


@dataclass(frozen=True)
class ContextUsage:
    """How much of the context window the next message would use.

    Attributes:
        used_tokens: Estimated tokens of the conversation and the attached
            documents, which are sent again with every message.
        budget_tokens: Tokens of text one request can hold
            (:func:`text_budget`).
        window_tokens: The context window (:func:`chat_context`).
        document_tokens: Estimated tokens of the attached documents.
    """

    used_tokens: int
    budget_tokens: int
    window_tokens: int
    document_tokens: int

    @property
    def fraction(self) -> float:
        """Return the used share of the budget; above 1 when it is full."""
        return self.used_tokens / self.budget_tokens

    @property
    def history_tokens(self) -> int:
        """Return the estimated tokens of the conversation alone."""
        return self.used_tokens - self.document_tokens

    @property
    def documents_split(self) -> bool:
        """Return True if the documents are too long for one request."""
        return self.document_tokens > self.budget_tokens

    @property
    def history_trimmed(self) -> bool:
        """Return True if the oldest messages no longer reach the model.

        Mirrors the answer: with documents read in parts, the conversation
        may use :data:`HISTORY_SHARE` of the budget, otherwise what the
        documents leave free.
        """
        if self.documents_split:
            return self.history_tokens > self.budget_tokens * HISTORY_SHARE
        return self.used_tokens > self.budget_tokens

    @property
    def suggest_new_chat(self) -> bool:
        """Return True if a new chat would give the model more room.

        That is the case when the conversation (not the documents, which a
        new chat sends again) fills the window from
        :data:`NEARLY_FULL_SHARE` on or pushes old messages out.
        """
        if self.history_tokens <= 0:
            return False
        if self.history_trimmed:
            return True
        return not self.documents_split and (
            self.fraction >= NEARLY_FULL_SHARE
        )


def context_usage(
    history: Sequence[dict[str, Any]], context: str = ""
) -> ContextUsage:
    """Estimate how much of the context window the next message uses.

    Args:
        history: The conversation so far.
        context: The attached documents' text, sent with every message.

    Returns:
        The usage; the new message itself is not counted.
    """
    document_tokens = estimate_tokens(context) if context else 0
    return ContextUsage(
        used_tokens=message_tokens(history) + document_tokens,
        budget_tokens=text_budget(),
        window_tokens=chat_context(),
        document_tokens=document_tokens,
    )


def group_by_budget(texts: Sequence[str], budget: int) -> list[list[str]]:
    """Group texts in order so that each group fits a token budget.

    A text larger than the budget forms a group of its own.

    Args:
        texts: The texts, e.g. partial answers.
        budget: Token budget per group.

    Returns:
        The groups, in order.
    """
    groups: list[list[str]] = []
    current: list[str] = []
    used = 0
    for text in texts:
        tokens = estimate_tokens(text)
        if current and used + tokens > budget:
            groups.append(current)
            current, used = [], 0
        current.append(text)
        used += tokens
    if current:
        groups.append(current)
    return groups
