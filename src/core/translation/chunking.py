"""Lossless translation boundaries, budgets, and truncation errors.

Token counting belongs to the backend: Hugging Face counts with its loaded
source tokenizer, including language prefixes and special tokens. Splitting
never decodes token slices or cuts a word to satisfy a budget.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Sequence
import re


MAX_SPLIT_RETRIES = 4


class TranslationLimitError(ValueError):
    """Translation cannot complete safely within a model's token budget."""


class InputTooLongError(TranslationLimitError):
    """An indivisible source span exceeds the encoder's input budget."""


class OutputTruncatedError(TranslationLimitError):
    """Generation exhausted its budget without completing the translation."""


# Whitespace is a word boundary; CJK sentence endings need no following space.
_BOUNDARY_RE = re.compile(r"\s+|[。！？]+[\"'”’）】]*")
_SENTENCE_RE = re.compile(r"[.!?][\"'”’)\]]*\s+|[。！？]+[\"'”’）】]*\s*")


def split_into_sentences(text: str) -> list[str]:
    """Split on conservative sentence boundaries without cutting words."""
    ends = [match.end() for match in _SENTENCE_RE.finditer(text)]
    ends.append(len(text))
    start = 0
    result = []
    for end in ends:
        if part := text[start:end].strip():
            result.append(part)
        start = end
    return result


def split_text(
    text: str,
    measure: Callable[[str], int],
    budget: int,
) -> list[str]:
    """Return exact source slices, each fitting ``measure(slice) <= budget``.

    Prefer sentence endings, then whitespace. Exponential probing followed
    by binary search avoids tokenizing an entire large document per chunk.
    An oversized word or unsegmented sentence fails rather than losing text.
    """
    if budget <= 0:
        raise ValueError("The translation input budget must be positive.")
    if not text or not text.strip():
        return [text] if text else []
    if measure(text) <= budget:
        return [text]

    boundaries = sorted({
        *(match.end() for match in _BOUNDARY_RE.finditer(text)), len(text),
    })
    sentences = [match.end() for match in _SENTENCE_RE.finditer(text)]
    chunks = []
    start = 0
    while start < len(text):
        first = bisect_right(boundaries, start)
        if measure(text[start:boundaries[first]]) > budget:
            raise InputTooLongError(
                "A word or unsegmented source span exceeds the model's "
                f"input budget ({budget}). Add a word/sentence boundary "
                "or choose a backend with a larger input window. "
                "No truncated translation was returned."
            )
        best = first
        step = 1
        upper = first + step
        while upper < len(boundaries):
            if measure(text[start:boundaries[upper]]) > budget:
                break
            best = upper
            step *= 2
            upper = first + step
        low, high = best + 1, min(upper, len(boundaries) - 1)
        while low <= high:
            middle = (low + high) // 2
            if measure(text[start:boundaries[middle]]) <= budget:
                best = middle
                low = middle + 1
            else:
                high = middle - 1

        end = boundaries[best]
        if end < len(text):
            sentence_index = bisect_right(sentences, end) - 1
            if sentence_index >= 0 and sentences[sentence_index] > start:
                candidate = sentences[sentence_index]
                if measure(text[start:candidate]) <= budget:
                    end = candidate
        chunk = text[start:end]
        # Token counts need not be strictly monotonic for every tokenizer.
        if measure(chunk) > budget:
            raise InputTooLongError("Source chunk exceeds the input budget.")
        chunks.append(chunk)
        start = end
    return chunks


def split_for_retry(text: str) -> list[str]:
    """Bisect a failed chunk at a word/sentence boundary, never mid-word."""
    ends = [
        match.end() for match in _BOUNDARY_RE.finditer(text)
        if text[:match.end()].strip() and text[match.end():].strip()
    ]
    if not ends:
        raise OutputTruncatedError(
            "The model reached its output-token limit for an indivisible "
            "source span. Choose another backend or shorten the source. "
            "No partial translation was returned."
        )
    end = min(ends, key=lambda value: abs(value - len(text) / 2))
    return [text[:end], text[end:]]


def join_translations(
    sources: Sequence[str], translations: Sequence[str],
) -> str:
    """Reassemble output with the whitespace at the original boundaries."""
    if len(sources) != len(translations):
        raise TranslationLimitError(
            "The backend did not return one translation per source chunk."
        )
    parts = []
    for source, translation in zip(sources, translations):
        leading = source[:len(source) - len(source.lstrip())]
        trailing = source[len(source.rstrip()):]
        parts.append(leading + translation.strip() + trailing)
    return "".join(parts)


def translate_lines(
    texts: Sequence[str], translate_parts: Callable[[list[str]], list[str]],
) -> list[str]:
    """Batch nonblank lines across inputs while preserving blank lines."""
    layouts = [re.split(r"(\n+)", text) for text in texts]
    sources = [part for parts in layouts for part in parts if part.strip()]
    if not sources:
        return list(texts)
    outputs = translate_parts(sources)
    if len(outputs) != len(sources):
        raise TranslationLimitError(
            "The backend returned an incomplete set of translations."
        )
    translated = iter(outputs)
    return [
        "".join(next(translated) if part.strip() else part for part in parts)
        for parts in layouts
    ]


def chunk_text_for_translation(
    text: str,
    max_chars: int = 1200,
    *,
    measure: Callable[[str], int] | None = None,
    max_tokens: int | None = None,
) -> list[str]:
    """Compatibility helper; inference backends supply their own counter.

    The legacy character budget remains available to callers, but no longer
    hard-splits words. Model inference uses ``measure`` and ``max_tokens``.
    """
    if measure is not None:
        if max_tokens is None:
            raise ValueError("A token counter requires max_tokens.")
        return split_text(text, measure, max_tokens)
    return split_text(text, len, max_chars)
