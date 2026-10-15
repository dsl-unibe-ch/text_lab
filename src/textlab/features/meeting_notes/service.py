"""Summarizing transcripts into cited notes: the Meeting Notes feature's API.

A transcript is a list of numbered segments (:mod:`.transcript`). The model
sees each line with its number and cites the lines every point is based on;
:func:`cite_summary` then checks those citations against the transcript.

A transcript that fits in one request is summarized in one streamed request
(:func:`get_summary_stream`). A longer one is summarized map-reduce style:
:func:`get_partial_notes` extracts cited notes from each chunk, then
:func:`get_synthesis_stream` streams the final summary from them. Use
:func:`needs_chunking` to choose.

Every request sets Ollama's context window (``num_ctx``) to one that holds
the whole prompt and the answer. Without it, Ollama uses its default window
and silently drops whatever does not fit, so a summary of a long meeting
would be based on part of it. The largest window is
:data:`DEFAULT_MAX_CONTEXT` tokens, or ``TEXTLAB_SUMMARY_NUM_CTX``; longer
transcripts are chunked to fit it.

The summary's structure depends on the mode (:data:`SUMMARY_MODES`), e.g.
meeting notes with decisions and action items, or lecture notes.

Streaming functions are plain generators of text, so any interface can
display them as they arrive.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from typing import Any

import ollama

from textlab.common.ollama import call_no_think, estimate_tokens, message_text
from textlab.common.progress import Progress, ProgressCallback, no_progress
from textlab.features.meeting_notes.citations import (
    cite_summary,
    format_citations,
    mark_uncited_points,
    with_footnotes,
)
from textlab.features.meeting_notes.models import CitedSummary, Segment
from textlab.features.meeting_notes.transcript import (
    numbered_text,
    rename_speakers,
    segment_label,
    segments_from_table,
    segments_from_text,
    speakers,
    transcript_text,
    word_count,
)

__all__ = [
    # Data classes
    "CitedSummary",
    "Segment",
    # Summary modes
    "SUMMARY_MODES",
    # Summarization
    "get_partial_notes",
    "get_summary_stream",
    "get_synthesis_stream",
    "needs_chunking",
    # Transcripts
    "build_speaker_context",
    "rename_speakers",
    "segment_label",
    "segments_from_table",
    "segments_from_text",
    "speakers",
    "transcript_text",
    "word_count",
    # Citations and export
    "cite_summary",
    "format_citations",
    "format_summary_document",
    "mark_uncited_points",
    "with_footnotes",
]

# Low temperature keeps summaries close to the transcript instead of
# paraphrasing creatively or adding detail that was never said.
SUMMARY_TEMPERATURE = 0.2


# ---------------------------------------------------------------------------
# Context window
# ---------------------------------------------------------------------------

#: Largest context window (tokens) a request may use, unless
#: ``TEXTLAB_SUMMARY_NUM_CTX`` sets another. Longer transcripts are chunked.
DEFAULT_MAX_CONTEXT = 32768
#: Context windows a request may use, smallest first. A request uses the
#: smallest that fits; keeping to a few sizes means Ollama reloads the model
#: rarely (it reloads whenever ``num_ctx`` changes).
_CONTEXT_STEPS = (8192, 16384, 32768, 65536, 131072)
#: Tokens kept free for the model's answer.
RESPONSE_TOKENS = 4096
#: Token estimates assume four characters per token, which undercounts many
#: languages other than English, so estimates are multiplied by this factor.
ESTIMATE_MARGIN = 1.3
#: Tokens of instructions around the transcript, for planning chunks.
PROMPT_OVERHEAD_TOKENS = 1000


def max_context() -> int:
    """Return the largest context window a request may use.

    Returns:
        ``TEXTLAB_SUMMARY_NUM_CTX`` if set to a number, otherwise
        :data:`DEFAULT_MAX_CONTEXT`; at least the smallest step.
    """
    raw = os.environ.get("TEXTLAB_SUMMARY_NUM_CTX", "").strip()
    try:
        value = int(raw) if raw else DEFAULT_MAX_CONTEXT
    except ValueError:
        value = DEFAULT_MAX_CONTEXT
    return max(value, _CONTEXT_STEPS[0])


def _context_sizes() -> tuple[int, ...]:
    """Return the context windows a request may use, smallest first."""
    limit = max_context()
    return tuple(size for size in _CONTEXT_STEPS if size < limit) + (limit,)


def _fits(prompt_tokens: int, num_ctx: int) -> bool:
    """Return True if a prompt and an answer fit in a context window."""
    return prompt_tokens * ESTIMATE_MARGIN + RESPONSE_TOKENS <= num_ctx


def _context_for(messages: Sequence[dict[str, str]]) -> int:
    """Return the smallest context window that holds a request.

    Args:
        messages: The request's messages.

    Returns:
        The window size; the largest one if none is big enough.
    """
    tokens = sum(estimate_tokens(message["content"]) for message in messages)
    sizes = _context_sizes()
    return next((size for size in sizes if _fits(tokens, size)), sizes[-1])


def _text_budget() -> int:
    """Return how many tokens of transcript or notes one request can hold."""
    usable = (max_context() - RESPONSE_TOKENS) / ESTIMATE_MARGIN
    return int(usable) - PROMPT_OVERHEAD_TOKENS


def _group_by_budget(texts: Sequence[str], budget: int) -> list[list[str]]:
    """Group texts in order so that each group fits a token budget.

    A text larger than the budget forms a group of its own.

    Args:
        texts: The texts, e.g. transcript lines or notes.
        budget: Token budget per group.

    Returns:
        The groups, in order.
    """
    groups: list[list[str]] = []
    current: list[str] = []
    used = 0
    for text in texts:
        tokens = estimate_tokens(text) + 1  # +1 for the joining newline
        if current and used + tokens > budget:
            groups.append(current)
            current, used = [], 0
        current.append(text)
        used += tokens
    if current:
        groups.append(current)
    return groups


# ---------------------------------------------------------------------------
# Summary mode registry
# ---------------------------------------------------------------------------

SUMMARY_MODES: dict[str, dict[str, str]] = {
    "general": {
        "label": "General Summary",
        "description": (
            "A concise, well-structured overview of the main topics, "
            "key points, and conclusions."
        ),
        "chunk_instruction": (
            "Extract the main ideas, key statements, and important "
            "information "
            "from this section of the transcript. Be specific and factual."
        ),
        "synthesis_instruction": (
            "Using the extracted notes from all parts, produce a "
            "well-structured summary "
            "with: a short introduction paragraph, a bulleted list of key "
            "points, "
            "and a brief conclusion."
        ),
    },
    "meeting_notes": {
        "label": "Meeting Notes",
        "description": (
            "Structured notes capturing decisions made, action items, "
            "topics discussed, and participants mentioned."
        ),
        "chunk_instruction": (
            "From this section of the transcript extract: "
            "decisions made, action items (with owner if mentioned), "
            "topics discussed, and any names of participants."
        ),
        "synthesis_instruction": (
            "Produce structured meeting notes with these sections:\n"
            "**Summary** (2-3 sentences)\n"
            "**Key Decisions**\n"
            "**Action Items** (include owner and deadline if mentioned)\n"
            "**Topics Discussed**\n"
            "**Participants** (if names were mentioned)"
        ),
    },
    "interview": {
        "label": "Interview Summary",
        "description": (
            "Highlights the main themes, key responses, and notable "
            "statements "
            "from an interview or conversation."
        ),
        "chunk_instruction": (
            "From this section of the transcript identify: "
            "the main themes raised, notable answers or statements, "
            "and any direct quotes worth preserving verbatim."
        ),
        "synthesis_instruction": (
            "Produce an interview summary with these sections:\n"
            "**Overview** (what the conversation was about)\n"
            "**Main Themes**\n"
            "**Key Statements and Notable Quotes** (use speaker names where "
            "available)\n"
            "**Conclusions or Outcomes**"
        ),
    },
    "academic": {
        "label": "Academic Abstract",
        "description": (
            "A structured abstract covering research question, methodology, "
            "findings, and conclusions."
        ),
        "chunk_instruction": (
            "From this section extract: the research question or objective, "
            "methodology described, findings reported, and any conclusions "
            "drawn."
        ),
        "synthesis_instruction": (
            "Write a structured academic summary with these sections:\n"
            "**Background and Objective**\n"
            "**Methods**\n"
            "**Findings**\n"
            "**Conclusions and Implications**"
        ),
    },
    "lecture": {
        "label": "Lecture Notes",
        "description": (
            "Organized study notes covering key concepts, definitions, "
            "examples, and topic structure."
        ),
        "chunk_instruction": (
            "From this part of the lecture transcript extract: "
            "main topics introduced, key concepts and definitions, "
            "examples given, and any important terms or formulas."
        ),
        "synthesis_instruction": (
            "Produce structured lecture notes with these sections:\n"
            "**Lecture Overview**\n"
            "**Topics Covered** (with sub-points for each topic)\n"
            "**Key Concepts and Definitions**\n"
            "**Important Examples**\n"
            "**Summary**"
        ),
    },
}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SUMMARIZER_SYSTEM_PROMPT = (
    "You are an expert summarizer and structured note-taker. "
    "You produce accurate, well-organized summaries of transcribed audio "
    "content. "
    "You are concise but thorough, preserve factual accuracy, and use clear "
    "Markdown formatting. Never fabricate information not present in the "
    "source text. "
    "If a section is unclear or contains transcription artifacts, note it "
    "briefly "
    "rather than guessing. "
    "Always follow the output language instruction given in the user message."
)

#: How to cite, for prompts that contain the numbered transcript.
TRANSCRIPT_CITATION_RULE = (
    "Each transcript line starts with its line number in square brackets, "
    "for example [12]. After every bullet point, decision, action item, "
    "quote and statement you write, cite the transcript lines it is based "
    "on, with their numbers in square brackets, for example [12] or "
    "[12, 15]. Cite only line numbers that appear in the transcript. Do not "
    "put citations in headings."
)

#: How to cite, for prompts that contain notes with citations.
NOTES_CITATION_RULE = (
    "The notes cite transcript lines by number in square brackets, for "
    "example [12]. After every bullet point, decision, action item, quote "
    "and statement you write, cite the lines it is based on, copying the "
    "numbers from the notes, for example [12] or [12, 15]. Use only numbers "
    "that appear in the notes; never invent new ones. Do not put citations "
    "in headings."
)

#: Task for merging notes when there are too many for one request.
CONDENSE_INSTRUCTION = (
    "Merge these notes into one shorter set of notes. Keep every decision, "
    "action item, name, number and date, and drop repetitions."
)


def _language_instruction(output_language: str | None) -> str:
    """Build the language directive that is appended to every LLM prompt.

    Args:
        output_language: The desired output language name (e.g. ``"English"``,
            ``"German"``), or ``None`` / ``"transcript"`` to instruct the model
            to match the language of the source transcript.

    Returns:
        A short imperative sentence to append to the user prompt.
    """
    if not output_language or output_language.lower() == "transcript":
        return (
            "Important: Write your entire response in the same language as "
            "the "
            "transcript content."
        )
    return f"Important: Write your entire response in {output_language}."


# ---------------------------------------------------------------------------
# Internal message builders
# ---------------------------------------------------------------------------


def _messages(prompt: str) -> list[dict[str, str]]:
    """Wrap a user prompt with the system prompt."""
    return [
        {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def _build_single_pass_messages(
    text: str,
    mode_key: str,
    speaker_context: str | None,
    output_language: str | None = "English",
) -> list[dict[str, str]]:
    """Build the messages for a single-pass summarization call.

    Args:
        text: The full numbered transcript (:func:`numbered_text`).
        mode_key: Key in SUMMARY_MODES selecting the summarization style.
        speaker_context: Optional sentence describing identified speakers.
        output_language: Desired language for the summary output. Pass
            ``None`` or ``"transcript"`` to match the transcript language.
            Defaults to ``"English"``.

    Returns:
        A list of role/content message dicts ready for ollama.chat().
    """
    mode = SUMMARY_MODES[mode_key]
    context_note = (
        f"\n\nContext about speakers: {speaker_context}"
        if speaker_context
        else ""
    )
    return _messages(
        f"Below is a transcript of an audio recording.{context_note}\n\n"
        f"--- Transcript Start ---\n{text}\n--- Transcript End ---\n\n"
        f"Task: {mode['synthesis_instruction']}\n\n"
        f"{TRANSCRIPT_CITATION_RULE}\n\n"
        f"{_language_instruction(output_language)}"
    )


def _build_chunk_messages(
    chunk: str,
    chunk_index: int,
    total_chunks: int,
    mode_key: str,
    speaker_context: str | None,
    output_language: str | None = "English",
) -> list[dict[str, str]]:
    """Build the messages for processing one chunk in the map phase.

    Args:
        chunk: Numbered transcript lines to analyze.
        chunk_index: 1-based position of this chunk.
        total_chunks: Total number of chunks.
        mode_key: Key in SUMMARY_MODES.
        speaker_context: Optional speaker description sentence.
        output_language: Desired language for the extracted notes. Pass
            ``None`` or ``"transcript"`` to match the transcript language.
            Defaults to ``"English"``.

    Returns:
        A list of role/content message dicts.
    """
    mode = SUMMARY_MODES[mode_key]
    context_note = (
        f"\nContext about speakers: {speaker_context}"
        if speaker_context
        else ""
    )
    return _messages(
        f"You are processing part {chunk_index} of {total_chunks} of a "
        "transcript."
        f"{context_note}\n\n"
        f"--- Transcript Part {chunk_index}/{total_chunks} ---\n{chunk}\n"
        f"--- End of Part {chunk_index}/{total_chunks} ---\n\n"
        f"Task: {mode['chunk_instruction']}\n\n"
        f"{TRANSCRIPT_CITATION_RULE} Keep the citations in your notes; the "
        "final summary is written from them.\n\n"
        f"{_language_instruction(output_language)}"
    )


def _notes_text(notes: Sequence[str]) -> str:
    """Join partial notes, each under a header with its part number."""
    return "\n\n".join(
        f"--- Notes from Part {i + 1} ---\n{note}"
        for i, note in enumerate(notes)
    )


def _build_synthesis_messages(
    partial_notes: Sequence[str],
    mode_key: str,
    output_language: str | None = "English",
) -> list[dict[str, str]]:
    """Build the messages for the reduce (synthesis) phase.

    Args:
        partial_notes: Extracted notes from each chunk, in order.
        mode_key: Key in SUMMARY_MODES defining the desired output structure.
        output_language: Desired language for the final summary. Pass
            ``None`` or ``"transcript"`` to match the transcript language.
            Defaults to ``"English"``.

    Returns:
        A list of role/content message dicts.
    """
    mode = SUMMARY_MODES[mode_key]
    return _messages(
        f"A transcript was split into {len(partial_notes)} parts and each "
        "was analyzed "
        f"separately. Below are the extracted notes.\n\n"
        f"{_notes_text(partial_notes)}\n\n"
        f"--- End of Partial Notes ---\n\n"
        f"Task: {mode['synthesis_instruction']}\n\n"
        f"{NOTES_CITATION_RULE}\n\n"
        f"{_language_instruction(output_language)}"
    )


def _build_condense_messages(
    notes: Sequence[str],
    output_language: str | None = "English",
) -> list[dict[str, str]]:
    """Build the messages for merging several partial notes into one.

    Args:
        notes: Consecutive partial notes.
        output_language: Desired language for the merged notes.

    Returns:
        A list of role/content message dicts.
    """
    return _messages(
        "Below are notes taken from consecutive parts of a transcript.\n\n"
        f"{_notes_text(notes)}\n\n"
        f"--- End of Notes ---\n\n"
        f"Task: {CONDENSE_INSTRUCTION}\n\n"
        f"{NOTES_CITATION_RULE}\n\n"
        f"{_language_instruction(output_language)}"
    )


# ---------------------------------------------------------------------------
# Internal Ollama call helpers
# ---------------------------------------------------------------------------


def _chat(
    model_name: str,
    messages: list[dict[str, str]],
    stream: bool,
    num_ctx: int,
    disable_thinking: bool = True,
) -> Any:
    """Call ``ollama.chat`` with the summarization options.

    Reasoning ("thinking") is disabled by default: a summary needs no
    chain-of-thought, and thinking tokens only add latency and use up the
    context window. Older ollama-python clients without the ``think``
    keyword are called without it.

    Args:
        model_name: The Ollama model identifier.
        messages: Role/content message dicts.
        stream: Whether to return a token stream instead of one response.
        num_ctx: Context window in tokens; it must hold the whole request
            and the answer, or Ollama drops part of the request.
        disable_thinking: Send ``think=False``. Callers pass ``False`` to
            retry when the server or model rejects the ``think`` argument.

    Returns:
        A chat response, or an iterator of response chunks when ``stream``
        is true. For streams the HTTP request is only sent on iteration.
    """
    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "stream": stream,
        "options": {"temperature": SUMMARY_TEMPERATURE, "num_ctx": num_ctx},
    }
    if disable_thinking:
        return call_no_think(ollama.chat, **kwargs)
    return ollama.chat(**kwargs)


def _is_think_rejection(exc: ollama.ResponseError) -> bool:
    """Return True if Ollama rejected the request because of ``think``."""
    return "think" in str(exc).lower()


def _response_text(response: Any) -> str:
    """Return the content of a chat response or stream chunk as a string.

    Reasoning-only chunks carry ``None`` content, which is mapped to an
    empty string so callers can concatenate safely.
    """
    if isinstance(response, dict):
        message = response.get("message")
    else:
        message = getattr(response, "message", None)
    return message_text(message)


def _complete_text(
    model_name: str, messages: list[dict[str, str]], num_ctx: int
) -> str:
    """Run one blocking chat call and return the reply text.

    Args:
        model_name: The Ollama model identifier.
        messages: Role/content message dicts.
        num_ctx: Context window in tokens.

    Returns:
        The model's reply, or an empty string if it produced no content.
    """
    try:
        response = _chat(model_name, messages, False, num_ctx)
    except ollama.ResponseError as exc:
        if not _is_think_rejection(exc):
            raise
        response = _chat(
            model_name, messages, False, num_ctx, disable_thinking=False
        )
    return _response_text(response)


def _stream_text(
    model_name: str, messages: list[dict[str, str]], num_ctx: int
) -> Iterator[str]:
    """Stream a chat reply as text tokens.

    If the server rejects ``think=False`` before any token arrives, the
    request is retried once without it.

    Args:
        model_name: The Ollama model identifier.
        messages: Role/content message dicts.
        num_ctx: Context window in tokens.

    Yields:
        Incremental string tokens (never ``None``).
    """
    started = False
    try:
        for chunk in _chat(model_name, messages, True, num_ctx):
            started = True
            yield _response_text(chunk)
        return
    except ollama.ResponseError as exc:
        if started or not _is_think_rejection(exc):
            raise
    for chunk in _chat(
        model_name, messages, True, num_ctx, disable_thinking=False
    ):
        yield _response_text(chunk)


# ---------------------------------------------------------------------------
# Public API: summarization
# ---------------------------------------------------------------------------


def needs_chunking(segments: Sequence[Segment]) -> bool:
    """Return True if a transcript is too long for one request.

    Args:
        segments: The transcript.

    Returns:
        True if the numbered transcript does not fit in the largest context
        window (:func:`max_context`) together with the instructions and the
        answer, so it must go through :func:`get_partial_notes`.
    """
    return estimate_tokens(numbered_text(segments)) > _text_budget()


def get_summary_stream(
    model_name: str,
    segments: Sequence[Segment],
    mode_key: str,
    speaker_context: str | None = None,
    output_language: str | None = "English",
) -> Iterator[str]:
    """Stream a cited summary of a transcript that fits in one request.

    Check with :func:`needs_chunking` first; for longer transcripts use
    :func:`get_partial_notes` and :func:`get_synthesis_stream`.

    Args:
        model_name: The Ollama model identifier.
        segments: The transcript.
        mode_key: Key in SUMMARY_MODES.
        speaker_context: Optional sentence describing identified speakers.
        output_language: Language for the summary output. Pass ``None`` or
            ``"transcript"`` to match the source transcript language.
            Defaults to ``"English"``.

    Yields:
        Incremental string tokens from the language model. Pass the joined
        text to :func:`cite_summary`.
    """
    messages = _build_single_pass_messages(
        numbered_text(segments), mode_key, speaker_context, output_language
    )
    yield from _stream_text(model_name, messages, _context_for(messages))


def get_partial_notes(
    model_name: str,
    segments: Sequence[Segment],
    mode_key: str,
    speaker_context: str | None = None,
    on_progress: ProgressCallback = no_progress,
    output_language: str | None = "English",
) -> list[str]:
    """Run the map phase of chunked summarization.

    Splits a long transcript into chunks of whole lines that fit in the
    largest context window, and extracts cited notes from each with one
    blocking request. All requests use the same window, so the model is
    loaded once. Follow with :func:`get_synthesis_stream`.

    Args:
        model_name: The Ollama model identifier.
        segments: The transcript (expected not to fit in one request).
        mode_key: Key in SUMMARY_MODES.
        speaker_context: Optional sentence describing identified speakers.
        on_progress: Receives an update before each chunk. The fraction
            counts the synthesis as one more step after the chunks.
        output_language: Language for extracted notes. Pass ``None`` or
            ``"transcript"`` to match the source transcript language.
            Defaults to ``"English"``.

    Returns:
        A list of partial note strings, one per chunk, in order.
    """
    lines = numbered_text(segments).splitlines()
    chunks = [
        "\n".join(group) for group in _group_by_budget(lines, _text_budget())
    ]
    total_steps = len(chunks) + 1  # +1 reserved for synthesis
    partial_notes: list[str] = []

    for i, chunk in enumerate(chunks, 1):
        on_progress(
            Progress(
                f"Analyzing part {i} of {len(chunks)}...",
                (i - 1) / total_steps,
            )
        )
        messages = _build_chunk_messages(
            chunk=chunk,
            chunk_index=i,
            total_chunks=len(chunks),
            mode_key=mode_key,
            speaker_context=speaker_context,
            output_language=output_language,
        )
        partial_notes.append(
            _complete_text(model_name, messages, max_context())
        )

    return partial_notes


def _condensed(
    model_name: str,
    partial_notes: Sequence[str],
    output_language: str | None,
) -> list[str]:
    """Merge consecutive notes until all of them fit in one request.

    Only needed for very long recordings, whose notes do not fit in the
    largest context window together.

    Args:
        model_name: The Ollama model identifier.
        partial_notes: Notes from :func:`get_partial_notes`.
        output_language: Language for the merged notes.

    Returns:
        The notes, merged as far as needed.
    """
    notes = list(partial_notes)
    budget = _text_budget()
    while sum(estimate_tokens(note) for note in notes) > budget:
        groups = _group_by_budget(notes, budget)
        if len(groups) == len(notes):
            break  # no two notes fit together; merging cannot help
        notes = [
            group[0]
            if len(group) == 1
            else _complete_text(
                model_name,
                _build_condense_messages(group, output_language),
                max_context(),
            )
            for group in groups
        ]
    return notes


def get_synthesis_stream(
    model_name: str,
    partial_notes: Sequence[str],
    mode_key: str,
    output_language: str | None = "English",
) -> Iterator[str]:
    """Stream the final cited summary from per-chunk partial notes.

    This is the reduce phase of the map-reduce summarization strategy. If
    the notes are too long for one request, consecutive notes are merged
    first. The request uses the same context window as the map phase.

    Args:
        model_name: The Ollama model identifier.
        partial_notes: Collected notes from get_partial_notes().
        mode_key: Key in SUMMARY_MODES.
        output_language: Language for the final summary. Pass ``None`` or
            ``"transcript"`` to match the source transcript language.
            Defaults to ``"English"``.

    Yields:
        Incremental string tokens from the language model. Pass the joined
        text to :func:`cite_summary`.
    """
    notes = _condensed(model_name, partial_notes, output_language)
    messages = _build_synthesis_messages(notes, mode_key, output_language)
    yield from _stream_text(model_name, messages, max_context())


# ---------------------------------------------------------------------------
# Public API: speakers and export
# ---------------------------------------------------------------------------


def build_speaker_context(speaker_labels: list[str]) -> str | None:
    """Build a short human-readable sentence describing identified speakers.

    This sentence is injected into LLM prompts to help the model understand
    the speaker structure of the transcript.

    Args:
        speaker_labels: List of speaker identifier strings.

    Returns:
        A descriptive sentence, or None if the list is empty.
    """
    if not speaker_labels:
        return None
    if len(speaker_labels) == 1:
        return f"There is one speaker: {speaker_labels[0]}."
    joined = ", ".join(speaker_labels[:-1]) + f" and {speaker_labels[-1]}"
    return f"There are {len(speaker_labels)} speakers: {joined}."


def format_summary_document(
    cited: CitedSummary,
    segments: Sequence[Segment],
    source_label: str,
    mode_key: str,
    duration_str: str | None = None,
) -> str:
    """Build the Markdown document with the summary and the transcript.

    Citations become footnotes that quote the cited lines
    (:func:`with_footnotes`), and the transcript lines carry the same
    labels, so every point can be traced back.

    Args:
        cited: The summary, checked by :func:`cite_summary`.
        segments: The transcript.
        source_label: The original file name, shown in the document header.
        mode_key: The summary mode key used, for display in the header.
        duration_str: Human-readable audio duration, e.g. ``"42m 10s"``.

    Returns:
        A Markdown-formatted string ready for download as a ``.md`` file.
    """
    mode_label = SUMMARY_MODES.get(mode_key, {}).get("label", mode_key)
    meta_lines = [
        f"# Audio-to-Summary: {source_label}",
        f"**Summary type**: {mode_label}",
    ]
    if duration_str:
        meta_lines.append(f"**Audio duration**: {duration_str}")

    meta_block = "\n\n".join(meta_lines)
    return (
        f"{meta_block}\n\n"
        f"---\n\n"
        f"## Summary\n\n"
        f"{with_footnotes(cited).rstrip()}\n\n"
        f"---\n\n"
        f"## Full Transcript\n\n"
        f"{transcript_text(segments)}\n"
    )
