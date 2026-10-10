"""Summarizing transcripts into notes: the Meeting Notes feature's API.

A transcript that fits in the model's context window is summarized in one
streamed request (:func:`get_summary_stream`). A longer one is summarized
map-reduce style: :func:`get_partial_notes` extracts notes from each chunk,
then :func:`get_synthesis_stream` streams the final summary from them. Use
:func:`needs_chunking` to choose.

The summary's structure depends on the mode (:data:`SUMMARY_MODES`), e.g.
meeting notes with decisions and action items, or lecture notes. Helpers
turn a transcript table into speaker-labeled text, apply names the user gave
the speakers, and build the Markdown document users download.

Streaming functions are plain generators of text, so any interface can
display them as they arrive.
"""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Iterator
from typing import Any

import ollama

from textlab.common.ollama import (
    MAX_CONTEXT_TOKENS,
    call_no_think,
    chunk_text,
    estimate_tokens,
    message_text,
)
from textlab.common.progress import Progress, ProgressCallback, no_progress

# Low temperature keeps summaries close to the transcript instead of
# paraphrasing creatively or adding detail that was never said.
SUMMARY_TEMPERATURE = 0.2


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
# System prompt
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


def _build_single_pass_messages(
    text: str,
    mode_key: str,
    speaker_context: str | None,
    output_language: str | None = "English",
) -> list[dict[str, str]]:
    """Build an Ollama message list for a single-pass summarization call.

    Args:
        text: The full transcript text.
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
    lang_note = _language_instruction(output_language)
    prompt = (
        f"Below is a transcript of an audio recording.{context_note}\n\n"
        f"--- Transcript Start ---\n{text}\n--- Transcript End ---\n\n"
        f"Task: {mode['synthesis_instruction']}\n\n"
        f"{lang_note}"
    )
    return [
        {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def _build_chunk_messages(
    chunk: str,
    chunk_index: int,
    total_chunks: int,
    mode_key: str,
    speaker_context: str | None,
    output_language: str | None = "English",
) -> list[dict[str, str]]:
    """Build an Ollama message list for processing one chunk in the map phase.

    Args:
        chunk: Text fragment to analyze.
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
    lang_note = _language_instruction(output_language)
    prompt = (
        f"You are processing part {chunk_index} of {total_chunks} of a "
        "transcript."
        f"{context_note}\n\n"
        f"--- Transcript Part {chunk_index}/{total_chunks} ---\n{chunk}\n"
        f"--- End of Part {chunk_index}/{total_chunks} ---\n\n"
        f"Task: {mode['chunk_instruction']}\n\n"
        f"{lang_note}"
    )
    return [
        {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def _build_synthesis_messages(
    partial_notes: list[str],
    mode_key: str,
    output_language: str | None = "English",
) -> list[dict[str, str]]:
    """Build an Ollama message list for the reduce (synthesis) phase.

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
    lang_note = _language_instruction(output_language)
    parts_text = "\n\n".join(
        f"--- Notes from Part {i + 1} ---\n{note}"
        for i, note in enumerate(partial_notes)
    )
    prompt = (
        f"A transcript was split into {len(partial_notes)} parts and each "
        "was analyzed "
        f"separately. Below are the extracted notes.\n\n"
        f"{parts_text}\n\n"
        f"--- End of Partial Notes ---\n\n"
        f"Task: {mode['synthesis_instruction']}\n\n"
        f"{lang_note}"
    )
    return [
        {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


# ---------------------------------------------------------------------------
# Internal Ollama call helpers
# ---------------------------------------------------------------------------


def _chat(
    model_name: str,
    messages: list[dict[str, str]],
    stream: bool,
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
        "options": {"temperature": SUMMARY_TEMPERATURE},
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


def _complete_text(model_name: str, messages: list[dict[str, str]]) -> str:
    """Run one blocking chat call and return the reply text.

    Args:
        model_name: The Ollama model identifier.
        messages: Role/content message dicts.

    Returns:
        The model's reply, or an empty string if it produced no content.
    """
    try:
        response = _chat(model_name, messages, stream=False)
    except ollama.ResponseError as exc:
        if not _is_think_rejection(exc):
            raise
        response = _chat(
            model_name, messages, stream=False, disable_thinking=False
        )
    return _response_text(response)


def _stream_text(
    model_name: str,
    messages: list[dict[str, str]],
) -> Iterator[str]:
    """Stream a chat reply as text tokens.

    If the server rejects ``think=False`` before any token arrives, the
    request is retried once without it.

    Args:
        model_name: The Ollama model identifier.
        messages: Role/content message dicts.

    Yields:
        Incremental string tokens (never ``None``).
    """
    started = False
    try:
        for chunk in _chat(model_name, messages, stream=True):
            started = True
            yield _response_text(chunk)
        return
    except ollama.ResponseError as exc:
        if started or not _is_think_rejection(exc):
            raise
    for chunk in _chat(
        model_name, messages, stream=True, disable_thinking=False
    ):
        yield _response_text(chunk)


# ---------------------------------------------------------------------------
# Public API: summarization
# ---------------------------------------------------------------------------


def needs_chunking(text: str) -> bool:
    """Return True if a transcript is too long for one summarization request.

    Args:
        text: The transcript text.

    Returns:
        True if it exceeds :data:`MAX_CONTEXT_TOKENS` by estimate, so it
        must go through :func:`get_partial_notes`.
    """
    return estimate_tokens(text) > MAX_CONTEXT_TOKENS


def get_summary_stream(
    model_name: str,
    text: str,
    mode_key: str,
    speaker_context: str | None = None,
    output_language: str | None = "English",
) -> Iterator[str]:
    """Stream a summary of a transcript that fits in one context window.

    Use estimate_tokens(text) <= MAX_CONTEXT_TOKENS to confirm the text is
    short enough before calling this function. For longer texts use the
    get_partial_notes / get_synthesis_stream pair instead.

    Args:
        model_name: The Ollama model identifier.
        text: The full transcript text.
        mode_key: Key in SUMMARY_MODES.
        speaker_context: Optional sentence describing identified speakers.
        output_language: Language for the summary output. Pass ``None`` or
            ``"transcript"`` to match the source transcript language.
            Defaults to ``"English"``.

    Yields:
        Incremental string tokens from the language model.
    """
    messages = _build_single_pass_messages(
        text, mode_key, speaker_context, output_language
    )
    yield from _stream_text(model_name, messages)


def get_partial_notes(
    model_name: str,
    text: str,
    mode_key: str,
    speaker_context: str | None = None,
    on_progress: ProgressCallback = no_progress,
    output_language: str | None = "English",
) -> list[str]:
    """Run the map phase of chunked summarization.

    Splits a long transcript into chunks and extracts partial notes from each
    using a single blocking LLM call per chunk. Designed to be followed by
    get_synthesis_stream() to produce the final summary.

    Args:
        model_name: The Ollama model identifier.
        text: The full transcript text (expected to exceed MAX_CONTEXT_TOKENS).
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
    chunks = chunk_text(text)
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
        partial_notes.append(_complete_text(model_name, messages))

    return partial_notes


def get_synthesis_stream(
    model_name: str,
    partial_notes: list[str],
    mode_key: str,
    output_language: str | None = "English",
) -> Iterator[str]:
    """Stream the final synthesized summary from per-chunk partial notes.

    This is the reduce phase of the map-reduce summarization strategy.

    Args:
        model_name: The Ollama model identifier.
        partial_notes: Collected notes from get_partial_notes().
        mode_key: Key in SUMMARY_MODES.
        output_language: Language for the final summary. Pass ``None`` or
            ``"transcript"`` to match the source transcript language.
            Defaults to ``"English"``.

    Yields:
        Incremental string tokens from the language model.
    """
    messages = _build_synthesis_messages(
        partial_notes, mode_key, output_language
    )
    yield from _stream_text(model_name, messages)


# ---------------------------------------------------------------------------
# Public API: transcript text utilities
# ---------------------------------------------------------------------------


def transcript_csv_to_speaker_text(csv_text: str) -> str:
    """Convert a WhisperX segment table into speaker-labeled plain text.

    Each segment is formatted as ``SPEAKER_XX: segment text``. If no Speaker
    column is present, plain text lines are returned without a label prefix.
    Consecutive segments from the same speaker are kept on separate lines so
    the LLM can follow speaker turns naturally.

    Args:
        csv_text: The TSV string produced by generate_transcription_csv().

    Returns:
        A multi-line string suitable for sending to the summarization LLM.
    """
    reader = csv.DictReader(io.StringIO(csv_text), delimiter="\t")
    rows = list(reader)
    if not rows:
        return ""

    has_speaker_col = "Speaker" in rows[0]
    lines: list[str] = []

    for row in rows:
        text = (row.get("Text") or row.get("text") or "").strip()
        if not text:
            continue
        if has_speaker_col:
            speaker = (row.get("Speaker") or "UNKNOWN").strip()
            lines.append(f"{speaker}: {text}")
        else:
            lines.append(text)

    return "\n".join(lines)


def apply_speaker_labels(text: str, label_map: dict[str, str]) -> str:
    """Replace generic speaker identifiers with human-readable labels.

    Applies whole-word replacement so that e.g. ``SPEAKER_00`` is replaced
    with ``Interviewer`` throughout the transcript text.

    Args:
        text: Transcript text containing SPEAKER_XX identifiers.
        label_map: Mapping from original label (e.g. ``SPEAKER_00``) to the
            desired display name (e.g. ``Interviewer``).

    Returns:
        The transcript text with speaker identifiers substituted.
    """
    for original, replacement in label_map.items():
        if not replacement or replacement.strip() == original:
            continue
        pattern = re.compile(re.escape(original), re.IGNORECASE)
        text = pattern.sub(replacement.strip(), text)
    return text


def extract_unique_speakers(text: str) -> list[str]:
    """Find all unique speaker identifiers in speaker-labeled transcript text.

    Matches patterns such as ``SPEAKER_00``, ``SPEAKER_01``, etc., which are
    the default labels produced by WhisperX diarization.

    Args:
        text: Speaker-labeled transcript text.

    Returns:
        Sorted list of unique speaker identifier strings found in the text.
    """
    pattern = re.compile(r"\bSPEAKER_\d+\b")
    return sorted(set(pattern.findall(text)))


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


# ---------------------------------------------------------------------------
# Public API: export
# ---------------------------------------------------------------------------


def format_summary_document(
    summary: str,
    transcript_text: str,
    source_label: str,
    mode_key: str,
    duration_str: str | None = None,
) -> str:
    """Build the Markdown document with the summary and the transcript.

    Args:
        summary: The generated summary text (may contain Markdown).
        transcript_text: The plain or speaker-labeled transcript text.
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
        f"{summary}\n\n"
        f"---\n\n"
        f"## Full Transcript\n\n"
        f"{transcript_text}\n"
    )
