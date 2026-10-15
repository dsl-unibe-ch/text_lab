"""Transcripts as numbered segments, the unit the summary cites.

A segment table from the Transcribe page becomes one :class:`Segment` per
row, with speaker and timestamps. Plain text becomes one segment per line;
very long lines, such as a pasted wall of text, are split into sentences so
citations stay precise. Segments are numbered from 1 in transcript order.

The model sees the transcript with these numbers (:func:`numbered_text`);
people see it with timestamps where there are any (:func:`transcript_text`),
and :func:`segment_label` names a segment the same way in both the summary
and the transcript.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable, Mapping, Sequence

from textlab.features.meeting_notes.models import Segment
from textlab.features.transcription.formats import (
    UNKNOWN_SPEAKER,
    load_tsv_rows,
    parse_timestamp,
)

#: Lines longer than this many characters are split into sentences.
LONG_LINE_CHARS = 400

#: Speaker labels produced by diarization, e.g. ``SPEAKER_00``.
_SPEAKER_LABEL = re.compile(r"^SPEAKER_\d+$")
#: A line of plain text that starts with a diarization label.
_SPEAKER_PREFIX = re.compile(r"^(SPEAKER_\d+)\s*:\s*(.*)$")
#: Whitespace after a sentence end.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


def segments_from_table(table_text: str) -> list[Segment]:
    """Read a segment table (``transcription.csv``) into segments.

    Rows without text are skipped. Speaker and timestamps are kept when the
    table has the columns.

    Args:
        table_text: The tab-separated table from the Transcribe page.

    Returns:
        The segments, numbered from 1.
    """
    rows = load_tsv_rows(table_text)
    if not rows:
        return []
    has_times = "Start" in rows[0]
    has_speakers = "Speaker" in rows[0]

    segments: list[Segment] = []
    for row in rows:
        text = (row.get("Text") or row.get("text") or "").strip()
        if not text:
            continue
        speaker = None
        if has_speakers:
            speaker = (row.get("Speaker") or "").strip() or UNKNOWN_SPEAKER
        start = end = None
        if has_times:
            start = parse_timestamp(row.get("Start"))
            end = parse_timestamp(row.get("End"))
        segments.append(Segment(len(segments) + 1, text, speaker, start, end))
    return segments


def segments_from_text(text: str) -> list[Segment]:
    """Split plain text into segments, one per non-empty line.

    A line that starts with a diarization label (``SPEAKER_00: ...``) gets
    that speaker. Lines longer than :data:`LONG_LINE_CHARS` are split into
    sentences.

    Args:
        text: The transcript text.

    Returns:
        The segments, numbered from 1, without timestamps.
    """
    segments: list[Segment] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        speaker = None
        match = _SPEAKER_PREFIX.match(line)
        if match:
            speaker, line = match.group(1), match.group(2).strip()
        for part in _split_long_line(line):
            segments.append(Segment(len(segments) + 1, part, speaker))
    return segments


def _split_long_line(line: str) -> list[str]:
    """Return a line as is, or its sentences if it is very long."""
    if len(line) <= LONG_LINE_CHARS:
        return [line] if line else []
    return [part for part in _SENTENCE_BREAK.split(line) if part.strip()]


def speakers(segments: Iterable[Segment]) -> list[str]:
    """Return the diarization labels (``SPEAKER_XX``) in the transcript.

    Args:
        segments: The transcript.

    Returns:
        The sorted unique labels; names the user already gave are not
        included.
    """
    return sorted(
        {
            segment.speaker
            for segment in segments
            if segment.speaker and _SPEAKER_LABEL.match(segment.speaker)
        }
    )


def rename_speakers(
    segments: Sequence[Segment], names: Mapping[str, str]
) -> list[Segment]:
    """Replace speaker labels with the names the user gave them.

    Args:
        segments: The transcript.
        names: Maps a label (e.g. ``SPEAKER_00``) to a name. Empty names are
            ignored.

    Returns:
        The segments with renamed speakers.
    """
    clean = {label: name.strip() for label, name in names.items()}
    return [
        dataclasses.replace(segment, speaker=clean[segment.speaker])
        if segment.speaker in clean and clean[segment.speaker]
        else segment
        for segment in segments
    ]


def format_time(seconds: float) -> str:
    """Format a time for people: ``MM:SS``, or ``H:MM:SS`` from one hour.

    Args:
        seconds: Time from the start of the recording.

    Returns:
        The formatted time.
    """
    total = int(seconds)
    hours, minutes, secs = total // 3600, (total % 3600) // 60, total % 60
    if hours:
        return f"{hours}:{minutes:02}:{secs:02}"
    return f"{minutes:02}:{secs:02}"


def segment_label(segment: Segment) -> str:
    """Name a segment for people: its start time, or its line number.

    Args:
        segment: The segment.

    Returns:
        E.g. ``"05:12"`` for a timed segment, ``"line 12"`` otherwise.
    """
    if segment.start is not None:
        return format_time(segment.start)
    return f"line {segment.id}"


def _line(segment: Segment, prefix: str) -> str:
    """Return one transcript line: prefix, speaker and text."""
    if segment.speaker:
        return f"[{prefix}] {segment.speaker}: {segment.text}"
    return f"[{prefix}] {segment.text}"


def numbered_text(segments: Iterable[Segment]) -> str:
    """Return the transcript for the model, each line with its number.

    Args:
        segments: The transcript.

    Returns:
        Lines such as ``[12] Anna: We ship on Friday.``
    """
    return "\n".join(_line(segment, str(segment.id)) for segment in segments)


def transcript_line(segment: Segment) -> str:
    """Return one transcript line for people, with the segment's label.

    The label is the one citations show (:func:`segment_label`), so a cited
    passage can be found in the transcript.

    Args:
        segment: The segment.

    Returns:
        E.g. ``[05:12] Anna: We ship on Friday.``, or ``[12] ...`` for a
        transcript without timestamps.
    """
    return _line(segment, segment_label(segment).removeprefix("line "))


def transcript_text(segments: Iterable[Segment]) -> str:
    """Return the transcript for people, one :func:`transcript_line` each.

    Args:
        segments: The transcript.

    Returns:
        The labeled lines joined by newlines.
    """
    return "\n".join(transcript_line(segment) for segment in segments)


def word_count(segments: Iterable[Segment]) -> int:
    """Return the number of words spoken in the transcript."""
    return sum(len(segment.text.split()) for segment in segments)
