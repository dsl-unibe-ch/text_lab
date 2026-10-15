"""Transcript segments and cited summaries: the Meeting Notes data classes.

A transcript is a list of :class:`Segment` objects, numbered from 1. The
numbers are shown to the model, which cites them in the summary; a
:class:`CitedSummary` is that summary with its citations checked against the
transcript.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Segment:
    """One line of a transcript: a speaker turn, a segment or a text line.

    Attributes:
        id: Line number, from 1, in transcript order. The model cites it.
        text: What was said.
        speaker: Speaker label or name, if known.
        start: Start time in seconds from the start of the recording, if
            the transcript has timestamps.
        end: End time in seconds, if known.
    """

    id: int
    text: str
    speaker: str | None = None
    start: float | None = None
    end: float | None = None


@dataclasses.dataclass(frozen=True)
class CitedSummary:
    """A summary whose citations were checked against the transcript.

    Attributes:
        text: The summary. Valid citations are normalized to the form
            ``[12, 15]``; citations of lines that do not exist are removed.
        sources: The cited segments, in transcript order, each once.
        invalid_citations: Number of cited line numbers that do not exist
            in the transcript and were removed.
        uncited_points: Number of bullet points that cite no line.
    """

    text: str
    sources: tuple[Segment, ...]
    invalid_citations: int
    uncited_points: int
