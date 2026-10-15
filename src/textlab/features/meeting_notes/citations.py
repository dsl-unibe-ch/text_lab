"""Citations in a summary: checking them and showing them.

The model cites transcript lines by number in square brackets, e.g. ``[12]``
or ``[12, 15]``. Models differ in how they write these, so the parser also
accepts ranges (``[12-15]``), a ``#`` before a number and a ``line`` prefix.
:func:`cite_summary` checks each cited number against the transcript,
removes the ones that do not exist, and counts the bullet points that cite
nothing, which are the ones a reader should check first.

:func:`format_citations` replaces the checked citations for display, and
:func:`with_footnotes` turns them into Markdown footnotes for the download.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence

from textlab.features.meeting_notes.models import CitedSummary, Segment
from textlab.features.meeting_notes.transcript import transcript_line

#: A range wider than this many lines is cited by its first and last line.
MAX_RANGE_LINES = 10

_PREFIX = r"(?:(?:lines?|l\.)\s*)?"
_ITEM = r"#?\d+(?:\s*[-–]\s*#?\d+)?"
#: A citation as the model may write it. Not followed by ``(``, so the text
#: of a Markdown link is never taken for a citation.
_RAW_MARKER = re.compile(
    rf"([ \t]*)\[\s*{_PREFIX}({_ITEM}(?:\s*[,;]\s*{_PREFIX}{_ITEM})*)\s*\]"
    r"(?!\()",
    re.IGNORECASE,
)
_RAW_ITEM = re.compile(r"(\d+)(?:\s*[-–]\s*#?(\d+))?")
#: A citation after :func:`cite_summary` normalized it.
_MARKER = re.compile(r"\[(\d+(?:, \d+)*)\](?!\()")
#: A bullet or numbered list item, with its indentation.
_BULLET = re.compile(r"^([ \t]*)(?:[-*+]|\d+[.)])[ \t]+\S")


def _cited_numbers(items: str) -> list[int]:
    """Return the line numbers in the inside of one raw citation."""
    numbers: list[int] = []
    for first, last in _RAW_ITEM.findall(items):
        low = int(first)
        high = int(last) if last else low
        low, high = min(low, high), max(low, high)
        if high - low + 1 > MAX_RANGE_LINES:
            numbers.extend((low, high))
        else:
            numbers.extend(range(low, high + 1))
    return numbers


def _uncited_bullets(lines: Sequence[str]) -> list[int]:
    """Return the indexes of bullet lines whose item cites no line.

    An item includes the more deeply indented lines below it, so a bullet
    whose sub-points carry the citations counts as cited.
    """
    uncited: list[int] = []
    for index, line in enumerate(lines):
        match = _BULLET.match(line)
        if not match:
            continue
        indent = len(match.group(1).expandtabs(4))
        cited = bool(_MARKER.search(line))
        for following in lines[index + 1 :]:
            depth = len(following) - len(following.lstrip())
            if not following.strip() or depth <= indent:
                break
            if _MARKER.search(following):
                cited = True
                break
        if not cited:
            uncited.append(index)
    return uncited


def cite_summary(summary: str, segments: Iterable[Segment]) -> CitedSummary:
    """Check the citations in a summary against the transcript.

    Args:
        summary: The summary as the model wrote it.
        segments: The transcript the model was given.

    Returns:
        The summary with valid citations normalized and invalid ones
        removed, the cited segments, and counts of removed citations and
        bullet points without a citation.
    """
    by_id = {segment.id: segment for segment in segments}
    cited: set[int] = set()
    invalid = 0

    def normalize(match: re.Match[str]) -> str:
        nonlocal invalid
        numbers = set(_cited_numbers(match.group(2)))
        valid = sorted(number for number in numbers if number in by_id)
        invalid += len(numbers) - len(valid)
        cited.update(valid)
        if not valid:
            return ""
        return f"{match.group(1)}[{', '.join(map(str, valid))}]"

    text = _RAW_MARKER.sub(normalize, summary)
    return CitedSummary(
        text=text,
        sources=tuple(by_id[number] for number in sorted(cited)),
        invalid_citations=invalid,
        uncited_points=len(_uncited_bullets(text.splitlines())),
    )


def format_citations(
    text: str,
    sources: Iterable[Segment],
    render: Callable[[list[Segment]], str],
) -> str:
    """Replace each citation in a checked summary.

    Args:
        text: The summary text of a :class:`CitedSummary`, possibly with
            uncited points marked (:func:`mark_uncited_points`).
        sources: The cited segments (``CitedSummary.sources``).
        render: Turns the segments of one citation into its replacement.

    Returns:
        The text with the citations replaced.
    """
    by_id = {segment.id: segment for segment in sources}

    def replace(match: re.Match[str]) -> str:
        numbers = (int(number) for number in match.group(1).split(", "))
        return render([by_id[number] for number in numbers])

    return _MARKER.sub(replace, text)


def mark_uncited_points(text: str, note: str) -> str:
    """Append a note to every bullet point that cites no transcript line.

    Use it on a summary checked by :func:`cite_summary`, before
    :func:`format_citations` replaces the citations.

    Args:
        text: The checked summary text.
        note: What to append, e.g. ``"(no source)"``.

    Returns:
        The text with the note after each uncited bullet point.
    """
    lines = text.splitlines()
    for index in _uncited_bullets(lines):
        lines[index] = f"{lines[index].rstrip()} {note}"
    return "\n".join(lines)


def with_footnotes(cited: CitedSummary) -> str:
    """Return the summary with its citations as Markdown footnotes.

    Each citation becomes footnote references such as ``[^12]``, and each
    cited segment a footnote that quotes it with its time and speaker.

    Args:
        cited: The checked summary.

    Returns:
        Markdown for the download; the text unchanged if nothing is cited.
    """
    body = format_citations(
        cited.text,
        cited.sources,
        lambda sources: "".join(f"[^{s.id}]" for s in sources),
    )
    if not cited.sources:
        return body
    notes = "\n".join(
        f"[^{segment.id}]: {transcript_line(segment)}"
        for segment in cited.sources
    )
    return f"{body.rstrip()}\n\n{notes}\n"
