"""Transcript files: writing the exports and reading them back.

The writers turn a :class:`~textlab.features.transcription.models.Transcript`
into the files users download: a segment table, a word table, an ELAN
import table, SubRip and WebVTT subtitles and plain text. Tables are
tab-separated even when named ``.csv``, as before. The readers load a
segment or word table for the review mode and for summarization.
"""

from __future__ import annotations

import csv
import io
import os
import zipfile
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from textlab.features.transcription.models import Transcript

#: Speaker label for words diarization could not attribute.
UNKNOWN_SPEAKER = "UNKNOWN"


# ---------------------------------------------------------------------------
# Time formatting
# ---------------------------------------------------------------------------


def format_time(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS.mmm`` for the transcript tables.

    Args:
        seconds: Time from the start of the recording.

    Returns:
        The formatted time.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:06.3f}"


def format_duration(seconds: float) -> str:
    """Format a duration for people, e.g. ``"1h 5m"``, ``"3m 12s"``.

    Args:
        seconds: The duration.

    Returns:
        The formatted duration.
    """
    total = int(round(seconds))
    hours, minutes, secs = total // 3600, (total % 3600) // 60, total % 60
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def parse_timestamp(value: Any) -> float:
    """Parse ``HH:MM:SS.mmm``, ``MM:SS.mmm`` or plain seconds.

    Args:
        value: The timestamp, as a string or number.

    Returns:
        The time in seconds; 0.0 for empty or unreadable values.
    """
    if value is None:
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    parts = text.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = (float(part) for part in parts)
        elif len(parts) == 2:
            hours = 0.0
            minutes, seconds = (float(part) for part in parts)
        else:
            return float(text)
    except ValueError:
        return 0.0
    return hours * 3600.0 + minutes * 60.0 + seconds


def _subtitle_time(seconds: float, separator: str) -> str:
    """Format seconds as ``HH:MM:SS,mmm`` (SRT) or ``HH:MM:SS.mmm`` (VTT)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def transcription_csv(transcript: Transcript) -> str:
    """Write the segment table: one row per segment or speaker turn.

    With speakers, a segment is split wherever the speaker changes.

    Args:
        transcript: The transcript.

    Returns:
        Tab-separated text with columns ``segment_id, Start, End, [Speaker,]
        Text``.
    """
    output, writer = _table()
    if transcript.has_speakers:
        writer.writerow(["segment_id", "Start", "End", "Speaker", "Text"])
        rows = _speaker_turns(transcript.segments)
    else:
        writer.writerow(["segment_id", "Start", "End", "Text"])
        rows = _plain_segments(transcript.segments)
    for segment_id, row in enumerate(rows, 1):
        writer.writerow([segment_id, *row])
    return output.getvalue()


def elan_tsv(transcript: Transcript) -> str:
    """Write the ELAN import table: the segment table without its ID column.

    Args:
        transcript: The transcript.

    Returns:
        Tab-separated text with columns ``Start, End, [Speaker,] Text``.
    """
    output, writer = _table()
    if transcript.has_speakers:
        writer.writerow(["Start", "End", "Speaker", "Text"])
        rows = _speaker_turns(transcript.segments)
    else:
        writer.writerow(["Start", "End", "Text"])
        rows = _plain_segments(transcript.segments)
    writer.writerows(rows)
    return output.getvalue()


def words_csv(transcript: Transcript) -> str:
    """Write the word table: one row per word with its timing and speaker.

    A new ``segment_id`` starts with each segment and each speaker change.

    Args:
        transcript: The transcript.

    Returns:
        Tab-separated text with columns ``segment_id, word_id, Start, End,
        Word, Speaker``.
    """
    output, writer = _table()
    writer.writerow(
        ["segment_id", "word_id", "Start", "End", "Word", "Speaker"]
    )
    segment_id = 1
    for segment in transcript.segments:
        words = segment.get("words", [])
        if not words:
            continue
        word_id = 1
        current_speaker = words[0].get("speaker") or UNKNOWN_SPEAKER
        for word in words:
            speaker = word.get("speaker") or UNKNOWN_SPEAKER
            if speaker != current_speaker:
                segment_id += 1
                word_id = 1
                current_speaker = speaker
            writer.writerow(
                [
                    segment_id,
                    word_id,
                    format_time(word.get("start", 0.0)),
                    format_time(word.get("end", 0.0)),
                    word.get("word", word.get("text", "")),
                    speaker,
                ]
            )
            word_id += 1
        segment_id += 1
    return output.getvalue()


def srt(transcript: Transcript) -> str:
    """Write SubRip subtitles, one cue per non-empty segment.

    Args:
        transcript: The transcript.

    Returns:
        The ``.srt`` file's text.
    """
    output = io.StringIO()
    cue = 1
    for start, end, text in _cues(transcript.segments):
        output.write(f"{cue}\n")
        output.write(
            f"{_subtitle_time(start, ',')} --> {_subtitle_time(end, ',')}\n"
        )
        output.write(f"{text}\n\n")
        cue += 1
    return output.getvalue()


def vtt(transcript: Transcript) -> str:
    """Write WebVTT subtitles, one cue per non-empty segment.

    Args:
        transcript: The transcript.

    Returns:
        The ``.vtt`` file's text.
    """
    output = io.StringIO()
    output.write("WEBVTT\n\n")
    for start, end, text in _cues(transcript.segments):
        output.write(
            f"{_subtitle_time(start, '.')} --> {_subtitle_time(end, '.')}\n"
        )
        output.write(f"{text}\n\n")
    return output.getvalue()


def transcript_files(transcript: Transcript, base_name: str) -> dict[str, str]:
    """Write every export of a transcript, named after the recording.

    Args:
        transcript: The transcript.
        base_name: File name stem, usually the recording's name without
            its extension.

    Returns:
        File names mapped to their text, in download order.
    """
    segments = transcription_csv(transcript)
    return {
        f"{base_name}_transcription.csv": segments,
        f"{base_name}_ELAN_compatible.tsv": elan_tsv(transcript),
        f"{base_name}_words.csv": words_csv(transcript),
        f"{base_name}_text.txt": transcription_text_from_csv(segments),
        f"{base_name}.srt": srt(transcript),
        f"{base_name}.vtt": vtt(transcript),
    }


def batch_zip(transcripts: Iterable[Transcript]) -> bytes:
    """Pack the exports of several transcripts, one folder per recording.

    Args:
        transcripts: The transcripts.

    Returns:
        The ZIP file's bytes.
    """
    files: dict[str, str] = {}
    for transcript in transcripts:
        base = base_name(transcript.name)
        for name, text in transcript_files(transcript, base).items():
            files[f"{base}/{name}"] = text
    return zip_bytes(files)


def zip_bytes(files: Mapping[str, str | bytes]) -> bytes:
    """Pack files into a compressed ZIP archive.

    Args:
        files: Archive paths mapped to text or bytes.

    Returns:
        The ZIP file's bytes.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buffer.getvalue()


def base_name(name: str) -> str:
    """Return a recording's file name without folders and extension.

    Args:
        name: The recording's name or path.

    Returns:
        The stem, used to name its exports.
    """
    return os.path.splitext(os.path.basename(name))[0]


def _table() -> tuple[io.StringIO, Any]:
    """Return a text buffer and a tab-separated writer for it."""
    output = io.StringIO()
    return output, csv.writer(output, delimiter="\t")


def _plain_segments(segments: list[dict[str, Any]]) -> Iterator[list[Any]]:
    """Yield ``[start, end, text]`` rows, one per segment."""
    for segment in segments:
        yield [
            format_time(segment.get("start", 0)),
            format_time(segment.get("end", 0)),
            segment.get("text", "").strip(),
        ]


def _speaker_turns(segments: list[dict[str, Any]]) -> Iterator[list[Any]]:
    """Yield ``[start, end, speaker, text]`` rows, one per speaker turn.

    Turns never span segments: each segment is split at speaker changes.
    """
    for segment in segments:
        words = segment.get("words", [])
        if not words:
            continue
        speaker = words[0].get("speaker") or UNKNOWN_SPEAKER
        text = words[0]["word"]
        start, end = words[0]["start"], words[0]["end"]
        for word in words[1:]:
            word_speaker = word.get("speaker") or UNKNOWN_SPEAKER
            if word_speaker != speaker:
                yield [
                    format_time(start),
                    format_time(end),
                    speaker,
                    text.strip(),
                ]
                speaker, text, start = (
                    word_speaker,
                    word["word"],
                    word["start"],
                )
            else:
                text += " " + word["word"]
            end = word["end"]
        yield [format_time(start), format_time(end), speaker, text.strip()]


def _cues(
    segments: list[dict[str, Any]],
) -> Iterator[tuple[float, float, str]]:
    """Yield ``(start, end, text)`` for each segment with text."""
    for segment in segments:
        text = segment.get("text", "").strip()
        if text:
            yield segment.get("start", 0), segment.get("end", 0), text


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def load_tsv_rows(text: str) -> list[dict[str, str]]:
    """Read a tab-separated table with a header row.

    Args:
        text: The table's text.

    Returns:
        One dictionary per row, keyed by column name.
    """
    return list(csv.DictReader(io.StringIO(text), delimiter="\t"))


def load_transcript_items(
    text: str, mode: str
) -> tuple[list[dict[str, Any]], bool]:
    """Read a segment or word table into items for the transcript player.

    Args:
        text: A segment table (``transcription.csv``) or word table
            (``words.csv``).
        mode: ``"words"`` for one item per word, anything else for one item
            per segment.

    Returns:
        ``(items, has_speakers)``. Items have ``start`` and ``end`` in
        seconds and ``text``; segment items also have ``speaker`` when the
        table has a filled ``Speaker`` column.
    """
    rows = load_tsv_rows(text)
    has_speakers = False
    if rows and ("Speaker" in rows[0] or "speaker" in rows[0]):
        first = rows[0].get("Speaker") or rows[0].get("speaker")
        has_speakers = bool(first and first.strip())

    items = []
    for row in rows:
        start = parse_timestamp(row.get("Start"))
        end = parse_timestamp(row.get("End"))
        if mode == "words":
            word = row.get("Word") or row.get("word") or row.get("Text") or ""
            items.append({"start": start, "end": end, "text": word})
            continue
        item_text = row.get("Text") or row.get("text") or row.get("Word") or ""
        item = {"start": start, "end": end, "text": item_text}
        if has_speakers:
            item["speaker"] = (
                row.get("Speaker") or row.get("speaker") or UNKNOWN_SPEAKER
            )
        items.append(item)
    return items, has_speakers


def transcription_text_from_csv(text: str) -> str:
    """Return the plain text of a segment table, one line per segment.

    Args:
        text: A segment table.

    Returns:
        The non-empty segment texts joined by newlines.
    """
    lines = []
    for row in load_tsv_rows(text):
        line = str(row.get("Text") or row.get("text") or "").strip()
        if line:
            lines.append(line)
    return "\n".join(lines)
