"""Options and results of a transcription run.

These data classes are the transcription feature's public interface: the
pages build :class:`TranscriptionOptions`, the service returns a
:class:`TranscriptionResult`. They convert to and from plain dictionaries so
they can cross the boundary to the worker process as JSON.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any


@dataclasses.dataclass(frozen=True)
class TranscriptionOptions:
    """How to transcribe a set of recordings.

    Attributes:
        model: WhisperX model name (e.g. ``"large-v3-turbo"``) or path.
        language: Text Lab language code (e.g. ``"de"``, ``"ch_de"``), or
            ``None`` to detect the language of each file.
        diarize: Identify speakers when a Hugging Face token is configured.
        min_speakers: Lower bound on the number of speakers, if known.
        max_speakers: Upper bound on the number of speakers, if known.
        vad_max_pause: Transcribe only the speech found by voice activity
            detection, merging pauses shorter than this many seconds;
            ``None`` transcribes the whole recording.
        batch_size: WhisperX batch size.
    """

    model: str
    language: str | None = None
    diarize: bool = True
    min_speakers: int | None = None
    max_speakers: int | None = None
    vad_max_pause: float | None = None
    batch_size: int = 16

    def to_dict(self) -> dict[str, Any]:
        """Return the options as a JSON-serializable dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TranscriptionOptions:
        """Rebuild options from :meth:`to_dict` output.

        Args:
            data: The dictionary.

        Returns:
            The options.
        """
        return cls(**data)


@dataclasses.dataclass(frozen=True)
class AudioFile:
    """A recording to transcribe.

    Attributes:
        path: Where the file is on disk (in the workspace).
        name: The name the user knows it by, used in messages and output
            file names.
    """

    path: Path
    name: str


@dataclasses.dataclass
class Transcript:
    """The transcript of one recording.

    Attributes:
        name: The recording's name, as in :attr:`AudioFile.name`.
        language: Language code the recording was aligned with (e.g.
            ``"de"``; Swiss German is aligned as German).
        duration_seconds: Length of the recording.
        segments: WhisperX segments after alignment: dictionaries with
            ``start``, ``end``, ``text`` and ``words``; words carry
            ``speaker`` when speakers were identified.
        has_speakers: Whether speaker diarization ran.
    """

    name: str
    language: str
    duration_seconds: float
    segments: list[dict[str, Any]]
    has_speakers: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the transcript as a JSON-serializable dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Transcript:
        """Rebuild a transcript from :meth:`to_dict` output.

        Args:
            data: The dictionary.

        Returns:
            The transcript.
        """
        return cls(**data)


@dataclasses.dataclass(frozen=True)
class SkippedFile:
    """A recording that could not be transcribed.

    Attributes:
        name: The recording's name.
        reason: Why it was skipped, for the user.
    """

    name: str
    reason: str


@dataclasses.dataclass(frozen=True)
class TimedWord:
    """A word of a transcript, with its place in the recording.

    Attributes:
        text: The word as transcribed.
        start: Where it starts, in seconds; ``None`` if alignment gave it
            no time (numbers and symbols often get none).
        end: Where it ends, in seconds, or ``None``.
    """

    text: str
    start: float | None = None
    end: float | None = None


@dataclasses.dataclass(frozen=True)
class Turn:
    """A passage of a transcript: a segment, or one speaker's part of it.

    Interfaces show a transcript as turns, for example next to an audio
    player that marks the word being spoken.

    Attributes:
        start: Where the passage starts, in seconds.
        end: Where it ends, in seconds.
        text: The passage's text.
        words: Its words with their times; empty when only the passage's
            times are known (a segment table).
        speaker: The speaker's label from diarization (``"SPEAKER_00"``),
            ``"UNKNOWN"`` for words diarization could not attribute, or
            ``None`` when speakers were not identified.
        speaker_number: The speaker's number in order of appearance,
            starting at 1; 0 for an unknown or no speaker.
    """

    start: float
    end: float
    text: str
    words: tuple[TimedWord, ...] = ()
    speaker: str | None = None
    speaker_number: int = 0


@dataclasses.dataclass(frozen=True)
class Playback:
    """A recording prepared for an audio player in the browser.

    Attributes:
        path: The recording as an MP3 file, which every browser plays.
        duration_seconds: The recording's length.
        levels: The waveform: the loudness of equal parts of the
            recording, each from 1 (quiet) to :data:`WAVEFORM_LEVELS`.
        wav: A 16 kHz WAV copy, if one was asked for (ELAN reads WAV).
    """

    path: Path
    duration_seconds: float
    levels: tuple[int, ...]
    wav: Path | None = None


#: Steps of loudness in :attr:`Playback.levels`.
WAVEFORM_LEVELS = 8


@dataclasses.dataclass
class TranscriptionResult:
    """Everything a transcription run produced.

    Attributes:
        transcripts: One transcript per recording that was transcribed, in
            input order.
        skipped: Recordings that could not be decoded.
        notes: Remarks for the user, e.g. that diarization was skipped.
    """

    transcripts: list[Transcript] = dataclasses.field(default_factory=list)
    skipped: list[SkippedFile] = dataclasses.field(default_factory=list)
    notes: list[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the result as a JSON-serializable dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TranscriptionResult:
        """Rebuild a result from :meth:`to_dict` output.

        Args:
            data: The dictionary.

        Returns:
            The result.
        """
        return cls(
            transcripts=[Transcript.from_dict(t) for t in data["transcripts"]],
            skipped=[SkippedFile(**s) for s in data["skipped"]],
            notes=list(data["notes"]),
        )
