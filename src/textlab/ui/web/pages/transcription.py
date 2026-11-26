"""The Transcription page: recordings to text, with the speakers told apart.

Three ways to use it, as on the Streamlit page:

- **One recording** is transcribed as a job. Its result shows the recording
  in a player beside the transcript, with the word being spoken marked; the
  plain text; and the models used. It offers every export, and a ZIP of all
  of them with a WAV copy of the recording for ELAN.
- **A ZIP of recordings**: each is transcribed with the same settings; the
  result is one ZIP with a folder per recording.
- **Review a transcript**: a recording and a transcript table Text Lab made
  earlier, uploaded or named by their paths on the cluster, shown in the
  same player. It needs no GPU, so it starts at once instead of waiting
  for the transcriptions ahead of it.

Uploads reach the server only when the form is sent, so the language is
detected in the job when none is chosen: the worker listens to the first
30 seconds of each recording.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
from functools import partial
from pathlib import Path

from fastapi import Request
from fastapi.responses import HTMLResponse
from markupsafe import Markup
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import FormData, UploadFile

from textlab.common import gpu_manager
from textlab.common.progress import Progress
from textlab.features.transcription import service
from textlab.ui.web import uploads
from textlab.ui.web.auth import Checked, protected_router
from textlab.ui.web.jobs import (
    Download,
    Job,
    JobContext,
    render_card,
    runner_of,
)
from textlab.ui.web.render import renderer

router = protected_router()

#: The page's id, which also names its jobs.
PAGE = "transcription"

#: Joining speech across pauses, in seconds: least, most and default.
VAD_PAUSE = (0.1, 1.0, 0.25)

#: Most speakers one can give as a bound.
MAX_SPEAKERS = 50

#: Largest transcript table the review reads, in bytes.
TABLE_LIMIT = 100 * 1000 * 1000

#: The exports of a transcript: the end of their names, the label of their
#: download, their media type.
EXPORTS = (
    ("_text.txt", "Text", "text/plain; charset=utf-8"),
    (".srt", "Subtitles", "application/x-subrip; charset=utf-8"),
    (".vtt", "Subtitles", "text/vtt; charset=utf-8"),
    ("_transcription.csv", "Passages", "text/csv; charset=utf-8"),
    ("_words.csv", "Words", "text/csv; charset=utf-8"),
    (
        "_ELAN_compatible.tsv",
        "ELAN",
        "text/tab-separated-values; charset=utf-8",
    ),
)


@dataclasses.dataclass(frozen=True)
class Listening:
    """A transcript beside its recording: what the player shows.

    The result of transcribing one recording and of reviewing a
    transcript.

    Attributes:
        audio: Name of the job's download that holds the recording as MP3.
        duration: The recording's length, in seconds.
        levels: Its waveform (see ``service.Playback``).
        turns: The transcript's passages.
        spaced: Whether words are separated by spaces.
        language: The language's name; empty for a review.
        detected: Whether the language was detected.
        source: For a review, the name of the transcript table.
        models: ``(what, model)`` pairs for the Models tab.
        notes: Remarks from the run, such as a skipped speaker step.
    """

    audio: str
    duration: float
    levels: tuple[int, ...]
    turns: list[service.Turn]
    spaced: bool = True
    language: str = ""
    detected: bool = False
    source: str = ""
    models: list[tuple[str, str]] = dataclasses.field(default_factory=list)
    notes: list[str] = dataclasses.field(default_factory=list)

    @property
    def speakers(self) -> int:
        """How many speakers were told apart."""
        return len({turn.speaker_number for turn in self.turns} - {0})

    @property
    def words(self) -> int:
        """How many words the transcript has."""
        return sum(
            len(turn.words) or len(turn.text.split()) for turn in self.turns
        )

    @property
    def timed_words(self) -> bool:
        """Whether the words have times, so the player can mark them."""
        return any(turn.words for turn in self.turns)


@dataclasses.dataclass(frozen=True)
class BatchRecording:
    """One recording of a batch, as the result lists it.

    Attributes:
        name: The recording's file name.
        folder: Its folder in the result's ZIP.
        language: The language's name.
        duration: Its length, in seconds.
        speakers: How many speakers were told apart; ``None`` if speakers
            were not identified.
    """

    name: str
    folder: str
    language: str
    duration: float
    speakers: int | None


@dataclasses.dataclass(frozen=True)
class BatchSummary:
    """The result of transcribing a ZIP of recordings.

    Attributes:
        recordings: The recordings transcribed.
        skipped: The recordings that could not be read.
        notes: Remarks from the run.
        model: The transcription model's name.
    """

    recordings: list[BatchRecording]
    skipped: list[service.SkippedFile]
    notes: list[str]
    model: str

    @property
    def duration(self) -> float:
        """The length of all recordings, in seconds."""
        return sum(item.duration for item in self.recordings)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/transcription")
def page(request: Request) -> HTMLResponse:
    """Show the form and the session's transcriptions.

    Args:
        request: The current request.

    Returns:
        The page.
    """
    jobs = runner_of(request).jobs(PAGE)
    return renderer(request).response(
        request,
        "transcription/page.html",
        page_id=PAGE,
        languages=languages(),
        default_model=service.DEFAULT_MODEL,
        audio_accept=",".join(service.AUDIO_EXTENSIONS),
        audio_formats=" ".join(
            sorted(
                {ext.lstrip(".").upper() for ext in service.AUDIO_EXTENSIONS}
                - {"OGA"}
            )
        ),
        table_accept=",".join(service.TABLE_EXTENSIONS),
        vad_pause=VAD_PAUSE,
        max_speakers=MAX_SPEAKERS,
        cards=Markup("".join(render_card(request, job) for job in jobs[::-1])),
    )


@router.post("/transcription")
async def submit(request: Request, session: Checked) -> HTMLResponse:
    """Start a transcription, a batch or a review from the form.

    Args:
        request: The form: ``mode`` (``"one"``, ``"batch"`` or
            ``"review"``), the files or paths for it, and for a
            transcription ``language``, ``model``, ``vad``, ``vad_pause``,
            ``min_speakers`` and ``max_speakers``.
        session: The session, with its CSRF token checked.

    Returns:
        The new job's card.

    Raises:
        UploadError: If a file or a value is missing or not valid.
    """
    form = await uploads.read_form(request, max_files=2, max_fields=20)
    mode = str(form.get("mode") or "one")
    if mode == "one":
        job = await _start_one(request, form)
    elif mode == "batch":
        job = await _start_batch(request, form)
    elif mode == "review":
        job = await _start_review(request, form)
    else:
        raise uploads.UploadError("Choose what to transcribe.")
    return HTMLResponse(render_card(request, job))


def languages() -> list[tuple[str, str, str]]:
    """Return the languages to choose from, with the model each uses.

    Returns:
        ``(code, name, model)`` triples in the order of
        ``service.LANGUAGES``. ``model`` is empty for a language whose model
        this installation does not have.
    """
    choices = []
    for name, code in service.LANGUAGES.items():
        try:
            model = service.model_label(service.default_model(code))
        except service.MissingSettingError:
            model = ""
        choices.append((code, name, model))
    return choices


def read_options(
    form: FormData, *, other_model: bool = True
) -> service.TranscriptionOptions:
    """Read the transcription settings from the form.

    Args:
        form: The form.
        other_model: Whether the form offers another Whisper model, in its
            ``model`` field. Without it the model follows from the
            language, whatever the form holds.

    Returns:
        The options.

    Raises:
        UploadError: If a value is not valid, or the language's model is
            missing.
    """
    language = str(form.get("language") or "").strip() or None
    if language is not None and language not in service.LANGUAGES.values():
        raise uploads.UploadError("Choose a language from the list.")
    model = str(form.get("model") or "").strip() if other_model else ""
    if not model:
        try:
            model = service.default_model(language)
        except service.MissingSettingError as error:
            raise uploads.UploadError(
                f"{service.language_name(language)} needs a model that this "
                "installation does not have. Choose another language."
            ) from error
    if service.converted_model_missing(language, model):
        raise uploads.UploadError(
            "The Flurin Swiss German model is not ready yet: it has to be "
            "converted once (see the guide). Choose Swiss German instead."
        )

    vad_max_pause = None
    if form.get("vad"):
        least, most, default = VAD_PAUSE
        vad_max_pause = _number(form, "vad_pause", default)
        if not least <= vad_max_pause <= most:
            raise uploads.UploadError(
                f"Join speech across pauses of {least:g} to {most:g} s."
            )
    low = _speakers(form, "min_speakers")
    high = _speakers(form, "max_speakers")
    if low is not None and high is not None and low > high:
        raise uploads.UploadError(
            "The least number of speakers is larger than the most."
        )
    return service.TranscriptionOptions(
        model=model,
        language=language,
        min_speakers=low,
        max_speakers=high,
        vad_max_pause=vad_max_pause,
    )


def _number(form: FormData, field: str, default: float) -> float:
    """Return a number from the form, or ``default`` if it is empty."""
    value = str(form.get(field) or "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError as error:
        raise uploads.UploadError(f"{value} is not a number.") from error


def _speakers(form: FormData, field: str) -> int | None:
    """Return a bound on the number of speakers, or ``None`` if empty."""
    value = str(form.get(field) or "").strip()
    if not value:
        return None
    if not value.isdigit() or not 1 <= int(value) <= MAX_SPEAKERS:
        raise uploads.UploadError(
            f"Give the number of speakers as a whole number from 1 to "
            f"{MAX_SPEAKERS}, or leave it empty."
        )
    return int(value)


# ---------------------------------------------------------------------------
# Starting jobs
# ---------------------------------------------------------------------------


async def _start_one(request: Request, form: FormData) -> Job:
    """Save the recording and queue its transcription."""
    options = read_options(form)
    upload = uploads.single_file(
        form, "recording", suffixes=service.AUDIO_EXTENSIONS
    )
    name = uploads.display_name(upload)
    runner = runner_of(request)
    job = runner.create(
        title=name,
        page=PAGE,
        gpu=gpu_manager.TRANSCRIBE,
        result_template="transcription/result.html",
    )
    try:
        path = await run_in_threadpool(uploads.save, upload, job.inputs)
    except BaseException:
        runner.discard(job)
        raise
    return runner.submit(job, partial(transcribe_one, path, name, options))


async def _start_batch(request: Request, form: FormData) -> Job:
    """Save the ZIP, check that it holds recordings, and queue it."""
    options = read_options(form)
    upload = uploads.single_file(form, "archive", suffixes=(".zip",))
    name = uploads.display_name(upload)
    runner = runner_of(request)
    job = runner.create(
        title=name,
        page=PAGE,
        gpu=gpu_manager.TRANSCRIBE,
        result_template="transcription/batch_result.html",
    )
    try:
        path = await run_in_threadpool(uploads.save, upload, job.inputs)
        try:
            found = await run_in_threadpool(service.zip_recordings, path)
        except ValueError as error:
            raise uploads.UploadError(str(error)) from error
        if not found:
            accepted = ", ".join(
                ext.lstrip(".") for ext in service.AUDIO_EXTENSIONS
            )
            raise uploads.UploadError(
                f"{name} holds no recordings. Accepted: {accepted}."
            )
    except BaseException:
        runner.discard(job)
        raise
    count = len(found)
    job.title = f"{name} ({count} recording{'' if count == 1 else 's'})"
    return runner.submit(job, partial(transcribe_batch, path, name, options))


async def _start_review(request: Request, form: FormData) -> Job:
    """Read the transcript, and start preparing the recording at once."""
    audio_upload, audio_file = review_input(
        form,
        "review_audio",
        "audio_path",
        service.AUDIO_EXTENSIONS,
        "recording",
    )
    table_upload, table_file = review_input(
        form,
        "review_transcript",
        "transcript_path",
        service.TABLE_EXTENSIONS,
        "transcript",
    )
    audio_name = (
        uploads.display_name(audio_upload) if audio_upload else audio_file.name
    )
    table_name = (
        uploads.display_name(table_upload) if table_upload else table_file.name
    )
    runner = runner_of(request)
    job = runner.create(
        title=f"{audio_name}, with {table_name}",
        page=PAGE,
        result_template="transcription/result.html",
    )
    try:
        if audio_upload:
            audio_file = await run_in_threadpool(
                uploads.save, audio_upload, job.inputs
            )
        if table_upload:
            table_file = await run_in_threadpool(
                uploads.save, table_upload, job.inputs
            )
        turns = await run_in_threadpool(read_table, table_file)
    except BaseException:
        runner.discard(job)
        raise
    return runner.submit(
        job,
        partial(review, audio_file, audio_name, turns, table_name),
        at_once=True,
    )


def review_input(
    form: FormData,
    upload_field: str,
    path_field: str,
    suffixes: tuple[str, ...],
    what: str,
) -> tuple[UploadFile | None, Path]:
    """Return a file for the review: an upload, or a path on the cluster.

    Args:
        form: The form.
        upload_field: The field of the upload.
        path_field: The field of the path, used without an upload.
        suffixes: Accepted file name endings.
        what: What the file is, for messages: ``"recording"``.

    Returns:
        ``(upload, path)``: the upload and a placeholder path until it is
        saved, or ``None`` and the file on the cluster.

    Raises:
        UploadError: If there is neither, or the file is not valid.
    """
    found = uploads.files(form, upload_field)
    if found:
        uploads.check_suffix(found[0], suffixes)
        return found[0], Path(uploads.display_name(found[0]))
    path = str(form.get(path_field) or "").strip()
    if not path:
        raise uploads.UploadError(
            f"Choose the {what}, or give its path on the cluster."
        )
    return None, cluster_file(path, suffixes, what)


def cluster_file(value: str, suffixes: tuple[str, ...], what: str) -> Path:
    """Check a path on the cluster that the form names.

    The server runs as the session's owner, so it reads what they can.

    Args:
        value: The path as typed.
        suffixes: Accepted file name endings.
        what: What the file is, for messages: ``"recording"``.

    Returns:
        The file.

    Raises:
        UploadError: If the path is not absolute, names no readable file,
            or a file of another type.
    """
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise uploads.UploadError(
            f"Give the {what}'s full path, starting with /."
        )
    if not path.is_file() or not os.access(path, os.R_OK):
        raise uploads.UploadError(
            f"There is no {what} at {value}, or it cannot be read."
        )
    if path.suffix.lower() not in suffixes:
        accepted = ", ".join(ext.lstrip(".") for ext in suffixes)
        raise uploads.UploadError(
            f"{path.name} is not a {what} this page reads. "
            f"Accepted: {accepted}."
        )
    return path


def read_table(path: Path) -> list[service.Turn]:
    """Read a transcript table into passages, for a review.

    Args:
        path: The table.

    Returns:
        The passages.

    Raises:
        UploadError: If the file is too large, not a transcript table, or
            holds no passages.
    """
    if path.stat().st_size > TABLE_LIMIT:
        raise uploads.UploadError(
            f"{path.name} is too large for a transcript table."
        )
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    try:
        turns = service.turns_from_table(text)
    except ValueError as error:
        raise uploads.UploadError(f"{path.name}: {error}") from error
    if not turns:
        raise uploads.UploadError(f"{path.name} holds no passages.")
    return turns


# ---------------------------------------------------------------------------
# The jobs' work
# ---------------------------------------------------------------------------


def transcribe_one(
    path: Path,
    name: str,
    options: service.TranscriptionOptions,
    context: JobContext,
) -> Listening:
    """Transcribe one recording and offer its exports.

    Args:
        path: The recording, in the job's inputs.
        name: Its name as uploaded.
        options: How to transcribe it.
        context: The job's context.

    Returns:
        What the player shows.

    Raises:
        RuntimeError: If the recording could not be transcribed.
    """
    stem = service.base_name(name)
    context.report(Progress("Preparing the recording for the player", 0.0))
    playback = service.prepare_playback(
        path, context.outputs, stem, wav=True, cancel=context.cancel
    )
    context.check_cancelled()
    result = service.run_transcription(
        [service.AudioFile(path, name)],
        options,
        on_progress=context.report,
        cancel=context.cancel,
    )
    if not result.transcripts:
        reason = (
            result.skipped[0].reason if result.skipped else "nothing came back"
        )
        raise RuntimeError(f"{name} could not be transcribed: {reason}.")
    transcript = result.transcripts[0]

    context.report(Progress("Writing the files", 1.0))
    files = service.save_transcript_files(transcript, context.outputs, stem)
    archive = context.outputs / f"{stem}_outputs.zip"
    contents = dict(files)
    if playback.wav is not None:
        contents[playback.wav.name] = playback.wav
    service.save_zip(contents, archive)
    if playback.wav is not None:
        playback.wav.unlink()
    for file_name, file_path in files.items():
        context.offer(export(file_name, file_path))
    context.offer(
        Download(archive.name, "All files", "application/zip", path=archive)
    )
    context.offer(recording(playback))

    return Listening(
        audio=playback.path.name,
        duration=transcript.duration_seconds or playback.duration_seconds,
        levels=playback.levels,
        turns=service.turns(transcript),
        spaced=service.writes_spaces(transcript.language),
        language=service.language_name(
            options.language or transcript.language
        ),
        detected=options.language is None,
        models=models(options, transcript),
        notes=[
            *result.notes,
            *(f"{item.name}: {item.reason}." for item in result.skipped),
        ],
    )


def transcribe_batch(
    archive: Path,
    name: str,
    options: service.TranscriptionOptions,
    context: JobContext,
) -> BatchSummary:
    """Transcribe the recordings in a ZIP and offer a ZIP of the results.

    Args:
        archive: The ZIP, in the job's inputs.
        name: Its name as uploaded.
        options: How to transcribe the recordings.
        context: The job's context.

    Returns:
        What was transcribed and what was skipped.

    Raises:
        RuntimeError: If no recording could be transcribed.
    """
    with service.staged_zip(archive) as files:
        context.report(Progress(f"Transcribing {len(files)} recordings", 0.0))
        result = service.run_transcription(
            files, options, on_progress=context.report, cancel=context.cancel
        )
    if not result.transcripts:
        reasons = "; ".join(
            f"{item.name}: {item.reason}" for item in result.skipped
        )
        raise RuntimeError(
            f"No recording in {name} could be transcribed ({reasons})."
        )

    context.report(Progress("Writing the files", 1.0))
    folder = context.outputs / "transcripts"
    contents: dict[str, Path] = {}
    recordings = []
    used: set[str] = set()
    for transcript in result.transcripts:
        stem = service.base_name(transcript.name)
        unique, counter = stem, 1
        while unique.casefold() in used:
            counter += 1
            unique = f"{stem}_{counter}"
        used.add(unique.casefold())
        target = folder / unique
        target.mkdir(parents=True)
        for file_name, path in service.save_transcript_files(
            transcript, target, stem
        ).items():
            contents[f"{unique}/{file_name}"] = path
        recordings.append(
            BatchRecording(
                name=transcript.name,
                folder=unique,
                language=service.language_name(
                    options.language or transcript.language
                ),
                duration=transcript.duration_seconds,
                speakers=(
                    len(
                        {
                            turn.speaker_number
                            for turn in service.turns(transcript)
                        }
                        - {0}
                    )
                    if transcript.has_speakers
                    else None
                ),
            )
        )
    zip_path = context.outputs / f"{service.base_name(name)}_transcripts.zip"
    service.save_zip(contents, zip_path)
    shutil.rmtree(folder)
    context.offer(
        Download(zip_path.name, "All files", "application/zip", path=zip_path)
    )
    return BatchSummary(
        recordings=recordings,
        skipped=list(result.skipped),
        notes=list(result.notes),
        model=service.model_label(options.model),
    )


def review(
    audio: Path,
    audio_name: str,
    turns: list[service.Turn],
    table_name: str,
    context: JobContext,
) -> Listening:
    """Prepare a recording for the player, beside a transcript.

    Args:
        audio: The recording, in the job's inputs or on the cluster.
        audio_name: Its name.
        turns: The transcript's passages.
        table_name: The transcript table's name.
        context: The job's context.

    Returns:
        What the player shows.
    """
    context.report(Progress("Preparing the recording for the player"))
    playback = service.prepare_playback(
        audio,
        context.outputs,
        service.base_name(audio_name),
        cancel=context.cancel,
    )
    context.offer(recording(playback))
    return Listening(
        audio=playback.path.name,
        duration=playback.duration_seconds,
        levels=playback.levels,
        turns=turns,
        source=table_name,
    )


def export(file_name: str, path: Path) -> Download:
    """Return the download of a transcript export.

    Args:
        file_name: The export's name.
        path: The file.

    Returns:
        The download, labelled by what the file is.
    """
    for ending, label, media_type in EXPORTS:
        if file_name.endswith(ending):
            return Download(file_name, label, media_type, path=path)
    return Download(file_name, "File", path=path)


def recording(playback: service.Playback) -> Download:
    """Return the player's recording as a download the card does not list.

    Args:
        playback: The prepared recording.

    Returns:
        The download.
    """
    return Download(
        playback.path.name,
        "Recording",
        "audio/mpeg",
        path=playback.path,
        listed=False,
    )


def models(
    options: service.TranscriptionOptions, transcript: service.Transcript
) -> list[tuple[str, str]]:
    """Return the models a transcript was made with, for the Models tab.

    Args:
        options: The options it was made with.
        transcript: The transcript.

    Returns:
        ``(what, model)`` pairs.
    """
    if options.vad_max_pause is None:
        silence = "not skipped"
    else:
        silence = (
            "skipped (Silero voice activity detection), speech joined "
            f"across pauses up to {options.vad_max_pause:g} s"
        )
    return [
        ("Transcription", service.model_label(options.model)),
        (
            "Word timing",
            "WhisperX alignment model for "
            f"{service.language_name(transcript.language)}",
        ),
        (
            "Speakers",
            service.DIARIZATION_MODEL
            if transcript.has_speakers
            else "not identified",
        ),
        ("Silence", silence),
    ]
