"""The Meeting Notes page: a recording or a transcript to cited notes.

Two ways to use it, as on the Streamlit page:

- **From a recording**: a job transcribes it (as Transcription does), then
  releases the GPU for the language model and summarizes the transcript.
  The result has the recording in the player: every citation in the
  summary plays the passage it cites.
- **From a transcript**: a segment table from Transcription, a text file,
  or pasted text, summarized by a job.

The summary appears on the job's card while the model writes it. Each
point cites the transcript lines it is based on; the citations are checked
(``cite_summary``) and shown as the passages' times. A result can be
summarized again, with names for the speakers, another summary type,
model or language, without transcribing again. When the summary fails,
the transcript is kept, so it can be summarized again too.
"""

from __future__ import annotations

import dataclasses
import datetime
import os
import shutil
from functools import partial
from pathlib import Path, PurePosixPath

from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse
from markupsafe import Markup, escape
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import FormData, UploadFile

from textlab.common import gpu_manager
from textlab.common.progress import CancelledError, Progress
from textlab.features.meeting_notes import service
from textlab.features.transcription import service as transcription
from textlab.ui.web import uploads
from textlab.ui.web.auth import Checked, protected_router
from textlab.ui.web.jobs import (
    Download,
    Job,
    JobContext,
    JobState,
    render_card,
    runner_of,
)
from textlab.ui.web.pages.transcription import (
    Listening,
    languages,
    read_options,
)
from textlab.ui.web.render import markdown, renderer

router = protected_router()

#: The page's id, which also names its jobs.
PAGE = "meeting_notes"

#: Transcript files the page reads: segment tables and plain text.
TRANSCRIPT_EXTENSIONS = (".csv", ".tsv", ".txt")

#: Largest transcript read, in bytes (an uploaded file or pasted text).
TRANSCRIPT_LIMIT = 20 * 1000 * 1000

#: How long submitting waits for the language model server, in attempts
#: half a second apart.
SERVER_ATTEMPTS = 4

#: Stand-ins in the Markdown for what becomes HTML after rendering: the
#: citations (numbered) and the mark of a point that cites nothing.
CITATION_MARK = "TLCITE{:05d}X"
NO_SOURCE_MARK = "TLNOSOURCEX"


@dataclasses.dataclass(frozen=True)
class Notes:
    """A summary with its transcript: what a Meeting Notes result shows.

    Attributes:
        source: What was summarized: the recording's or the file's name, or
            "Pasted text".
        mode: The summary type, a key of ``service.SUMMARY_MODES``.
        model: The language model.
        output_language: The summary's language; ``None`` for the
            transcript's.
        base_segments: The transcript as read, with the diarization's
            speaker labels; summarizing again starts from it.
        names: The names the speakers were given, by label.
        segments: The transcript as summarized, with the names.
        summary: The checked summary, or ``None`` if summarizing failed.
        html: The summary as HTML, with its citations as links.
        models: The models to choose from when summarizing again.
        listening: The recording in the player, for a recording.
        duration: The recording's length, in seconds.
        remarks: Remarks from the transcription, such as a skipped
            speaker step.
        error: Why summarizing failed, for the user.
        error_details: The details, for support.
    """

    source: str
    mode: str
    model: str
    output_language: str | None
    base_segments: list[service.Segment]
    names: dict[str, str]
    segments: list[service.Segment]
    summary: service.CitedSummary | None
    html: Markup
    models: list[str]
    listening: Listening | None = None
    duration: float | None = None
    remarks: list[str] = dataclasses.field(default_factory=list)
    error: str = ""
    error_details: str = ""

    @property
    def mode_label(self) -> str:
        """The summary type's name."""
        return service.SUMMARY_MODES[self.mode]["label"]

    @property
    def modes(self) -> list[tuple[str, str, str]]:
        """The summary types to choose from: key, name, description."""
        return summary_modes()

    @property
    def words(self) -> int:
        """How many words the transcript has."""
        return service.word_count(self.segments)

    @property
    def labels(self) -> list[str]:
        """The diarization's speaker labels, in order of appearance."""
        return speaker_labels(self.base_segments)

    @property
    def playable(self) -> bool:
        """Whether citations play the recording."""
        return self.listening is not None

    def label(self, segment: service.Segment) -> str:
        """Return how a passage is named: its time, or its line number.

        Args:
            segment: A passage of the transcript.

        Returns:
            For example ``"05:12"`` or ``"line 12"``.
        """
        return service.segment_label(segment)

    def link(self, segment: service.Segment) -> Markup:
        """Return a cited passage's link, as the summary shows it.

        Args:
            segment: A passage of the transcript.

        Returns:
            The link (see :func:`source_link`).
        """
        return source_link(segment, playable=self.playable)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/meeting-notes")
def page(request: Request) -> HTMLResponse:
    """Show the form and the session's notes.

    Args:
        request: The current request.

    Returns:
        The page.
    """
    jobs = runner_of(request).jobs(PAGE)
    return renderer(request).response(
        request,
        "meeting_notes/page.html",
        page_id=PAGE,
        languages=languages(),
        default_model=transcription.DEFAULT_MODEL,
        audio_accept=",".join(transcription.AUDIO_EXTENSIONS),
        audio_formats=" ".join(
            sorted(
                {
                    ext.lstrip(".").upper()
                    for ext in transcription.AUDIO_EXTENSIONS
                }
                - {"OGA"}
            )
        ),
        transcript_accept=",".join(TRANSCRIPT_EXTENSIONS),
        modes=summary_modes(),
        models=service.summary_models(),
        server_ready=service.server_ready(),
        cards=Markup("".join(render_card(request, job) for job in jobs[::-1])),
    )


@router.post("/meeting-notes")
async def submit(request: Request, session: Checked) -> HTMLResponse:
    """Start notes from a recording or a transcript.

    Args:
        request: The form: ``source`` (``"audio"`` or ``"transcript"``);
            for a recording ``recording``, ``language``, ``min_speakers``
            and ``max_speakers``; for a transcript ``transcript`` (a file)
            or ``text``; and ``mode``, ``llm`` (the language model) and
            ``output_language``.
        session: The session, with its CSRF token checked.

    Returns:
        The new job's card.

    Raises:
        UploadError: If a file or a value is missing or not valid, or the
            language model server does not answer.
    """
    form = await uploads.read_form(request, max_files=1, max_fields=20)
    source = str(form.get("source") or "audio")
    if source not in ("audio", "transcript"):
        raise uploads.UploadError("Choose what to summarize.")
    choice = read_choice(form)
    await _check_server()
    if source == "audio":
        job = await _start_from_audio(request, form, choice)
    else:
        job = await _start_from_transcript(request, form, choice)
    return HTMLResponse(render_card(request, job))


@router.post("/meeting-notes/{job_id}/again")
async def summarize_again(
    request: Request, job_id: str, session: Checked
) -> HTMLResponse:
    """Summarize a result's transcript again, with other settings.

    Args:
        request: The form: ``mode``, ``llm``, ``output_language`` and a
            ``name_<label>`` for each speaker.
        job_id: The finished job whose transcript to summarize.
        session: The session, with its CSRF token checked.

    Returns:
        The new job's card.

    Raises:
        HTTPException: If there is no such finished job of this page (404).
        UploadError: If a value is not valid, or the language model server
            does not answer.
    """
    runner = runner_of(request)
    earlier = runner.get(job_id)
    if (
        earlier is None
        or earlier.page != PAGE
        or earlier.state is not JobState.DONE
        or not isinstance(earlier.result, Notes)
    ):
        raise HTTPException(
            status_code=404,
            detail="This result no longer exists. It may have been removed.",
        )
    notes: Notes = earlier.result
    form = await uploads.read_form(request, max_files=0, max_fields=60)
    choice = read_choice(form)
    names = {
        label: str(form.get(f"name_{label}") or "").strip()[:80]
        or notes.names.get(label, label)
        for label in notes.labels
    }
    await _check_server()

    mode = service.SUMMARY_MODES[choice.mode]["label"]
    job = runner.create(
        title=f"{notes.source}, as {mode}",
        page=PAGE,
        gpu=gpu_manager.LLM,
        ollama_model=choice.model,
        result_template="meeting_notes/result.html",
    )
    listening = notes.listening
    try:
        if listening is not None:
            listening = await run_in_threadpool(
                _share_recording, earlier, job, listening
            )
    except BaseException:
        runner.discard(job)
        raise
    runner.submit(
        job,
        partial(
            summarize_transcript,
            Source(
                name=notes.source,
                segments=notes.base_segments,
                listening=listening,
                duration=notes.duration,
            ),
            choice,
            names,
        ),
    )
    return HTMLResponse(render_card(request, job))


# ---------------------------------------------------------------------------
# The form
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Choice:
    """How to summarize: the summary type, the model and the language.

    Attributes:
        mode: A key of ``service.SUMMARY_MODES``.
        model: The Ollama model.
        output_language: ``"English"``, or ``None`` for the transcript's
            language.
    """

    mode: str
    model: str
    output_language: str | None


@dataclasses.dataclass(frozen=True)
class Source:
    """A transcript to summarize, with its recording if there is one.

    Attributes:
        name: What it is, for the result: a file's name, "Pasted text".
        segments: The transcript, with the diarization's speaker labels.
        listening: The recording in the player, if there is one.
        duration: The recording's length, in seconds.
    """

    name: str
    segments: list[service.Segment]
    listening: Listening | None = None
    duration: float | None = None


def summary_modes() -> list[tuple[str, str, str]]:
    """Return the summary types to choose from.

    Returns:
        ``(key, name, description)`` triples, in the service's order.
    """
    return [
        (key, mode["label"], mode["description"])
        for key, mode in service.SUMMARY_MODES.items()
    ]


def read_choice(form: FormData) -> Choice:
    """Read the summary type, the model and the language from the form.

    Args:
        form: The form.

    Returns:
        The choice.

    Raises:
        UploadError: If one of them is not one offered.
    """
    mode = str(form.get("mode") or "")
    if mode not in service.SUMMARY_MODES:
        raise uploads.UploadError("Choose a summary type from the list.")
    models = service.summary_models()
    model = str(form.get("llm") or "")
    if model not in models:
        raise uploads.UploadError("Choose a language model from the list.")
    language = str(form.get("output_language") or "English")
    return Choice(
        mode=mode,
        model=model,
        output_language=None if language == "transcript" else "English",
    )


async def _check_server() -> None:
    """Refuse to start when the language model server does not answer."""
    ready = await run_in_threadpool(service.server_ready, SERVER_ATTEMPTS)
    if not ready:
        raise uploads.UploadError(
            "The language model server (Ollama) does not answer. Try again "
            "in a minute; if it keeps failing, send ollama.log from the "
            "session's folder to support."
        )


def speaker_labels(segments: list[service.Segment]) -> list[str]:
    """Return the diarization's speaker labels in order of appearance.

    Args:
        segments: The transcript.

    Returns:
        Labels such as ``SPEAKER_00``, each once.
    """
    known = set(service.speakers(segments))
    return list(
        dict.fromkeys(
            segment.speaker for segment in segments if segment.speaker in known
        )
    )


def default_names(segments: list[service.Segment]) -> dict[str, str]:
    """Name the speakers "Speaker 1", "Speaker 2" in order of appearance.

    The same names as Transcription shows, so the summary, its sources
    and the transcript agree until the user names the speakers.

    Args:
        segments: The transcript.

    Returns:
        The names, by label.
    """
    return {
        label: f"Speaker {number}"
        for number, label in enumerate(speaker_labels(segments), 1)
    }


async def _start_from_audio(
    request: Request, form: FormData, choice: Choice
) -> Job:
    """Save the recording and queue its transcription and summary."""
    # The Whisper model follows from the language: the page offers no other.
    options = read_options(form, other_model=False)
    upload = uploads.single_file(
        form, "recording", suffixes=transcription.AUDIO_EXTENSIONS
    )
    name = uploads.display_name(upload)
    runner = runner_of(request)
    # The language model is loaded only after the transcription, so the
    # transcription has the whole GPU.
    job = runner.create(
        title=name,
        page=PAGE,
        gpu=gpu_manager.TRANSCRIBE,
        result_template="meeting_notes/result.html",
    )
    try:
        path = await run_in_threadpool(uploads.save, upload, job.inputs)
    except BaseException:
        runner.discard(job)
        raise
    return runner.submit(
        job, partial(notes_from_audio, path, name, options, choice)
    )


async def _start_from_transcript(
    request: Request, form: FormData, choice: Choice
) -> Job:
    """Read the transcript and queue its summary."""
    found = uploads.files(form, "transcript")
    pasted = str(form.get("text") or "").strip()
    if found:
        upload = found[0]
        uploads.check_suffix(upload, TRANSCRIPT_EXTENSIONS)
        name = uploads.display_name(upload)
        segments = await run_in_threadpool(read_transcript, upload, name)
    elif pasted:
        if len(pasted.encode()) > TRANSCRIPT_LIMIT:
            raise uploads.UploadError("The pasted text is too long.")
        name = "Pasted text"
        segments = service.segments_from_text(pasted)
    else:
        raise uploads.UploadError("Choose a transcript, or paste its text.")
    if not segments:
        raise uploads.UploadError(f"{name} holds no text to summarize.")

    runner = runner_of(request)
    job = runner.create(
        title=name,
        page=PAGE,
        gpu=gpu_manager.LLM,
        ollama_model=choice.model,
        result_template="meeting_notes/result.html",
    )
    return runner.submit(
        job,
        partial(
            summarize_transcript,
            Source(name=name, segments=segments),
            choice,
            None,
        ),
    )


def read_transcript(upload: UploadFile, name: str) -> list[service.Segment]:
    """Read an uploaded transcript into segments.

    Args:
        upload: The upload: a segment table (``.csv``, ``.tsv``) or text.
        name: Its name.

    Returns:
        The segments.

    Raises:
        UploadError: If it is too large, or a table without the columns of
            a Text Lab transcript.
    """
    upload.file.seek(0, os.SEEK_END)
    if upload.file.tell() > TRANSCRIPT_LIMIT:
        raise uploads.UploadError(f"{name} is too large for a transcript.")
    upload.file.seek(0)
    text = upload.file.read().decode("utf-8-sig", errors="replace")
    if PurePosixPath(name).suffix.lower() == ".txt":
        return service.segments_from_text(text)
    segments = service.segments_from_table(text)
    if not segments and text.strip():
        raise uploads.UploadError(
            f"{name} is not a transcript table: it needs the columns Start, "
            "End and Text, as Transcription writes them in "
            "_transcription.csv."
        )
    return segments


def _share_recording(
    earlier: Job, job: Job, listening: Listening
) -> Listening | None:
    """Give a new job the recording of an earlier one, for its player.

    The file is linked, not copied, where the file system allows it, so
    removing either job leaves the other's player working.

    Returns:
        The player of the new job; ``None`` if the recording is gone.
    """
    item = earlier.download(listening.audio)
    if item is None or item.path is None or not item.path.exists():
        return None
    target = job.outputs / item.path.name
    try:
        os.link(item.path, target)
    except OSError:
        shutil.copyfile(item.path, target)
    return listening


# ---------------------------------------------------------------------------
# The jobs' work
# ---------------------------------------------------------------------------


def notes_from_audio(
    path: Path,
    name: str,
    options: transcription.TranscriptionOptions,
    choice: Choice,
    context: JobContext,
) -> Notes:
    """Transcribe a recording, then summarize the transcript.

    Args:
        path: The recording, in the job's inputs.
        name: Its name as uploaded.
        options: How to transcribe it.
        choice: How to summarize it.
        context: The job's context.

    Returns:
        The notes.

    Raises:
        RuntimeError: If the recording could not be transcribed.
    """
    stem = transcription.base_name(name)
    context.report(Progress("Preparing the recording for the player", 0.0))
    playback = transcription.prepare_playback(
        path, context.outputs, stem, cancel=context.cancel
    )
    context.check_cancelled()
    result = transcription.run_transcription(
        [transcription.AudioFile(path, name)],
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
    files = transcription.save_transcript_files(
        transcript, context.inputs, stem
    )
    table = files[f"{stem}_transcription.csv"]
    segments = service.segments_from_table(table.read_text(encoding="utf-8"))
    kept = context.outputs / table.name
    shutil.move(table, kept)
    context.offer(
        Download(
            kept.name,
            "Transcript table",
            "text/csv; charset=utf-8",
            path=kept,
        )
    )
    listening = Listening(
        audio=playback.path.name,
        duration=transcript.duration_seconds or playback.duration_seconds,
        levels=playback.levels,
        turns=transcription.turns(transcript),
        spaced=transcription.writes_spaces(transcript.language),
    )
    context.use_gpu(gpu_manager.LLM, choice.model)
    return summarize_transcript(
        Source(
            name=name,
            segments=segments,
            listening=listening,
            duration=listening.duration,
        ),
        choice,
        None,
        context,
        remarks=[
            *result.notes,
            *(f"{item.name}: {item.reason}." for item in result.skipped),
        ],
    )


def summarize_transcript(
    source: Source,
    choice: Choice,
    names: dict[str, str] | None,
    context: JobContext,
    remarks: list[str] | None = None,
) -> Notes:
    """Summarize a transcript and offer the summary's files.

    A failed summary is kept as the result, with the transcript, so it can
    be summarized again; a cancelled one stops the job.

    Args:
        source: The transcript, with its recording if there is one.
        choice: How to summarize it.
        names: The speakers' names by label; by default "Speaker 1" ...
        context: The job's context.
        remarks: Remarks from the transcription, shown with the result.

    Returns:
        The notes.
    """
    names = names or default_names(source.segments)
    segments = service.rename_speakers(source.segments, names)
    labels = speaker_labels(source.segments)
    common = dict(
        source=source.name,
        mode=choice.mode,
        model=choice.model,
        output_language=choice.output_language,
        base_segments=source.segments,
        names=names,
        segments=segments,
        models=service.summary_models(),
        listening=source.listening,
        duration=source.duration,
        remarks=list(remarks or []),
    )
    stem = (
        "pasted_text"
        if source.name == "Pasted text"
        else transcription.base_name(source.name)
    )
    if source.listening is not None:
        # The player's recording, in this job's outputs (summarizing again
        # links it there).
        recording = context.outputs / source.listening.audio
        context.offer(
            Download(
                recording.name,
                "Recording",
                "audio/mpeg",
                path=recording,
                listed=False,
            )
        )
    transcript_file = context.outputs / f"{stem}_transcript.txt"
    transcript_file.write_text(
        service.transcript_text(segments) + "\n", encoding="utf-8"
    )

    try:
        text = service.summarize(
            choice.model,
            segments,
            choice.mode,
            speaker_context=service.build_speaker_context(
                [names[label] for label in labels]
            ),
            output_language=choice.output_language,
            on_progress=context.report,
            on_text=context.preview,
            cancel=context.cancel,
        )
    except CancelledError:
        raise
    except Exception as error:
        context.offer(
            Download(
                transcript_file.name,
                "Transcript",
                "text/plain; charset=utf-8",
                path=transcript_file,
            )
        )
        return Notes(
            **common,
            summary=None,
            html=Markup(""),
            error=str(error) or type(error).__name__,
            error_details=getattr(error, "details", "") or repr(error),
        )

    cited = service.cite_summary(text, segments)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    duration = (
        transcription.format_duration(source.duration)
        if source.duration is not None
        else None
    )
    files = [
        (
            f"{stem}_summary_{stamp}.md",
            "Summary",
            service.with_footnotes(cited),
        ),
        (
            f"{stem}_notes_{stamp}.md",
            "Summary and transcript",
            service.format_summary_document(
                cited=cited,
                segments=segments,
                source_label=source.name,
                mode_key=choice.mode,
                duration_str=duration,
            ),
        ),
    ]
    for file_name, label, content in files:
        target = context.outputs / file_name
        target.write_text(content, encoding="utf-8")
        context.offer(
            Download(
                file_name, label, "text/markdown; charset=utf-8", path=target
            )
        )
    context.offer(
        Download(
            transcript_file.name,
            "Transcript",
            "text/plain; charset=utf-8",
            path=transcript_file,
        )
    )
    return Notes(
        **common,
        summary=cited,
        html=summary_html(cited, playable=source.listening is not None),
        error="",
        error_details="",
    )


# ---------------------------------------------------------------------------
# Showing the summary
# ---------------------------------------------------------------------------


def summary_html(cited: service.CitedSummary, *, playable: bool) -> Markup:
    """Render a checked summary as HTML, with its citations as links.

    The Markdown is rendered with citations replaced by stand-ins, which
    then become the links: buttons that play the cited passage when there
    is a recording, links to the transcript line otherwise. Points that
    cite nothing are marked.

    Args:
        cited: The summary, checked by ``cite_summary``.
        playable: Whether the result has a recording to play.

    Returns:
        The HTML.
    """
    citations: dict[str, list[service.Segment]] = {}

    def stand_in(sources: list[service.Segment]) -> str:
        mark = CITATION_MARK.format(len(citations))
        citations[mark] = sources
        return mark

    text = service.mark_uncited_points(cited.text, NO_SOURCE_MARK)
    text = service.format_citations(text, cited.sources, stand_in)
    html = str(markdown(text))
    for mark, sources in citations.items():
        html = html.replace(mark, str(citation(sources, playable=playable)))
    html = html.replace(
        NO_SOURCE_MARK, '<span class="cite cite--none">no source</span>'
    )
    return Markup(html)


def citation(sources: list[service.Segment], *, playable: bool) -> Markup:
    """Return the links of one citation, such as ``(05:12, 06:40)``.

    Args:
        sources: The cited segments.
        playable: Whether a recording plays them.

    Returns:
        The links in brackets.
    """
    links = [source_link(segment, playable=playable) for segment in sources]
    return (
        Markup('<span class="cites">(')
        + Markup(", ").join(links)
        + Markup(")</span>")
    )


def source_link(segment: service.Segment, *, playable: bool) -> Markup:
    """Return the link of one cited segment.

    Args:
        segment: The segment.
        playable: Whether a recording plays it.

    Returns:
        A button that plays it from its start, or one that shows its line
        in the transcript (``data-goto``: the element with that
        ``data-line`` in the same result, see ``app.js``).
    """
    label = escape(service.segment_label(segment))
    quote = escape(segment.text[:120])
    if playable and segment.start is not None:
        return Markup(
            f'<button class="cite" type="button" '
            f'data-seek="{segment.start:.2f}" title="Play: {quote}">'
            f"{label}</button>"
        )
    return Markup(
        f'<button class="cite" type="button" data-goto="{segment.id}" '
        f'title="{quote}">{label}</button>'
    )
