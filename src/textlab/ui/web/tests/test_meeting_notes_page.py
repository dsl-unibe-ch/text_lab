"""The Meeting Notes page: notes from a recording or a transcript.

The transcription is faked as for the Transcription page (the ``backend``
fixture), and ``summarize`` returns a fixed summary with citations, so the
tests check what the page adds: the form, the jobs, the GPU switch between
the two steps, the summary's citations, summarizing again.
"""

import re
import threading
from types import SimpleNamespace

import pytest

from textlab.common.progress import Progress, check_cancelled
from textlab.features.meeting_notes import service
from textlab.ui.web.jobs import JobState
from textlab.ui.web.tests.conftest import BASE, wait_for
from textlab.ui.web.tests.test_job_pages import htmx

SUMMARY = (
    "## Points\n\n"
    "- First point [1]\n"
    "- Second point [2, 3]\n"
    "- A point without a source\n\n"
    "Some <b>bold</b> HTML and [a link](https://example.org)."
)

TABLE = (
    "segment_id\tStart\tEnd\tSpeaker\tText\r\n"
    "1\t00:00:01.000\t00:00:02.000\tSPEAKER_01\tWe ship on Friday.\r\n"
    "2\t00:00:03.000\t00:00:04.000\tSPEAKER_00\tAgreed.\r\n"
    "3\t00:00:05.000\t00:00:06.000\tSPEAKER_01\tAnna writes the notes.\r\n"
)

CHOICE = {"mode": "general", "llm": "small:8b", "output_language": "English"}


@pytest.fixture
def llm(monkeypatch):
    """Fake the language model: a fixed summary; record the calls."""
    calls = SimpleNamespace(runs=[], gate=None, fail=None)

    def summarize(
        model,
        segments,
        mode,
        *,
        speaker_context=None,
        output_language="English",
        on_progress,
        on_text=None,
        cancel=None,
    ):
        calls.runs.append(
            SimpleNamespace(
                model=model,
                segments=list(segments),
                mode=mode,
                speaker_context=speaker_context,
                output_language=output_language,
            )
        )
        on_progress(Progress("Writing the summary"))
        on_text("- First point")
        if calls.gate is not None:
            while not calls.gate.wait(0.01):
                check_cancelled(cancel)
        if calls.fail is not None:
            raise calls.fail
        return SUMMARY

    monkeypatch.setattr(service, "summarize", summarize)
    monkeypatch.setattr(service, "server_ready", lambda attempts=1: True)
    monkeypatch.setattr(
        service, "summary_models", lambda: ["small:8b", "big:70b"]
    )
    return calls


def post(client, data, files=None, path="/meeting-notes"):
    return client.post(
        f"{BASE}{path}", data=data, files=files, headers=htmx(client)
    )


def submit(client, data, files=None, path="/meeting-notes"):
    response = post(client, data, files, path)
    assert response.status_code == 200, response.text[:500]
    return re.search(r'id="job-([0-9a-f]+)"', response.text).group(1)


def finished(runner, job_id):
    job = runner.get(job_id)
    wait_for(lambda: not job.active)
    assert job.state is JobState.DONE, job.error_details
    return job


def listed(card):
    return re.findall(r'class="file__name">([^<]+)<', card)


def test_the_page_offers_both_sources_and_the_summary_types(client, llm):
    page = client.get(f"{BASE}/meeting-notes").text
    assert 'name="source" value="audio" checked' in page
    assert 'name="source" value="transcript"' in page
    for key, mode in service.SUMMARY_MODES.items():
        assert f'value="{key}"' in page
        assert mode["label"] in page
    assert "<option selected>small:8b</option>" in page
    assert "does not answer yet" not in page


def test_the_page_warns_while_the_server_does_not_answer(
    client, llm, monkeypatch
):
    monkeypatch.setattr(service, "server_ready", lambda attempts=1: False)
    assert "does not answer yet" in client.get(f"{BASE}/meeting-notes").text
    response = post(
        client, {"source": "transcript", "text": "Hello.", **CHOICE}
    )
    assert response.status_code == 422
    assert "does not answer" in response.text


def test_pasted_text_is_summarized_with_citations(
    client, runner, llm, released
):
    job_id = submit(
        client,
        {
            "source": "transcript",
            "text": "Hello there.\nWe decided to ship.\nAnything else?",
            "mode": "interview",
            "llm": "big:70b",
            "output_language": "transcript",
        },
    )
    finished(runner, job_id)
    [run] = llm.runs
    assert (run.model, run.mode, run.output_language) == (
        "big:70b",
        "interview",
        None,
    )
    assert [segment.text for segment in run.segments] == [
        "Hello there.",
        "We decided to ship.",
        "Anything else?",
    ]
    assert released == ["llm:big:70b"]

    card = client.get(f"{BASE}/jobs/{job_id}").text
    assert "<h2>Points</h2>" in card
    assert "&lt;b&gt;bold&lt;/b&gt;" in card
    assert 'target="_blank" rel="noopener noreferrer"' in card
    assert '<button class="cite" type="button" data-goto="1"' in card
    assert 'data-goto="2"' in card and 'data-goto="3"' in card
    assert '<span class="cite cite--none">no source</span>' in card
    assert '<li class="turn" data-line="2">' in card
    assert "TLCITE" not in card
    assert "1 point cites nothing" in card
    names = listed(card)
    assert re.fullmatch(r"pasted_text_summary_[\d_-]+\.md", names[0])
    assert re.fullmatch(r"pasted_text_notes_[\d_-]+\.md", names[1])
    assert names[2] == "pasted_text_transcript.txt"
    summary = client.get(f"{BASE}/jobs/{job_id}/files/{names[0]}").text
    assert "[^1]" in summary
    assert "[^1]: [1] Hello there." in summary


def test_a_transcript_table_keeps_its_speakers_and_times(client, runner, llm):
    job_id = submit(
        client,
        {"source": "transcript", **CHOICE},
        {"transcript": ("standup_transcription.csv", TABLE, "text/csv")},
    )
    finished(runner, job_id)
    [run] = llm.runs
    # Named in order of appearance, as Transcription shows them.
    assert [segment.speaker for segment in run.segments] == [
        "Speaker 1",
        "Speaker 2",
        "Speaker 1",
    ]
    assert run.speaker_context == (
        "There are 2 speakers: Speaker 1 and Speaker 2."
    )
    card = client.get(f"{BASE}/jobs/{job_id}").text
    assert 'data-goto="1" title="We ship on Friday.">00:01</button>' in card
    assert '<span class="speaker speaker--1">Speaker 1</span>' in card
    assert 'name="name_SPEAKER_01" value="Speaker 1"' in card


def test_notes_from_a_recording_play_their_sources(
    client, runner, backend, llm, released
):
    job_id = submit(
        client,
        {
            "source": "audio",
            "language": "de",
            "min_speakers": "2",
            **CHOICE,
            "mode": "meeting_notes",
        },
        {"recording": ("standup.m4a", b"audio", "audio/mp4")},
    )
    job = finished(runner, job_id)
    assert released == ["transcribe", "llm:small:8b"]
    [(names, options)] = backend.runs
    assert options.min_speakers == 2
    # The language model is not a Whisper model: the transcription uses the
    # language's own.
    assert options.model == "large-v3-turbo"
    [run] = llm.runs
    assert [(s.speaker, s.text) for s in run.segments] == [
        ("Speaker 1", "Guten Morgen."),
        ("Speaker 2", "Hallo."),
    ]

    card = client.get(f"{BASE}/jobs/{job_id}").text
    assert 'class="result" data-player data-duration="12.50"' in card
    assert '<button class="cite" type="button" data-seek="1.20"' in card
    assert "data-goto" not in card
    assert "1 citation of passages that do not exist was removed" in card
    assert f'src="{BASE}/jobs/{job_id}/files/standup.mp3"' in card
    assert "standup_transcription.csv" in listed(card)
    assert "standup.mp3" not in listed(card)
    assert not job.inputs.exists()


def test_a_failed_summary_keeps_the_transcript(client, runner, backend, llm):
    llm.fail = RuntimeError("model 'small:8b' not found")
    job_id = submit(
        client,
        {"source": "audio", "language": "de", **CHOICE},
        {"recording": ("standup.m4a", b"audio", "audio/mp4")},
    )
    finished(runner, job_id)
    card = client.get(f"{BASE}/jobs/{job_id}").text
    assert "The summary failed" in card
    assert "model &#39;small:8b&#39; not found" in card
    assert 'id="' + job_id + '-tab-transcript" aria-selected="true"' in card
    assert '<details class="disclosure" open>' in card
    assert sorted(listed(card)) == [
        "standup_transcript.txt",
        "standup_transcription.csv",
    ]


def test_summarizing_again_names_the_speakers_and_keeps_the_recording(
    client, runner, backend, llm
):
    first = submit(
        client,
        {"source": "audio", "language": "de", **CHOICE},
        {"recording": ("standup.m4a", b"audio", "audio/mp4")},
    )
    finished(runner, first)
    again = submit(
        client,
        {
            "mode": "interview",
            "llm": "big:70b",
            "output_language": "English",
            "name_SPEAKER_00": "Anna",
            "name_SPEAKER_01": "",
        },
        path=f"/meeting-notes/{first}/again",
    )
    job = finished(runner, again)
    assert job.title == "standup.m4a, as Interview Summary"
    assert len(backend.runs) == 1
    assert [s.speaker for s in llm.runs[1].segments] == ["Anna", "Speaker 2"]

    client.post(f"{BASE}/jobs/{first}/remove", headers=htmx(client))
    recording = client.get(f"{BASE}/jobs/{again}/files/standup.mp3")
    assert recording.status_code == 200
    card = client.get(f"{BASE}/jobs/{again}").text
    assert '<span class="speaker speaker--1">Anna</span>' in card


def test_only_finished_notes_can_be_summarized_again(client, runner, llm):
    response = post(client, CHOICE, path="/meeting-notes/nothing/again")
    assert response.status_code == 404


@pytest.mark.parametrize(
    ("fields", "files", "message"),
    [
        ({"mode": "poem"}, None, "Choose a summary type"),
        ({"llm": "other:1b"}, None, "Choose a language model"),
        ({"source": "video"}, None, "Choose what to summarize"),
        ({"text": ""}, None, "Choose a transcript, or paste its text"),
        ({"text": "   \n  "}, None, "Choose a transcript, or paste its text"),
        (
            {"text": ""},
            {"transcript": ("a.csv", "name\tage\nx\t1\n", "text/csv")},
            "not a transcript table",
        ),
        (
            {"text": ""},
            {"transcript": ("a.pdf", b"%PDF", "application/pdf")},
            "not a file this page reads",
        ),
    ],
)
def test_the_form_is_checked_before_anything_runs(
    client, runner, llm, fields, files, message
):
    data = {"source": "transcript", "text": "Hello.", **CHOICE, **fields}
    response = post(client, data, files)
    assert response.status_code == 422
    assert message in response.text
    assert runner.jobs() == []


def test_the_summary_shows_while_it_is_written_and_can_be_cancelled(
    client, runner, llm
):
    llm.gate = threading.Event()
    job_id = submit(client, {"source": "transcript", "text": "Hi.", **CHOICE})
    job = runner.get(job_id)
    wait_for(lambda: job.preview == "- First point")
    card = client.get(f"{BASE}/jobs/{job_id}").text
    assert "Written so far" in card
    assert "- First point" in card

    client.post(f"{BASE}/jobs/{job_id}/cancel", headers=htmx(client))
    wait_for(lambda: job.state is JobState.CANCELLED)
    assert job.downloads == []


def test_the_language_model_never_reaches_the_transcription(
    client, runner, backend, llm
):
    # A field called "model" (as on Transcription) is not read here.
    job_id = submit(
        client,
        {"source": "audio", "language": "en", "model": "x", **CHOICE},
        {"recording": ("talk.mp3", b"audio", "audio/mpeg")},
    )
    finished(runner, job_id)
    assert backend.runs[0][1].model == "large-v3-turbo"
    assert llm.runs[0].model == "small:8b"
