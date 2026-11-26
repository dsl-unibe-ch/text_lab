"""The Transcription page: one recording, a ZIP batch, a review.

The backend is faked: ``run_transcription`` returns a transcript with two
speakers, and ``prepare_playback`` writes stand-in files, so the tests need
neither WhisperX nor ffmpeg. They check what the page adds: the form's
settings, the files, the jobs and their results.
"""

import io
import os
import re
import threading
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

from textlab.common import storage
from textlab.common.config import get_settings
from textlab.features.transcription import audio, service
from textlab.ui.web.jobs import JobState
from textlab.ui.web.tests.conftest import BASE, wait_for
from textlab.ui.web.tests.test_job_pages import htmx, read_events

WORDS_TABLE = (
    "segment_id\tword_id\tStart\tEnd\tWord\tSpeaker\r\n"
    "1\t1\t00:00:00.000\t00:00:00.400\tHello\tSPEAKER_00\r\n"
    "1\t2\t00:00:00.500\t00:00:01.000\tthere.\tSPEAKER_00\r\n"
    "2\t1\t00:00:01.200\t00:00:01.600\tYes.\tSPEAKER_01\r\n"
)


def transcript_for(name):
    """A transcript with two speakers, as WhisperX gives it."""
    return service.Transcript(
        name=name,
        language="de",
        duration_seconds=12.5,
        has_speakers=True,
        segments=[
            {
                "start": 0.0,
                "end": 1.6,
                "text": "Guten Morgen. Hallo.",
                "words": [
                    {
                        "word": "Guten",
                        "start": 0.0,
                        "end": 0.4,
                        "speaker": "SPEAKER_00",
                    },
                    {
                        "word": "Morgen.",
                        "start": 0.5,
                        "end": 1.0,
                        "speaker": "SPEAKER_00",
                    },
                    {
                        "word": "Hallo.",
                        "start": 1.2,
                        "end": 1.6,
                        "speaker": "SPEAKER_01",
                    },
                ],
            }
        ],
    )


@pytest.fixture
def backend(monkeypatch, tmp_path):
    """Fake the transcription and the playback; record the calls."""
    monkeypatch.setenv("TEXT_LAB_WORKDIR", str(tmp_path / "work"))
    get_settings.cache_clear()
    storage.get_workspace.cache_clear()
    calls = SimpleNamespace(runs=[], gate=None)

    def prepare_playback(source, folder, name, *, wav=False, cancel=None):
        assert source.is_file()
        mp3 = folder / f"{name}.mp3"
        mp3.write_bytes(b"ID3" + bytes(997))
        wav_path = None
        if wav:
            wav_path = folder / f"{name}.wav"
            wav_path.write_bytes(b"RIFF" + bytes(96))
        return service.Playback(
            path=mp3, duration_seconds=12.5, levels=(1, 4, 8, 2), wav=wav_path
        )

    def run_transcription(files, options, *, on_progress, cancel=None):
        calls.runs.append(([file.name for file in files], options))
        for file in files:
            assert file.path.is_file()
        if calls.gate is not None:
            assert calls.gate.wait(10)
        return service.TranscriptionResult(
            transcripts=[
                transcript_for(file.name)
                for file in files
                if not file.name.startswith("broken")
            ],
            skipped=[
                service.SkippedFile(file.name, "could not decode audio")
                for file in files
                if file.name.startswith("broken")
            ],
        )

    monkeypatch.setattr(service, "prepare_playback", prepare_playback)
    monkeypatch.setattr(service, "run_transcription", run_transcription)
    yield calls
    get_settings.cache_clear()
    storage.get_workspace.cache_clear()


def post(client, data, files=None):
    return client.post(
        f"{BASE}/transcription",
        data=data,
        files=files,
        headers=htmx(client),
    )


def submit(client, data, files=None):
    response = post(client, data, files)
    assert response.status_code == 200, response.text[:500]
    return re.search(r'id="job-([0-9a-f]+)"', response.text).group(1)


def finished(runner, job_id):
    job = runner.get(job_id)
    wait_for(lambda: not job.active)
    assert job.state is JobState.DONE, job.error_details
    return job


def zip_of(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def test_the_page_offers_three_ways_and_the_languages(client):
    page = client.get(f"{BASE}/transcription").text
    for mode in ("one", "batch", "review"):
        assert f'name="mode" value="{mode}"' in page
    assert '<option value="" data-readout-text="large-v3-turbo" selected>' in (
        page
    )
    assert '<option value="de" data-readout-text="large-v3-turbo">' in page
    assert f'hx-post="{BASE}/transcription"' in page


def test_one_recording_is_transcribed_with_every_file(
    client, runner, backend, released
):
    job_id = submit(
        client,
        {
            "mode": "one",
            "language": "de",
            "vad": "1",
            "vad_pause": "0.3",
            "min_speakers": "2",
            "max_speakers": "3",
        },
        {"recording": ("Interview 1.m4a", b"audio", "audio/mp4")},
    )
    job = finished(runner, job_id)
    [(names, options)] = backend.runs
    assert names == ["Interview 1.m4a"]
    assert options.language == "de"
    assert options.model == "large-v3-turbo"
    assert options.vad_max_pause == 0.3
    assert (options.min_speakers, options.max_speakers) == (2, 3)
    assert released == ["transcribe"]
    assert not job.inputs.exists()

    card = client.get(f"{BASE}/jobs/{job_id}").text
    assert "German</li>" in card
    assert "2 speakers" in card
    assert 'class="speaker speaker--2">Speaker 2</span>' in card
    assert 'data-s="0.50" data-e="1.00" data-t="00:00">Morgen.</span>' in card
    assert f'src="{BASE}/jobs/{job_id}/files/Interview%201.mp3"' in card
    listed = re.findall(r'class="file__name">([^<]+)<', card)
    assert listed == [
        "Interview 1_text.txt",
        "Interview 1.srt",
        "Interview 1.vtt",
        "Interview 1_transcription.csv",
        "Interview 1_words.csv",
        "Interview 1_ELAN_compatible.tsv",
        "Interview 1_outputs.zip",
    ]

    text = client.get(f"{BASE}/jobs/{job_id}/files/Interview%201_text.txt")
    assert text.text == "Guten Morgen.\nHallo."
    archive = client.get(
        f"{BASE}/jobs/{job_id}/files/Interview%201_outputs.zip"
    )
    with zipfile.ZipFile(io.BytesIO(archive.content)) as opened:
        assert sorted(opened.namelist()) == sorted(
            [*listed[:-1], "Interview 1.wav"]
        )
    assert not (job.outputs / "Interview 1.wav").exists()


def test_the_player_can_jump_into_the_recording(client, runner, backend):
    job_id = submit(
        client,
        {"mode": "one", "language": "de"},
        {"recording": ("talk.mp3", b"audio", "audio/mpeg")},
    )
    finished(runner, job_id)
    part = client.get(
        f"{BASE}/jobs/{job_id}/files/talk.mp3", headers={"Range": "bytes=0-99"}
    )
    assert part.status_code == 206
    assert len(part.content) == 100
    assert part.headers["content-type"] == "audio/mpeg"


def test_the_language_is_detected_unless_one_is_chosen(
    client, runner, backend
):
    job_id = submit(
        client,
        {"mode": "one", "language": ""},
        {"recording": ("talk.wav", b"audio", "audio/wav")},
    )
    finished(runner, job_id)
    assert backend.runs[0][1].language is None
    assert "German, detected" in client.get(f"{BASE}/jobs/{job_id}").text


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"language": "xx"}, "Choose a language from the list"),
        ({"min_speakers": "3", "max_speakers": "2"}, "larger than the most"),
        ({"min_speakers": "two"}, "whole number"),
        ({"max_speakers": "0"}, "whole number"),
        ({"vad": "1", "vad_pause": "5"}, "Join speech across pauses"),
        ({"vad": "1", "vad_pause": "soon"}, "not a number"),
        ({"mode": "all"}, "Choose what to transcribe"),
    ],
)
def test_settings_are_checked_before_anything_runs(
    client, runner, backend, fields, message
):
    data = {"mode": "one", "language": "de", **fields}
    response = post(
        client, data, {"recording": ("a.mp3", b"audio", "audio/mpeg")}
    )
    assert response.status_code == 422
    assert message in response.text
    assert runner.jobs() == []
    assert backend.runs == []


def test_a_file_of_another_type_is_refused(client, runner, backend):
    response = post(
        client,
        {"mode": "one", "language": "de"},
        {"recording": ("notes.txt", b"text", "text/plain")},
    )
    assert response.status_code == 422
    assert "not a file this page reads" in response.text
    assert runner.jobs() == []


def test_the_flurin_model_must_be_converted_first(
    client, runner, backend, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        service, "default_model", lambda language: str(tmp_path / "missing")
    )
    response = post(
        client,
        {"mode": "one", "language": "ch_de_flurin"},
        {"recording": ("a.mp3", b"audio", "audio/mpeg")},
    )
    assert response.status_code == 422
    assert "Flurin" in response.text


def test_another_model_can_be_named(client, runner, backend):
    job_id = submit(
        client,
        {"mode": "one", "language": "en", "model": "large-v3"},
        {"recording": ("a.mp3", b"audio", "audio/mpeg")},
    )
    finished(runner, job_id)
    assert backend.runs[0][1].model == "large-v3"


def test_a_zip_of_recordings_gives_one_zip(client, runner, backend):
    archive = zip_of(
        {
            "a.mp3": b"a",
            "folder/b.wav": b"b",
            "other/a.wav": b"a2",
            "broken.ogg": b"x",
            "readme.txt": b"r",
            "__MACOSX/._a.mp3": b"m",
        }
    )
    job_id = submit(
        client,
        {"mode": "batch", "language": ""},
        {"archive": ("Focus groups.zip", archive, "application/zip")},
    )
    job = finished(runner, job_id)
    assert job.title == "Focus groups.zip (4 recordings)"
    assert sorted(backend.runs[0][0]) == [
        "a.mp3",
        "a.wav",
        "b.wav",
        "broken.ogg",
    ]

    card = client.get(f"{BASE}/jobs/{job_id}").text
    assert "3 recordings transcribed" in card
    assert "broken.ogg: could not decode audio" in card
    assert re.findall(r'class="file__name">([^<]+)<', card) == [
        "Focus groups_transcripts.zip"
    ]
    response = client.get(
        f"{BASE}/jobs/{job_id}/files/Focus%20groups_transcripts.zip"
    )
    with zipfile.ZipFile(io.BytesIO(response.content)) as opened:
        names = opened.namelist()
    assert "a/a_text.txt" in names
    assert "a_2/a_words.csv" in names
    assert "b/b.srt" in names
    assert not (job.outputs / "transcripts").exists()


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (zip_of({"readme.txt": b"r"}), "holds no recordings"),
        (b"not a zip", "is not a ZIP file"),
    ],
)
def test_a_zip_without_recordings_is_refused(
    client, runner, backend, content, message
):
    response = post(
        client,
        {"mode": "batch", "language": ""},
        {"archive": ("batch.zip", content, "application/zip")},
    )
    assert response.status_code == 422
    assert message in response.text
    assert runner.jobs() == []


def test_a_review_opens_while_a_transcription_runs(client, runner, backend):
    backend.gate = threading.Event()
    long = submit(
        client,
        {"mode": "one", "language": "de"},
        {"recording": ("long.mp3", b"audio", "audio/mpeg")},
    )
    wait_for(lambda: runner.get(long).state is JobState.RUNNING)

    review = submit(
        client,
        {"mode": "review"},
        {
            "review_audio": ("talk.wav", b"audio", "audio/wav"),
            "review_transcript": ("talk_words.csv", WORDS_TABLE, "text/csv"),
        },
    )
    job = finished(runner, review)
    assert runner.get(long).state is JobState.RUNNING
    assert job.title == "talk.wav, with talk_words.csv"
    card = client.get(f"{BASE}/jobs/{review}").text
    assert "talk_words.csv</li>" in card
    assert 'data-t="00:00">Hello</span>' in card
    assert "Speaker 2" in card
    assert 'name="' + review + '-view" value="words"' in card
    assert "file__name" not in card
    backend.gate.set()
    finished(runner, long)


def test_a_review_reads_files_on_the_cluster(
    client, runner, backend, tmp_path
):
    audio = tmp_path / "talk.mp3"
    audio.write_bytes(b"audio")
    table = tmp_path / "talk_transcription.csv"
    table.write_text("Start\tEnd\tText\n00:00:01.000\t00:00:02.000\tHi.\n")
    job_id = submit(
        client,
        {
            "mode": "review",
            "audio_path": str(audio),
            "transcript_path": str(table),
        },
    )
    finished(runner, job_id)
    card = client.get(f"{BASE}/jobs/{job_id}").text
    assert "Hi.</p>" in card
    assert 'value="words"' not in card
    assert audio.exists() and table.exists()


@pytest.mark.parametrize(
    ("fields", "files", "message"),
    [
        ({}, None, "Choose the recording"),
        ({"audio_path": "talk.mp3"}, None, "full path"),
        ({"audio_path": "/nonexistent/talk.mp3"}, None, "cannot be read"),
        (
            {},
            {"review_audio": ("a.mp3", b"audio", "audio/mpeg")},
            "Choose the transcript",
        ),
        (
            {},
            {
                "review_audio": ("a.mp3", b"audio", "audio/mpeg"),
                "review_transcript": (
                    "a.csv",
                    "name\tage\nx\t1\n",
                    "text/csv",
                ),
            },
            "not a transcript table",
        ),
        (
            {},
            {
                "review_audio": ("a.mp3", b"audio", "audio/mpeg"),
                "review_transcript": ("a.pdf", b"%PDF", "application/pdf"),
            },
            "not a file this page reads",
        ),
    ],
)
def test_a_review_needs_a_recording_and_a_transcript(
    client, runner, backend, fields, files, message
):
    response = post(client, {"mode": "review", **fields}, files)
    assert response.status_code == 422
    assert message in response.text
    assert runner.jobs() == []


def test_home_lists_the_jobs_and_follows_them(client, runner, backend):
    home = client.get(f"{BASE}/").text
    assert 'id="recent-jobs"' in home
    assert "No jobs yet" in home

    job_id = submit(
        client,
        {"mode": "one", "language": "de"},
        {"recording": ("talk.mp3", b"audio", "audio/mpeg")},
    )
    finished(runner, job_id)
    home = client.get(f"{BASE}/").text
    assert f'href="{BASE}/transcription#job-{job_id}"' in home
    updates = "".join(data for name, data in read_events(client, page="home"))
    assert 'id="recent-jobs" hx-swap-oob="true"' in updates
    assert "talk.mp3" in updates


HF_HUB = os.path.join(os.environ.get("HF_HOME", "/opt/huggingface"), "hub")


@pytest.fixture
def workspace_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("TEXT_LAB_WORKDIR", str(tmp_path / "work"))
    get_settings.cache_clear()
    storage.get_workspace.cache_clear()
    yield
    get_settings.cache_clear()
    storage.get_workspace.cache_clear()


@pytest.mark.container
@pytest.mark.slow
@pytest.mark.skipif(
    not (os.path.isdir(HF_HUB) and os.listdir(HF_HUB)),
    reason="the Hugging Face model store is not mounted",
)
def test_a_recording_goes_through_the_real_pipeline(
    client, runner, workspace_dir
):
    # A tone, so no words are expected: the test checks that the page,
    # the worker, ffmpeg and the files work together, not the transcript.
    seconds = 3
    time = np.arange(audio.SAMPLE_RATE * seconds) / audio.SAMPLE_RATE
    tone = (0.1 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    job_id = submit(
        client,
        {"mode": "one", "language": "en", "model": "tiny"},
        {"recording": ("tone.wav", audio.to_wav_bytes(tone), "audio/wav")},
    )
    job = runner.get(job_id)
    wait_for(lambda: not job.active, timeout=600)
    assert job.state is JobState.DONE, job.error_details

    card = client.get(f"{BASE}/jobs/{job_id}").text
    assert "00:03 of audio" in card
    assert 'data-duration="3.00"' in card
    recording = client.get(f"{BASE}/jobs/{job_id}/files/tone.mp3")
    assert recording.status_code == 200
    assert recording.content[:3] == b"ID3"
    archive = client.get(f"{BASE}/jobs/{job_id}/files/tone_outputs.zip")
    with zipfile.ZipFile(io.BytesIO(archive.content)) as opened:
        assert "tone.wav" in opened.namelist()
        assert "tone_words.csv" in opened.namelist()
