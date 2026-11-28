"""What interfaces get around a run: playback, files, languages, ZIPs."""

import shutil
import zipfile

import numpy as np
import pytest

from textlab.features.transcription import audio, service
from textlab.features.transcription.models import WAVEFORM_LEVELS, Transcript

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="needs ffmpeg"
)

TRANSCRIPT = Transcript(
    name="Interview 1.m4a",
    language="de",
    duration_seconds=2.0,
    segments=[
        {
            "start": 0.0,
            "end": 1.0,
            "text": "Guten Tag.",
            "words": [
                {"word": "Guten", "start": 0.0, "end": 0.4},
                {"word": "Tag.", "start": 0.5, "end": 1.0},
            ],
        }
    ],
)


def tone(seconds, sr=audio.SAMPLE_RATE):
    """A tone that grows louder: quiet at the start, full at the end."""
    times = np.arange(int(seconds * sr)) / sr
    return (np.sin(2 * np.pi * 440 * times) * times / seconds).astype(
        np.float32
    )


def test_the_waveform_follows_the_loudness():
    levels = audio.waveform_levels(tone(2.0), bars=10)
    assert len(levels) == 10
    assert levels == tuple(sorted(levels))
    assert levels[-1] == WAVEFORM_LEVELS
    assert min(levels) >= 1


def test_silence_and_short_audio_have_a_waveform_too():
    assert (
        audio.waveform_levels(np.zeros(1000, np.float32), bars=5) == (1,) * 5
    )
    assert len(audio.waveform_levels(np.ones(3, np.float32), bars=120)) == 3
    assert audio.waveform_levels(np.zeros(0, np.float32)) == ()


@needs_ffmpeg
def test_a_recording_is_prepared_for_the_browser(tmp_path):
    source = tmp_path / "source.wav"
    audio.write_wav(tone(3.0), source)
    playback = service.prepare_playback(source, tmp_path, "talk", wav=True)

    assert playback.path == tmp_path / "talk.mp3"
    assert playback.path.read_bytes()[:3] in (b"ID3", b"\xff\xfb", b"\xff\xf3")
    assert playback.duration_seconds == pytest.approx(3.0, abs=0.01)
    assert len(playback.levels) == audio.WAVEFORM_BARS
    assert playback.wav.read_bytes()[:4] == b"RIFF"
    decoded = audio.load_audio(playback.path)
    assert len(decoded) / audio.SAMPLE_RATE == pytest.approx(3.0, abs=0.1)


@needs_ffmpeg
def test_a_recording_that_cannot_be_read_is_reported(tmp_path):
    source = tmp_path / "broken.mp3"
    source.write_bytes(b"not audio")
    with pytest.raises(service.AudioDecodeError):
        service.prepare_playback(source, tmp_path, "broken")
    assert not (tmp_path / "broken.mp3.mp3").exists()


def test_every_export_is_written_in_download_order(tmp_path):
    files = service.save_transcript_files(TRANSCRIPT, tmp_path)
    assert list(files) == [
        "Interview 1_text.txt",
        "Interview 1.srt",
        "Interview 1.vtt",
        "Interview 1_transcription.csv",
        "Interview 1_words.csv",
        "Interview 1_ELAN_compatible.tsv",
    ]
    assert files["Interview 1_text.txt"].read_text() == "Guten Tag."
    assert all(path.parent == tmp_path for path in files.values())


def test_files_are_zipped_from_disk(tmp_path):
    files = service.save_transcript_files(TRANSCRIPT, tmp_path, "a")
    target = service.save_zip(
        {f"folder/{name}": path for name, path in files.items()},
        tmp_path / "all.zip",
    )
    with zipfile.ZipFile(target) as archive:
        assert archive.read("folder/a_text.txt") == b"Guten Tag."


def test_the_recordings_in_a_zip_are_listed_without_extracting(tmp_path):
    path = tmp_path / "batch.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("a.MP3", b"a")
        archive.writestr("folder/b.wav", b"b")
        archive.writestr("notes.txt", b"n")
        archive.writestr("__MACOSX/._a.MP3", b"m")
        archive.writestr("empty/", b"")
    assert service.zip_recordings(path) == ["a.MP3", "b.wav"]

    other = tmp_path / "other.zip"
    other.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="not a ZIP file"):
        service.zip_recordings(other)


@pytest.mark.parametrize(
    ("code", "name"),
    [("de", "German"), ("ch_de", "Swiss German"), ("xx", "xx"), (None, "")],
)
def test_languages_have_names(code, name):
    assert service.language_name(code) == name


def test_some_languages_are_written_without_spaces():
    assert service.writes_spaces("de")
    assert not service.writes_spaces("zh")
    assert not service.writes_spaces("ja")


def test_the_languages_lead_with_the_common_ones():
    assert list(service.LANGUAGES)[:2] == ["English", "German"]
    assert service.LANGUAGES["Swiss German"] == "ch_de"


def test_interfaces_find_everything_they_need_in_the_service():
    for name in service.__all__:
        assert hasattr(service, name), name
