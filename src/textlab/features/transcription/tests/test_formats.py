"""Transcript exports keep their exact format; readers load them back.

The expected strings pin the formats users rely on (ELAN imports, subtitle
players, scripts reading the tables), so a change to them is deliberate.
Tables are tab-separated with CRLF line ends, as Python's csv module writes.
"""

import io
import zipfile

import pytest

from textlab.features.transcription import formats
from textlab.features.transcription.models import Transcript

PLAIN = Transcript(
    name="interview.mp3",
    language="en",
    duration_seconds=3662.0,
    segments=[
        {
            "start": 0.0,
            "end": 1.5,
            "text": " Hello world. ",
            "words": [
                {"word": "Hello", "start": 0.0, "end": 0.5},
                {"word": "world.", "start": 0.6, "end": 1.5},
            ],
        },
        {"start": 2.0, "end": 3.25, "text": "", "words": []},
        {
            "start": 3661.5,
            "end": 3662.0,
            "text": "Late.",
            "words": [{"word": "Late.", "start": 3661.5, "end": 3662.0}],
        },
    ],
)

SPEAKERS = Transcript(
    name="meeting.wav",
    language="de",
    duration_seconds=6.0,
    has_speakers=True,
    segments=[
        {
            "start": 0.0,
            "end": 1.8,
            "text": "Hi there. Yes. um",
            "words": [
                {
                    "word": "Hi",
                    "start": 0.0,
                    "end": 0.4,
                    "speaker": "SPEAKER_00",
                },
                {
                    "word": "there.",
                    "start": 0.5,
                    "end": 1.0,
                    "speaker": "SPEAKER_00",
                },
                {
                    "word": "Yes.",
                    "start": 1.2,
                    "end": 1.6,
                    "speaker": "SPEAKER_01",
                },
                {"word": "um", "start": 1.7, "end": 1.8},
            ],
        },
        {
            "start": 5.0,
            "end": 5.5,
            "text": "Next.",
            "words": [
                {
                    "word": "Next.",
                    "start": 5.0,
                    "end": 5.5,
                    "speaker": "SPEAKER_01",
                }
            ],
        },
    ],
)


def rows(*lines):
    return "".join(line + "\r\n" for line in lines)


def test_segment_table_without_speakers():
    assert formats.transcription_csv(PLAIN) == rows(
        "segment_id\tStart\tEnd\tText",
        "1\t00:00:00.000\t00:00:01.500\tHello world.",
        "2\t00:00:02.000\t00:00:03.250\t",
        "3\t01:01:01.500\t01:01:02.000\tLate.",
    )


def test_segment_table_splits_speaker_turns():
    assert formats.transcription_csv(SPEAKERS) == rows(
        "segment_id\tStart\tEnd\tSpeaker\tText",
        "1\t00:00:00.000\t00:00:01.000\tSPEAKER_00\tHi there.",
        "2\t00:00:01.200\t00:00:01.600\tSPEAKER_01\tYes.",
        "3\t00:00:01.700\t00:00:01.800\tUNKNOWN\tum",
        "4\t00:00:05.000\t00:00:05.500\tSPEAKER_01\tNext.",
    )


def test_elan_table_is_the_segment_table_without_ids():
    assert formats.elan_tsv(SPEAKERS) == rows(
        "Start\tEnd\tSpeaker\tText",
        "00:00:00.000\t00:00:01.000\tSPEAKER_00\tHi there.",
        "00:00:01.200\t00:00:01.600\tSPEAKER_01\tYes.",
        "00:00:01.700\t00:00:01.800\tUNKNOWN\tum",
        "00:00:05.000\t00:00:05.500\tSPEAKER_01\tNext.",
    )
    assert formats.elan_tsv(PLAIN).startswith("Start\tEnd\tText\r\n")


def test_word_table_numbers_segments_and_speaker_changes():
    assert formats.words_csv(SPEAKERS) == rows(
        "segment_id\tword_id\tStart\tEnd\tWord\tSpeaker",
        "1\t1\t00:00:00.000\t00:00:00.400\tHi\tSPEAKER_00",
        "1\t2\t00:00:00.500\t00:00:01.000\tthere.\tSPEAKER_00",
        "2\t1\t00:00:01.200\t00:00:01.600\tYes.\tSPEAKER_01",
        "3\t1\t00:00:01.700\t00:00:01.800\tum\tUNKNOWN",
        "4\t1\t00:00:05.000\t00:00:05.500\tNext.\tSPEAKER_01",
    )


def test_word_table_skips_segments_without_words():
    assert formats.words_csv(PLAIN) == rows(
        "segment_id\tword_id\tStart\tEnd\tWord\tSpeaker",
        "1\t1\t00:00:00.000\t00:00:00.500\tHello\tUNKNOWN",
        "1\t2\t00:00:00.600\t00:00:01.500\tworld.\tUNKNOWN",
        "2\t1\t01:01:01.500\t01:01:02.000\tLate.\tUNKNOWN",
    )


def test_srt_numbers_only_cues_with_text():
    assert formats.srt(PLAIN) == (
        "1\n00:00:00,000 --> 00:00:01,500\nHello world.\n\n"
        "2\n01:01:01,500 --> 01:01:02,000\nLate.\n\n"
    )


def test_vtt():
    assert formats.vtt(PLAIN) == (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:01.500\nHello world.\n\n"
        "01:01:01.500 --> 01:01:02.000\nLate.\n\n"
    )


def test_transcript_files_names_and_order():
    files = formats.transcript_files(PLAIN, "interview")
    assert list(files) == [
        "interview_transcription.csv",
        "interview_ELAN_compatible.tsv",
        "interview_words.csv",
        "interview_text.txt",
        "interview.srt",
        "interview.vtt",
    ]
    assert files["interview_text.txt"] == "Hello world.\nLate."


def test_batch_zip_has_one_folder_per_recording():
    archive = zipfile.ZipFile(io.BytesIO(formats.batch_zip([PLAIN, SPEAKERS])))
    names = archive.namelist()
    assert "interview/interview_transcription.csv" in names
    assert "meeting/meeting.vtt" in names
    assert len(names) == 12


@pytest.mark.parametrize(
    "seconds,expected",
    [(0.4, "0s"), (75, "1m 15s"), (3725, "1h 2m")],
)
def test_format_duration(seconds, expected):
    assert formats.format_duration(seconds) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("01:01:01.500", 3661.5),
        ("02:03.5", 123.5),
        ("7.25", 7.25),
        (3, 3.0),
        ("", 0.0),
        (None, 0.0),
        ("a:b", 0.0),
    ],
)
def test_parse_timestamp(value, expected):
    assert formats.parse_timestamp(value) == expected


def test_segment_items_read_back_with_speakers():
    items, has_speakers = formats.load_transcript_items(
        formats.transcription_csv(SPEAKERS), "segments"
    )
    assert has_speakers
    assert items[0] == {
        "start": 0.0,
        "end": 1.0,
        "text": "Hi there.",
        "speaker": "SPEAKER_00",
    }


def test_word_items_read_back():
    items, _ = formats.load_transcript_items(formats.words_csv(PLAIN), "words")
    assert [item["text"] for item in items] == ["Hello", "world.", "Late."]
    assert items[2]["start"] == 3661.5


def test_plain_text_from_segment_table():
    text = formats.transcription_text_from_csv(
        formats.transcription_csv(SPEAKERS)
    )
    assert text == "Hi there.\nYes.\num\nNext."
