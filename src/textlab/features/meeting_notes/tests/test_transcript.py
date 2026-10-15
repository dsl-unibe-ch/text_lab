"""Transcripts as numbered segments."""

from textlab.features.meeting_notes import transcript
from textlab.features.meeting_notes.models import Segment

TABLE = (
    "segment_id\tStart\tEnd\tSpeaker\tText\r\n"
    "1\t00:00:00.000\t00:00:01.000\tSPEAKER_00\tHello everyone.\r\n"
    "2\t00:00:01.000\t00:00:02.000\tSPEAKER_01\t \r\n"
    "3\t00:01:05.500\t00:01:07.000\tSPEAKER_01\tThanks.\r\n"
)


def test_segments_from_a_table_skip_empty_rows():
    assert transcript.segments_from_table(TABLE) == [
        Segment(1, "Hello everyone.", "SPEAKER_00", 0.0, 1.0),
        Segment(2, "Thanks.", "SPEAKER_01", 65.5, 67.0),
    ]


def test_segments_from_a_table_without_speakers():
    table = "segment_id\tStart\tEnd\tText\r\n1\t0\t1\tJust text.\r\n"
    assert transcript.segments_from_table(table) == [
        Segment(1, "Just text.", None, 0.0, 1.0)
    ]
    assert transcript.segments_from_table("") == []


def test_segments_from_text_read_speaker_labels():
    text = "SPEAKER_00: Hi there\n\n  Anna: plain line  \n"
    assert transcript.segments_from_text(text) == [
        Segment(1, "Hi there", "SPEAKER_00"),
        Segment(2, "Anna: plain line"),
    ]


def test_long_lines_are_split_into_sentences():
    sentence = "This is a sentence of moderate length. "
    line = sentence * 12  # longer than LONG_LINE_CHARS
    segments = transcript.segments_from_text(line)
    assert len(segments) == 12
    assert [s.id for s in segments] == list(range(1, 13))
    assert segments[0].text == sentence.strip()


def test_speakers_and_renaming():
    segments = transcript.segments_from_table(TABLE)
    assert transcript.speakers(segments) == ["SPEAKER_00", "SPEAKER_01"]
    renamed = transcript.rename_speakers(
        segments, {"SPEAKER_00": " Anna ", "SPEAKER_01": ""}
    )
    assert [s.speaker for s in renamed] == ["Anna", "SPEAKER_01"]
    # Names are not diarization labels, so they are not offered again.
    assert transcript.speakers(renamed) == ["SPEAKER_01"]


def test_labels_use_time_or_line_number():
    assert transcript.format_time(65.9) == "01:05"
    assert transcript.format_time(3725) == "1:02:05"
    assert transcript.segment_label(Segment(4, "x", start=65.0)) == "01:05"
    assert transcript.segment_label(Segment(4, "x")) == "line 4"


def test_numbered_and_display_text():
    segments = transcript.segments_from_table(TABLE)
    assert transcript.numbered_text(segments) == (
        "[1] SPEAKER_00: Hello everyone.\n[2] SPEAKER_01: Thanks."
    )
    assert transcript.transcript_text(segments) == (
        "[00:00] SPEAKER_00: Hello everyone.\n[01:05] SPEAKER_01: Thanks."
    )
    plain = transcript.segments_from_text("one two\nthree")
    assert transcript.transcript_text(plain) == "[1] one two\n[2] three"
    assert transcript.word_count(plain) == 3
