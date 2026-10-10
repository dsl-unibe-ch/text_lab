"""Summaries and transcript helpers, with Ollama faked."""

from types import SimpleNamespace

import ollama as client
import pytest

from textlab.common.ollama import CHARS_PER_TOKEN, MAX_CONTEXT_TOKENS
from textlab.features.meeting_notes import service

CSV = (
    "segment_id\tStart\tEnd\tSpeaker\tText\r\n"
    "1\t00:00:00.000\t00:00:01.000\tSPEAKER_00\tHello everyone.\r\n"
    "2\t00:00:01.000\t00:00:02.000\tSPEAKER_01\t \r\n"
    "3\t00:00:02.000\t00:00:03.000\tSPEAKER_01\tThanks.\r\n"
)


def test_speaker_text_from_a_transcript_table():
    assert service.transcript_csv_to_speaker_text(CSV) == (
        "SPEAKER_00: Hello everyone.\nSPEAKER_01: Thanks."
    )


def test_speaker_text_without_speaker_column():
    table = "segment_id\tStart\tEnd\tText\r\n1\t0\t1\tJust text.\r\n"
    assert service.transcript_csv_to_speaker_text(table) == "Just text."
    assert service.transcript_csv_to_speaker_text("") == ""


def test_speaker_labels_are_applied_case_insensitively():
    text = "SPEAKER_00: hi\nspeaker_00: again\nSPEAKER_01: yes"
    labels = {"SPEAKER_00": "Anna", "SPEAKER_01": "SPEAKER_01"}
    assert service.apply_speaker_labels(text, labels) == (
        "Anna: hi\nAnna: again\nSPEAKER_01: yes"
    )


def test_unique_speakers_and_context():
    text = "SPEAKER_01: a\nSPEAKER_00: b\nSPEAKER_01: c"
    speakers = service.extract_unique_speakers(text)
    assert speakers == ["SPEAKER_00", "SPEAKER_01"]
    assert service.build_speaker_context([]) is None
    assert service.build_speaker_context(["Anna"]) == (
        "There is one speaker: Anna."
    )
    assert service.build_speaker_context(["A", "B", "C"]) == (
        "There are 3 speakers: A, B and C."
    )


def test_needs_chunking_follows_the_context_budget():
    limit = MAX_CONTEXT_TOKENS * CHARS_PER_TOKEN
    assert not service.needs_chunking("x" * limit)
    assert service.needs_chunking("x" * (limit + CHARS_PER_TOKEN))


def test_summary_document_layout():
    document = service.format_summary_document(
        summary="- point",
        transcript_text="Anna: hi",
        source_label="meeting.wav",
        mode_key="meeting_notes",
        duration_str="3m 2s",
    )
    assert document.startswith("# Audio-to-Summary: meeting.wav\n\n")
    assert "**Summary type**: Meeting Notes" in document
    assert "**Audio duration**: 3m 2s" in document
    assert document.endswith("## Full Transcript\n\nAnna: hi\n")


@pytest.mark.parametrize(
    "language,expected",
    [
        ("English", "Write your entire response in English."),
        (None, "same language as the transcript"),
        ("transcript", "same language as the transcript"),
    ],
)
def test_language_instruction_is_in_every_prompt(language, expected):
    messages = service._build_single_pass_messages(
        "text", "general", None, language
    )
    assert expected in messages[1]["content"]


@pytest.fixture
def fake_chat(monkeypatch):
    calls = []

    def chat(**kwargs):
        calls.append(kwargs)
        reply = {"message": {"content": f"notes {len(calls)}"}}
        if kwargs["stream"]:
            return iter([{"message": {"content": "sum"}}, reply])
        return reply

    monkeypatch.setattr(client, "chat", chat)
    return calls


def test_partial_notes_one_call_per_chunk_with_progress(
    fake_chat, monkeypatch
):
    monkeypatch.setattr(service, "chunk_text", lambda text: ["a", "b", "c"])
    updates = []
    notes = service.get_partial_notes(
        "m", "long text", "general", on_progress=updates.append
    )
    assert notes == ["notes 1", "notes 2", "notes 3"]
    assert [u.message for u in updates] == [
        "Analyzing part 1 of 3...",
        "Analyzing part 2 of 3...",
        "Analyzing part 3 of 3...",
    ]
    assert [u.fraction for u in updates] == [0.0, 0.25, 0.5]
    assert all(call["think"] is False for call in fake_chat)
    assert (
        fake_chat[0]["options"]["temperature"] == service.SUMMARY_TEMPERATURE
    )


def test_summary_stream_yields_text(fake_chat):
    stream = service.get_summary_stream("m", "text", "lecture")
    assert "".join(stream) == "sumnotes 1"


def test_stream_retries_without_think_when_rejected(monkeypatch):
    calls = []

    def chat(**kwargs):
        calls.append(kwargs)
        if "think" in kwargs:
            raise client.ResponseError("model does not support think")
        return iter([SimpleNamespace(message=SimpleNamespace(content="ok"))])

    monkeypatch.setattr(client, "chat", chat)
    assert "".join(service.get_synthesis_stream("m", ["n"], "general")) == "ok"
    assert len(calls) == 2
