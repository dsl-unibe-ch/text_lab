"""Summaries, prompts and context windows, with Ollama faked."""

from types import SimpleNamespace

import ollama as client
import pytest

from textlab.common.ollama import CHARS_PER_TOKEN
from textlab.features.meeting_notes import service
from textlab.features.meeting_notes.models import Segment


def _segments(*texts: str) -> list[Segment]:
    return [Segment(i, text) for i, text in enumerate(texts, 1)]


def test_speaker_context():
    assert service.build_speaker_context([]) is None
    assert service.build_speaker_context(["Anna"]) == (
        "There is one speaker: Anna."
    )
    assert service.build_speaker_context(["A", "B", "C"]) == (
        "There are 3 speakers: A, B and C."
    )


# ---------------------------------------------------------------------------
# Context window
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("", service.DEFAULT_MAX_CONTEXT),
        ("65536", 65536),
        ("not a number", service.DEFAULT_MAX_CONTEXT),
        ("100", 8192),
    ],
)
def test_max_context_from_the_environment(monkeypatch, value, expected):
    monkeypatch.setenv("TEXTLAB_SUMMARY_NUM_CTX", value)
    assert service.max_context() == expected


def test_context_sizes_end_at_the_maximum(monkeypatch):
    monkeypatch.setenv("TEXTLAB_SUMMARY_NUM_CTX", "20000")
    assert service._context_sizes() == (8192, 16384, 20000)


def test_needs_chunking_follows_the_context_budget(monkeypatch):
    monkeypatch.delenv("TEXTLAB_SUMMARY_NUM_CTX", raising=False)
    budget_chars = service._text_budget() * CHARS_PER_TOKEN
    # "[1] " adds four characters to the line.
    assert not service.needs_chunking(_segments("x" * (budget_chars - 4)))
    assert service.needs_chunking(_segments("x" * (budget_chars + 100)))


def test_short_request_uses_the_smallest_window(monkeypatch):
    monkeypatch.delenv("TEXTLAB_SUMMARY_NUM_CTX", raising=False)
    messages = [{"role": "user", "content": "short"}]
    assert service._context_for(messages) == 8192
    long = [{"role": "user", "content": "x" * 30_000 * CHARS_PER_TOKEN}]
    assert service._context_for(long) == service.DEFAULT_MAX_CONTEXT


def test_groups_respect_the_budget():
    texts = ["a" * 40, "b" * 40, "c" * 40]  # ten tokens each, plus one
    assert service._group_by_budget(texts, 22) == [texts[:2], texts[2:]]
    assert service._group_by_budget(["x" * 400], 5) == [["x" * 400]]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


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


def test_every_prompt_asks_for_citations():
    single = service._build_single_pass_messages("[1] hi", "general", None)
    chunk = service._build_chunk_messages("[1] hi", 1, 2, "general", None)
    synthesis = service._build_synthesis_messages(["- a [1]"], "general")
    condense = service._build_condense_messages(["- a [1]", "- b [2]"])
    for messages in (single, chunk):
        assert service.TRANSCRIPT_CITATION_RULE in messages[1]["content"]
    for messages in (synthesis, condense):
        assert service.NOTES_CITATION_RULE in messages[1]["content"]


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


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
    monkeypatch.delenv("TEXTLAB_SUMMARY_NUM_CTX", raising=False)
    return calls


def test_partial_notes_one_call_per_chunk_with_progress(
    fake_chat, monkeypatch
):
    monkeypatch.setattr(
        service,
        "_group_by_budget",
        lambda lines, budget: [["a"], ["b"], ["c"]],
    )
    updates = []
    notes = service.get_partial_notes(
        "m", _segments("long"), "general", on_progress=updates.append
    )
    assert notes == ["notes 1", "notes 2", "notes 3"]
    assert [u.message for u in updates] == [
        "Analyzing part 1 of 3...",
        "Analyzing part 2 of 3...",
        "Analyzing part 3 of 3...",
    ]
    assert [u.fraction for u in updates] == [0.0, 0.25, 0.5]
    assert all(call["think"] is False for call in fake_chat)
    options = fake_chat[0]["options"]
    assert options["temperature"] == service.SUMMARY_TEMPERATURE
    # Every chunk uses the largest window, so the model loads only once.
    assert {call["options"]["num_ctx"] for call in fake_chat} == {
        service.DEFAULT_MAX_CONTEXT
    }


def test_chunks_keep_whole_numbered_lines(fake_chat, monkeypatch):
    monkeypatch.setattr(service, "_text_budget", lambda: 10)
    service.get_partial_notes("m", _segments("a" * 20, "b" * 20), "general")
    assert len(fake_chat) == 2
    assert "[1] " + "a" * 20 in fake_chat[0]["messages"][1]["content"]
    assert "[2] " + "b" * 20 in fake_chat[1]["messages"][1]["content"]


def test_summary_stream_sends_numbered_lines_and_a_window(fake_chat):
    stream = service.get_summary_stream("m", _segments("hi", "yes"), "lecture")
    assert "".join(stream) == "sumnotes 1"
    prompt = fake_chat[0]["messages"][1]["content"]
    assert "[1] hi\n[2] yes" in prompt
    assert fake_chat[0]["options"]["num_ctx"] == 8192


def test_synthesis_condenses_notes_that_do_not_fit(fake_chat, monkeypatch):
    monkeypatch.setattr(service, "_text_budget", lambda: 25)
    notes = ["a" * 40, "b" * 40, "c" * 40, "d" * 40]  # 11 tokens each
    "".join(service.get_synthesis_stream("m", notes, "general"))
    # Two merges of two notes each, then the streamed synthesis.
    assert [call["stream"] for call in fake_chat] == [False, False, True]
    final = fake_chat[-1]["messages"][1]["content"]
    assert "notes 1" in final and "notes 2" in final


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
    assert calls[1]["options"]["num_ctx"] == calls[0]["options"]["num_ctx"]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_summary_document_layout():
    segments = [Segment(1, "hi", "Anna", 62.0, 63.0)]
    cited = service.cite_summary("- point [1]", segments)
    document = service.format_summary_document(
        cited=cited,
        segments=segments,
        source_label="meeting.wav",
        mode_key="meeting_notes",
        duration_str="3m 2s",
    )
    assert document.startswith("# Audio-to-Summary: meeting.wav\n\n")
    assert "**Summary type**: Meeting Notes" in document
    assert "**Audio duration**: 3m 2s" in document
    assert "- point [^1]" in document
    assert "[^1]: [01:02] Anna: hi" in document
    assert document.endswith("## Full Transcript\n\n[01:02] Anna: hi\n")
