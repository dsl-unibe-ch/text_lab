"""Ollama helpers, against a fake client: no server or model is needed."""

from types import SimpleNamespace

import ollama as client
import pytest

from textlab.common import ollama


@pytest.mark.parametrize(
    "value,expected",
    [
        ("127.0.0.1:16912", ("127.0.0.1", 16912)),
        ("http://localhost:11500/", ("localhost", 11500)),
        ("node12", ("node12", 11434)),
        ("", ("127.0.0.1", 11434)),
    ],
)
def test_server_address(value, expected):
    assert ollama.server_address(value) == expected


def test_server_address_defaults_to_the_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:12345")
    assert ollama.server_address() == ("127.0.0.1", 12345)


def test_message_text_treats_missing_content_as_empty():
    assert ollama.message_text(None) == ""
    assert ollama.message_text({"content": None}) == ""
    assert ollama.message_text({"content": "hi"}) == "hi"
    assert ollama.message_text(SimpleNamespace(content="hey")) == "hey"


def test_estimate_tokens_is_at_least_one():
    assert ollama.estimate_tokens("") == 1
    assert ollama.estimate_tokens("x" * 400) == 100


def test_chunk_text_prefers_paragraph_breaks():
    text = "a" * 30 + "\n\n" + "b" * 30 + "\n" + "c" * 30
    chunks = ollama.chunk_text(text, chunk_size_tokens=10)  # 40 chars
    assert chunks[0] == "a" * 30
    assert "".join(chunks).replace("\n", "") == text.replace("\n", "")
    assert all(chunks)


def test_chunk_text_keeps_short_text_whole():
    assert ollama.chunk_text("short text") == ["short text"]


def test_chat_no_think_disables_thinking(monkeypatch):
    calls = []

    def chat(**kwargs):
        calls.append(kwargs)
        return {"message": {"role": "assistant", "content": "ok"}}

    monkeypatch.setattr(client, "chat", chat)
    reply = ollama.chat_no_think("m", [{"role": "user", "content": "hi"}])
    assert reply["message"]["content"] == "ok"
    assert calls[0]["think"] is False


def test_chat_no_think_retries_without_think_when_rejected(monkeypatch):
    calls = []

    def chat(**kwargs):
        calls.append(kwargs)
        if "think" in kwargs:
            raise client.ResponseError('"m" does not support thinking')
        return {"message": {"role": "assistant", "content": "ok"}}

    monkeypatch.setattr(client, "chat", chat)
    assert ollama.chat_no_think("m", [])["message"]["content"] == "ok"
    assert len(calls) == 2 and "think" not in calls[1]


def test_chat_no_think_does_not_retry_other_errors(monkeypatch):
    def chat(**kwargs):
        raise client.ResponseError("error parsing tool call")

    monkeypatch.setattr(client, "chat", chat)
    with pytest.raises(client.ResponseError, match="tool call"):
        ollama.chat_no_think("m", [])


def test_chat_no_think_normalizes_message_objects(monkeypatch):
    message = SimpleNamespace(role="assistant", content=None, tool_calls=None)
    response = SimpleNamespace(message=message, model="m", done=True)
    monkeypatch.setattr(client, "chat", lambda **kwargs: response)
    reply = ollama.chat_no_think("m", [])
    assert reply == {
        "message": {"role": "assistant", "content": ""},
        "model": "m",
        "done": True,
    }


@pytest.mark.parametrize(
    "entry,expected",
    [
        (SimpleNamespace(model="qwen3:8b"), "qwen3:8b"),
        ({"name": "llama3"}, "llama3"),
        ("gemma3", "gemma3"),
        (("mistral", 1), "mistral"),
    ],
)
def test_extract_model_name(entry, expected):
    assert ollama.extract_model_name(entry) == expected


def test_loaded_models_and_prefix_match(monkeypatch):
    running = {"models": [{"name": "qwen3:8b"}]}
    monkeypatch.setattr(client, "ps", lambda: running)
    assert ollama.get_loaded_models() == ["qwen3:8b"]
    assert ollama.is_model_loaded("qwen3")
    assert not ollama.is_model_loaded("llama3")


def test_loaded_models_is_empty_without_a_server(monkeypatch):
    def unreachable():
        raise ConnectionError

    monkeypatch.setattr(client, "ps", unreachable)
    assert ollama.get_loaded_models() == []
