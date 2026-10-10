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


def test_call_no_think_falls_back_for_clients_without_think():
    calls = []

    def legacy_chat(*, model, messages):
        calls.append(model)
        return "reply"

    assert ollama.call_no_think(legacy_chat, model="m", messages=[]) == (
        "reply"
    )
    assert calls == ["m"]


@pytest.mark.parametrize(
    "error",
    [
        TypeError("Client.chat() got an unexpected keyword argument 'think'"),
        TypeError("API failure"),
        RuntimeError("model does not support thinking"),
    ],
)
def test_call_no_think_propagates_errors_raised_inside_the_client(error):
    calls = []

    def chat(**request):
        calls.append(request)
        raise error

    with pytest.raises(type(error)) as caught:
        ollama.call_no_think(chat, model="m")
    assert caught.value is error
    assert calls == [{"think": False, "model": "m"}]


@pytest.mark.parametrize(
    "name,expected",
    [
        ("llama3", "llama3:latest"),
        ("qwen3:8b", "qwen3:8b"),
        ("registry.io:5000/team/model", "registry.io:5000/team/model:latest"),
    ],
)
def test_canonical_model_name(name, expected):
    assert ollama.canonical_model_name(name) == expected


def test_running_model_names_are_canonical():
    server = SimpleNamespace(
        ps=lambda: {"models": [{"model": "llama3"}, {"name": "qwen3:8b"}]}
    )
    assert ollama.running_model_names(server) == {
        "llama3:latest",
        "qwen3:8b",
    }


@pytest.mark.parametrize(
    "response",
    [{"models": None}, {"models": [{"model": ""}]}, ConnectionError()],
)
def test_running_model_names_are_unknown_when_untrustworthy(response):
    def ps():
        if isinstance(response, Exception):
            raise response
        return response

    assert ollama.running_model_names(SimpleNamespace(ps=ps)) is None


def test_installed_model_names_are_canonical():
    server = SimpleNamespace(
        list=lambda: SimpleNamespace(
            models=[SimpleNamespace(model="glm-ocr"), {"name": "qwen3:8b"}]
        )
    )
    assert ollama.installed_model_names(server) == {
        "glm-ocr:latest",
        "qwen3:8b",
    }


@pytest.mark.parametrize("response", [{"models": None}, ConnectionError()])
def test_installed_model_names_are_unknown_without_an_answer(response):
    def listing():
        if isinstance(response, Exception):
            raise response
        return response

    assert ollama.installed_model_names(SimpleNamespace(list=listing)) is None


class FakeServer:
    """Loaded models that leave memory a few polls after being unloaded."""

    def __init__(self, loaded, polls_until_free=2):
        self.loaded = list(loaded)
        self.unloaded = []
        self.polls_until_free = polls_until_free

    def ps(self):
        if self.unloaded:
            self.polls_until_free -= 1
            if self.polls_until_free < 0:
                self.loaded = [
                    m for m in self.loaded if m not in self.unloaded
                ]
        return {"models": [{"name": name} for name in self.loaded]}

    def generate(self, model, prompt, keep_alive):
        assert keep_alive == 0
        self.unloaded.append(model)


def test_release_models_waits_until_memory_is_free(monkeypatch):
    server = FakeServer(["vision:20b", "qwen3:latest"])
    monkeypatch.setattr(client, "ps", server.ps)
    monkeypatch.setattr(client, "generate", server.generate)
    monkeypatch.setattr(ollama.time, "sleep", lambda seconds: None)
    released = ollama.release_models(keep={"qwen3:latest"})
    assert released == ["vision:20b"]
    assert server.unloaded == ["vision:20b"]
    assert server.loaded == ["qwen3:latest"]


def test_release_models_does_nothing_when_nothing_is_loaded(monkeypatch):
    server = FakeServer([])
    monkeypatch.setattr(client, "ps", server.ps)
    monkeypatch.setattr(client, "generate", server.generate)
    assert ollama.release_models() == []
    assert server.unloaded == []


def test_release_models_gives_up_after_the_timeout(monkeypatch):
    server = FakeServer(["stuck:latest"], polls_until_free=10**9)
    monkeypatch.setattr(client, "ps", server.ps)
    monkeypatch.setattr(client, "generate", server.generate)
    clock = iter(range(0, 1000, 1))
    monkeypatch.setattr(ollama.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(ollama.time, "sleep", lambda seconds: None)
    assert ollama.release_models(timeout=5) == ["stuck:latest"]
