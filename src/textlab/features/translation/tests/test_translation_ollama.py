"""Offline Ollama budgets, completion checks, and lossless retry coverage."""

import sys
from types import SimpleNamespace

import pytest

from textlab.features.translation import ollama_backend as backend
from textlab.features.translation.chunking import (
    MAX_SPLIT_RETRIES,
    InputTooLongError,
    OutputTruncatedError,
)


def _source(request):
    return request["messages"][1]["content"]


def _response(content="translated", *, as_object=False, **metadata):
    result = {
        "message": {"content": content},
        "done": True,
        "done_reason": "stop",
        "eval_count": 10,
        "prompt_eval_count": 50,
    }
    result.update(metadata)
    if as_object:
        if isinstance(result["message"], dict):
            result["message"] = SimpleNamespace(**result["message"])
        return SimpleNamespace(**result)
    return result


@pytest.fixture(autouse=True)
def fake_ollama(monkeypatch):
    """Never import a real Ollama client or contact a service."""
    state = SimpleNamespace(calls=[], respond=None)

    def chat(**request):
        state.calls.append(request)
        if state.respond is not None:
            return state.respond(request)
        return _response(_source(request))

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=chat))
    return state


def _translate(text, **kwargs):
    return backend.translate_ollama(
        text, "English", "German", "test-model", **kwargs,
    )


@pytest.mark.parametrize("object_response", [False, True])
@pytest.mark.parametrize("object_message", [False, True])
def test_stop_success_and_explicit_options(
    fake_ollama, object_response, object_message,
):
    message = {"content": "  Guten Tag!  "}
    if object_message:
        message = SimpleNamespace(**message)
    response = {"message": message, "done_reason": "stop"}
    if object_response:
        response = SimpleNamespace(**response)
    fake_ollama.respond = lambda request: response
    progress = []
    statuses = []

    result = _translate(
        "Hello!", progress_cb=lambda *args: progress.append(args),
        status_cb=statuses.append,
    )

    assert type(result) is str
    assert result == "Guten Tag!"
    assert progress == [(1, 1)]
    assert statuses == []
    assert len(fake_ollama.calls) == 1
    request = fake_ollama.calls[0]
    assert request["model"] == "test-model"
    assert request["think"] is False
    assert request["messages"][1] == {"role": "user", "content": "Hello!"}
    assert request["options"] == {
        "temperature": 0.2,
        "num_ctx": 4096,
        "num_predict": 1024,
    }


@pytest.mark.parametrize(
    ("formality", "instruction"),
    [
        ("default", ""),
        ("unknown", ""),
        (
            "formal",
            " Use a formal, professional register throughout — polite"
            " pronouns, complete sentences, no colloquialisms.",
        ),
        (
            "informal",
            " Use a casual, conversational register — everyday vocabulary,"
            " contractions where natural, informal pronouns.",
        ),
    ],
)
def test_original_language_and_formality_prompt(
    fake_ollama, formality, instruction,
):
    backend.translate_ollama(
        "Grüezi", "Swiss German", "English", "test-model", formality,
    )
    assert fake_ollama.calls[0]["messages"][0] == {
        "role": "system",
        "content": (
            "You are a professional translator. Translate the user's text "
            "from Swiss German into English. Preserve meaning, tone, "
            "formatting (paragraphs, lists) and named "
            f"entities.{instruction} Do not add commentary. "
            "Return ONLY the translated text."
        ),
    }


@pytest.mark.parametrize("text", ["", " ", "\n \t\r\n\n"])
def test_blank_text_needs_no_client_or_callbacks(
    fake_ollama, monkeypatch, text,
):
    monkeypatch.setitem(sys.modules, "ollama", None)
    events = []
    assert _translate(
        text, progress_cb=lambda *args: events.append(args),
        status_cb=events.append,
    ) == text
    assert not events
    assert not fake_ollama.calls


def test_line_layout_and_boundary_whitespace_survive(fake_ollama):
    translated = {"alpha": "eins", "beta": "zwei"}
    fake_ollama.respond = lambda request: _response(
        translated[_source(request).strip()],
    )
    progress = []
    result = _translate(
        "\n \t\n  alpha \t\r\n\r\nbeta\r\n\n\t",
        progress_cb=lambda *args: progress.append(args),
    )
    assert result == "\n \t\n  eins \t\r\n\r\nzwei\r\n\n\t"
    assert [_source(call) for call in fake_ollama.calls] == [
        "  alpha \t\r", "beta\r",
    ]
    assert progress == [(1, 2), (2, 2)]


def test_utf8_estimate_is_bytes_not_characters():
    assert backend._estimate_tokens("é漢🙂") == 9
    assert backend._estimate_tokens("abc") == 3


@pytest.mark.parametrize(
    "text",
    [
        " \t" + "naïve  東京。 Weiter!\t" * 120 + "Ende \t",
        "最初の文。次の文！最後の文？" * 100,
    ],
)
def test_chunks_preserve_source_without_word_splits(fake_ollama, text):
    progress = []
    assert _translate(
        text, progress_cb=lambda *args: progress.append(args),
    ) == text
    sources = [_source(call) for call in fake_ollama.calls]
    assert len(sources) > 1
    assert "".join(sources) == text
    for source in sources[:-1]:
        assert source[-1].isspace() or source[-1] in "。！？"
    for call in fake_ollama.calls:
        system = call["messages"][0]["content"]
        estimate = backend._estimate_tokens(_source(call))
        assert estimate <= backend._SOURCE_BYTE_CAP == 2048
        assert call["think"] is False
        assert (
            estimate + backend._estimate_tokens(system)
            + backend._TEMPLATE_TOKEN_RESERVE + backend._NUM_PREDICT
        ) <= backend._NUM_CTX
    assert progress == [(i, len(sources)) for i in range(1, len(sources) + 1)]


@pytest.mark.parametrize(("words", "requests"), [(400, 1), (600, 2)])
def test_initial_chunks_use_context_instead_of_output_expansion(
    fake_ollama, words, requests,
):
    system = backend._system_prompt("English", "German", "default")
    assert backend._input_budget(system) == 2048
    text = "word " * words
    assert _translate(text) == text
    assert len(fake_ollama.calls) == requests
    sources = [_source(call) for call in fake_ollama.calls]
    assert "".join(sources) == text
    assert 1024 < len(sources[0].encode("utf-8")) <= 2048


def test_output_expansion_splits_only_after_observed_limit(fake_ollama):
    text = "word " * 400

    def respond(request):
        source = _source(request)
        if source == text:
            return _response("discarded partial", done_reason="length")
        return _response(source.upper())

    fake_ollama.respond = respond
    progress = []
    assert _translate(
        text, progress_cb=lambda *args: progress.append(args),
    ) == text.upper()
    sources = [_source(call) for call in fake_ollama.calls]
    assert sources == [text, "word " * 200, "word " * 200]
    assert progress == [(1, 1)]
    for call in fake_ollama.calls:
        assert call["think"] is False
        assert call["options"]["num_ctx"] == 4096
        assert call["options"]["num_predict"] == 1024


def test_whitespace_only_chunk_is_not_duplicated_or_sent(fake_ollama):
    system = backend._system_prompt("English", "German", "default")
    text = " " * (backend._input_budget(system) - 1) + "word"
    assert _translate(text) == text
    assert [_source(call) for call in fake_ollama.calls] == ["word"]


@pytest.mark.parametrize("formality", ["default", "formal", "informal"])
def test_system_template_and_output_are_reserved(
    fake_ollama, monkeypatch, formality,
):
    source, target = "Français", "日本語"
    system = backend._system_prompt(source, target, formality)
    monkeypatch.setattr(
        backend, "_NUM_CTX",
        backend._estimate_tokens(system) + backend._TEMPLATE_TOKEN_RESERVE
        + backend._NUM_PREDICT + 20,
    )
    assert backend._input_budget(system) == 20
    text = "small words " * 6
    assert backend.translate_ollama(
        text, source, target, "test-model", formality,
    ) == text
    assert len(fake_ollama.calls) > 1
    for call in fake_ollama.calls:
        assert backend._estimate_tokens(_source(call)) <= 20
        assert call["messages"][0]["content"] == system


def test_oversized_system_fails_before_request(fake_ollama):
    with pytest.raises(InputTooLongError, match="system prompt"):
        backend.translate_ollama(
            "small", "é" * backend._NUM_CTX, "German", "test-model",
        )
    assert not fake_ollama.calls


def test_all_source_lines_are_preflighted_before_requests(fake_ollama):
    system = backend._system_prompt("English", "German", "default")
    budget = backend._input_budget(system)
    oversized = "é" * (budget // 2 + 1)
    with pytest.raises(InputTooLongError, match="input budget"):
        _translate("small\n" + oversized)
    assert not fake_ollama.calls


def test_indivisible_word_at_exact_byte_budget_is_allowed(fake_ollama):
    system = backend._system_prompt("English", "German", "default")
    word = "x" * backend._input_budget(system)
    assert _translate(word) == word
    assert [_source(call) for call in fake_ollama.calls] == [word]


@pytest.mark.parametrize("as_object", [False, True])
@pytest.mark.parametrize(
    "metadata",
    [
        {"done_reason": "length", "eval_count": 1},
        {"done_reason": None, "eval_count": backend._NUM_PREDICT},
        {"done_reason": "stop", "eval_count": backend._NUM_PREDICT},
        {"eval_count": backend._NUM_PREDICT + 10},
        {"prompt_eval_count": backend._NUM_CTX - backend._NUM_PREDICT},
        {"prompt_eval_count": backend._NUM_CTX},
        {"done": False},
        {"done_reason": "unknown reason"},
        {"done_reason": None, "eval_count": None},
    ],
)
def test_limit_metadata_discards_content_and_retries(
    fake_ollama, caplog, as_object, metadata,
):
    text = "private-alpha \tprivate-beta"

    def respond(request):
        if _source(request) == text:
            return _response("discarded partial", as_object=as_object,
                             **metadata)
        return _response(_source(request).upper(), as_object=as_object)

    fake_ollama.respond = respond
    statuses = []
    progress = []
    assert _translate(
        text, status_cb=statuses.append,
        progress_cb=lambda *args: progress.append(args),
    ) == text.upper()
    assert [_source(call) for call in fake_ollama.calls] == [
        text, "private-alpha \t", "private-beta",
    ]
    assert all(
        call["options"] == fake_ollama.calls[0]["options"]
        and call["think"] is False
        for call in fake_ollama.calls
    )
    assert progress == [(1, 1)]
    assert "Retrying" in statuses[0]
    assert "recovered" in statuses[-1]
    assert "private-alpha" not in " ".join(statuses) + caplog.text
    assert "discarded partial" not in " ".join(statuses) + caplog.text


@pytest.mark.parametrize("as_object", [False, True])
def test_legacy_completion_count_below_budget_is_accepted(
    fake_ollama, as_object,
):
    fake_ollama.respond = lambda request: _response(
        "fertig", as_object=as_object, done_reason=None,
        eval_count=backend._NUM_PREDICT - 1,
        prompt_eval_count=backend._NUM_CTX - backend._NUM_PREDICT - 1,
    )
    assert _translate("ready") == "fertig"
    assert len(fake_ollama.calls) == 1


@pytest.mark.parametrize(
    "response",
    [
        {"message": {"content": "unverified"}},
        _response(" "),
        _response(None),
        _response(message={}),
        _response(message=None),
        _response(done=None, done_reason=None),
        _response(done_reason=None, eval_count=0),
        _response(eval_count=-1),
        _response(eval_count=True),
        _response(eval_count="10"),
        _response(prompt_eval_count=1.5),
    ],
)
def test_unverifiable_or_empty_output_fails_clearly(fake_ollama, response):
    fake_ollama.respond = lambda request: response
    with pytest.raises(OutputTruncatedError, match="No partial translation"):
        _translate("word")
    assert len(fake_ollama.calls) == 1


@pytest.mark.parametrize("text", ["word", " \tword \t", "漢字"])
def test_indivisible_truncated_input_is_not_repeated(fake_ollama, text):
    fake_ollama.respond = lambda request: _response(done_reason="length")
    statuses = []
    with pytest.raises(OutputTruncatedError, match="indivisible"):
        _translate(text, status_cb=statuses.append)
    assert len(fake_ollama.calls) == 1
    assert len(statuses) == 1
    assert "No partial translation" in statuses[0]


def test_exhausted_retries_are_bounded_and_return_no_partial(fake_ollama):
    fake_ollama.respond = lambda request: _response(
        "never return this", done_reason="length",
    )
    progress = []
    statuses = []
    with pytest.raises(
        OutputTruncatedError,
        match=f"after {MAX_SPLIT_RETRIES} smaller-chunk retries",
    ):
        _translate(
            "word " * 64, status_cb=statuses.append,
            progress_cb=lambda *args: progress.append(args),
        )
    assert len(fake_ollama.calls) == MAX_SPLIT_RETRIES + 1
    lengths = [len(_source(call)) for call in fake_ollama.calls]
    assert all(left > right for left, right in zip(lengths, lengths[1:]))
    assert not progress
    assert "No partial translation" in statuses[-1]


def test_full_retry_tree_remains_bounded_and_lossless(fake_ollama):
    accepted = []

    def respond(request):
        source = _source(request)
        if len(source.split()) > 2:
            return _response("discard me", done_reason="length")
        accepted.append(source)
        return _response(source.upper())

    fake_ollama.respond = respond
    text = "word " * (2 ** (MAX_SPLIT_RETRIES + 1))
    assert _translate(text) == text.upper()
    assert "".join(accepted) == text
    assert len(fake_ollama.calls) == 2 ** (MAX_SPLIT_RETRIES + 1) - 1


@pytest.mark.parametrize("truncate", [False, True])
def test_legacy_client_keyword_fallback_keeps_completion_checks(
    fake_ollama, monkeypatch, truncate,
):
    def legacy_chat(*, model, messages, options):
        request = {"model": model, "messages": messages, "options": options}
        fake_ollama.calls.append(request)
        source = _source(request)
        if truncate and source == "alpha beta":
            return _response("discard me", done_reason="length")
        return _response(source.upper())

    monkeypatch.setattr(sys.modules["ollama"], "chat", legacy_chat)
    assert _translate("alpha beta") == "ALPHA BETA"
    assert len(fake_ollama.calls) == (3 if truncate else 1)
    for call in fake_ollama.calls:
        assert "think" not in call
        assert call["options"]["num_ctx"] == 4096
        assert call["options"]["num_predict"] == 1024


def test_legacy_client_api_exception_propagates(monkeypatch):
    calls = []
    error = RuntimeError("API failure")

    def legacy_chat(*, model, messages, options):
        calls.append(messages)
        raise error

    monkeypatch.setattr(sys.modules["ollama"], "chat", legacy_chat)
    with pytest.raises(RuntimeError) as caught:
        _translate("alpha beta")
    assert caught.value is error
    assert len(calls) == 1


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("API failure"),
        OutputTruncatedError("API failure"),
        TypeError("API failure"),
        TypeError("Client.chat() got an unexpected keyword argument 'think'"),
        TypeError(
            "Client.chat() got an unexpected keyword argument 'options'"
        ),
        RuntimeError("model does not support thinking (status code: 400)"),
    ],
)
@pytest.mark.parametrize("during_retry", [False, True])
def test_api_exceptions_propagate_unchanged(
    fake_ollama, caplog, error, during_retry,
):
    def respond(request):
        if during_retry and len(fake_ollama.calls) == 1:
            return _response(done_reason="length")
        raise error

    fake_ollama.respond = respond
    with pytest.raises(type(error)) as caught:
        _translate("alpha beta")
    assert caught.value is error
    assert len(fake_ollama.calls) == (2 if during_retry else 1)
    assert str(error) not in caplog.text
