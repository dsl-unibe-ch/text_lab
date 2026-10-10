"""Helpers for the session's Ollama server, shared by all LLM features.

Every Text Lab session starts its own Ollama server; the launch script
exports its address as ``OLLAMA_HOST``. This module wraps the calls that
several features need: checking the server, chatting without reasoning
output, listing and unloading models, and keeping long texts within a
model's context window.

The module is named after the server, not the ``ollama`` client package it
imports; absolute imports keep the two apart.
"""

from __future__ import annotations

import functools
import os
import socket
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import ollama

#: Address used when ``OLLAMA_HOST`` is not set (Ollama's own default).
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11434

#: Rough characters per token, for budgeting without a tokenizer.
CHARS_PER_TOKEN = 4
#: Texts above this many tokens are processed in chunks (about 300k chars).
MAX_CONTEXT_TOKENS = 75_000
#: Tokens per chunk when a text is split (about 56k characters).
CHUNK_SIZE_TOKENS = 14_000

#: Seconds to wait at most for Ollama to free a model's memory.
UNLOAD_WAIT_TIMEOUT = float(
    os.environ.get("TEXTLAB_UNLOAD_WAIT_TIMEOUT", "60")
)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


def server_address(value: str | None = None) -> tuple[str, int]:
    """Return the host and port of the Ollama server.

    Args:
        value: An address such as ``"127.0.0.1:16912"`` or
            ``"http://localhost:11434"``; defaults to ``OLLAMA_HOST``.

    Returns:
        ``(host, port)``, with Ollama's defaults for missing parts.
    """
    if value is None:
        value = os.environ.get("OLLAMA_HOST", "")
    address = value.replace("http://", "").replace("https://", "").strip("/")
    if not address:
        return DEFAULT_HOST, DEFAULT_PORT
    host, _, port = address.partition(":")
    return host or DEFAULT_HOST, int(port) if port else DEFAULT_PORT


def check_ollama_server(attempts: int = 20, delay: float = 0.5) -> bool:
    """Check that the Ollama server accepts connections.

    The server may still be starting when a page first loads, so the check
    is retried.

    Args:
        attempts: How many times to try.
        delay: Seconds between attempts.

    Returns:
        True as soon as the server accepts a connection, False if it never
        does.
    """
    host, port = server_address()
    for attempt in range(attempts):
        if _port_open(host, port):
            return True
        if attempt < attempts - 1:
            time.sleep(delay)
    return False


def _port_open(host: str, port: int) -> bool:
    """Return True if something accepts TCP connections on host:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        return connection.connect_ex((host, port)) == 0


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


def message_text(message: Any) -> str:
    """Return the text of a chat message, or ``""`` when it has none.

    Ollama 0.28 and later put a reasoning model's thinking in a separate
    field and leave ``content`` as ``None`` on reasoning-only chunks, so
    callers that concatenate content must treat ``None`` as empty.

    Args:
        message: A message object or dictionary, or ``None``.

    Returns:
        The message content as a string.
    """
    if message is None:
        return ""
    if isinstance(message, Mapping):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)
    return content or ""


def chat_no_think(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    options: dict[str, Any] | None = None,
    timeout: float | None = None,
    json_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Send one non-streaming chat request with reasoning disabled.

    Tool routing and agent steps need a fast, deterministic answer, not a
    chain of thought: with thinking on, reasoning models fill the context
    window and often stop before producing a tool call. If the client,
    server or model does not accept the ``think`` argument, the request is
    sent again without it.

    Args:
        model: The Ollama model name.
        messages: The chat history to send.
        tools: Tool definitions in Ollama's function-calling format.
        options: Ollama runtime options such as ``temperature``. A
            ``num_ctx`` different from the loaded one makes Ollama reload
            the model.
        timeout: HTTP timeout in seconds; without it, a request Ollama never
            answers blocks forever.
        json_schema: JSON schema passed as Ollama's ``format``, so the reply
            is valid JSON for it.

    Returns:
        The response as a plain dictionary with a plain-dictionary
        ``message``.

    Raises:
        ollama.ResponseError: If Ollama rejects the request for another
            reason, e.g. a tool call it cannot parse.
        httpx.TimeoutException: If ``timeout`` elapses first.
    """
    chat = _timeout_client(timeout).chat if timeout else ollama.chat
    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if tools is not None:
        kwargs["tools"] = tools
    if options:
        kwargs["options"] = options
    if json_schema is not None:
        kwargs["format"] = json_schema
    try:
        return _normalize_response(call_no_think(chat, **kwargs))
    except ollama.ResponseError as exc:
        # Only a rejected ``think`` is worth retrying; any other error would
        # recur with the identical request.
        if "think" not in str(exc).lower():
            raise
        return _normalize_response(chat(**kwargs))


def call_no_think(chat: Callable[..., Any], **request: Any) -> Any:
    """Call a chat function with ``think=False``, as far as it supports it.

    Clients older than the ``think`` keyword are called again without it.
    Only that exact rejection, raised while binding the arguments, is
    retried: a ``TypeError`` from inside the client, and every error from
    the server, propagates unchanged. The response is returned as the client
    gave it.

    Args:
        chat: ``ollama.chat`` or a client's ``chat`` method.
        **request: The request arguments (``model``, ``messages``, ...).

    Returns:
        The chat response, or a stream when ``stream=True`` is requested.
    """
    try:
        return chat(think=False, **request)
    except TypeError as error:
        if not _rejects_think_keyword(error):
            raise
    return chat(**request)


def _rejects_think_keyword(error: TypeError) -> bool:
    """Return True if ``error`` is Python refusing the ``think`` keyword.

    Such an error is raised before the called function runs, so its
    traceback holds only the caller's frame.
    """
    traceback = error.__traceback__
    return (
        str(error).endswith(" got an unexpected keyword argument 'think'")
        and traceback is not None
        and traceback.tb_next is None
    )


@functools.lru_cache(maxsize=4)
def _timeout_client(timeout: float) -> ollama.Client:
    """Return a shared client whose requests fail after ``timeout`` s."""
    return ollama.Client(timeout=timeout)


def _normalize_response(response: Any) -> dict[str, Any]:
    """Convert a ``ChatResponse`` (or dictionary) into a plain dictionary."""
    if isinstance(response, dict):
        if "message" in response and not isinstance(response["message"], dict):
            response["message"] = _normalize_message(response["message"])
        return response
    message = getattr(response, "message", None)
    return {
        "message": _normalize_message(message) if message is not None else {},
        "model": getattr(response, "model", ""),
        "done": getattr(response, "done", True),
    }


def _normalize_message(message: Any) -> dict[str, Any]:
    """Convert a ``Message`` object into a plain dictionary.

    The client returns Pydantic objects that support ``message["role"]`` but
    not ``message.get(...)``; plain dictionaries work everywhere and can be
    sent back in the next request.
    """
    if isinstance(message, dict):
        return message
    if hasattr(message, "model_dump"):  # Pydantic v2 keeps every field.
        return message.model_dump(exclude_none=True)
    result: dict[str, Any] = {
        "role": getattr(message, "role", "assistant"),
        "content": getattr(message, "content", None) or "",
    }
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        result["tool_calls"] = [_normalize_tool_call(c) for c in tool_calls]
    return result


def _normalize_tool_call(call: Any) -> dict[str, Any]:
    """Convert a ``ToolCall`` object into a plain dictionary."""
    if isinstance(call, dict):
        return call
    function = getattr(call, "function", None) or {}
    if not isinstance(function, dict):
        function = {
            "name": getattr(function, "name", ""),
            "arguments": getattr(function, "arguments", {}),
        }
    return {"function": function}


# ---------------------------------------------------------------------------
# Models in memory
# ---------------------------------------------------------------------------


def extract_model_name(entry: Any) -> str:
    """Return the model name from any of the client's model entry formats.

    Args:
        entry: A model object, dictionary, string or tuple.

    Returns:
        The model name.
    """
    name = getattr(entry, "model", None)
    if isinstance(name, str):
        return name
    if isinstance(entry, dict) and "name" in entry:
        return entry["name"]
    if isinstance(entry, str):
        return entry
    if isinstance(entry, tuple | list) and entry:
        return str(entry[0])
    return str(entry)


def canonical_model_name(name: str) -> str:
    """Return a model name with its tag, adding Ollama's default ``latest``.

    Args:
        name: A model name such as ``"llama3"`` or ``"llama3:8b"``.

    Returns:
        The name as Ollama lists loaded models, e.g. ``"llama3:latest"``.
    """
    return name if ":" in name.rsplit("/", 1)[-1] else name + ":latest"


def running_model_names(client: Any = ollama) -> set[str] | None:
    """Return the canonical names of the models a server holds in memory.

    Unlike :func:`get_loaded_models`, an answer that cannot be trusted is
    reported as unknown rather than as "nothing loaded", for callers that
    must not unload a model unless they are sure of the server's state.

    Args:
        client: The ``ollama`` module or an ``ollama.Client``.

    Returns:
        The names, or ``None`` if the server cannot be reached or its answer
        has an unexpected shape.
    """
    try:
        response = client.ps()
        models = _field(response, "models")
        if not isinstance(models, list | tuple):
            return None
        names = set()
        for model in models:
            name = _field(model, "model") or _field(model, "name")
            if not isinstance(name, str) or not name:
                return None
            names.add(canonical_model_name(name))
        return names
    except Exception:
        # An old client without ps(), or a server that is not running.
        return None


def _field(value: Any, name: str) -> Any:
    """Return a field of a response object or dictionary, or ``None``."""
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def get_loaded_models() -> list[str]:
    """Return the names of the models Ollama currently holds in memory.

    Returns:
        The model names; empty if none are loaded or the server cannot be
        reached.
    """
    try:
        running = ollama.ps()
    except Exception:
        return []
    if isinstance(running, dict):
        models = running.get("models", [])
    else:
        models = getattr(running, "models", [])
    return [extract_model_name(model) for model in models or []]


def is_model_loaded(model_name: str) -> bool:
    """Return True if Ollama holds ``model_name`` in memory.

    Args:
        model_name: The model, with or without a tag.

    Returns:
        True if a loaded model matches the name exactly or by tag prefix.
    """
    return any(
        name == model_name or name.startswith(f"{model_name}:")
        for name in get_loaded_models()
    )


def unload_model(model_name: str) -> bool:
    """Ask Ollama to release a model from memory immediately.

    Args:
        model_name: The model to unload.

    Returns:
        True if the request was accepted.
    """
    try:
        ollama.generate(model=model_name, prompt="", keep_alive=0)
    except Exception:
        return False
    return True


def unload_all_models() -> list[str]:
    """Release every model Ollama holds in memory.

    Returns:
        The names of the models that were loaded.
    """
    loaded = get_loaded_models()
    for name in loaded:
        unload_model(name)
    return loaded


def release_models(
    keep: Iterable[str] = (), timeout: float | None = None
) -> list[str]:
    """Unload every model but ``keep`` and wait until Ollama has let go.

    Ollama answers an unload request while the model is still in memory;
    its runner takes seconds to exit. Code that is about to put something
    else on the GPU (an OCR worker, another feature's model) must wait for
    that, or it runs out of memory while loading.

    Args:
        keep: Canonical names (``name:tag``) of models to leave loaded.
        timeout: Seconds to wait at most; defaults to
            :data:`UNLOAD_WAIT_TIMEOUT`.

    Returns:
        The names of the models that were unloaded.
    """
    if timeout is None:
        timeout = UNLOAD_WAIT_TIMEOUT
    keep = set(keep)
    resident = [name for name in get_loaded_models() if name not in keep]
    if not resident:
        return []
    for name in resident:
        unload_model(name)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not set(get_loaded_models()) - keep:
            break
        time.sleep(0.25)
    return resident


def warm_model(model_name: str, keep_alive: int = 300) -> bool:
    """Load a model ahead of use, so the next real request starts at once.

    Args:
        model_name: The model to load.
        keep_alive: Seconds Ollama keeps it loaded afterwards.

    Returns:
        True if the request was accepted.
    """
    try:
        ollama.generate(model=model_name, prompt="", keep_alive=keep_alive)
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# Context budget
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text from its length.

    Args:
        text: The text.

    Returns:
        The estimate, at least 1.
    """
    return max(1, len(text) // CHARS_PER_TOKEN)


def chunk_text(
    text: str, chunk_size_tokens: int = CHUNK_SIZE_TOKENS
) -> list[str]:
    """Split a text into chunks of about ``chunk_size_tokens`` each.

    Chunks end at a paragraph break where possible, otherwise at a line
    break, so that they stay coherent.

    Args:
        text: The text to split.
        chunk_size_tokens: Target size of a chunk in tokens.

    Returns:
        The non-empty chunks, in order.
    """
    chunk_size_chars = chunk_size_tokens * CHARS_PER_TOKEN
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size_chars
        if end < len(text):
            breakpoint_ = text.rfind("\n\n", start, end)
            if breakpoint_ == -1:
                breakpoint_ = text.rfind("\n", start, end)
            if breakpoint_ > start:
                end = breakpoint_
        chunks.append(text[start:end].strip())
        start = end
    return [chunk for chunk in chunks if chunk]
