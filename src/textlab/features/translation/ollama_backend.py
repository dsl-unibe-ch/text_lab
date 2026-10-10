"""Budgeted Ollama translation without a local model tokenizer.

One UTF-8 byte is conservatively estimated as one token, NOT an exact token
count or a guarantee for every tokenizer/template. The 4096-token context
reserves 1024 output tokens, 256 chat-template tokens, and the estimated
system prompt. Initial source chunks use the remaining context, capped at
2048 bytes, rather than an assumed output expansion ratio. This avoids
unnecessary requests; actual output-limit signals trigger smaller-chunk
retries without increasing the context/output budgets or their VRAM demand.
No model metadata is fetched; the selected model must support this context.
An unreported server-side context reduction or silent source omission cannot
be detected.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import logging
from typing import Any

from .gpu_memory import serialized
from .chunking import (
    MAX_SPLIT_RETRIES,
    InputTooLongError,
    OutputTruncatedError,
    join_translations,
    split_for_retry,
    split_text,
    translate_lines,
)


_NUM_CTX = 4096
_NUM_PREDICT = 1024
_TEMPLATE_TOKEN_RESERVE = 256
_SOURCE_BYTE_CAP = 2048
_LOG = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    """Use UTF-8 bytes as a conservative estimate, not exact tokenization."""
    return len(text.encode("utf-8"))


def _system_prompt(source: str, target: str, formality: str) -> str:
    instruction = {
        "formal": (
            " Use a formal, professional register throughout — polite"
            " pronouns, complete sentences, no colloquialisms."
        ),
        "informal": (
            " Use a casual, conversational register — everyday vocabulary,"
            " contractions where natural, informal pronouns."
        ),
    }.get(formality, "")
    return (
        "You are a professional translator. Translate the user's text from "
        f"{source} into {target}. "
        "Preserve meaning, tone, formatting (paragraphs, lists) and named "
        f"entities.{instruction} Do not add commentary. "
        "Return ONLY the translated text."
    )


def _input_budget(system: str) -> int:
    available = (
        _NUM_CTX - _NUM_PREDICT - _TEMPLATE_TOKEN_RESERVE
        - _estimate_tokens(system)
    )
    budget = min(available, _SOURCE_BYTE_CAP)
    if budget <= 0:
        raise InputTooLongError(
            "The Ollama system prompt and reserved template/output tokens "
            "leave no source budget. Shorten the language names or choose "
            "another backend. No truncated translation was returned."
        )
    return budget


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _token_count(response: Any, name: str) -> int | None:
    count = _field(response, name)
    if count is not None and (type(count) is not int or count < 0):
        raise OutputTruncatedError(
            "Ollama returned invalid token-count metadata."
        )
    return count


def _completed_text(response: Any) -> str:
    reason = _field(response, "done_reason")
    done = _field(response, "done")
    generated = _token_count(response, "eval_count")
    prompt = _token_count(response, "prompt_eval_count")
    if reason == "length" or (
        generated is not None and generated >= _NUM_PREDICT
    ):
        raise OutputTruncatedError("Ollama reached its output-token limit.")
    if prompt is not None and prompt >= _NUM_CTX - _NUM_PREDICT:
        raise OutputTruncatedError(
            "Ollama's reported prompt leaves insufficient context for "
            "the reserved output tokens."
        )
    if done is False or reason not in (None, "stop"):
        raise OutputTruncatedError(
            "Ollama did not report a completed translation."
        )
    # Older servers may omit done_reason; require completion plus a count.
    if reason is None and not (
        done is True and generated is not None and generated > 0
    ):
        raise OutputTruncatedError(
            "Ollama returned no verifiable completion metadata."
        )
    content = _field(_field(response, "message"), "content")
    if not isinstance(content, str) or not content.strip():
        raise OutputTruncatedError(
            "Ollama returned no translated text for a nonblank source."
        )
    return content


def _chat_no_think(chat: Callable[..., Any], **request: Any) -> Any:
    """Keep reasoning out of the output budget when the client supports it.

    Only Python argument binding rejecting exactly ``think`` allows a plain
    retry. Legacy clients cannot explicitly disable thinking. Client-internal
    TypeErrors and model/server errors propagate; completion checks remain
    mandatory in either case.
    """
    try:
        return chat(think=False, **request)
    except TypeError as error:
        traceback = error.__traceback__
        if (
            not str(error).endswith(
                " got an unexpected keyword argument 'think'"
            )
            or traceback is None
            or traceback.tb_next is not None
        ):
            raise
    return chat(**request)


def _report(
    message: str,
    status_cb: Callable[[str], None] | None,
    level: int = logging.WARNING,
) -> None:
    _LOG.log(level, message)
    if status_cb is not None:
        status_cb(message)


def _translate_chunk(
    text: str,
    system: str,
    model_name: str,
    chat: Callable[..., Any],
    status_cb: Callable[[str], None] | None,
    depth: int = 0,
) -> str:
    if not text.strip():
        return text
    response = _chat_no_think(
        chat,
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        options={
            "temperature": 0.2,
            "num_ctx": _NUM_CTX,
            "num_predict": _NUM_PREDICT,
        },
    )
    try:
        content = _completed_text(response)
    except OutputTruncatedError as error:
        if depth >= MAX_SPLIT_RETRIES:
            message = (
                f"{error} Ollama translation failed after "
                f"{MAX_SPLIT_RETRIES} smaller-chunk retries. Choose another "
                "model/backend or shorten the source. "
                "No partial translation was returned."
            )
            _report(message, status_cb)
            raise OutputTruncatedError(message) from error
        try:
            pieces = split_for_retry(text)
        except OutputTruncatedError:
            message = (
                f"{error} The source span is indivisible without splitting "
                "a word. Choose another model/backend or shorten the "
                "source. No partial translation was returned."
            )
            _report(message, status_cb)
            raise OutputTruncatedError(message) from error
        _report(
            f"{error} Retrying with smaller source chunks "
            f"({depth + 1}/{MAX_SPLIT_RETRIES}).",
            status_cb,
        )
        translations = [
            _translate_chunk(
                part, system, model_name, chat, status_cb, depth + 1,
            )
            for part in pieces
        ]
        result = join_translations(pieces, translations)
        if depth == 0:
            _report(
                "Ollama translation recovered using smaller source chunks.",
                status_cb,
                logging.INFO,
            )
        return result
    return join_translations([text], [content])


_ACTIVE_OLLAMA = None
_OWNED_OLLAMA = None


def _canonical_model(name: str) -> str:
    return name if ":" in name.rsplit("/", 1)[-1] else name + ":latest"


def _running_models(client):
    """Unknown residency must never be treated as ownership permission."""
    try:
        response = client.ps()
        models = _field(response, "models")
        if not isinstance(models, (list, tuple)):
            return None
        names = set()
        for model in models:
            name = _field(model, "model") or _field(model, "name")
            if not isinstance(name, str) or not name:
                return None
            names.add(_canonical_model(name))
        return names
    except Exception:
        # Missing legacy API / disconnected service: cannot prove ownership.
        return None


@serialized
def release_ollama_model() -> None:
    """Unload only the model this translator observed absent before loading.

    Never enumerate-and-unload other jobs, kill a server, or unload a model
    already resident before translation. Ollama exposes no ownership leases:
    another process sharing the *same* model cannot be isolated by this
    process-local guard. Use a dedicated service for cross-process isolation.
    """
    global _ACTIVE_OLLAMA, _OWNED_OLLAMA

    if _OWNED_OLLAMA is not None:
        client, name = _OWNED_OLLAMA
        running = _running_models(client)
        if running is None:
            raise RuntimeError(
                "Cannot verify the translation-owned Ollama model's "
                "residency; GPU handoff stopped rather than unloading "
                "an unverified model."
            )
        if _canonical_model(name) in running:
            unload = getattr(client, "generate", None)
            if unload is None:
                raise RuntimeError(
                    "This Ollama client cannot release the translation "
                    "model. GPU handoff stopped."
                )
            unload(model=name, keep_alive=0)
            remaining = _running_models(client)
            if (remaining is None
                    or _canonical_model(name) in remaining):
                raise RuntimeError(
                    "Ollama has not released the translation model; "
                    "GPU handoff stopped."
                )
    _OWNED_OLLAMA = None
    _ACTIVE_OLLAMA = None


@serialized
def prepare_ollama_model(model_name: str) -> None:
    """Evict HF before Ollama use and conservatively track model ownership."""
    global _ACTIVE_OLLAMA, _OWNED_OLLAMA

    import ollama
    from . import engine

    current = (ollama, _canonical_model(model_name))
    if _ACTIVE_OLLAMA != current:
        release_ollama_model()
    if engine._ACTIVE_HF_SIGNATURE is not None:
        engine._free_hf_cache()
    running = _running_models(ollama)
    if running is not None and current[1] not in running:
        _OWNED_OLLAMA = (ollama, model_name)
    _ACTIVE_OLLAMA = current


@serialized
def ollama_model_is_loaded(model_name: str) -> bool:
    """Check actual residency, including server eviction/keep-alive expiry."""
    if _ACTIVE_OLLAMA is None:
        return False
    client, active = _ACTIVE_OLLAMA
    canonical = _canonical_model(model_name)
    running = _running_models(client)
    return active == canonical and running is not None and canonical in running


@serialized
def translate_ollama(
    text: str,
    src_lang_name: str,
    tgt_lang_name: str,
    model_name: str,
    formality: str = "default",
    progress_cb: Callable[[int, int], None] | None = None,
    *,
    status_cb: Callable[[str], None] | None = None,
) -> str:
    """Translate with lossless source boundaries and explicit token budgets.

    Language names and formality keep the original translation prompt.
    Progress counts initial chunks, including successful smaller-chunk
    recoveries, not failed attempts. Status messages report retries/recovery
    without including source text. API exceptions propagate unchanged.
    Unrecoverable budget/completion failures raise a translation-limit error,
    never returning accumulated partial output. Ollama is imported lazily.
    """
    if not text.strip():
        return text
    system = _system_prompt(src_lang_name, tgt_lang_name, formality)
    budget = _input_budget(system)

    def translate_parts(lines: list[str]) -> list[str]:
        chunks = [split_text(line, _estimate_tokens, budget) for line in lines]
        import ollama

        prepare_ollama_model(model_name)
        total = sum(len(parts) for parts in chunks)
        completed = 0
        translated = []
        for parts in chunks:
            outputs = []
            for part in parts:
                outputs.append(_translate_chunk(
                    part, system, model_name, ollama.chat, status_cb,
                ))
                completed += 1
                if progress_cb is not None:
                    progress_cb(completed, total)
            translated.append("".join(outputs))
        return translated

    return translate_lines([text], translate_parts)[0]
