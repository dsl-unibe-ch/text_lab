"""Token-budgeted, batched generation for encoder-decoder translators.

Model loading and language selection stay in ``engine``. This module owns
input measurement, complete-output validation, and bounded split/retry.
"""

from __future__ import annotations

from collections.abc import Callable
import logging

from .chunking import (
    InputTooLongError,
    MAX_SPLIT_RETRIES,
    OutputTruncatedError,
    join_translations,
    split_for_retry,
    split_text,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_INPUT_TOKENS = 512
DEFAULT_OUTPUT_TOKENS = 512
DEFAULT_NUM_BEAMS = 1
DEFAULT_BATCH_SIZE = 16


def input_token_limit(tokenizer, model) -> int:
    """Respect smaller model windows without increasing the VRAM budget."""
    limits = [DEFAULT_INPUT_TOKENS]
    for value in (
        getattr(tokenizer, "model_max_length", None),
        getattr(model.config, "max_position_embeddings", None),
        getattr(model.config, "max_encoder_position_embeddings", None),
    ):
        if isinstance(value, int) and value > 0:
            limits.append(value)
    return min(limits)


def _eos_ids(model, tokenizer) -> set[int]:
    for config in (
        getattr(model, "generation_config", None), model.config, tokenizer,
    ):
        value = getattr(config, "eos_token_id", None)
        if value is not None:
            return {value} if isinstance(value, int) else set(value)
    raise OutputTruncatedError(
        "This model has no end-of-sequence token configured; translation "
        "completion cannot be verified."
    )


def _notify(status_cb, message: str) -> None:
    LOGGER.warning(message)
    if status_cb is not None:
        status_cb(message)


def generate_translations(
    model,
    tokenizer,
    texts: list[str],
    device: str,
    *,
    source_prefix: str = "",
    forced_bos_token_id: int | None = None,
    num_beams: int = DEFAULT_NUM_BEAMS,
    max_new_tokens: int = DEFAULT_OUTPUT_TOKENS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress_cb: Callable[[int, int], None] | None = None,
    status_cb: Callable[[str], None] | None = None,
) -> list[str]:
    """Translate complete inputs; never accept input or output truncation.

    Token counts include special tokens and the MADLAD target prefix. GPU
    work remains batched; only unfinished outputs are retried on smaller
    source spans, with unchanged per-call token and memory budgets.
    """
    if batch_size <= 0 or max_new_tokens <= 0:
        raise ValueError(
            "Batch size and output-token budget must be positive.")
    if not texts:
        return []

    import torch

    limit = input_token_limit(tokenizer, model)
    eos_ids = _eos_ids(model, tokenizer)

    def measure(text: str) -> int:
        encoded = tokenizer(
            source_prefix + text,
            add_special_tokens=True,
            truncation=False,
        )
        return len(encoded["input_ids"])

    pieces = [split_text(text, measure, limit) for text in texts]
    flat = [chunk for chunks in pieces for chunk in chunks]
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        # Marian/NLLB may otherwise force EOS at the last allowed token,
        # making a length-limited output look like a natural completion.
        "forced_eos_token_id": None,
    }
    if forced_bos_token_id is not None:
        kwargs["forced_bos_token_id"] = forced_bos_token_id

    def generate_batch(chunks: list[str], depth: int) -> list[str]:
        prepared = [source_prefix + chunk for chunk in chunks]
        encoded = tokenizer(
            prepared,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
            truncation=False,
        )
        if encoded["input_ids"].shape[-1] > limit:
            raise InputTooLongError(
                "The encoded source exceeds the model's input budget. "
                "Translation stopped rather than truncating the input."
            )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(**encoded, **kwargs)
        rows = generated.tolist()
        if len(rows) != len(chunks):
            raise OutputTruncatedError(
                "The model returned an incomplete batch of translations."
            )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        retry_sources = []
        retry_slots = []
        results = list(decoded)
        for index, (row, text) in enumerate(zip(rows, decoded)):
            # Encoder-decoder outputs begin with a decoder start token,
            # which itself can be EOS (notably for NLLB).
            complete = any(token in eos_ids for token in row[1:])
            if complete and text.strip():
                continue
            if depth >= MAX_SPLIT_RETRIES:
                raise OutputTruncatedError(
                    "Translation still reaches its output-token limit or "
                    f"returns no complete text after {MAX_SPLIT_RETRIES} "
                    "smaller-chunk retries. Try another backend. "
                    "No partial translation was returned."
                )
            smaller = split_for_retry(chunks[index])
            # Recheck independently: tokenization can change at boundaries.
            smaller = [
                part for piece in smaller
                for part in split_text(piece, measure, limit)
            ]
            retry_slots.append((index, smaller, len(retry_sources)))
            retry_sources.extend(smaller)

        if retry_sources:
            _notify(
                status_cb,
                "The model did not finish within its output-token budget. "
                "Retrying the affected text in smaller chunks "
                f"(attempt {depth + 1}/{MAX_SPLIT_RETRIES}); "
                "partial outputs are discarded.",
            )
            recovered = []
            for start in range(0, len(retry_sources), batch_size):
                recovered.extend(generate_batch(
                    retry_sources[start:start + batch_size], depth + 1,
                ))
            for index, smaller, start in retry_slots:
                results[index] = join_translations(
                    smaller, recovered[start:start + len(smaller)],
                )
        return results

    translated = []
    for start in range(0, len(flat), batch_size):
        batch = flat[start:start + batch_size]
        translated.extend(generate_batch(batch, 0))
        if progress_cb is not None:
            progress_cb(start + len(batch), len(flat))

    results = []
    offset = 0
    for chunks in pieces:
        results.append(join_translations(
            chunks, translated[offset:offset + len(chunks)],
        ))
        offset += len(chunks)
    return results
