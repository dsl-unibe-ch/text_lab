"""Token-budgeted, batched generation for encoder-decoder translators.

Model loading and language selection stay in ``engine``. This module owns
input measurement, complete-output validation, and bounded split/retry.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from .chunking import (
    MAX_SPLIT_RETRIES,
    InputTooLongError,
    OutputTruncatedError,
    join_translations,
    sentence_slices,
    split_for_retry,
    split_text,
)
from .gpu_memory import (
    clear_cuda_cache,
    discard_exception_tensors,
    is_cuda_oom,
    serialized,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_INPUT_TOKENS = 512
DEFAULT_OUTPUT_TOKENS = 512
DEFAULT_NUM_BEAMS = 1
DEFAULT_BATCH_SIZE = 16
# ``batch_size`` is calibrated for full-window inputs; short sentences can
# share that token budget, up to this many sentences per budgeted slot.
SENTENCES_PER_SLOT = 8


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
        getattr(model, "generation_config", None),
        model.config,
        tokenizer,
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


@serialized
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
    cache: dict[str, str] | None = None,
) -> list[str]:
    """Translate complete inputs; never accept input or output truncation.

    Inputs are translated sentence by sentence (what these models are
    trained on). Identical sentences are translated once, and ``cache``
    (normalized sentence -> translation) carries results across calls, e.g.
    between the Markdown and PDF outputs of one document. Sentences are
    sorted by length and batched by total tokens (``batch_size`` full-window
    inputs), so short sentences share a batch without padding waste.

    Token counts include special tokens and the MADLAD target prefix. Only
    unfinished outputs are retried on smaller source spans, with unchanged
    per-call token and memory budgets. CUDA OOM halves the microbatch
    independently of split-retry depth, stopping at one; failed
    tensors/tracebacks are released before clearing the cache.
    """
    if batch_size <= 0 or max_new_tokens <= 0:
        raise ValueError(
            "Batch size and output-token budget must be positive."
        )
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

    lengths: dict[str, int] = {}

    def measured(text: str) -> int:
        if text not in lengths:
            lengths[text] = measure(text)
        return lengths[text]

    def model_pieces(text: str) -> list[str]:
        result = []
        for sentence in sentence_slices(text):
            if measured(sentence) <= limit:
                result.append(sentence)
            else:
                result.extend(split_text(sentence, measured, limit))
        return result

    pieces = [model_pieces(text) for text in texts]
    store = cache if cache is not None else {}
    key_tokens: dict[str, int] = {}
    for chunks in pieces:
        for chunk in chunks:
            key = " ".join(chunk.split())
            if key and key not in store:
                key_tokens.setdefault(key, measured(chunk))
    # Longest first: padding stays minimal and any OOM surfaces early.
    pending = sorted(key_tokens, key=key_tokens.__getitem__, reverse=True)
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        # Marian/NLLB may otherwise force EOS at the last allowed token,
        # making a length-limited output look like a natural completion.
        "forced_eos_token_id": None,
    }
    if forced_bos_token_id is not None:
        kwargs["forced_bos_token_id"] = forced_bos_token_id

    max_items = batch_size * SENTENCES_PER_SLOT
    token_budget = batch_size * limit
    batches: list[list[str]] = []
    for key in pending:
        if batches and (
            len(batches[-1]) < max_items
            and (len(batches[-1]) + 1) * key_tokens[batches[-1][0]]
            <= token_budget
        ):
            batches[-1].append(key)
        else:
            batches.append([key])
    microbatch_size = max_items

    def infer(chunks: list[str]):
        # Return CPU data only so output retries never retain GPU tensors.
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
        if len(decoded) != len(chunks):
            raise OutputTruncatedError(
                "The tokenizer returned an incomplete batch of translations."
            )
        return rows, decoded

    def attempt(chunks: list[str]):
        try:
            return infer(chunks)
        except RuntimeError as error:
            if not is_cuda_oom(error, device, torch):
                raise
            # Failed frames can own tensors even after infer has unwound.
            discard_exception_tensors(error)
        # Leave the exception scope before collection/empty_cache in caller.
        return None

    def generate_pending(chunks: list[str], depth: int) -> list[str]:
        nonlocal microbatch_size
        outputs = []
        offset = 0
        while offset < len(chunks):
            batch = chunks[offset : offset + microbatch_size]
            outcome = attempt(batch)
            if outcome is None:
                clear_cuda_cache(device)
                if len(batch) == 1:
                    message = (
                        "CUDA out of memory at translation microbatch 1. "
                        "Free GPU memory or choose a smaller/CPU backend. "
                        "No partial translation was returned."
                    )
                    _notify(status_cb, message)
                    raise RuntimeError(message) from None
                microbatch_size = max(1, len(batch) // 2)
                _notify(
                    status_cb,
                    "CUDA out of memory; retrying the same source with "
                    f"microbatch {microbatch_size}. Token budgets unchanged.",
                )
                continue
            rows, decoded = outcome
            outputs.extend(finish_batch(batch, rows, decoded, depth))
            offset += len(batch)
        return outputs

    def finish_batch(chunks, rows, decoded, depth: int) -> list[str]:
        retry_sources = []
        retry_slots = []
        results = list(decoded)
        for index, (row, text) in enumerate(zip(rows, decoded, strict=False)):
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
                part
                for piece in smaller
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
            recovered = generate_pending(retry_sources, depth + 1)
            for index, smaller, start in retry_slots:
                results[index] = join_translations(
                    smaller,
                    recovered[start : start + len(smaller)],
                )
        return results

    done = 0
    for batch in batches:
        for key, output in zip(
            batch, generate_pending(batch, 0), strict=False
        ):
            store[key] = output
        done += len(batch)
        if progress_cb is not None:
            progress_cb(done, len(pending))

    return [
        join_translations(
            chunks,
            [store.get(" ".join(chunk.split()), "") for chunk in chunks],
        )
        for chunks in pieces
    ]
