"""Sentence-transformer embeddings for BERTopic.

The two default MiniLM models are in the shared model store; any other model
the user names is downloaded to their own Hugging Face cache. Documents
longer than a model's context window are truncated by the model, or embedded
in chunks whose embeddings are averaged.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from textlab.features.topic_modeling.models import (
    DEFAULT_EMBEDDING_MODEL_ENGLISH,
    DEFAULT_EMBEDDING_MODEL_MULTILINGUAL,
    SHARED_EMBEDDING_MODELS,
    Algorithm,
    Notice,
    TopicModelingConfig,
)

LOGGER = logging.getLogger(__name__)

#: Share of documents above the context window that triggers a notice.
TRUNCATION_NOTICE_RATIO = 0.10

#: Models with a long context window, suggested for long documents.
LONG_CONTEXT_MODELS = (
    "`jinaai/jina-embeddings-v2-base-en` (8192 tokens), "
    "`BAAI/bge-m3` (8192 tokens) or "
    "`nomic-ai/nomic-embed-text-v1.5` (8192 tokens)"
)


def get_embedding_model_name(config: TopicModelingConfig) -> str:
    """Return the name of the embedding model a configuration uses.

    Args:
        config: The topic modeling configuration.

    Returns:
        The resolved embedding model name.
    """
    if config.algorithm == Algorithm.BERTOPIC:
        return resolve_bertopic_embedding_model_id(config)

    if config.algorithm == Algorithm.TOP2VEC:
        if config.top2vec_backend == "transformer":
            return (
                DEFAULT_EMBEDDING_MODEL_ENGLISH
                if config.language == "English"
                else DEFAULT_EMBEDDING_MODEL_MULTILINGUAL
            )
        return "Doc2Vec (Trained from scratch)"

    return "N/A"


def resolve_bertopic_embedding_model_id(config: TopicModelingConfig) -> str:
    """Resolve the HuggingFace sentence-transformer model ID used by BERTopic.

    Args:
        config: The topic modeling configuration.

    Returns:
        The HuggingFace model ID (e.g. "all-MiniLM-L6-v2").
    """
    if config.embedding_model_id and config.embedding_model_id.strip():
        return config.embedding_model_id.strip()

    return (
        DEFAULT_EMBEDDING_MODEL_ENGLISH
        if config.language == "English"
        else DEFAULT_EMBEDDING_MODEL_MULTILINGUAL
    )


def is_shared_embedding_model(model_id: str) -> bool:
    """Check whether a model is expected in the shared read-only cache.

    Non-shared models must be downloaded to the calling user's home cache.

    Args:
        model_id: The HuggingFace model ID.

    Returns:
        True if the model is one of the pre-downloaded shared models.
    """
    return model_id in SHARED_EMBEDDING_MODELS


def get_user_hf_cache_dir() -> str:
    """Return the user's writable Hugging Face cache, creating it if needed.

    This is used for user-selected custom embedding models so they do not need
    write access to the shared model cache.

    Returns:
        The absolute path of the cache directory.
    """
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError as exc:
        LOGGER.warning(
            "Could not create user HF cache dir '%s': %s", cache_dir, exc
        )
    return cache_dir


def load_sentence_transformer(
    model_id: str, trust_remote_code: bool = False
) -> Any:
    """Instantiate a SentenceTransformer for the given model ID.

    Shared/curated models resolve through the container-level HF_HOME (the
    read-only shared cache). Custom models are directed to the user's own
    HuggingFace cache directory so they can be downloaded without write access
    to the shared cache.

    Args:
        model_id: The HuggingFace sentence-transformer model ID.
        trust_remote_code: Whether a custom model may run Python code shipped
            in its repository (needed by some long-context models such as
            Jina or Nomic). Ignored for the shared models.

    Returns:
        A ready-to-use SentenceTransformer instance.
    """
    from sentence_transformers import SentenceTransformer

    if is_shared_embedding_model(model_id):
        return SentenceTransformer(model_id)

    return SentenceTransformer(
        model_id,
        cache_folder=get_user_hf_cache_dir(),
        trust_remote_code=trust_remote_code,
    )


def count_docs_exceeding_context(
    embedding_model: Any,
    texts: list[str],
) -> tuple[int, int, int]:
    """Count how many documents would be truncated by the embedding model.

    The tokenizer associated with the sentence-transformer is used to count
    WordPiece/BPE tokens for each document. Documents whose token count exceeds
    the model's ``max_seq_length`` will be silently truncated by the encoder
    and only their leading portion will contribute to the topic embedding.

    Args:
        embedding_model: A SentenceTransformer (or compatible) instance.
        texts: The documents to inspect.

    Returns:
        A tuple ``(over_count, total_count, max_seq_length)``.
    """
    if not texts:
        return 0, 0, 0

    max_seq_length = int(getattr(embedding_model, "max_seq_length", 0) or 0)
    tokenizer = getattr(embedding_model, "tokenizer", None)
    if not max_seq_length or tokenizer is None:
        return 0, len(texts), max_seq_length

    over = 0
    for text in texts:
        try:
            n_tokens = len(
                tokenizer.encode(
                    text, add_special_tokens=True, truncation=False
                )
            )
        except Exception:
            # Fall back to a coarse whitespace heuristic on tokenizer error.
            n_tokens = len(str(text).split())
        if n_tokens > max_seq_length:
            over += 1

    return over, len(texts), max_seq_length


def split_into_token_chunks(
    text: str,
    tokenizer: Any,
    max_tokens: int,
) -> list[tuple[str, int]]:
    """Split a document into pieces that fit a model's context window.

    Args:
        text: The document to split.
        tokenizer: The HuggingFace tokenizer of the embedding model.
        max_tokens: The maximum number of tokens per piece, excluding the
            special tokens the model adds.

    Returns:
        ``(chunk_text, token_count)`` pairs. A document that already fits is
        returned unchanged as a single chunk.
    """
    token_ids = tokenizer.encode(
        text, add_special_tokens=False, truncation=False
    )
    if len(token_ids) <= max_tokens:
        return [(text, max(len(token_ids), 1))]

    chunks = []
    for start in range(0, len(token_ids), max_tokens):
        window = token_ids[start : start + max_tokens]
        chunks.append(
            (tokenizer.decode(window, skip_special_tokens=True), len(window))
        )
    return chunks


def average_chunk_embeddings(
    chunk_embeddings: np.ndarray,
    owners: np.ndarray,
    weights: np.ndarray,
    n_documents: int,
) -> np.ndarray:
    """Combine chunk embeddings into one embedding per document.

    Chunks are averaged with their token counts as weights, so a short final
    chunk does not count as much as a full one. Averaging shortens vectors
    whose chunks point in different directions, so each average is rescaled
    to the weighted mean length of its chunks; long documents then stay on
    the same scale as documents that were embedded in one piece.

    Args:
        chunk_embeddings: One embedding per chunk, shape ``(n_chunks, dim)``.
        owners: The document index of every chunk.
        weights: The token count of every chunk.
        n_documents: The number of documents.

    Returns:
        One embedding per document, shape ``(n_documents, dim)``.
    """
    weighted = chunk_embeddings * weights[:, None]
    sums = np.zeros((n_documents, chunk_embeddings.shape[1]), dtype=np.float64)
    np.add.at(sums, owners, weighted)
    totals = np.bincount(owners, weights=weights, minlength=n_documents)
    averages = sums / totals[:, None]

    chunk_norms = np.linalg.norm(chunk_embeddings, axis=1)
    target_norms = (
        np.bincount(
            owners, weights=weights * chunk_norms, minlength=n_documents
        )
        / totals
    )
    current_norms = np.linalg.norm(averages, axis=1)
    scale = np.divide(
        target_norms,
        current_norms,
        out=np.ones_like(current_norms),
        where=current_norms > 0,
    )
    return (averages * scale[:, None]).astype(chunk_embeddings.dtype)


def _special_token_count(tokenizer: Any) -> int:
    """Return how many special tokens the tokenizer adds to a single text.

    Args:
        tokenizer: A HuggingFace tokenizer.

    Returns:
        The number of special tokens, or 2 (the usual start and end tokens)
        if the tokenizer cannot tell.
    """
    try:
        return int(tokenizer.num_special_tokens_to_add(pair=False))
    except (AttributeError, TypeError):
        return 2


def embed_documents(
    embedding_model: Any,
    texts: list[str],
    chunk_long_documents: bool = False,
) -> np.ndarray:
    """Embed documents with a sentence-transformer.

    By default the model truncates documents longer than its context window.
    With ``chunk_long_documents`` those documents are split into pieces that
    fit, every piece is embedded, and the pieces are averaged with
    :func:`average_chunk_embeddings`, so the whole document contributes.

    Args:
        embedding_model: A SentenceTransformer (or compatible) instance.
        texts: The documents to embed.
        chunk_long_documents: Whether to embed long documents in chunks.

    Returns:
        One embedding per document, shape ``(len(texts), dim)``.
    """
    max_seq_length = int(getattr(embedding_model, "max_seq_length", 0) or 0)
    tokenizer = getattr(embedding_model, "tokenizer", None)
    if not chunk_long_documents or not max_seq_length or tokenizer is None:
        return embedding_model.encode(
            texts, show_progress_bar=False, convert_to_numpy=True
        )

    max_tokens = max(max_seq_length - _special_token_count(tokenizer), 1)
    chunk_texts: list[str] = []
    owners: list[int] = []
    weights: list[int] = []
    for index, text in enumerate(texts):
        for chunk_text, n_tokens in split_into_token_chunks(
            text, tokenizer, max_tokens
        ):
            chunk_texts.append(chunk_text)
            owners.append(index)
            weights.append(n_tokens)

    chunk_embeddings = embedding_model.encode(
        chunk_texts, show_progress_bar=False, convert_to_numpy=True
    )
    return average_chunk_embeddings(
        chunk_embeddings,
        np.asarray(owners),
        np.asarray(weights, dtype=np.float64),
        len(texts),
    )


def load_embedding_model(config: TopicModelingConfig) -> tuple[Any, str]:
    """Load the sentence-transformer a BERTopic run uses.

    Args:
        config: The run's configuration.

    Returns:
        The model and its ID.

    Raises:
        ValueError: If the model cannot be loaded, with a message for users.
    """
    model_id = resolve_bertopic_embedding_model_id(config)
    try:
        model = load_sentence_transformer(
            model_id, trust_remote_code=config.trust_remote_code
        )
    except Exception as exc:
        message = f"Failed to load embedding model '{model_id}': {exc}"
        if "trust_remote_code" in str(exc) and not config.trust_remote_code:
            message += (
                "\n\nThis model ships its own code. If you trust the "
                "repository, enable 'Allow custom code from this model "
                "repository' in the Embedding Model section and run again."
            )
        raise ValueError(message) from exc
    return model, model_id


def long_document_notice(
    embedding_model: Any,
    model_id: str,
    texts: list[str],
    chunk_long_documents: bool,
) -> Notice | None:
    """Tell users how documents above the context window are handled.

    Args:
        embedding_model: The loaded sentence-transformer.
        model_id: The model ID, named in the notice.
        texts: The documents.
        chunk_long_documents: Whether long documents are embedded in chunks.

    Returns:
        A notice, or ``None`` when at most
        :data:`TRUNCATION_NOTICE_RATIO` of the documents are affected.
    """
    over_count, total_count, max_seq_length = count_docs_exceeding_context(
        embedding_model, texts
    )
    if not total_count or over_count / total_count <= TRUNCATION_NOTICE_RATIO:
        return None

    summary = (
        f"**{over_count} of {total_count} documents "
        f"({over_count / total_count:.0%}) exceed the "
        f"{max_seq_length}-token context window of `{model_id}`.**"
    )
    if chunk_long_documents:
        return Notice(
            "info",
            f"{summary} They will be split into chunks that fit the window "
            "and the chunk embeddings averaged, so encoding takes longer. "
            "A model with a larger context window needs fewer chunks, for "
            f"example {LONG_CONTEXT_MODELS}.",
        )
    return Notice(
        "warning",
        f"{summary} Only the leading portion of each of those documents "
        "will be used to generate the topic embedding, which may bias topic "
        "assignments toward document openings.\n\n"
        "To capture the full content of long documents, either:\n\n"
        "- enable *Embed long documents in chunks* in the *Embedding Model* "
        "section, which embeds each document piece by piece and averages "
        "the results, or\n"
        "- pick a sentence-transformer with a larger context window from the "
        f"same section, for example {LONG_CONTEXT_MODELS}. Custom models are "
        "downloaded to your own home cache (`~/.cache/huggingface`).",
    )
