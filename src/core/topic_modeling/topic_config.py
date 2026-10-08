"""Configuration and result types shared by the topic modeling modules."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, NotRequired, TypedDict

import pandas as pd

#: Default number of intervals for the topics-over-time chart.
DEFAULT_TIME_BINS = 20


class Algorithm(StrEnum):
    """Topic modeling algorithms offered by the page."""

    BERTOPIC = "bertopic"
    TOP2VEC = "top2vec"
    LDA = "lda"

    @property
    def label(self) -> str:
        """Return the human-readable name used in the UI and reports."""
        return _ALGORITHM_LABELS[self]


_ALGORITHM_LABELS = {
    Algorithm.BERTOPIC: "BERTopic (Transformer Embeddings)",
    Algorithm.TOP2VEC: "Top2Vec (Joint Semantic Embeddings)",
    Algorithm.LDA: "Latent Dirichlet Allocation (LDA)",
}

#: Human-readable names of the Top2Vec embedding backends.
TOP2VEC_BACKEND_LABELS = {
    "transformer": "Transformer (Pre-trained)",
    "doc2vec": "Doc2Vec (Train from scratch)",
}


@dataclass
class TopicModelingConfig:
    """
    Store topic modeling configuration selected from the UI.

    ``num_topics`` depends on the algorithm: the exact topic count for LDA;
    a target count or ``"auto"`` for Top2Vec; and for BERTopic with HDBSCAN a
    target count, ``"auto"`` or ``None`` to keep every cluster found (KMeans
    takes its count from ``clustering_params["n_clusters"]``).
    """

    algorithm: Algorithm
    language: str
    text_column: str
    enable_dtm: bool = False
    date_column: str | None = None
    time_bins: int = DEFAULT_TIME_BINS
    custom_stopwords: str = ""
    num_topics: int | str | None = 10
    dim_reduction_algo: str = "UMAP"
    dim_params: dict[str, Any] = field(default_factory=dict)
    clustering_algo: str = "HDBSCAN"
    clustering_params: dict[str, Any] = field(default_factory=dict)
    ngram_range: tuple[int, int] = (1, 1)
    min_df: int = 1
    reduce_frequent: bool = True
    reduce_outliers: bool = False
    top2vec_backend: str = "doc2vec"
    top2vec_speed: str = "learn"
    use_bigrams: bool = False
    passes: int = 10
    random_state: int | None = 42
    # Optional HuggingFace sentence-transformer model ID for BERTopic embeddings.
    # None means the language-based default (MiniLM). Custom IDs are downloaded
    # to the user's home cache rather than the shared read-only model cache.
    embedding_model_id: str | None = None
    # Whether a custom embedding model may run code from its repository.
    trust_remote_code: bool = False


@dataclass
class TopicKeywords:
    """Describe one topic by its display number, keywords and size."""

    topic: int | str
    keywords: list[str]
    count: int | None = None


class TopicModelingRunResult(TypedDict):
    """Store topic modeling outputs returned by the execution pipeline."""

    topic_df: pd.DataFrame
    docs_df: pd.DataFrame
    dashboard_assets: dict[str, str]
    # Keywords per topic, in the same order as ``topic_df``.
    topic_keywords: list[list[str]]

    # Optional keys returned only by the LDA pipeline for Perplexity evaluation
    lda_model: NotRequired[Any]
    corpus: NotRequired[Any]
    # Pre-tokenized corpus, if the pipeline already tokenized for training,
    # which allows :func:`evaluate_topic_quality` to skip re-tokenizing.
    tokenized_texts: NotRequired[list[list[str]]]
