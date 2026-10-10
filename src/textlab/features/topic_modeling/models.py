"""Settings, results and choices of the topic modeling feature.

Imported by the page through :mod:`.service`, so it needs only pandas.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, NotRequired, TypedDict

import pandas as pd

#: Default number of intervals for the topics-over-time chart.
DEFAULT_TIME_BINS = 20

#: Total number of runs compared by the topic stability evaluation.
STABILITY_RUNS = 3

SUPPORTED_LANGUAGES: list[str] = [
    "English",
    "German",
    "French",
    "Spanish",
    "Italian",
    "Dutch",
    "Portuguese",
    "Russian",
    "Arabic",
    "Chinese",
    "Other / Mixed",
]

# Sentence-transformer models pre-downloaded into the shared read-only cache.
# Anything else is treated as a "custom" model and will be downloaded to the
# calling user's own HuggingFace cache directory.
SHARED_EMBEDDING_MODELS: set[str] = {
    "all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}

DEFAULT_EMBEDDING_MODEL_ENGLISH: str = "all-MiniLM-L6-v2"

DEFAULT_EMBEDDING_MODEL_MULTILINGUAL: str = (
    "paraphrase-multilingual-MiniLM-L12-v2"
)

#: Tabular file extensions accepted by :func:`data.read_uploaded_table`.
TABLE_EXTENSIONS: tuple[str, ...] = ("csv", "xlsx")


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
    """Store topic modeling configuration selected from the UI.

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
    # Words seen fewer times than this in the whole collection are ignored.
    top2vec_min_count: int = 10
    use_bigrams: bool = False
    passes: int = 10
    random_state: int | None = 42
    # Optional HuggingFace sentence-transformer model ID for BERTopic
    # embeddings. None means the language-based default (MiniLM). Custom IDs
    # are downloaded to the user's home cache rather than the shared read-only
    # model cache.
    embedding_model_id: str | None = None
    # Whether a custom embedding model may run code from its repository.
    trust_remote_code: bool = False
    # Embed documents longer than the model's context window in chunks and
    # average the chunk embeddings, instead of truncating them.
    chunk_long_documents: bool = False
    # Train the model STABILITY_RUNS times and compare the topics.
    evaluate_stability: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the configuration as JSON types, for a worker request."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TopicModelingConfig:
        """Rebuild a configuration from :meth:`to_dict` output.

        Args:
            data: The dictionary.

        Returns:
            The configuration.
        """
        values = dict(data)
        values["algorithm"] = Algorithm(values["algorithm"])
        values["ngram_range"] = tuple(values["ngram_range"])
        return cls(**values)


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


@dataclass(frozen=True)
class Notice:
    """Something users should know about a run, shown with its results.

    Attributes:
        level: ``"info"`` or ``"warning"``.
        message: The text, in Markdown.
    """

    level: Literal["info", "warning"]
    message: str


@dataclass
class TopicModelingResult:
    """The outcome of a run, as the page shows and downloads it.

    Attributes:
        algorithm: The algorithm that ran.
        enable_dtm: Whether topics over time were analyzed.
        topic_df: One row per topic: number, size (not for LDA), keywords.
        dashboard_assets: HTML charts by file name.
        evaluation_metrics: Metric values by name; ``None`` when a metric
            could not be computed.
        zip_bytes: The download: report, tables and charts.
        notices: What users should know about the run, such as skipped
            rows or truncated documents.
    """

    algorithm: Algorithm
    enable_dtm: bool
    topic_df: pd.DataFrame
    dashboard_assets: dict[str, str]
    evaluation_metrics: dict[str, float | None]
    zip_bytes: bytes
    notices: list[Notice] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the result as JSON types, without the ZIP.

        The worker writes the ZIP to a file next to its result.
        """
        return {
            "algorithm": str(self.algorithm),
            "enable_dtm": self.enable_dtm,
            "topic_df": self.topic_df.to_dict(orient="split", index=False),
            "dashboard_assets": self.dashboard_assets,
            "evaluation_metrics": self.evaluation_metrics,
            "notices": [dataclasses.asdict(notice) for notice in self.notices],
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], zip_bytes: bytes
    ) -> TopicModelingResult:
        """Rebuild a result from :meth:`to_dict` output and the ZIP.

        Args:
            data: The dictionary.
            zip_bytes: The ZIP the worker wrote.

        Returns:
            The result.
        """
        table = data["topic_df"]
        return cls(
            algorithm=Algorithm(data["algorithm"]),
            enable_dtm=data["enable_dtm"],
            topic_df=pd.DataFrame(table["data"], columns=table["columns"]),
            dashboard_assets=data["dashboard_assets"],
            evaluation_metrics=data["evaluation_metrics"],
            zip_bytes=zip_bytes,
            notices=[Notice(**notice) for notice in data["notices"]],
        )
