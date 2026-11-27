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

    @property
    def description(self) -> str:
        """Return what the algorithm does and suits, in a sentence or two."""
        return _ALGORITHM_DESCRIPTIONS[self]


_ALGORITHM_LABELS = {
    Algorithm.BERTOPIC: "BERTopic (Transformer Embeddings)",
    Algorithm.TOP2VEC: "Top2Vec (Joint Semantic Embeddings)",
    Algorithm.LDA: "Latent Dirichlet Allocation (LDA)",
}

_ALGORITHM_DESCRIPTIONS = {
    Algorithm.BERTOPIC: (
        "Groups the documents by meaning with a language model, then names "
        "each group by its typical words. Finds the number of topics by "
        "itself, and can follow topics over time."
    ),
    Algorithm.TOP2VEC: (
        "Places documents and words in one space and finds where documents "
        "gather. Finds the number of topics by itself; learns from your "
        "texts or uses a language model."
    ),
    Algorithm.LDA: (
        "The classic statistical model: each document is a mix of topics, "
        "each topic a mix of words. Fast; you set the number of topics. "
        "Suits longer texts best."
    ),
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
    """Describe one topic by its display number, keywords and size.

    ``weights`` gives each keyword's weight in the topic, in the same
    order, as the engine scores it (c-TF-IDF for BERTopic, probability for
    LDA, similarity for Top2Vec); empty when the engine gives none.
    """

    topic: int | str
    keywords: list[str]
    count: int | None = None
    weights: list[float] = field(default_factory=list)


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

    # What the overview draws, only for the run that is shown: the topics
    # with their keyword weights, one vector per topic in the same order,
    # and what each algorithm's own library draws (see ``overview``).
    topics: NotRequired[list[TopicKeywords]]
    topic_vectors: NotRequired[Any]
    document_vectors: NotRequired[Any]
    time_rows: NotRequired[list[list[Any]]]
    ldavis: NotRequired[Any]
    document_topics: NotRequired[Any]
    topic_words: NotRequired[list[list[list[Any]]]]


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
class TopicSummary:
    """One topic, as the overview shows it.

    The first fields are known for every algorithm; the last ones only for
    the algorithm whose library draws them (they stay empty otherwise).

    Attributes:
        topic: Its number, as in the topic table.
        keywords: Its keywords, most typical first.
        weights: Each keyword's weight, scaled so the first is 1; empty
            when the engine gives none.
        documents: How many documents have it as their main topic.
        examples: The beginning of a few of those documents, the most
            typical first when the engine says how typical they are.
        prevalence: Its share of the collection, from 0 to 1: of the
            words for LDA (as pyLDAvis sizes topics), else of the
            documents.
        cloud: Top2Vec's word cloud: up to 30 words with their weight,
            scaled so the first is 1, as ``[word, weight]``.
        mixtures: LDA: for each example, the topics it mixes, as
            ``[topic index, share]``, the largest first.
        terms: LDA: the terms pyLDAvis may rank for the topic, as ``[term,
            frequency in the topic, frequency overall, log probability,
            log lift]``; their relevance for a weight lambda is ``lambda *
            log probability + (1 - lambda) * log lift``.
        time_words: BERTopic over time: per interval, the topic's words
            then, or "" when it has no documents there.
    """

    topic: int | str
    keywords: list[str]
    weights: list[float]
    documents: int
    examples: list[str]
    prevalence: float = 0.0
    cloud: list[list[Any]] = field(default_factory=list)
    mixtures: list[list[list[float]]] = field(default_factory=list)
    terms: list[list[Any]] = field(default_factory=list)
    time_words: list[str] = field(default_factory=list)


@dataclass
class Hierarchy:
    """How the topics merge into larger ones: a dendrogram.

    Attributes:
        order: The topics' indices in the order of the leaves, top first.
        links: Each merge as its outline: four distances and four leaf
            positions, ``[[d1, d2, d3, d4], [p1, p2, p3, p4]]``; a leaf
            position is its place in ``order`` (0, 1, ...), a merge's is
            between those it joins.
        height: The distance of the last merge.
    """

    order: list[int]
    links: list[list[list[float]]]
    height: float


@dataclass
class DocumentMap:
    """The documents placed where documents of alike meaning lie close.

    Attributes:
        points: Each document drawn, as ``[x, y, topic index]`` with x and
            y between 0 and 1 and topic index -1 for no topic.
        excerpts: The beginning of each, in the same order.
        documents: How many documents there are in all; more than
            ``points`` when only a sample is drawn.
    """

    points: list[list[float]]
    excerpts: list[str]
    documents: int


@dataclass
class TopicOverview:
    """What a run found, ready to draw: topics, their map, their timeline.

    Attributes:
        documents: How many documents were analyzed.
        outliers: How many have no topic (outliers, or empty for LDA).
        topics: The topics, in the topic table's order.
        similarity: How alike each two topics are, from 0 to 1, in the
            order of ``topics``; ``None`` when there are fewer than two
            topics or the engine gives no topic vectors.
        positions: Each topic's place on a map where alike topics lie
            close, as ``[x, y]`` between 0 and 1: pyLDAvis's for LDA, else
            from ``similarity``; ``None`` without either.
        time_labels: The intervals of the timeline, for people; empty
            without topics over time.
        over_time: Per topic, how many of its documents fall into each
            interval.
        hierarchy: How the topics merge (from ``similarity``), or
            ``None``.
        document_map: The documents placed by meaning (BERTopic, Top2Vec),
            or ``None``.
        salient: LDA: pyLDAvis's most salient terms of the whole
            collection, as ``[term, frequency]``.
        term_topics: LDA: for each term a topic may rank, how its
            occurrences spread over the topics, as ``[topic index,
            share]``.
        origin: LDA: where pyLDAvis's axes cross on the map, ``[x, y]``.
    """

    documents: int
    outliers: int
    topics: list[TopicSummary]
    similarity: list[list[float]] | None = None
    positions: list[list[float]] | None = None
    time_labels: list[str] = field(default_factory=list)
    over_time: list[list[int]] = field(default_factory=list)
    hierarchy: Hierarchy | None = None
    document_map: DocumentMap | None = None
    salient: list[list[Any]] = field(default_factory=list)
    term_topics: dict[str, list[list[float]]] = field(default_factory=dict)
    origin: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the overview as JSON types."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TopicOverview:
        """Rebuild an overview from :meth:`to_dict` output.

        Args:
            data: The dictionary.

        Returns:
            The overview.
        """
        values = dict(data)
        values["topics"] = [TopicSummary(**item) for item in data["topics"]]
        if data.get("hierarchy"):
            values["hierarchy"] = Hierarchy(**data["hierarchy"])
        if data.get("document_map"):
            values["document_map"] = DocumentMap(**data["document_map"])
        return cls(**values)


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
        overview: The topics, their map and timeline, to draw; ``None``
            if it could not be made.
    """

    algorithm: Algorithm
    enable_dtm: bool
    topic_df: pd.DataFrame
    dashboard_assets: dict[str, str]
    evaluation_metrics: dict[str, float | None]
    zip_bytes: bytes
    notices: list[Notice] = field(default_factory=list)
    overview: TopicOverview | None = None

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
            "overview": self.overview.to_dict() if self.overview else None,
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
            overview=TopicOverview.from_dict(data["overview"])
            if data.get("overview")
            else None,
        )
