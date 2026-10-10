"""BERTopic engine: training, topic extraction and visualizations."""

import html
import logging
import re
from typing import Any

import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from textlab.features.topic_modeling import small_corpus
from textlab.features.topic_modeling.models import TopicKeywords

LOGGER = logging.getLogger(__name__)


_SUPPORTED_DIM_ALGOS = {"UMAP", "PCA", "Truncated SVD", "None"}
_SUPPORTED_CLUSTERING_ALGOS = {"HDBSCAN", "KMeans"}

OUTLIER_LABEL = "Outlier"
_TOP_N_KEYWORDS = 10
_LABEL_KEYWORDS = 3
_LABEL_MAX_CHARS = 32
# BERTopic hard-codes its 0-based IDs as "Topic N" in a few chart elements.
_RAW_TOPIC_REFERENCE = re.compile(r"Topic (-?\d+)")
# scikit-learn messages raised when ``min_df`` prunes the whole vocabulary.
_PRUNING_SIGNATURES = (
    "max_df corresponds to < documents than min_df",
    "after pruning, no terms remain",
)


def display_topic_id(topic_id: int) -> int | str:
    """Map a BERTopic topic ID to the identifier shown to users.

    BERTopic numbers topics from 0 and marks outliers as -1. The page numbers
    topics from 1 for every algorithm, so BERTopic IDs are shifted by one.

    Args:
        topic_id: The topic ID assigned by BERTopic.

    Returns:
        The 1-based topic number, or ``OUTLIER_LABEL`` for the outlier topic.
    """
    return OUTLIER_LABEL if topic_id == -1 else topic_id + 1


def _topic_title(topic_id: int) -> str:
    """Build the short title ("Topic N" or "Outlier") for a BERTopic topic ID.

    Args:
        topic_id: The topic ID assigned by BERTopic.

    Returns:
        The title using the 1-based display number.
    """
    return OUTLIER_LABEL if topic_id == -1 else f"Topic {topic_id + 1}"


def _renumber_topic_references(text: str) -> str:
    """Replace BERTopic's 0-based "Topic N" references in a chart string.

    Args:
        text: A label or hover text produced by a BERTopic visualization.

    Returns:
        The text with every "Topic N" reference renumbered for display.
    """
    return _RAW_TOPIC_REFERENCE.sub(
        lambda match: _topic_title(int(match.group(1))),
        text,
    )


def _apply_display_labels(topic_model: BERTopic) -> None:
    """Set custom topic labels that use the 1-based display numbering.

    Visualizations called with ``custom_labels=True`` then show the same topic
    numbers as the result tables, e.g. "Topic 3: tax, budget, spending".

    Args:
        topic_model: A fitted BERTopic model.
    """
    labels = []
    for topic_id in sorted(set(topic_model.topics_)):
        if topic_id == -1:
            labels.append(OUTLIER_LABEL)
            continue

        words = [
            word
            for word, _ in (topic_model.get_topic(topic_id) or [])[
                :_LABEL_KEYWORDS
            ]
            if word
        ]
        label = f"{_topic_title(topic_id)}: {', '.join(words)}"
        if len(label) > _LABEL_MAX_CHARS:
            label = label[: _LABEL_MAX_CHARS - 3] + "..."
        labels.append(label)

    topic_model.set_topic_labels(labels)


def _renumber_intertopic_map(fig: Any) -> None:
    """Renumber the topic IDs that the intertopic distance map hard-codes.

    The hover header reads the raw topic ID from ``customdata[0]`` and the
    slider steps are labelled "Topic N"; neither honours custom labels.

    Args:
        fig: The plotly figure returned by ``BERTopic.visualize_topics``.
    """
    for trace in fig.data:
        customdata = getattr(trace, "customdata", None)
        if customdata is None:
            continue
        rows = [list(row) for row in customdata]
        for row in rows:
            row[0] = display_topic_id(int(row[0]))
        trace.customdata = rows

    for slider in fig.layout.sliders or ():
        for step in slider.steps:
            if step.label:
                step.label = _renumber_topic_references(step.label)


def _renumber_hover_text(fig: Any) -> None:
    """Renumber the "Topic N" references in the hover text of every trace.

    Args:
        fig: A plotly figure produced by a BERTopic visualization.
    """
    for trace in fig.data:
        hovertext = getattr(trace, "hovertext", None)
        if hovertext is None or isinstance(hovertext, str):
            continue
        trace.hovertext = [
            _renumber_topic_references(str(text)) for text in hovertext
        ]


def _is_vocabulary_pruning_error(exc: BaseException) -> bool:
    """Check whether *exc* means ``min_df`` removed the whole vocabulary.

    Args:
        exc: The exception raised while extracting topic keywords.

    Returns:
        True if the exception matches a known scikit-learn pruning failure.
    """
    message = str(exc).lower()
    return any(signature in message for signature in _PRUNING_SIGNATURES)


def _vocabulary_pruning_error(min_df: int) -> ValueError:
    """Build the user-facing error for a vocabulary pruned away by ``min_df``.

    Args:
        min_df: The minimum topic frequency that was requested.

    Returns:
        A ValueError with guidance on how to fix the configuration.
    """
    return ValueError(
        f"No topic keywords could be extracted with a Minimum Topic Frequency "
        f"(min_df) of {min_df}. BERTopic applies this threshold to topics, "
        "not "
        f"documents: a word must appear in at least {min_df} topic(s) to be "
        "kept, so the value must not exceed the number of topics found. "
        "Lower "
        f"it (1 is a safe default) or remove some custom stopwords, then run "
        f"again."
    )


def _notice_html(message: str) -> str:
    """Return a styled HTML notice block for visualization messages.

    Args:
        message: The message content to display inside the notice.

    Returns:
        A string containing the HTML for the formatted notice block.
    """
    return (
        "<div style='padding: 30px; font-family: sans-serif; color: #555; "
        "background: #f9f9f9; border-radius: 8px;'>"
        "<h3>Visualization Notice</h3>"
        f"<p>{message}</p>"
        "</div>"
    )


def train_bertopic_model(
    texts: list[str],
    language: str,
    num_topics: int | str | None,
    stop_words_set: set[str],
    dim_reduction_algo: str = "UMAP",
    dim_params: dict[str, Any] | None = None,
    clustering_algo: str = "HDBSCAN",
    clustering_params: dict[str, Any] | None = None,
    ngram_range: tuple[int, int] = (1, 1),
    min_df: int = 1,
    reduce_outliers: bool = False,
    reduce_frequent_words: bool = True,
    random_state: int | None = 42,
    embedding_model: Any = None,
    precomputed_embeddings: np.ndarray | None = None,
) -> tuple[BERTopic, list[int], np.ndarray | None]:
    """Train a BERTopic model.

    Dimensionality reduction and clustering are configurable.

    Args:
        texts: The input documents to model.
        language: The language setting used to select the default embedding
            model when ``embedding_model`` is not provided.
        num_topics: The desired number of topics for BERTopic when applicable.
        stop_words_set: A set of stop words used by the vectorizer.
        dim_reduction_algo: The dimensionality reduction algorithm. One of
            ``"UMAP"``, ``"PCA"``, ``"Truncated SVD"``, ``"None"``.
        dim_params: Optional parameters for the dimensionality reduction model.
        clustering_algo: One of ``"HDBSCAN"`` or ``"KMeans"``.
        clustering_params: Optional parameters for the clustering model.
        ngram_range: The lower and upper boundary of the n-grams to extract.
        min_df: Minimum number of topics a word must appear in to be used as
            a keyword. BERTopic fits the vectorizer on one joined document
            per topic, so this counts topics rather than documents.
        reduce_outliers: Whether to reduce outlier assignments after fitting
            when using HDBSCAN. Topic sizes, keywords and embeddings are then
            recomputed from the new assignments.
        reduce_frequent_words: Whether to reduce frequent words in the
            class-based TF-IDF transformer.
        random_state: Random seed for the dimensionality-reduction step. Set
            to ``None`` to make the run non-deterministic (used by the
            stability evaluation).
        embedding_model: Optional pre-instantiated SentenceTransformer to use
            as the embedding backbone. When ``None``, BERTopic will construct
            its own default model based on ``language``.
        precomputed_embeddings: Optional matrix of document embeddings. When
            provided, the embedding step is skipped and this matrix is passed
            directly to ``fit_transform``.

    Returns:
        A tuple ``(topic_model, topics, probabilities)`` where
        ``probabilities`` may be ``None`` if outlier reduction invalidated the
        original probability matrix. The model carries custom topic labels
        that use the 1-based display numbering.

    Raises:
        ValueError: If no texts are provided, ``ngram_range`` is invalid, an
            unsupported dimensionality-reduction/clustering algorithm is
            requested, the corpus is too small to cluster, or ``min_df``
            leaves no keywords to extract.
    """
    if clustering_params is None:
        clustering_params = {}
    if dim_params is None:
        dim_params = {}

    if not texts:
        raise ValueError("No texts were provided to BERTopic.")
    if ngram_range[0] > ngram_range[1]:
        raise ValueError(
            "Invalid ngram_range: lower bound cannot be greater than upper "
            "bound."
        )
    if dim_reduction_algo not in _SUPPORTED_DIM_ALGOS:
        raise ValueError(
            f"Unsupported dimensionality-reduction algorithm: "
            f"'{dim_reduction_algo}'. Expected one of "
            f"{sorted(_SUPPORTED_DIM_ALGOS)}."
        )
    if clustering_algo not in _SUPPORTED_CLUSTERING_ALGOS:
        raise ValueError(
            f"Unsupported clustering algorithm: '{clustering_algo}'. "
            f"Expected one of {sorted(_SUPPORTED_CLUSTERING_ALGOS)}."
        )

    # Fall back to BERTopic's language shortcut only when no explicit model is
    # provided.
    embedding_language = "english" if language == "English" else "multilingual"

    tokenizer = None
    if language == "Chinese":
        try:
            import jieba

            def tokenize_zh(text: str) -> list[str]:
                tokens = jieba.lcut(text)
                if stop_words_set:
                    return [t for t in tokens if t not in stop_words_set]
                return tokens

            tokenizer = tokenize_zh
        except ImportError:
            LOGGER.warning(
                "'jieba' is missing. Default tokenization will be used for "
                "Chinese."
            )

    vectorizer_model = CountVectorizer(
        stop_words=sorted(stop_words_set)
        if stop_words_set and language != "Chinese"
        else None,
        ngram_range=ngram_range,
        min_df=min_df,
        tokenizer=tokenizer,
    )

    ctfidf_model = ClassTfidfTransformer(
        reduce_frequent_words=reduce_frequent_words
    )

    # Configure Dimensionality Reduction. A ``random_state`` in ``dim_params``
    # (the UI's "Lock Seed" option) takes precedence; the top-level
    # ``random_state`` is only used when ``dim_params`` does not set one.
    n_components = int(dim_params.get("n_components", 5))
    effective_random_state = dim_params.get("random_state", random_state)

    if dim_reduction_algo == "PCA":
        dim_model = PCA(
            n_components=n_components,
            random_state=effective_random_state,
        )
    elif dim_reduction_algo == "Truncated SVD":
        dim_model = TruncatedSVD(
            n_components=n_components,
            random_state=effective_random_state,
        )
    elif dim_reduction_algo == "None":
        dim_model = BaseDimensionalityReduction()
    else:  # UMAP
        n_neighbors = int(dim_params.get("n_neighbors", 15))
        min_dist = float(dim_params.get("min_dist", 0.0))
        dim_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            metric="cosine",
            random_state=effective_random_state,
        )

    # Configure Clustering Step
    if clustering_algo == "KMeans":
        n_clusters = int(clustering_params.get("n_clusters", 10))
        cluster_model = KMeans(
            n_clusters=n_clusters,
            random_state=effective_random_state,
            n_init="auto",
        )
        bertopic_nr_topics = None
    else:
        min_cluster_size = int(clustering_params.get("min_cluster_size", 10))
        min_samples = clustering_params.get("min_samples")
        if min_samples is not None:
            min_samples = int(min_samples)

        cluster_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
        bertopic_nr_topics = num_topics

    topic_model = BERTopic(
        language=embedding_language if embedding_model is None else None,
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        umap_model=dim_model,
        hdbscan_model=cluster_model,
        ctfidf_model=ctfidf_model,
        nr_topics=bertopic_nr_topics,
        calculate_probabilities=True,
    )

    try:
        topics, probabilities = topic_model.fit_transform(
            texts,
            embeddings=precomputed_embeddings,
        )
    except (ValueError, TypeError) as e:
        # Same UMAP/HDBSCAN stack as Top2Vec, so the same failures on a corpus
        # with too few documents to cluster.
        if small_corpus.is_corpus_too_small(e):
            raise small_corpus.too_small_error(len(texts), "BERTopic") from e
        if _is_vocabulary_pruning_error(e):
            raise _vocabulary_pruning_error(min_df) from e
        raise

    if reduce_outliers and clustering_algo == "HDBSCAN":
        topics = topic_model.reduce_outliers(
            texts,
            topics,
            strategy="c-tf-idf",
        )
        # Recompute topic sizes, keywords and embeddings from the new
        # assignments. The original vectorizer and c-TF-IDF models must be
        # passed again, otherwise BERTopic falls back to defaults and drops the
        # stopword, n-gram and min_df settings.
        try:
            topic_model.update_topics(
                texts,
                topics=topics,
                vectorizer_model=vectorizer_model,
                ctfidf_model=ctfidf_model,
            )
        except ValueError as e:
            if _is_vocabulary_pruning_error(e):
                raise _vocabulary_pruning_error(min_df) from e
            raise
        # HDBSCAN probabilities correspond to the *original* topic assignments;
        # once outliers are re-assigned via c-TF-IDF those confidences are no
        # longer meaningful, so drop them rather than mislead the user.
        probabilities = None

    _apply_display_labels(topic_model)

    return topic_model, topics, probabilities


def extract_bertopic_topics(topic_model: BERTopic) -> list[TopicKeywords]:
    """Extract the keywords and sizes of every BERTopic topic.

    The outlier topic (-1) is skipped. BERTopic pads topics that have fewer
    than ten distinct words with empty strings; those are dropped so that
    they do not count as keywords in the evaluation metrics.

    Args:
        topic_model: A fitted BERTopic model.

    Returns:
        One entry per topic, numbered with :func:`display_topic_id`.
    """
    topics = []
    for _, row in topic_model.get_topic_info().iterrows():
        topic_id = int(row["Topic"])
        if topic_id == -1:
            continue

        words = topic_model.get_topic(topic_id) or []
        keywords = [word for word, _ in words[:_TOP_N_KEYWORDS] if word]
        if keywords:
            topics.append(
                TopicKeywords(
                    topic=display_topic_id(topic_id),
                    keywords=keywords,
                    count=int(row["Count"]),
                )
            )

    return topics


def generate_bertopic_document_topics_df(
    topics: list[int],
    probabilities: np.ndarray | None,
    original_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach BERTopic topic assignments to the original DataFrame.

    ``Dominant_Topic`` and ``Topic_Confidence`` (when available) are placed
    as the first two columns.
    """
    if len(topics) != len(original_df):
        raise ValueError(
            f"Topic assignment length ({len(topics)}) does not match "
            f"dataframe length ({len(original_df)})."
        )

    formatted_topics = [display_topic_id(int(t)) for t in topics]

    result_df = original_df.copy()
    # Drop any pre-existing columns with our reserved names to avoid
    # collisions.
    result_df = result_df.drop(
        columns=[
            c
            for c in ("Dominant_Topic", "Topic_Confidence")
            if c in result_df.columns
        ],
        errors="ignore",
    )
    result_df["Dominant_Topic"] = formatted_topics

    if probabilities is not None:
        if isinstance(probabilities, np.ndarray) and probabilities.ndim == 2:
            confidence = np.max(probabilities, axis=1)
        else:
            confidence = probabilities
        result_df["Topic_Confidence"] = [
            round(float(p), 4) for p in confidence
        ]
    else:
        result_df["Topic_Confidence"] = None

    other_cols = [
        c
        for c in result_df.columns
        if c not in ("Dominant_Topic", "Topic_Confidence")
    ]
    return result_df[["Dominant_Topic", "Topic_Confidence", *other_cols]]


def generate_bertopic_visualizations(topic_model: BERTopic) -> dict[str, str]:
    """Generate BERTopic visualizations as HTML strings.

    This function attempts to create three BERTopic visualizations:
    intertopic distance map, topic word-score bar chart, and topic
    similarity heatmap. If the model contains fewer than two valid
    topics, placeholder notice HTML is returned for all visualizations.
    If generation of an individual visualization fails, a notice HTML
    message is returned for that specific visualization instead.

    Args:
        topic_model: A fitted BERTopic model.

    Returns:
        A dictionary mapping visualization names to HTML strings. The
        returned keys are:
            - "distance_map"
            - "barchart"
            - "heatmap"
    """
    visualizations: dict[str, str] = {}
    topic_info = topic_model.get_topic_info()
    valid_topics = topic_info[topic_info["Topic"] != -1]

    if len(valid_topics) < 2:
        error_html = _notice_html(
            "The visual maps require at least <strong>2 distinct "
            "topics</strong>. "
            f"The model only identified {len(valid_topics)} valid topic(s) in "
            "this dataset."
        )
        return {
            "distance_map": error_html,
            "barchart": error_html,
            "heatmap": error_html,
        }

    try:
        fig_distance = topic_model.visualize_topics()
        _renumber_intertopic_map(fig_distance)
        visualizations["distance_map"] = fig_distance.to_html(
            full_html=False,
            include_plotlyjs="cdn",
        )
    except Exception:
        LOGGER.exception("Could not generate the BERTopic distance map.")
        visualizations["distance_map"] = _notice_html(
            "Could not generate the intertopic distance map."
        )

    try:
        fig_barchart = topic_model.visualize_barchart(
            top_n_topics=12,
            custom_labels=True,
        )
        visualizations["barchart"] = fig_barchart.to_html(
            full_html=False,
            include_plotlyjs="cdn",
        )
    except Exception:
        LOGGER.exception("Could not generate the BERTopic barchart.")
        visualizations["barchart"] = _notice_html(
            "Could not generate the topic word-score chart."
        )

    try:
        fig_heatmap = topic_model.visualize_heatmap(custom_labels=True)
        visualizations["heatmap"] = fig_heatmap.to_html(
            full_html=False,
            include_plotlyjs="cdn",
        )
    except Exception:
        LOGGER.exception("Could not generate the BERTopic heatmap.")
        visualizations["heatmap"] = _notice_html(
            "Could not generate the topic similarity heatmap."
        )

    return visualizations


def generate_topics_over_time_html(
    topic_model: BERTopic,
    texts: list[str],
    timestamps: list[Any],
    nr_bins: int | None = None,
) -> str:
    """Return an HTML chart of topics over time from a BERTopic model.

    Args:
        topic_model: A fitted BERTopic model.
        texts: A list of input texts used for the topics-over-time analysis.
        timestamps: A list of timestamps corresponding to each input text.
        nr_bins: Number of equal-width time intervals to group the
            timestamps into, or ``None`` to keep every distinct timestamp.

    Returns:
        An HTML string containing the topics-over-time visualization. If an
        error occurs during generation, an HTML error message is returned
        instead.

    Raises:
        ValueError: If the number of texts does not match the number of
            timestamps.
    """
    if len(texts) != len(timestamps):
        raise ValueError(
            f"Text count ({len(texts)}) does not match "
            f"timestamp count ({len(timestamps)})."
        )

    try:
        topics_over_time = topic_model.topics_over_time(
            texts,
            timestamps,
            nr_bins=nr_bins,
        )
        fig = topic_model.visualize_topics_over_time(
            topics_over_time,
            custom_labels=True,
        )
        _renumber_hover_text(fig)
        return fig.to_html(full_html=False, include_plotlyjs="cdn")
    except Exception as exc:
        LOGGER.exception("Could not generate the topics-over-time chart.")
        return _notice_html(
            f"Failed to generate topics over time: {html.escape(str(exc))}"
        )
