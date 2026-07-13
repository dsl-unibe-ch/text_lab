import sys
import traceback
from typing import Any

import pandas as pd
import numpy as np
from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
from hdbscan import HDBSCAN
from umap import UMAP


_SUPPORTED_DIM_ALGOS = {"UMAP", "PCA", "Truncated SVD", "None"}
_SUPPORTED_CLUSTERING_ALGOS = {"HDBSCAN", "KMeans"}


def _notice_html(message: str) -> str:
    """
    Helper function to generate a styled HTML notice block for visualization messages.

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
    """
    Train a BERTopic model with configurable dimensionality reduction and
    clustering settings.

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
        min_df: Minimum document frequency for the vectorizer.
        reduce_outliers: Whether to reduce outlier assignments after fitting
            when using HDBSCAN.
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
        original probability matrix.

    Raises:
        ValueError: If no texts are provided, ``ngram_range`` is invalid, or
            an unsupported dimensionality-reduction/clustering algorithm is
            requested.
    """
    if clustering_params is None:
        clustering_params = {}
    if dim_params is None:
        dim_params = {}

    if not texts:
        raise ValueError("No texts were provided to BERTopic.")
    if ngram_range[0] > ngram_range[1]:
        raise ValueError(
            "Invalid ngram_range: lower bound cannot be greater than upper bound."
        )
    if dim_reduction_algo not in _SUPPORTED_DIM_ALGOS:
        raise ValueError(
            f"Unsupported dimensionality-reduction algorithm: "
            f"'{dim_reduction_algo}'. Expected one of {sorted(_SUPPORTED_DIM_ALGOS)}."
        )
    if clustering_algo not in _SUPPORTED_CLUSTERING_ALGOS:
        raise ValueError(
            f"Unsupported clustering algorithm: '{clustering_algo}'. "
            f"Expected one of {sorted(_SUPPORTED_CLUSTERING_ALGOS)}."
        )

    # Fall back to BERTopic's language shortcut only when no explicit model is provided.
    embedding_language = (
        "english"
        if language == "English"
        else "multilingual"
    )

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
            print(
                "Warning: 'jieba' library is missing. Default tokenization "
                "will be used for Chinese.",
                file=sys.stderr,
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

    # Configure Dimensionality Reduction. The top-level ``random_state`` is the
    # source of truth; only fall back to ``dim_params`` if the caller did not
    # pass ``random_state`` explicitly through ``dim_params`` either.
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

    topics, probabilities = topic_model.fit_transform(
        texts,
        embeddings=precomputed_embeddings,
    )

    if reduce_outliers and clustering_algo == "HDBSCAN":
        topics = topic_model.reduce_outliers(
            texts,
            topics,
            strategy="c-tf-idf",
        )
        # HDBSCAN probabilities correspond to the *original* topic assignments;
        # once outliers are re-assigned via c-TF-IDF those confidences are no
        # longer meaningful, so drop them rather than mislead the user.
        probabilities = None

    return topic_model, topics, probabilities


def generate_bertopic_keywords_df(topic_model: BERTopic) -> pd.DataFrame:
    """
    Generate a DataFrame containing BERTopic topic keywords and counts.

    This function extracts topic information from a fitted BERTopic model,
    skips the outlier topic (-1), and creates a DataFrame with the topic
    number, document count, and top keywords for each topic.

    Args:
        topic_model: A fitted BERTopic model.

    Returns:
        A pandas DataFrame with the columns:
            - "Topic"
            - "Count"
            - "Keywords"
    """
    topic_info = topic_model.get_topic_info()
    topic_data = []

    for _, row in topic_info.iterrows():
        topic_id = int(row["Topic"])
        if topic_id == -1:
            continue

        words = topic_model.get_topic(topic_id)
        if words:
            topic_keywords = ", ".join(word for word, _ in words[:10])
            topic_data.append(
                {
                    "Topic": topic_id + 1,
                    "Count": int(row["Count"]),
                    "Keywords": topic_keywords,
                }
            )

    return pd.DataFrame(topic_data)


def generate_bertopic_document_topics_df(
    topics: list[int],
    probabilities: np.ndarray | None,
    original_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Attach BERTopic topic assignments (and confidence when available) to the
    original DataFrame. ``Dominant_Topic`` and ``Topic_Confidence`` are placed
    as the first two columns.
    """
    if len(topics) != len(original_df):
        raise ValueError(
            f"Topic assignment length ({len(topics)}) does not match "
            f"dataframe length ({len(original_df)})."
        )

    formatted_topics = [t + 1 if t != -1 else "Outlier" for t in topics]

    result_df = original_df.copy()
    # Drop any pre-existing columns with our reserved names to avoid collisions.
    result_df = result_df.drop(
        columns=[c for c in ("Dominant_Topic", "Topic_Confidence") if c in result_df.columns],
        errors="ignore",
    )
    result_df["Dominant_Topic"] = formatted_topics

    if probabilities is not None:
        if isinstance(probabilities, np.ndarray) and probabilities.ndim == 2:
            confidence = np.max(probabilities, axis=1)
        else:
            confidence = probabilities
        result_df["Topic_Confidence"] = [round(float(p), 4) for p in confidence]
    else:
        result_df["Topic_Confidence"] = None

    other_cols = [c for c in result_df.columns if c not in ("Dominant_Topic", "Topic_Confidence")]
    return result_df[["Dominant_Topic", "Topic_Confidence", *other_cols]]


def generate_bertopic_visualizations(topic_model: BERTopic) -> dict[str, str]:
    """
    Generate BERTopic visualizations as HTML strings.

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
            "The visual maps require at least <strong>2 distinct topics</strong>. "
            f"The model only identified {len(valid_topics)} valid topic(s) in this dataset."
        )
        return {
            "distance_map": error_html,
            "barchart": error_html,
            "heatmap": error_html,
        }

    try:
        fig_distance = topic_model.visualize_topics()
        visualizations["distance_map"] = fig_distance.to_html(
            full_html=False,
            include_plotlyjs="cdn",
        )
    except Exception:
        print(
            f"--- BERTopic distance map error ---\n{traceback.format_exc()}",
            file=sys.stderr,
        )
        visualizations["distance_map"] = _notice_html(
            "Could not generate the intertopic distance map."
        )

    try:
        fig_barchart = topic_model.visualize_barchart(top_n_topics=12)
        visualizations["barchart"] = fig_barchart.to_html(
            full_html=False,
            include_plotlyjs="cdn",
        )
    except Exception:
        print(
            f"--- BERTopic barchart error ---\n{traceback.format_exc()}",
            file=sys.stderr,
        )
        visualizations["barchart"] = _notice_html(
            "Could not generate the topic word-score chart."
        )

    try:
        fig_heatmap = topic_model.visualize_heatmap()
        visualizations["heatmap"] = fig_heatmap.to_html(
            full_html=False,
            include_plotlyjs="cdn",
        )
    except Exception:
        print(
            f"--- BERTopic heatmap error ---\n{traceback.format_exc()}",
            file=sys.stderr,
        )
        visualizations["heatmap"] = _notice_html(
            "Could not generate the topic similarity heatmap."
        )

    return visualizations


def generate_topics_over_time_html(
    topic_model: BERTopic,
    texts: list[str],
    timestamps: list[Any],
) -> str:
    """
    Generate an HTML visualization of topics over time using a BERTopic model.

    Args:
        topic_model: A fitted BERTopic model.
        texts: A list of input texts used for the topics-over-time analysis.
        timestamps: A list of timestamps corresponding to each input text.

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
        topics_over_time = topic_model.topics_over_time(texts, timestamps)
        fig = topic_model.visualize_topics_over_time(topics_over_time)
        return fig.to_html(full_html=False, include_plotlyjs="cdn")
    except Exception as e:
        print(
            f"--- Topics Over Time Error ---\n{traceback.format_exc()}",
            file=sys.stderr,
        )
        return (
            "<div style='padding:20px; color:red;'>"
            f"Failed to generate topics over time: {str(e)}"
            "</div>"
        )