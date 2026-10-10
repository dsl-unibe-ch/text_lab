"""Run the selected topic modeling algorithm and collect its outputs."""

import copy
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from .bertopic_engine import (
    extract_bertopic_topics,
    generate_bertopic_document_topics_df,
    generate_bertopic_visualizations,
    generate_topics_over_time_html,
    train_bertopic_model,
)
from .evaluation import calculate_jaccard_stability
from .lda_engine import (
    extract_lda_topics,
    generate_lda_document_topics_df,
    generate_lda_html,
    train_lda_model,
)
from .top2vec_engine import (
    extract_top2vec_topics,
    generate_top2vec_barchart_html,
    generate_top2vec_document_topics_df,
    train_top2vec_model,
)
from .topic_config import Algorithm, TopicModelingConfig, TopicModelingRunResult
from .topic_utils import (
    build_topic_table,
    get_stopword_set,
    preprocess_texts_for_lda,
    resolve_time_bins,
    validate_minimum_documents,
)

#: Total number of runs compared by the topic stability evaluation.
STABILITY_RUNS = 3


def run_topic_modeling_pipeline(
    df: pd.DataFrame,
    config: TopicModelingConfig,
    timestamps: list[Any] | None = None,
    embedding_model: Any = None,
    precomputed_embeddings: np.ndarray | None = None,
    include_dashboards: bool = True,
) -> TopicModelingRunResult:
    """
    Execute the selected topic modeling pipeline.

    Args:
        df: The prepared input DataFrame.
        config: The topic modeling configuration.
        timestamps: Optional parsed timestamps used for BERTopic
            topics-over-time analysis.
        embedding_model: Optional pre-instantiated SentenceTransformer instance
            (BERTopic only).
        precomputed_embeddings: Optional precomputed document embeddings
            (BERTopic only). When provided, the embedding step is skipped and
            the same matrix can be reused across stability runs.
        include_dashboards: Whether to generate the HTML dashboards. Runs
            that only need topics and keywords (e.g. the stability check) skip
            them, and return an empty ``dashboard_assets`` dictionary.

    Returns:
        The topic table, document table, keyword lists and dashboard assets
        (plus the LDA model, corpus and tokens for LDA evaluation).
    """
    raw_texts = df[config.text_column].astype(str).tolist()
    validate_minimum_documents(raw_texts)

    if config.algorithm == Algorithm.LDA:
        return _run_lda_pipeline(df, raw_texts, config, include_dashboards)

    if config.algorithm == Algorithm.TOP2VEC:
        return _run_top2vec_pipeline(df, raw_texts, config, include_dashboards)

    return _run_bertopic_pipeline(
        df,
        raw_texts,
        config,
        timestamps=timestamps,
        embedding_model=embedding_model,
        precomputed_embeddings=precomputed_embeddings,
        include_dashboards=include_dashboards,
    )


def evaluate_topic_stability(
    df: pd.DataFrame,
    config: TopicModelingConfig,
    base_keywords: list[list[str]],
    embedding_model: Any = None,
    precomputed_embeddings: np.ndarray | None = None,
) -> float:
    """
    Measure how reproducible the topics are across independent runs.

    The model is trained ``STABILITY_RUNS - 1`` more times with unlocked
    random seeds. Every pair of runs (including the base run) is compared
    with :func:`calculate_jaccard_stability` and the scores are averaged.
    BERTopic embeddings are reused, so only the variance of dimensionality
    reduction and clustering is measured.

    Args:
        df: The prepared input DataFrame used for the base run.
        config: The configuration of the base run.
        base_keywords: Keywords per topic from the base run.
        embedding_model: Optional SentenceTransformer (BERTopic only).
        precomputed_embeddings: Optional document embeddings (BERTopic only).

    Returns:
        The average Jaccard stability, between 0 and 1.
    """
    keyword_runs = [base_keywords]
    for _ in range(STABILITY_RUNS - 1):
        result = run_topic_modeling_pipeline(
            df,
            _unseeded_copy(config),
            embedding_model=embedding_model,
            precomputed_embeddings=precomputed_embeddings,
            include_dashboards=False,
        )
        keyword_runs.append(result["topic_keywords"])

    scores = [
        calculate_jaccard_stability(first, second)
        for first, second in combinations(keyword_runs, 2)
    ]
    return round(sum(scores) / len(scores), 4)


def _unseeded_copy(config: TopicModelingConfig) -> TopicModelingConfig:
    """
    Copy a configuration with every random seed unlocked.

    Args:
        config: The configuration to copy.

    Returns:
        A deep copy without a random seed.
    """
    unseeded = copy.deepcopy(config)
    unseeded.random_state = None
    # The dimensionality-reduction seed takes precedence over random_state, so
    # a locked UMAP seed would otherwise make every run identical to the first.
    unseeded.dim_params.pop("random_state", None)
    return unseeded


def _run_lda_pipeline(
    df: pd.DataFrame,
    raw_texts: list[str],
    config: TopicModelingConfig,
    include_dashboards: bool,
) -> TopicModelingRunResult:
    """
    Execute the LDA topic modeling pipeline.

    Args:
        df: The prepared input DataFrame.
        raw_texts: The documents, in DataFrame order.
        config: The topic modeling configuration.
        include_dashboards: Whether to generate the pyLDAvis dashboard.

    Returns:
        The run result, including the model and corpus for perplexity.
    """
    processed_texts = preprocess_texts_for_lda(
        raw_texts,
        config.language,
        config.custom_stopwords,
        config.use_bigrams,
    )
    lda_model, corpus, id2word = train_lda_model(
        processed_texts,
        config.num_topics,
        config.passes,
        random_state=config.random_state,
    )

    topics = extract_lda_topics(lda_model, config.num_topics)
    dashboard_assets: dict[str, str] = {}
    if include_dashboards:
        dashboard_assets["lda_dashboard.html"] = generate_lda_html(
            lda_model, corpus, id2word
        )

    return {
        "topic_df": build_topic_table(topics, with_counts=False),
        "docs_df": generate_lda_document_topics_df(lda_model, corpus, df),
        "dashboard_assets": dashboard_assets,
        "topic_keywords": [topic.keywords for topic in topics],
        "lda_model": lda_model,
        "corpus": corpus,
        "tokenized_texts": processed_texts,
    }


def _run_top2vec_pipeline(
    df: pd.DataFrame,
    raw_texts: list[str],
    config: TopicModelingConfig,
    include_dashboards: bool,
) -> TopicModelingRunResult:
    """
    Execute the Top2Vec topic modeling pipeline.

    Args:
        df: The prepared input DataFrame.
        raw_texts: The documents, in DataFrame order.
        config: The topic modeling configuration.
        include_dashboards: Whether to generate the keyword bar chart.

    Returns:
        The run result.
    """
    topic_model = train_top2vec_model(
        texts=raw_texts,
        language=config.language,
        embedding_backend=config.top2vec_backend,
        speed=config.top2vec_speed,
        min_count=config.top2vec_min_count,
        target_topics=config.num_topics,
    )

    topics = extract_top2vec_topics(topic_model)
    dashboard_assets: dict[str, str] = {}
    if include_dashboards:
        dashboard_assets["top2vec_barchart.html"] = generate_top2vec_barchart_html(
            topic_model
        )

    return {
        "topic_df": build_topic_table(topics),
        "docs_df": generate_top2vec_document_topics_df(topic_model, df),
        "dashboard_assets": dashboard_assets,
        "topic_keywords": [topic.keywords for topic in topics],
    }


def _run_bertopic_pipeline(
    df: pd.DataFrame,
    raw_texts: list[str],
    config: TopicModelingConfig,
    timestamps: list[Any] | None = None,
    embedding_model: Any = None,
    precomputed_embeddings: np.ndarray | None = None,
    include_dashboards: bool = True,
) -> TopicModelingRunResult:
    """
    Execute the BERTopic topic modeling pipeline.

    Args:
        df: The prepared input DataFrame.
        raw_texts: The documents, in DataFrame order.
        config: The topic modeling configuration.
        timestamps: Optional parsed timestamps for topics over time.
        embedding_model: Optional SentenceTransformer instance.
        precomputed_embeddings: Optional precomputed document embeddings.
        include_dashboards: Whether to generate the HTML charts.

    Returns:
        The run result.
    """
    stop_set = get_stopword_set(config.language, config.custom_stopwords)

    topic_model, topic_ids, probabilities = train_bertopic_model(
        texts=raw_texts,
        language=config.language,
        num_topics=config.num_topics,
        stop_words_set=stop_set,
        dim_reduction_algo=config.dim_reduction_algo,
        dim_params=config.dim_params,
        clustering_algo=config.clustering_algo,
        clustering_params=config.clustering_params,
        ngram_range=config.ngram_range,
        min_df=config.min_df,
        reduce_outliers=config.reduce_outliers,
        reduce_frequent_words=config.reduce_frequent,
        random_state=config.random_state,
        embedding_model=embedding_model,
        precomputed_embeddings=precomputed_embeddings,
    )

    topics = extract_bertopic_topics(topic_model)
    dashboard_assets: dict[str, str] = {}
    if include_dashboards:
        visualizations = generate_bertopic_visualizations(topic_model)
        dashboard_assets = {
            "intertopic_distance.html": visualizations.get("distance_map", ""),
            "topic_barchart.html": visualizations.get("barchart", ""),
            "similarity_heatmap.html": visualizations.get("heatmap", ""),
        }
        if config.enable_dtm and timestamps is not None:
            dashboard_assets["topics_over_time.html"] = generate_topics_over_time_html(
                topic_model,
                raw_texts,
                timestamps,
                nr_bins=resolve_time_bins(timestamps, config.time_bins),
            )

    return {
        "topic_df": build_topic_table(topics),
        "docs_df": generate_bertopic_document_topics_df(topic_ids, probabilities, df),
        "dashboard_assets": dashboard_assets,
        "topic_keywords": [topic.keywords for topic in topics],
    }
