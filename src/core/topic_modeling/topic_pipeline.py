from typing import Any

import pandas as pd

from .bertopic_engine import (
    generate_bertopic_document_topics_df,
    generate_bertopic_keywords_df,
    generate_bertopic_visualizations,
    generate_topics_over_time_html,
    train_bertopic_model,
)
from .lda_engine import (
    generate_lda_document_topics_df,
    generate_lda_html,
    generate_lda_keywords_df,
    train_lda_model,
)
from .top2vec_engine import (
    generate_top2vec_barchart_html,
    generate_top2vec_document_topics_df,
    generate_top2vec_keywords_df,
    train_top2vec_model,
)
from .topic_config import TopicModelingConfig, TopicModelingRunResult
from .topic_utils import (
    get_stopword_set,
    preprocess_texts_for_lda,
    validate_minimum_documents,
)


def run_topic_modeling_pipeline(
    df: pd.DataFrame,
    config: TopicModelingConfig,
    timestamps: list[Any] | None = None,
) -> TopicModelingRunResult:
    """
    Execute the selected topic modeling pipeline.

    Args:
        df: The prepared input DataFrame.
        config: The topic modeling configuration.
        timestamps: Optional parsed timestamps used for BERTopic
            topics-over-time analysis.

    Returns:
        A dictionary containing the generated topic table, document table,
        and dashboard assets. (Also returns raw models for LDA evaluation).
    """
    raw_texts = df[config.text_column].astype(str).tolist()
    validate_minimum_documents(raw_texts)

    if "LDA" in config.algorithm:
        return _run_lda_pipeline(df, raw_texts, config)

    if "Top2Vec" in config.algorithm:
        return _run_top2vec_pipeline(df, raw_texts, config)

    return _run_bertopic_pipeline(df, raw_texts, config, timestamps=timestamps)


def _run_lda_pipeline(
    df: pd.DataFrame,
    raw_texts: list[str],
    config: TopicModelingConfig,
) -> TopicModelingRunResult:
    """
    Execute the LDA topic modeling pipeline.
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
        random_state=config.random_state
    )

    topic_df = generate_lda_keywords_df(lda_model, config.num_topics)
    docs_df = generate_lda_document_topics_df(lda_model, corpus, df)
    html_string = generate_lda_html(lda_model, corpus, id2word)

    return {
        "topic_df": topic_df,
        "docs_df": docs_df,
        "dashboard_assets": {
            "lda_dashboard.html": html_string,
        },
        "lda_model": lda_model,  # Added for Perplexity Calculation
        "corpus": corpus,        # Added for Perplexity Calculation
    }


def _run_top2vec_pipeline(
    df: pd.DataFrame,
    raw_texts: list[str],
    config: TopicModelingConfig,
) -> TopicModelingRunResult:
    """
    Execute the Top2Vec topic modeling pipeline.
    """
    topic_model = train_top2vec_model(
        texts=raw_texts,
        language=config.language,
        embedding_backend=config.top2vec_backend,
        speed=config.top2vec_speed,
        target_topics=config.num_topics,
    )

    topic_df = generate_top2vec_keywords_df(topic_model, config.num_topics)
    docs_df = generate_top2vec_document_topics_df(topic_model, df, config.num_topics)
    html_string = generate_top2vec_barchart_html(topic_model, config.num_topics)

    return {
        "topic_df": topic_df,
        "docs_df": docs_df,
        "dashboard_assets": {
            "top2vec_barchart.html": html_string,
        },
    }


def _run_bertopic_pipeline(
    df: pd.DataFrame,
    raw_texts: list[str],
    config: TopicModelingConfig,
    timestamps: list[Any] | None = None,
) -> TopicModelingRunResult:
    """
    Execute the BERTopic topic modeling pipeline.
    """
    stop_set = get_stopword_set(config.language, config.custom_stopwords)

    topic_model, topics, probabilities = train_bertopic_model(
        texts=raw_texts,
        language=config.language,
        num_topics=config.bertopic_nr_topics,
        stop_words_set=stop_set,
        dim_reduction_algo=config.dim_reduction_algo,
        dim_params=config.dim_params,
        clustering_algo=config.clustering_algo,
        clustering_params=config.clustering_params,
        ngram_range=config.ngram_range,
        min_df=config.min_df,
        reduce_outliers=config.reduce_outliers,
        reduce_frequent_words=config.reduce_frequent,
        random_state=config.random_state
    )

    topic_df = generate_bertopic_keywords_df(topic_model)
    docs_df = generate_bertopic_document_topics_df(topics, probabilities, df)

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
        )

    return {
        "topic_df": topic_df,
        "docs_df": docs_df,
        "dashboard_assets": dashboard_assets,
    }