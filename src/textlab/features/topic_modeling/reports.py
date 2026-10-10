"""The topic table, the run report and the result ZIP."""

from __future__ import annotations

import datetime
import io
import zipfile

import pandas as pd

from textlab.features.topic_modeling.models import (
    TOP2VEC_BACKEND_LABELS,
    Algorithm,
    TopicKeywords,
    TopicModelingConfig,
)


def build_topic_table(
    topics: list[TopicKeywords],
    with_counts: bool = True,
) -> pd.DataFrame:
    """Build the topic table shown on the page and exported as CSV.

    Args:
        topics: The topics, in display order.
        with_counts: Whether to include the ``Count`` column.

    Returns:
        A DataFrame with the columns ``Topic``, ``Count`` (optional) and
        ``Keywords`` (comma-separated).
    """
    columns = (
        ["Topic", "Count", "Keywords"]
        if with_counts
        else ["Topic", "Keywords"]
    )
    rows = [
        {
            "Topic": topic.topic,
            "Count": topic.count,
            "Keywords": ", ".join(topic.keywords),
        }
        for topic in topics
    ]
    return pd.DataFrame(rows, columns=columns)


def _report_header(filename: str, config: TopicModelingConfig) -> list[str]:
    """Build the source and core-settings part of the metadata report.

    Args:
        filename: The source file name.
        config: The topic modeling configuration.

    Returns:
        The report lines.
    """
    lines = [
        "=========================================",
        " TEXT LAB - TOPIC MODELING CONFIGURATION ",
        "=========================================",
        f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Source File: {filename}",
        f"Target Text Column: {config.text_column}",
    ]

    if config.enable_dtm and config.date_column:
        lines.append(f"Timestamp Column (DTM): {config.date_column}")
        lines.append(f"Time Bins (DTM): {config.time_bins}")

    stopword_text = (
        config.custom_stopwords if config.custom_stopwords.strip() else "None"
    )
    lines.extend(
        [
            "",
            "--- CORE SETTINGS ---",
            f"Framework: {config.algorithm.label}",
            f"Primary Language: {config.language}",
            f"Custom Stopwords: {stopword_text}",
            "",
        ]
    )
    return lines


def _report_bertopic_parameters(
    config: TopicModelingConfig,
    embedding_model_name: str,
) -> list[str]:
    """Build the BERTopic-specific part of the metadata report.

    Args:
        config: The topic modeling configuration.
        embedding_model_name: The resolved embedding model name.

    Returns:
        The report lines.
    """
    lines = [
        "--- BERTOPIC PARAMETERS ---",
        f"Target Topics: {config.num_topics}",
        f"Embedding Model: {embedding_model_name}",
        f"Trust Remote Code: {config.trust_remote_code}",
        f"Chunk Long Documents: {config.chunk_long_documents}",
        f"N-Gram Range: {config.ngram_range}",
        f"Min Topic Frequency (min_df): {config.min_df}",
        "Reduce Frequent Words (ClassTfidfTransformer): "
        f"{config.reduce_frequent}",
        "",
        f"--- DIMENSIONALITY REDUCTION ({config.dim_reduction_algo}) ---",
    ]

    if config.dim_reduction_algo != "None":
        lines.append(f"N Components: {config.dim_params.get('n_components')}")
        if config.dim_reduction_algo == "UMAP":
            lines.append(
                f"N Neighbors: {config.dim_params.get('n_neighbors')}"
            )
            lines.append(f"Min Distance: {config.dim_params.get('min_dist')}")
        lines.append(
            f"Random State: {config.dim_params.get('random_state', 'None')}"
        )

    lines.extend(["", f"--- CLUSTERING ({config.clustering_algo}) ---"])

    params = config.clustering_params
    if config.clustering_algo == "KMeans":
        lines.append(f"N Clusters: {params.get('n_clusters')}")
    else:
        min_samples = params.get(
            "min_samples", "Default (equals min_cluster_size)"
        )
        lines.extend(
            [
                f"Min Cluster Size: {params.get('min_cluster_size')}",
                f"Min Samples: {min_samples}",
                f"Force-reduce Outliers: {config.reduce_outliers}",
            ]
        )
    return lines


def generate_metadata_report(
    filename: str,
    config: TopicModelingConfig,
    embedding_model_name: str,
    evaluation_metrics: dict[str, float | None] | None = None,
) -> str:
    """Compile a formatted metadata report for reproducibility.

    Args:
        filename: The source file name.
        config: The topic modeling configuration.
        embedding_model_name: The resolved embedding model name.
        evaluation_metrics: Metric values by name, if any were computed.

    Returns:
        A formatted metadata report string.
    """
    report = _report_header(filename, config)

    if config.algorithm == Algorithm.LDA:
        report.extend(
            [
                "--- LDA PARAMETERS ---",
                f"Number of Topics: {config.num_topics}",
                f"Training Passes: {config.passes}",
                f"Extract Bigrams: {config.use_bigrams}",
            ]
        )
    elif config.algorithm == Algorithm.TOP2VEC:
        backend_label = TOP2VEC_BACKEND_LABELS.get(
            config.top2vec_backend, config.top2vec_backend
        )
        report.extend(
            [
                "--- TOP2VEC PARAMETERS ---",
                f"Target Topics: {config.num_topics}",
                f"Embedding Backend: {backend_label}",
                f"Embedding Model: {embedding_model_name}",
                f"Training Speed: {config.top2vec_speed}",
                f"Minimum Word Count (min_count): {config.top2vec_min_count}",
            ]
        )
    else:
        report.extend(
            _report_bertopic_parameters(config, embedding_model_name)
        )

    if evaluation_metrics:
        report.extend(
            [
                "",
                "=========================================",
                " MODEL EVALUATION METRICS                ",
                "=========================================",
            ]
        )
        for metric_name, score in evaluation_metrics.items():
            report.append(
                f"{metric_name}: {'N/A' if score is None else score}"
            )

    return "\n".join(report)


def build_results_zip(
    metadata_report: str,
    docs_df: pd.DataFrame,
    topic_df: pd.DataFrame,
    dashboard_assets: dict[str, str],
) -> bytes:
    """Build a ZIP archive containing modeling outputs.

    Args:
        metadata_report: The run configuration report.
        docs_df: The document-level topic assignments.
        topic_df: The topic keywords table.
        dashboard_assets: HTML dashboard artifacts.

    Returns:
        ZIP archive content as bytes.
    """
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(
        zip_buffer, "w", compression=zipfile.ZIP_DEFLATED
    ) as zf:
        zf.writestr("run_configuration.txt", metadata_report.encode("utf-8"))
        zf.writestr(
            "document_topics.csv",
            docs_df.to_csv(index=False).encode("utf-8-sig"),
        )
        zf.writestr(
            "topic_keywords.csv",
            topic_df.to_csv(index=False).encode("utf-8-sig"),
        )

        for filename, html_data in dashboard_assets.items():
            if html_data:
                zf.writestr(filename, html_data.encode("utf-8"))

    return zip_buffer.getvalue()
