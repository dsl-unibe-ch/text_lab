"""Top2Vec engine: training, topic extraction and the keyword bar chart."""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from top2vec import Top2Vec

from textlab.features.topic_modeling import small_corpus
from textlab.features.topic_modeling.models import TopicKeywords

_TOP_N_KEYWORDS = 10


def _is_reduced_model(topic_model: Top2Vec) -> bool:
    """Determine whether hierarchical topic reduction was applied to the model.

    Reduction is only performed when fewer topics are requested than Top2Vec
    found. Asking Top2Vec for reduced topics when no reduction took place
    raises an error, so the presence of the reduction hierarchy is checked
    directly.

    Args:
        topic_model: A trained Top2Vec model.

    Returns:
        True if the reduced topics should be used, otherwise False.
    """
    return getattr(topic_model, "hierarchy", None) is not None


def train_top2vec_model(
    texts: list[str],
    language: str,
    embedding_backend: str = "doc2vec",
    speed: str = "learn",
    min_count: int = 10,
    target_topics: int | str = "auto",
) -> Top2Vec:
    """Train a Top2Vec model on the provided texts.

    Args:
        texts: The input documents to model.
        language: The language of the documents.
        embedding_backend: The embedding backend to use.
        speed: The Top2Vec training speed mode.
        min_count: Words that appear fewer times than this in the whole
            collection are ignored.
        target_topics: The target number of topics, or "auto".

    Returns:
        A trained Top2Vec model.

    Raises:
        ValueError: If no texts are provided, or if the corpus is too small for
            Top2Vec to model.
    """
    if not texts:
        raise ValueError("No texts were provided to Top2Vec.")

    if embedding_backend == "doc2vec":
        embed_model = "doc2vec"
    else:
        embed_model = (
            "all-MiniLM-L6-v2"
            if language == "English"
            else "paraphrase-multilingual-MiniLM-L12-v2"
        )

    # Use only the CPU cores allocated to the Slurm job, not every core of
    # the node.
    if hasattr(os, "sched_getaffinity"):
        workers = len(os.sched_getaffinity(0))
    else:
        workers = os.cpu_count() or 1

    try:
        topic_model = Top2Vec(
            documents=texts,
            speed=speed,
            min_count=min_count,
            embedding_model=embed_model,
            workers=workers,
        )
    except (ValueError, TypeError) as e:
        if small_corpus.is_corpus_too_small(e):
            raise small_corpus.too_small_error(len(texts), "Top2Vec") from e
        raise

    if target_topics != "auto" and isinstance(target_topics, int):
        num_found = topic_model.get_num_topics()
        if target_topics < num_found:
            topic_model.hierarchical_topic_reduction(num_topics=target_topics)

    return topic_model


def extract_top2vec_topics(topic_model: Top2Vec) -> list[TopicKeywords]:
    """Extract the keywords and sizes of every Top2Vec topic.

    Reduced topics are used when hierarchical topic reduction was applied.

    Args:
        topic_model: A trained Top2Vec model.

    Returns:
        One entry per topic, numbered from 1.
    """
    is_reduced = _is_reduced_model(topic_model)

    topic_words, _, topic_nums = topic_model.get_topics(reduced=is_reduced)
    topic_sizes, size_topic_nums = topic_model.get_topic_sizes(
        reduced=is_reduced
    )
    size_map = {
        int(topic_num): int(size)
        for size, topic_num in zip(topic_sizes, size_topic_nums, strict=False)
    }

    return [
        TopicKeywords(
            topic=int(t_num) + 1,
            keywords=[
                str(word) for word in topic_words[idx][:_TOP_N_KEYWORDS]
            ],
            count=size_map.get(int(t_num), 0),
        )
        for idx, t_num in enumerate(topic_nums)
    ]


def generate_top2vec_document_topics_df(
    topic_model: Top2Vec,
    original_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate a DataFrame with the dominant Top2Vec topic for each document.

    Reduced topics are used when hierarchical topic reduction was applied.

    Args:
        topic_model: A trained Top2Vec model.
        original_df: The original DataFrame containing the source documents.

    Returns:
        A copy of the original DataFrame with two additional columns:
            - "Dominant_Topic"
            - "Topic_Confidence"

        These two columns are placed at the beginning of the returned
        DataFrame.

    Raises:
        ValueError: If the number of inferred topic assignments does not
            match the number of rows in the input DataFrame.
    """
    is_reduced = _is_reduced_model(topic_model)

    doc_topics, doc_dist, _, _ = topic_model.get_documents_topics(
        doc_ids=list(range(len(original_df))),
        reduced=is_reduced,
        num_topics=1,
    )

    doc_topics_flat = np.array(doc_topics).reshape(-1)
    doc_dist_flat = np.array(doc_dist).reshape(-1)

    if len(doc_topics_flat) != len(original_df):
        raise ValueError(
            f"Top2Vec output length ({len(doc_topics_flat)}) does not match "
            f"dataframe length ({len(original_df)})."
        )

    result_df = original_df.copy()
    result_df["Dominant_Topic"] = [int(t) + 1 for t in doc_topics_flat]
    result_df["Topic_Confidence"] = [round(float(d), 4) for d in doc_dist_flat]

    cols = result_df.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    return result_df[cols]


def generate_top2vec_barchart_html(topic_model: Top2Vec) -> str:
    """Generate an HTML bar chart visualization for Top2Vec topics.

    Reduced topics are used when hierarchical topic reduction was applied.

    Args:
        topic_model: A trained Top2Vec model.

    Returns:
        An HTML string containing the Top2Vec bar chart visualization.
    """
    is_reduced = _is_reduced_model(topic_model)

    num_topics = topic_model.get_num_topics(reduced=is_reduced)
    display_topics = min(num_topics, 12)

    topic_words, word_scores, topic_nums = topic_model.get_topics(
        num_topics=display_topics,
        reduced=is_reduced,
    )

    cols = 4
    rows = (display_topics + cols - 1) // cols

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"Topic {t_num + 1}" for t_num in topic_nums],
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
    )

    for i in range(display_topics):
        row = (i // cols) + 1
        col = (i % cols) + 1

        words = topic_words[i][:8][::-1]
        scores = word_scores[i][:8][::-1]

        fig.add_trace(
            go.Bar(
                x=scores,
                y=words,
                orientation="h",
                marker_color="#4A90E2",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        fig.update_xaxes(showticklabels=False, row=row, col=col)
        fig.update_yaxes(tickfont=dict(size=11), row=row, col=col)

    fig.update_layout(
        height=300 * rows,
        title_text="Top2Vec Word Similarity Scores",
        template="plotly_white",
        margin=dict(t=50, b=20, l=20, r=20),
    )

    return fig.to_html(full_html=False, include_plotlyjs="cdn")
