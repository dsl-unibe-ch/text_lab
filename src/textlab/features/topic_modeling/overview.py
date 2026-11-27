"""What a run found, ready to draw: topic sizes, examples, map, timeline.

The overview is built the same way for every algorithm, from what each
engine gives: the topics with their keyword weights, every document's main
topic (the document table), and one vector per topic (BERTopic's topic
embeddings, LDA's topic-word distributions, Top2Vec's topic vectors).
Topics whose vectors point the same way are alike; the map places them
with classical multidimensional scaling, as pyLDAvis does.

Runs in the worker; it needs only NumPy and pandas.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from textlab.features.topic_modeling.data import resolve_time_bins
from textlab.features.topic_modeling.models import (
    TopicKeywords,
    TopicOverview,
    TopicSummary,
)

#: The document table's columns of the main topic and its confidence.
TOPIC_COLUMN = "Dominant_Topic"
CONFIDENCE_COLUMN = "Topic_Confidence"

#: Example documents shown per topic, and their length in characters.
EXAMPLES_PER_TOPIC = 3
EXAMPLE_CHARACTERS = 280


def build_overview(
    topics: Sequence[TopicKeywords],
    docs_df: pd.DataFrame,
    text_column: str,
    *,
    topic_vectors: Any = None,
    timestamps: Sequence[Any] | None = None,
    time_bins: int | None = None,
) -> TopicOverview:
    """Return what a run found, ready to draw.

    Args:
        topics: The topics, in the topic table's order, with weights.
        docs_df: The document table: the analyzed rows with their main
            topic (``Dominant_Topic``) and, if known, its confidence.
        text_column: The column of the documents' text.
        topic_vectors: One vector per topic, in the order of ``topics``,
            as an array or a sparse matrix; ``None`` for no map.
        timestamps: Each row's timestamp, for the timeline.
        time_bins: How many intervals the timeline has at most.

    Returns:
        The overview.
    """
    assigned = docs_df[TOPIC_COLUMN] if TOPIC_COLUMN in docs_df else None
    summaries = []
    for topic in topics:
        mask = (
            assigned == topic.topic
            if assigned is not None
            else pd.Series(False, index=docs_df.index)
        )
        summaries.append(
            TopicSummary(
                topic=topic.topic,
                keywords=list(topic.keywords),
                weights=scaled_weights(topic.weights),
                documents=int(mask.sum()),
                examples=examples(docs_df.loc[mask], text_column),
            )
        )
    in_topics = sum(summary.documents for summary in summaries)

    similarity = positions = None
    if topic_vectors is not None and len(topics) >= 2:
        matrix = similarity_matrix(topic_vectors)
        if matrix.shape == (len(topics), len(topics)):
            similarity = np.round(matrix, 3).tolist()
            positions = np.round(map_positions(matrix), 4).tolist()

    labels: list[str] = []
    counts: list[list[int]] = []
    if timestamps is not None and assigned is not None and len(timestamps):
        labels, counts = topics_over_time(
            [topic.topic for topic in topics],
            assigned.tolist(),
            timestamps,
            time_bins,
        )

    return TopicOverview(
        documents=len(docs_df),
        outliers=len(docs_df) - in_topics,
        topics=summaries,
        similarity=similarity,
        positions=positions,
        time_labels=labels,
        over_time=counts,
    )


def scaled_weights(weights: Sequence[float]) -> list[float]:
    """Return keyword weights scaled so that the largest is 1.

    Args:
        weights: The engine's weights; any scale, all of one sign.

    Returns:
        The weights between 0 and 1, rounded; empty if there are none or
        they are all zero.
    """
    values = np.abs(np.asarray(list(weights), dtype=float))
    if not values.size or not values.max():
        return []
    return np.round(values / values.max(), 3).tolist()


def examples(rows: pd.DataFrame, text_column: str) -> list[str]:
    """Return the beginning of a topic's most typical documents.

    Args:
        rows: The documents whose main topic it is.
        text_column: The column of their text.

    Returns:
        Up to :data:`EXAMPLES_PER_TOPIC` excerpts, the most confident
        first when the table says how confident the engine is.
    """
    if text_column not in rows or rows.empty:
        return []
    if CONFIDENCE_COLUMN in rows and rows[CONFIDENCE_COLUMN].notna().any():
        rows = rows.sort_values(
            CONFIDENCE_COLUMN, ascending=False, kind="stable"
        )
    found = []
    for value in rows[text_column]:
        text = excerpt(str(value))
        if text:
            found.append(text)
        if len(found) == EXAMPLES_PER_TOPIC:
            break
    return found


def excerpt(text: str, limit: int = EXAMPLE_CHARACTERS) -> str:
    """Return the beginning of a text, cut at a word, on one line.

    Args:
        text: The text.
        limit: Its longest length, in characters.

    Returns:
        The text with runs of white space made one space; cut at the last
        word that fits, with an ellipsis, if it is longer.
    """
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    cut = flat[:limit].rsplit(" ", 1)[0] or flat[:limit]
    return cut.rstrip(" ,.;:") + "…"


def similarity_matrix(vectors: Any) -> np.ndarray:
    """Return how alike each two topic vectors are (cosine similarity).

    Args:
        vectors: One vector per topic, as an array or a sparse matrix.

    Returns:
        A square matrix between 0 and 1 (negative similarities count as
        0), with 1 on the diagonal.
    """
    if hasattr(vectors, "toarray"):
        vectors = vectors.toarray()
    matrix = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = matrix / norms
    similarity = np.clip(unit @ unit.T, 0.0, 1.0)
    np.fill_diagonal(similarity, 1.0)
    return similarity


def map_positions(similarity: np.ndarray) -> np.ndarray:
    """Place topics on a map where alike topics lie close.

    Classical multidimensional scaling of the distances ``1 -
    similarity``, kept to two dimensions and scaled into the unit square
    (both axes by the same factor, centred).

    Args:
        similarity: A square similarity matrix.

    Returns:
        One ``[x, y]`` per topic, between 0 and 1.
    """
    count = len(similarity)
    distance = np.clip(1.0 - np.asarray(similarity, dtype=float), 0.0, None)
    np.fill_diagonal(distance, 0.0)
    centring = np.eye(count) - np.full((count, count), 1.0 / count)
    gram = -0.5 * centring @ (distance**2) @ centring
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1][:2]
    coords = vectors[:, order] * np.sqrt(np.clip(values[order], 0.0, None))
    if coords.shape[1] < 2:
        coords = np.hstack([coords, np.zeros((count, 2 - coords.shape[1]))])
    coords = coords - coords.min(axis=0)
    span = coords.max()
    if span > 0:
        coords = coords / span
    return coords + (1.0 - coords.max(axis=0)) / 2


def topics_over_time(
    topic_ids: Sequence[int | str],
    assigned: Sequence[Any],
    timestamps: Sequence[Any],
    requested_bins: int | None,
) -> tuple[list[str], list[list[int]]]:
    """Count each topic's documents per interval of time.

    Args:
        topic_ids: The topics, in order.
        assigned: Each document's main topic.
        timestamps: Each document's timestamp, in the same order.
        requested_bins: How many equal intervals to use at most; with
            fewer distinct timestamps (or ``None``), each timestamp is its
            own interval.

    Returns:
        The intervals' labels (their start), and per topic the count of
        its documents in each interval.
    """
    stamps = pd.to_datetime(pd.Series(list(timestamps)))
    nanos = stamps.astype("int64").to_numpy()
    bins = (
        resolve_time_bins(list(stamps), requested_bins)
        if requested_bins
        else None
    )
    if bins is None or nanos.min() == nanos.max():
        starts = np.unique(nanos)
        index = np.searchsorted(starts, nanos)
        step = None
    else:
        edges = np.linspace(nanos.min(), nanos.max(), bins + 1)
        starts = edges[:-1]
        index = np.clip(
            np.searchsorted(edges, nanos, side="right") - 1, 0, bins - 1
        )
        step = pd.Timedelta(int(edges[1] - edges[0]), unit="ns")
    labels = _time_labels(
        pd.to_datetime(starts.astype("int64")),
        yearly=bool(
            ((stamps.dt.month == 1) & (stamps.dt.day == 1)).all()
            and (stamps == stamps.dt.normalize()).all()
        ),
        step=step,
    )
    position = {topic: row for row, topic in enumerate(topic_ids)}
    counts = np.zeros((len(topic_ids), len(starts)), dtype=int)
    for topic, column in zip(assigned, index, strict=True):
        row = position.get(topic)
        if row is not None:
            counts[row, column] += 1
    return labels, counts.tolist()


def _time_labels(
    starts: pd.DatetimeIndex, *, yearly: bool, step: pd.Timedelta | None
) -> list[str]:
    """Return intervals' starts as briefly as the data is precise.

    Args:
        starts: The intervals' starts.
        yearly: Whether the timestamps are whole years.
        step: The intervals' length, or ``None`` when each timestamp is
            its own interval.

    Returns:
        Years for years, dates for intervals of a day or more and for
        days, else dates with the time.
    """
    if yearly:
        return [str(stamp.year) for stamp in starts]
    daily = step is not None and step >= pd.Timedelta(days=1)
    if daily or all(stamp == stamp.normalize() for stamp in starts):
        return [stamp.strftime("%Y-%m-%d") for stamp in starts]
    return [stamp.strftime("%Y-%m-%d %H:%M") for stamp in starts]
