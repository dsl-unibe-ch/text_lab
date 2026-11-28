"""What a run found, ready to draw: the views each algorithm's library uses.

Every run gets the parts all algorithms share, built the same way from
what each engine gives: the topics with their keyword weights, every
document's main topic (the document table), and one vector per topic
(BERTopic's topic embeddings, LDA's topic-word distributions, Top2Vec's
topic vectors). From these come each topic's size and most typical
documents, how alike each two topics are (cosine similarity), a map where
alike topics lie close (classical multidimensional scaling, as pyLDAvis
does), how the topics merge (a dendrogram), and, with timestamps, each
topic's documents per interval.

Each algorithm adds what its own library draws:

- **LDA**, pyLDAvis's view (:func:`ldavis_parts`): the topics placed and
  sized as pyLDAvis places them (by their share of the words), the terms
  it ranks by relevance for each topic, and how each term spreads over the
  topics; and, as LDA mixes topics in every document, each example's
  mixture.
- **BERTopic**, its document map (documents placed by their embeddings,
  as ``visualize_documents`` does) and its topics over time, with each
  topic's words in each interval.
- **Top2Vec**, its document map and its word clouds.

Runs in the worker; it needs NumPy and pandas, and SciPy and UMAP (both in
the modeling environment) for the dendrogram and the document map.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from textlab.features.topic_modeling.data import resolve_time_bins
from textlab.features.topic_modeling.models import (
    DocumentMap,
    Hierarchy,
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

#: Documents drawn on the document map at most, and their excerpts' length.
MAP_DOCUMENTS = 1500
MAP_EXCERPT_CHARACTERS = 80

#: Words in a Top2Vec word cloud.
CLOUD_WORDS = 30

#: The smallest share of a document a topic must have to show in its
#: mixture.
MIXTURE_FLOOR = 0.05

#: Reduces document vectors to two dimensions: rows in, ``(n, 2)`` out.
Reducer = Callable[[np.ndarray], np.ndarray]


def build_overview(
    topics: Sequence[TopicKeywords],
    docs_df: pd.DataFrame,
    text_column: str,
    *,
    topic_vectors: Any = None,
    timestamps: Sequence[Any] | None = None,
    time_bins: int | None = None,
    time_rows: Sequence[Sequence[Any]] | None = None,
    document_vectors: Any = None,
    reducer: Reducer | None = None,
    ldavis: Any = None,
    document_topics: Any = None,
    topic_words: Sequence[Sequence[Sequence[Any]]] | None = None,
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
        time_rows: BERTopic's topics over time, as ``[topic, timestamp in
            nanoseconds, words]`` rows (:func:`.bertopic_engine.
            topics_over_time_rows`), for each interval's words.
        document_vectors: One vector per row, for the document map.
        reducer: Places document vectors in two dimensions; UMAP by
            default.
        ldavis: pyLDAvis's prepared data, with topics in model order.
        document_topics: LDA: each row's topic mixture, one column per
            topic in the order of ``topics``.
        topic_words: Top2Vec: per topic, its words with their score, the
            most typical first, as ``[word, score]``.

    Returns:
        The overview.
    """
    count = len(docs_df)
    texts = (
        docs_df[text_column].astype(str).tolist()
        if text_column in docs_df
        else [""] * count
    )
    position = {topic.topic: index for index, topic in enumerate(topics)}
    assigned = (
        docs_df[TOPIC_COLUMN].tolist()
        if TOPIC_COLUMN in docs_df
        else [None] * count
    )
    topic_of = np.array(
        [position.get(value, -1) for value in assigned], dtype=int
    )
    confidence = (
        pd.to_numeric(docs_df[CONFIDENCE_COLUMN], errors="coerce").to_numpy()
        if CONFIDENCE_COLUMN in docs_df
        else None
    )
    mixtures_of = (
        np.asarray(document_topics, dtype=float)
        if document_topics is not None
        else None
    )

    summaries = []
    for index, topic in enumerate(topics):
        members = np.flatnonzero(topic_of == index)
        rows = example_rows(members, confidence, texts)
        summaries.append(
            TopicSummary(
                topic=topic.topic,
                keywords=list(topic.keywords),
                weights=scaled_weights(topic.weights),
                documents=len(members),
                examples=[excerpt(texts[row]) for row in rows],
                prevalence=round(len(members) / count, 4) if count else 0.0,
                mixtures=(
                    [mixture(mixtures_of[row]) for row in rows]
                    if mixtures_of is not None
                    else []
                ),
                cloud=(
                    cloud(topic_words[index])
                    if topic_words is not None and index < len(topic_words)
                    else []
                ),
            )
        )
    in_topics = sum(summary.documents for summary in summaries)

    similarity = positions = None
    hierarchy = None
    if topic_vectors is not None and len(topics) >= 2:
        matrix = similarity_matrix(topic_vectors)
        if matrix.shape == (len(topics), len(topics)):
            similarity = np.round(matrix, 3).tolist()
            positions = np.round(map_positions(matrix), 4).tolist()
            hierarchy = topic_hierarchy(matrix)

    overview = TopicOverview(
        documents=count,
        outliers=count - in_topics,
        topics=summaries,
        similarity=similarity,
        positions=positions,
        hierarchy=hierarchy,
    )

    if timestamps is not None and len(timestamps):
        labels, index, locate = intervals(timestamps, time_bins)
        overview.time_labels = labels
        overview.over_time = interval_counts(
            topic_of, index, len(topics), len(labels)
        )
        if time_rows:
            for row, words in enumerate(
                interval_words(
                    time_rows, position, locate, len(topics), len(labels)
                )
            ):
                summaries[row].time_words = words

    if document_vectors is not None and count:
        overview.document_map = document_map(
            document_vectors, topic_of, texts, reducer=reducer
        )

    if ldavis is not None:
        parts = ldavis_parts(ldavis, len(topics))
        overview.positions = parts["positions"]
        overview.origin = parts["origin"]
        overview.salient = parts["salient"]
        overview.term_topics = parts["term_topics"]
        for summary, share, terms in zip(
            summaries, parts["prevalence"], parts["terms"], strict=False
        ):
            summary.prevalence = share
            summary.terms = terms
    return overview


# ---------------------------------------------------------------------------
# What every algorithm gets
# ---------------------------------------------------------------------------


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


def example_rows(
    members: np.ndarray, confidence: np.ndarray | None, texts: Sequence[str]
) -> list[int]:
    """Return the rows of a topic's most typical documents.

    Args:
        members: The rows whose main topic it is.
        confidence: Each row's confidence in its main topic, or ``None``.
        texts: Each row's text.

    Returns:
        Up to :data:`EXAMPLES_PER_TOPIC` rows with text, the most
        confident first when the confidence is known.
    """
    if confidence is not None and len(members):
        values = confidence[members]
        if not np.isnan(values).all():
            order = np.argsort(
                -np.nan_to_num(values, nan=-np.inf), kind="stable"
            )
            members = members[order]
    rows = []
    for row in members:
        if excerpt(texts[row]):
            rows.append(int(row))
        if len(rows) == EXAMPLES_PER_TOPIC:
            break
    return rows


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
    (:func:`unit_square`).

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
    return unit_square(coords)


def unit_square(
    coords: np.ndarray, *, extra: np.ndarray | None = None
) -> np.ndarray:
    """Scale points into the unit square, both axes alike, centred.

    Args:
        coords: The points, ``(n, 2)``.
        extra: More points, such as an origin, that are scaled the same
            way but do not set the scale.

    Returns:
        The points, then the extra ones, between 0 and 1 (extra points may
        lie outside).
    """
    coords = np.asarray(coords, dtype=float)
    low = coords.min(axis=0)
    span = (coords - low).max()
    scale = 1.0 / span if span > 0 else 1.0
    shift = (1.0 - (coords - low).max(axis=0) * scale) / 2
    points = coords if extra is None else np.vstack([coords, extra])
    return (points - low) * scale + shift


def topic_hierarchy(similarity: np.ndarray) -> Hierarchy | None:
    """Return how the topics merge into larger ones, as a dendrogram.

    Average linkage of the distances ``1 - similarity``, with the leaves
    in an order that keeps alike topics together (SciPy's optimal
    ordering).

    Args:
        similarity: A square similarity matrix of at least two topics.

    Returns:
        The dendrogram, or ``None`` with fewer than two topics.
    """
    if len(similarity) < 2:
        return None
    from scipy.cluster import hierarchy as clustering
    from scipy.spatial.distance import squareform

    distance = np.clip(1.0 - np.asarray(similarity, dtype=float), 0.0, None)
    np.fill_diagonal(distance, 0.0)
    distance = (distance + distance.T) / 2
    links = clustering.linkage(
        squareform(distance, checks=False),
        method="average",
        optimal_ordering=True,
    )
    tree = clustering.dendrogram(links, no_plot=True, orientation="left")
    # SciPy places leaves at 5, 15, 25, ...; make them 0, 1, 2, ...
    return Hierarchy(
        order=[int(leaf) for leaf in tree["leaves"]],
        links=[
            [
                [round(float(value), 4) for value in distances],
                [round((float(value) - 5) / 10, 3) for value in places],
            ]
            for distances, places in zip(
                tree["dcoord"], tree["icoord"], strict=True
            )
        ],
        height=round(float(links[-1][2]), 4),
    )


# ---------------------------------------------------------------------------
# Over time
# ---------------------------------------------------------------------------


def intervals(
    timestamps: Sequence[Any], requested_bins: int | None
) -> tuple[list[str], np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """Divide time into intervals, and place each document in one.

    Args:
        timestamps: Each document's timestamp.
        requested_bins: How many equal intervals to use at most; with
            fewer distinct timestamps (or ``None``), each timestamp is its
            own interval.

    Returns:
        The intervals' labels (their start), each document's interval,
        and a function that places other timestamps (in nanoseconds) in
        the same intervals.
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
        step = None

        def locate(values: np.ndarray) -> np.ndarray:
            places = np.searchsorted(starts, values)
            return np.clip(places, 0, len(starts) - 1)

    else:
        edges = np.linspace(nanos.min(), nanos.max(), bins + 1)
        starts = edges[:-1]
        step = pd.Timedelta(int(edges[1] - edges[0]), unit="ns")

        def locate(values: np.ndarray) -> np.ndarray:
            places = np.searchsorted(edges, values, side="right") - 1
            return np.clip(places, 0, bins - 1)

    labels = _time_labels(
        pd.to_datetime(starts.astype("int64")),
        yearly=bool(
            ((stamps.dt.month == 1) & (stamps.dt.day == 1)).all()
            and (stamps == stamps.dt.normalize()).all()
        ),
        step=step,
    )
    return labels, locate(nanos), locate


def interval_counts(
    topic_of: np.ndarray, index: np.ndarray, topics: int, columns: int
) -> list[list[int]]:
    """Count each topic's documents per interval.

    Args:
        topic_of: Each document's topic index, -1 for none.
        index: Each document's interval.
        topics: How many topics there are.
        columns: How many intervals there are.

    Returns:
        Per topic, its documents in each interval.
    """
    counts = np.zeros((topics, columns), dtype=int)
    for topic, column in zip(topic_of, index, strict=True):
        if topic >= 0:
            counts[topic, column] += 1
    return counts.tolist()


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
        requested_bins: How many equal intervals to use at most
            (:func:`intervals`).

    Returns:
        The intervals' labels (their start), and per topic the count of
        its documents in each interval.
    """
    position = {topic: row for row, topic in enumerate(topic_ids)}
    topic_of = np.array([position.get(value, -1) for value in assigned])
    labels, index, _locate = intervals(timestamps, requested_bins)
    return labels, interval_counts(
        topic_of, index, len(topic_ids), len(labels)
    )


def interval_words(
    rows: Sequence[Sequence[Any]],
    position: Mapping[Any, int],
    locate: Callable[[np.ndarray], np.ndarray],
    topics: int,
    columns: int,
) -> list[list[str]]:
    """Return each topic's words in each interval, from BERTopic's rows.

    Args:
        rows: ``[topic, timestamp in nanoseconds, words]`` rows.
        position: Each topic's index, by its number.
        locate: Places timestamps in the intervals (:func:`intervals`).
        topics: How many topics there are.
        columns: How many intervals there are.

    Returns:
        Per topic, its words in each interval, "" where it has none.
    """
    grid = [[""] * columns for _ in range(topics)]
    for topic, stamp, words in rows:
        row = position.get(topic)
        if row is None:
            continue
        column = int(locate(np.array([int(stamp)]))[0])
        if not grid[row][column]:
            grid[row][column] = str(words)
    return grid


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


# ---------------------------------------------------------------------------
# The documents placed by meaning (BERTopic, Top2Vec)
# ---------------------------------------------------------------------------


def document_map(
    vectors: Any,
    topic_of: np.ndarray,
    texts: Sequence[str],
    *,
    reducer: Reducer | None = None,
    limit: int = MAP_DOCUMENTS,
) -> DocumentMap | None:
    """Place the documents where documents of alike meaning lie close.

    Collections larger than ``limit`` are drawn as a random sample (the
    same sample each time).

    Args:
        vectors: One vector per document.
        topic_of: Each document's topic index, -1 for none.
        texts: Each document's text.
        reducer: Places vectors in two dimensions; UMAP by default.
        limit: How many documents to draw at most.

    Returns:
        The map, or ``None`` if the vectors do not match the documents.
    """
    if hasattr(vectors, "toarray"):
        vectors = vectors.toarray()
    vectors = np.asarray(vectors, dtype=float)
    count = len(topic_of)
    if vectors.ndim != 2 or len(vectors) != count or count < 3:
        return None
    rows = np.arange(count)
    if count > limit:
        rows = np.sort(
            np.random.default_rng(42).choice(count, limit, replace=False)
        )
    coords = (reducer or reduce_documents)(vectors[rows])
    coords = np.round(unit_square(coords), 4)
    return DocumentMap(
        points=[
            [float(x), float(y), int(topic_of[row])]
            for (x, y), row in zip(coords, rows, strict=True)
        ],
        excerpts=[excerpt(texts[row], MAP_EXCERPT_CHARACTERS) for row in rows],
        documents=count,
    )


def reduce_documents(vectors: np.ndarray) -> np.ndarray:
    """Place document vectors in two dimensions, as BERTopic does.

    UMAP with BERTopic's settings for its document chart (10 neighbors,
    cosine, no minimum distance, a fixed seed); principal components for
    very small collections or without UMAP.

    Args:
        vectors: One vector per document.

    Returns:
        One ``[x, y]`` per document.
    """
    if len(vectors) >= 15:
        try:
            from umap import UMAP
        except ImportError:
            pass
        else:
            return UMAP(
                n_neighbors=10,
                n_components=2,
                min_dist=0.0,
                metric="cosine",
                random_state=42,
            ).fit_transform(vectors)
    return principal_components(vectors)


def principal_components(vectors: np.ndarray) -> np.ndarray:
    """Return the vectors' first two principal components.

    Args:
        vectors: One vector per row.

    Returns:
        One ``[x, y]`` per row.
    """
    centred = vectors - vectors.mean(axis=0)
    _left, _values, axes = np.linalg.svd(centred, full_matrices=False)
    coords = centred @ axes[:2].T
    if coords.shape[1] < 2:
        coords = np.hstack(
            [coords, np.zeros((len(coords), 2 - coords.shape[1]))]
        )
    return coords


# ---------------------------------------------------------------------------
# What LDA and Top2Vec add
# ---------------------------------------------------------------------------


def mixture(shares: np.ndarray) -> list[list[float]]:
    """Return the topics a document mixes, the largest first.

    Args:
        shares: Its share of each topic.

    Returns:
        ``[topic index, share]`` for each topic with at least
        :data:`MIXTURE_FLOOR` of the document.
    """
    total = float(np.sum(shares)) or 1.0
    found = [
        [int(index), round(float(share) / total, 3)]
        for index, share in enumerate(shares)
        if share / total >= MIXTURE_FLOOR
    ]
    return sorted(found, key=lambda item: -item[1])


def cloud(words: Sequence[Sequence[Any]]) -> list[list[Any]]:
    """Return a word cloud: the first words with weights scaled to 1.

    Args:
        words: ``[word, score]`` pairs, the most typical first.

    Returns:
        Up to :data:`CLOUD_WORDS` ``[word, weight]`` pairs.
    """
    chosen = [(str(word), float(score)) for word, score in words[:CLOUD_WORDS]]
    weights = scaled_weights([score for _, score in chosen])
    return [
        [word, weight]
        for (word, _), weight in zip(chosen, weights, strict=False)
    ]


def ldavis_parts(prepared: Any, topics: int) -> dict[str, Any]:
    """Return what pyLDAvis draws, from its prepared data.

    pyLDAvis must have kept the topics in model order (``sort_topics=
    False``), so that its topic N is the table's topic N.

    Args:
        prepared: pyLDAvis's ``PreparedData``: its topic coordinates,
            topic information and token table.
        topics: How many topics there are.

    Returns:
        ``positions`` (each topic's place, between 0 and 1), ``origin``
        (where the axes cross), ``prevalence`` (each topic's share of the
        words), ``salient`` (the most salient terms, as ``[term,
        frequency]``), ``terms`` (per topic, the terms it may rank, as
        ``[term, frequency in the topic, frequency, log probability, log
        lift]``) and ``term_topics`` (per term, ``[topic index, share]``).
    """
    coordinates = prepared.topic_coordinates.sort_values("topics")
    coords = coordinates[["x", "y"]].to_numpy(dtype=float)
    placed = unit_square(coords, extra=np.zeros((1, 2)))
    info = prepared.topic_info
    salient = info[info["Category"] == "Default"]
    terms = []
    for number in range(1, topics + 1):
        rows = info[info["Category"] == f"Topic{number}"]
        terms.append(
            [
                [
                    str(row.Term),
                    round(float(row.Freq), 2),
                    round(float(row.Total), 2),
                    round(float(row.logprob), 4),
                    round(float(row.loglift), 4),
                ]
                for row in rows.itertuples(index=False)
            ]
        )
    term_topics: dict[str, list[list[float]]] = {}
    for row in prepared.token_table.itertuples(index=False):
        term_topics.setdefault(str(row.Term), []).append(
            [int(row.Topic) - 1, round(float(row.Freq), 3)]
        )
    return {
        "positions": np.round(placed[:-1], 4).tolist(),
        "origin": np.round(placed[-1], 4).tolist(),
        "prevalence": [
            round(float(value) / 100, 4) for value in coordinates["Freq"]
        ],
        "salient": [
            [str(row.Term), round(float(row.Total), 2)]
            for row in salient.itertuples(index=False)
        ],
        "terms": terms,
        "term_topics": term_topics,
    }
