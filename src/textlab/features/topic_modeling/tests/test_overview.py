"""The overview of a run: sizes, examples, similarity, map and timeline.

Built from plain tables and arrays, the same way for every algorithm, so
these tests need no modeling library.
"""

import numpy as np
import pandas as pd
import pytest

from textlab.features.topic_modeling import overview
from textlab.features.topic_modeling.models import TopicKeywords


def documents():
    """Six documents: three in topic 1, two in topic 2, one outlier."""
    return pd.DataFrame(
        {
            "Text": [
                "taxes rise",
                "taxes  fall\nagain",
                "budget and taxes",
                "football final",
                "football cup",
                "noise",
            ],
            "Dominant_Topic": [1, 1, 1, 2, 2, "Outlier"],
            "Topic_Confidence": [0.2, 0.9, 0.5, 0.7, 0.8, None],
        }
    )


TOPICS = [
    TopicKeywords(1, ["taxes", "budget"], 3, [0.5, 0.25]),
    TopicKeywords(2, ["football", "cup"], 2, [0.8, 0.4]),
]


def test_each_topic_has_its_size_weights_and_most_typical_examples():
    found = overview.build_overview(TOPICS, documents(), "Text")
    first, second = found.topics
    assert (found.documents, found.outliers) == (6, 1)
    assert (first.documents, second.documents) == (3, 2)
    assert first.weights == [1.0, 0.5]
    # The most confident first; white space made one space.
    assert first.examples == [
        "taxes fall again",
        "budget and taxes",
        "taxes rise",
    ]
    assert found.similarity is None and found.positions is None
    assert found.time_labels == [] and found.over_time == []


def test_alike_topics_lie_close_on_the_map():
    vectors = np.array([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 0.0, 1.0]])
    topics = [*TOPICS, TopicKeywords(3, ["noise"], 1, [1.0])]
    found = overview.build_overview(
        topics, documents(), "Text", topic_vectors=vectors
    )
    similarity = np.array(found.similarity)
    assert similarity.shape == (3, 3)
    assert np.allclose(np.diag(similarity), 1.0)
    assert similarity[0, 1] > 0.9 and similarity[0, 2] == 0.0
    positions = np.array(found.positions)
    assert positions.min() >= 0.0 and positions.max() <= 1.0
    near = np.linalg.norm(positions[0] - positions[1])
    far = np.linalg.norm(positions[0] - positions[2])
    assert near < far


def test_sparse_topic_vectors_are_read_too():
    sparse = pytest.importorskip("scipy.sparse")
    vectors = sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
    found = overview.build_overview(
        TOPICS, documents(), "Text", topic_vectors=vectors
    )
    assert found.similarity == [[1.0, 0.0], [0.0, 1.0]]


def test_vectors_that_do_not_match_the_topics_draw_no_map():
    found = overview.build_overview(
        TOPICS, documents(), "Text", topic_vectors=np.ones((3, 2))
    )
    assert found.similarity is None


def test_topics_are_counted_per_year():
    stamps = pd.to_datetime(
        ["2020-01-01", "2021-01-01", "2021-01-01", "2020-01-01", "2022-01-01"]
        + ["2022-01-01"]
    ).tolist()
    found = overview.build_overview(
        TOPICS, documents(), "Text", timestamps=stamps, time_bins=20
    )
    assert found.time_labels == ["2020", "2021", "2022"]
    assert found.over_time == [[1, 2, 0], [1, 0, 1]]


def test_many_timestamps_are_grouped_into_intervals():
    labels, counts = overview.topics_over_time(
        [1],
        [1] * 10,
        pd.date_range("2024-01-01", periods=10, freq="D").tolist(),
        2,
    )
    assert labels == ["2024-01-01", "2024-01-05"]
    assert counts == [[5, 5]]


def test_weights_are_scaled_to_the_largest():
    assert overview.scaled_weights([2.0, 1.0, 0.5]) == [1.0, 0.5, 0.25]
    assert overview.scaled_weights([-0.4, -0.2]) == [1.0, 0.5]
    assert overview.scaled_weights([]) == []
    assert overview.scaled_weights([0.0]) == []


def test_long_texts_are_cut_at_a_word():
    text = "word " * 100
    cut = overview.excerpt(text, limit=22)
    assert cut == "word word word word…"
    assert overview.excerpt("short") == "short"
