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


# --- What each algorithm's library draws ----------------------------------


def prepared_ldavis():
    """pyLDAvis's prepared data for two topics, in model order."""
    from types import SimpleNamespace

    return SimpleNamespace(
        topic_coordinates=pd.DataFrame(
            {
                "x": [0.2, -0.2],
                "y": [0.0, 0.1],
                "topics": [1, 2],
                "cluster": [1, 1],
                "Freq": [60.0, 40.0],
            }
        ),
        topic_info=pd.DataFrame(
            {
                "Term": ["taxes", "football", "taxes", "budget", "cup"],
                "Freq": [30.0, 20.0, 25.0, 9.0, 12.0],
                "Total": [30.0, 20.0, 30.0, 10.0, 12.0],
                "Category": [
                    "Default",
                    "Default",
                    "Topic1",
                    "Topic1",
                    "Topic2",
                ],
                "logprob": [2, 1, -1.2, -2.3, -0.9],
                "loglift": [2, 1, 0.4, 0.9, 1.1],
            }
        ),
        token_table=pd.DataFrame(
            {
                "Topic": [1, 2, 1, 2],
                "Freq": [0.8, 0.2, 0.9, 1.0],
                "Term": ["taxes", "taxes", "budget", "cup"],
            }
        ),
    )


def test_lda_is_drawn_from_pyldavis_data():
    found = overview.build_overview(
        TOPICS,
        documents(),
        "Text",
        topic_vectors=np.eye(2),
        ldavis=prepared_ldavis(),
        document_topics=np.array(
            [
                [0.9, 0.1],
                [0.6, 0.4],
                [0.97, 0.03],
                [0.2, 0.8],
                [0.5, 0.5],
                [0.5, 0.5],
            ]
        ),
    )
    first, second = found.topics
    # Sized by their share of the words, placed where pyLDAvis put them.
    assert (first.prevalence, second.prevalence) == (0.6, 0.4)
    assert found.positions[0][0] > found.positions[1][0]
    assert found.origin == [0.5, 0.375]
    assert found.salient == [["taxes", 30.0], ["football", 20.0]]
    assert first.terms == [
        ["taxes", 25.0, 30.0, -1.2, 0.4],
        ["budget", 9.0, 10.0, -2.3, 0.9],
    ]
    assert found.term_topics["taxes"] == [[0, 0.8], [1, 0.2]]
    # Each example's mixture, topics under 5% left out.
    assert first.mixtures[0] == [[0, 0.6], [1, 0.4]]
    assert [[0, 0.97]] in first.mixtures


def test_top2vec_words_make_a_cloud():
    words = [[["tax", 0.8], ["budget", 0.4]], [["cup", 0.5]]]
    found = overview.build_overview(
        TOPICS, documents(), "Text", topic_words=words
    )
    assert found.topics[0].cloud == [["tax", 1.0], ["budget", 0.5]]
    assert found.topics[1].cloud == [["cup", 1.0]]


def test_bertopic_words_of_each_interval_are_kept():
    stamps = pd.to_datetime(
        ["2020-01-01", "2021-01-01", "2021-01-01", "2020-01-01"]
        + ["2022-01-01", "2022-01-01"]
    )
    rows = [
        [1, pd.Timestamp("2020-01-01").value, "taxes, rise"],
        [1, pd.Timestamp("2021-01-01").value, "taxes, budget"],
        [2, pd.Timestamp("2022-01-01").value, "football, cup"],
        ["Outlier", pd.Timestamp("2022-01-01").value, "noise"],
    ]
    found = overview.build_overview(
        TOPICS,
        documents(),
        "Text",
        timestamps=stamps.tolist(),
        time_bins=20,
        time_rows=rows,
    )
    assert found.topics[0].time_words == ["taxes, rise", "taxes, budget", ""]
    assert found.topics[1].time_words == ["", "", "football, cup"]


def test_documents_are_placed_and_sampled():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(6, 4))
    found = overview.build_overview(
        TOPICS,
        documents(),
        "Text",
        document_vectors=vectors,
        reducer=overview.principal_components,
    )
    points = found.document_map.points
    assert len(points) == 6 and found.document_map.documents == 6
    assert [point[2] for point in points] == [0, 0, 0, 1, 1, -1]
    assert all(0 <= point[0] <= 1 and 0 <= point[1] <= 1 for point in points)
    assert found.document_map.excerpts[1] == "taxes fall again"

    sampled = overview.document_map(
        rng.normal(size=(50, 3)),
        np.zeros(50, dtype=int),
        ["text"] * 50,
        reducer=overview.principal_components,
        limit=10,
    )
    assert len(sampled.points) == 10 and sampled.documents == 50


def test_document_vectors_that_do_not_match_draw_no_map():
    found = overview.build_overview(
        TOPICS, documents(), "Text", document_vectors=np.ones((2, 3))
    )
    assert found.document_map is None


def test_alike_topics_merge_first_in_the_hierarchy():
    similarity = np.array([[1.0, 0.1, 0.9], [0.1, 1.0, 0.2], [0.9, 0.2, 1.0]])
    tree = overview.topic_hierarchy(similarity)
    order = tree.order
    # Topics 0 and 2 are alike: they sit next to each other.
    assert abs(order.index(0) - order.index(2)) == 1
    assert len(tree.links) == 2
    first_merge = min(tree.links, key=lambda link: max(link[0]))
    assert max(first_merge[0]) == pytest.approx(0.1)
    assert tree.height == pytest.approx(0.85)
    assert overview.topic_hierarchy(np.ones((1, 1))) is None


def test_a_mixture_keeps_the_large_shares_in_order():
    assert overview.mixture(np.array([0.02, 0.3, 0.68])) == [
        [2, 0.68],
        [1, 0.3],
    ]


def test_points_are_scaled_together_with_extra_ones():
    placed = overview.unit_square(
        np.array([[0.0, 0.0], [2.0, 1.0]]), extra=np.array([[1.0, 0.5]])
    )
    assert placed.tolist() == [[0.0, 0.25], [1.0, 0.75], [0.5, 0.5]]
