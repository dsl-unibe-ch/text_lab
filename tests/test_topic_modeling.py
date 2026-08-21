"""Topic-modeling guard rails.

Kept to :mod:`core.topic_modeling.small_corpus`, which deliberately imports
nothing heavy: pulling in either engine costs about half a minute.
"""

import conftest_path  # noqa: F401

import pytest

from core.topic_modeling import small_corpus

# Verbatim from the three failures observed on real uploads. They come from
# three different libraries and two different exception types, and not one of
# them mentions the corpus.
UMAP_SPECTRAL = (
    "Cannot use scipy.linalg.eigh for sparse A with k >= N. "
    "Use scipy.linalg.eigh(A.toarray()) or reduce k."
)
HDBSCAN_NO_CLUSTER = "need at least one array to concatenate"
ALL_OUTLIERS = "Found array with 0 sample(s) (shape=(0, 384)) while a minimum of 1 is required."


@pytest.mark.parametrize(
    "exc",
    [
        TypeError(UMAP_SPECTRAL),  # Top2Vec and BERTopic, <= 6 documents
        ValueError(HDBSCAN_NO_CLUSTER),  # no dense region found
        ValueError(ALL_OUTLIERS),  # BERTopic, every document an outlier
    ],
)
def test_a_corpus_too_small_is_recognised(exc):
    assert small_corpus.is_corpus_too_small(exc) is True


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("No texts were provided to BERTopic."),
        ValueError("Invalid ngram_range: lower bound cannot be greater than upper bound."),
        TypeError("unsupported operand type(s) for +: 'int' and 'str'"),
        KeyError("Topic"),
    ],
)
def test_an_unrelated_failure_is_left_alone(exc):
    """A real bug has to keep reaching the traceback rather than be explained
    away as a small dataset."""
    assert small_corpus.is_corpus_too_small(exc) is False


def test_the_message_says_what_to_do_instead():
    err = small_corpus.too_small_error(4, "Top2Vec")
    assert isinstance(err, ValueError)
    text = str(err)
    assert "Top2Vec" in text and "4 document(s)" in text
    # The page renders a ValueError's message as markdown, and the whole point
    # is to send the user to the algorithm that does work on small datasets.
    assert "Latent Dirichlet Allocation (LDA)" in text
    assert "BERTopic" in str(small_corpus.too_small_error(9, "BERTopic"))
