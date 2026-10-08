"""Topic-modeling guard rails.

Kept to :mod:`core.topic_modeling.small_corpus` and
:mod:`core.topic_modeling.topic_utils`, neither of which imports a modeling
engine: pulling in either engine costs about half a minute.
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


def _prepare_timestamps():
    """Import lazily so the small-corpus tests run without spaCy and NLTK."""
    pytest.importorskip("spacy")
    pytest.importorskip("nltk")
    from core.topic_modeling.topic_utils import prepare_timestamps

    return prepare_timestamps


def test_integer_years_are_read_as_years():
    """pandas reads a bare integer as nanoseconds since 1970, which put every
    document at the same instant without any error."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"Text": ["a", "b", "c"], "Year": [2021, 2019, 2020]})

    _, timestamps, dropped = _prepare_timestamps()(df, "Year")

    assert [ts.year for ts in timestamps] == [2019, 2020, 2021]
    assert dropped == 0


def test_float_years_with_gaps_are_read_as_years():
    """A year column with empty cells is loaded as floats (2019.0, NaN)."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"Text": ["a", "b", "c"], "Year": [2019.0, None, 2021.0]})

    _, timestamps, dropped = _prepare_timestamps()(df, "Year")

    assert [ts.year for ts in timestamps] == [2019, 2021]
    assert dropped == 1


@pytest.mark.parametrize("values", [[1.5e9, 1.6e9], [20190531, 20200101], [2019.5, 2020.0]])
def test_numbers_that_are_not_years_are_rejected(values):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"Text": ["a", "b"], "When": values})

    with pytest.raises(ValueError, match="not years"):
        _prepare_timestamps()(df, "When")


def test_date_strings_are_still_parsed():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"Text": ["a", "b", "c"], "Date": ["2020-03-01", "oops", "2019-12-31"]})

    _, timestamps, dropped = _prepare_timestamps()(df, "Date")

    assert [str(ts.date()) for ts in timestamps] == ["2019-12-31", "2020-03-01"]
    assert dropped == 1
