"""Topic-modeling guard rails.

Kept to :mod:`core.topic_modeling.small_corpus`,
:mod:`core.topic_modeling.topic_utils` and
:mod:`core.topic_modeling.evaluation`, none of which imports a modeling
engine: pulling in either engine costs about half a minute.
"""

import io
import zipfile

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


def _topic_utils():
    """Import lazily so the small-corpus tests run without spaCy and NLTK."""
    pytest.importorskip("spacy")
    pytest.importorskip("nltk")
    from core.topic_modeling import topic_utils

    return topic_utils


def _prepare_timestamps():
    return _topic_utils().prepare_timestamps


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


@pytest.mark.parametrize(
    "content",
    [
        "Text;Year\nfirst doc;2019\nsecond doc;2020\n",
        "Text\tYear\nfirst doc\t2019\nsecond doc\t2020\n",
        "Text,Year\nfirst doc,2019\nsecond doc,2020\n",
    ],
    ids=["semicolon", "tab", "comma"],
)
def test_csv_delimiter_is_detected(content):
    df = _topic_utils().read_uploaded_table("data.csv", content.encode("utf-8"))

    assert df.columns.tolist() == ["Text", "Year"]
    assert df["Text"].tolist() == ["first doc", "second doc"]


def test_quoted_text_with_semicolons_keeps_comma_delimiter():
    content = 'Text,Year\n"one; two; three",2019\n"four; five",2020\n'

    df = _topic_utils().read_uploaded_table("data.csv", content.encode("utf-8"))

    assert df.columns.tolist() == ["Text", "Year"]
    assert df["Text"].tolist() == ["one; two; three", "four; five"]


def test_windows_encoded_csv_keeps_accents():
    """Excel on Windows saves CSV as cp1252, which is not valid UTF-8."""
    content = "Text;Ort\nÜber die Brücke;Zürich\n".encode("cp1252")

    df = _topic_utils().read_uploaded_table("data.csv", content)

    assert df.loc[0, "Text"] == "Über die Brücke"
    assert df.loc[0, "Ort"] == "Zürich"


def test_legacy_xls_is_rejected_with_guidance():
    with pytest.raises(ValueError, match=".xlsx"):
        _topic_utils().read_uploaded_table("data.xls", b"irrelevant")


def test_zip_text_files_are_decoded_without_losing_characters():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("utf8.txt", "Café crème".encode("utf-8"))
        archive.writestr("windows.txt", "Café crème".encode("cp1252"))
        archive.writestr("empty.txt", b"   ")

    df = _topic_utils().load_zip_texts(buffer.getvalue())

    assert sorted(df["Filename"]) == ["utf8.txt", "windows.txt"]
    assert set(df["Text"]) == {"Café crème"}


@pytest.mark.parametrize("max_chars", [5, 7, 1000])
def test_long_text_is_split_without_losing_content(max_chars):
    text = "alpha beta gamma delta epsilon zetaetaeta"

    chunks = _topic_utils().split_long_text(text, max_chars)

    assert "".join(chunks) == text
    assert all(len(chunk) <= max_chars for chunk in chunks)


def test_long_text_is_split_at_spaces():
    chunks = _topic_utils().split_long_text("aaa bbb ccc", 6)

    assert chunks == ["aaa", " bbb", " ccc"]


def test_time_bins_only_apply_when_there_are_more_timestamps():
    resolve_time_bins = _topic_utils().resolve_time_bins

    assert resolve_time_bins([2019, 2020, 2020, 2021], 20) is None
    assert resolve_time_bins(list(range(50)), 20) == 20


def test_topic_table_joins_keywords():
    from core.topic_modeling.topic_config import TopicKeywords

    topics = [
        TopicKeywords(topic=1, keywords=["tax", "budget"], count=12),
        TopicKeywords(topic=2, keywords=["rail"], count=3),
    ]
    utils = _topic_utils()

    with_counts = utils.build_topic_table(topics)
    without_counts = utils.build_topic_table(topics, with_counts=False)

    assert with_counts.columns.tolist() == ["Topic", "Count", "Keywords"]
    assert with_counts["Keywords"].tolist() == ["tax, budget", "rail"]
    assert without_counts.columns.tolist() == ["Topic", "Keywords"]


def test_empty_topic_table_keeps_its_columns():
    table = _topic_utils().build_topic_table([])

    assert table.empty
    assert table.columns.tolist() == ["Topic", "Count", "Keywords"]


def _evaluation():
    _topic_utils()
    pytest.importorskip("gensim")
    from core.topic_modeling import evaluation

    return evaluation


def test_identical_runs_are_perfectly_stable():
    topics = [["tax", "budget"], ["rail", "train"]]

    assert _evaluation().calculate_jaccard_stability(topics, topics) == 1.0


def test_disjoint_runs_have_zero_stability():
    stability = _evaluation().calculate_jaccard_stability([["a", "b"]], [["c", "d"]])

    assert stability == 0.0


def test_metrics_without_keywords_are_not_reported_as_zero():
    """0.0 would read as a perfect U_mass score."""
    metrics = _evaluation().evaluate_topic_quality(
        topic_keywords=[],
        raw_texts=["some text"],
        language="English",
        custom_stopwords_str="",
    )

    assert set(metrics.values()) == {None}


class _WordTokenizer:
    """Stand-in for a HuggingFace tokenizer: one token per word."""

    def encode(self, text, add_special_tokens=False, truncation=False):
        return text.split()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)

    def num_special_tokens_to_add(self, pair=False):
        return 2


class _FakeEmbeddingModel:
    """Stand-in for a SentenceTransformer with a six-token context window."""

    max_seq_length = 6
    tokenizer = _WordTokenizer()

    def __init__(self):
        self.encoded = []

    def encode(self, texts, show_progress_bar=False, convert_to_numpy=True):
        import numpy as np

        self.encoded.append(list(texts))
        vectors = np.array([[len(text.split()), 1.0] for text in texts])
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_long_document_is_split_into_token_chunks():
    chunks = _topic_utils().split_into_token_chunks(
        "a b c d e f g h i j", _WordTokenizer(), 4
    )

    assert chunks == [("a b c d", 4), ("e f g h", 4), ("i j", 2)]


def test_short_documents_are_embedded_as_before():
    np = pytest.importorskip("numpy")
    utils = _topic_utils()
    texts = ["a b", "c d e"]

    plain = utils.embed_documents(_FakeEmbeddingModel(), texts)
    model = _FakeEmbeddingModel()
    chunked = utils.embed_documents(model, texts, chunk_long_documents=True)

    assert np.allclose(plain, chunked)
    assert model.encoded == [texts]


def test_long_documents_are_embedded_in_chunks_that_fit():
    model = _FakeEmbeddingModel()
    long_text = "w1 w2 w3 w4 w5 w6 w7 w8 w9"

    embeddings = _topic_utils().embed_documents(
        model, ["a b", long_text], chunk_long_documents=True
    )

    # Six tokens minus two special tokens leaves four words per chunk.
    assert model.encoded == [["a b", "w1 w2 w3 w4", "w5 w6 w7 w8", "w9"]]
    assert embeddings.shape == (2, 2)


def test_chunk_average_is_weighted_by_length_and_keeps_scale():
    np = pytest.importorskip("numpy")
    chunk_embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [3.0, 4.0]])

    averages = _topic_utils().average_chunk_embeddings(
        chunk_embeddings,
        owners=np.array([0, 0, 1]),
        weights=np.array([3.0, 1.0, 2.0]),
        n_documents=2,
    )

    expected_direction = np.array([3.0, 1.0]) / np.linalg.norm([3.0, 1.0])
    assert np.allclose(averages[0], expected_direction)
    assert np.allclose(averages[1], [3.0, 4.0])
