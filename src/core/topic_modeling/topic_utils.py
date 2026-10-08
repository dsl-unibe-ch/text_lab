"""Data loading, text preprocessing and reporting helpers for topic modeling.

This module deliberately imports no modeling engine, so it stays cheap to
import from the page and from tests.
"""

import csv
import datetime
import io
import logging
import os
import re
import zipfile
from functools import lru_cache
from typing import Any

import nltk
import pandas as pd
import spacy
from nltk.corpus import stopwords

from core import upload_safety
from .topic_config import (
    TOP2VEC_BACKEND_LABELS,
    Algorithm,
    TopicKeywords,
    TopicModelingConfig,
)

LOGGER = logging.getLogger(__name__)

if "NLTK_DATA" in os.environ:
    nltk.data.path.append(os.environ["NLTK_DATA"])

SUPPORTED_LANGUAGES: list[str] = [
    "English",
    "German",
    "French",
    "Spanish",
    "Italian",
    "Dutch",
    "Portuguese",
    "Russian",
    "Arabic",
    "Chinese",
    "Other / Mixed",
]

SPACY_MODELS: dict[str, str] = {
    "English": "en_core_web_sm",
    "German": "de_core_news_sm",
    "French": "fr_core_news_sm",
}

NLTK_LANGUAGES: dict[str, str] = {
    "Spanish": "spanish",
    "Italian": "italian",
    "Dutch": "dutch",
    "Portuguese": "portuguese",
    "Russian": "russian",
    "Arabic": "arabic",
}

# Sentence-transformer models pre-downloaded into the shared read-only cache.
# Anything else is treated as a "custom" model and will be downloaded to the
# calling user's own HuggingFace cache directory.
SHARED_EMBEDDING_MODELS: set[str] = {
    "all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}

DEFAULT_EMBEDDING_MODEL_ENGLISH: str = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_MODEL_MULTILINGUAL: str = "paraphrase-multilingual-MiniLM-L12-v2"

#: Tabular file extensions accepted by :func:`read_uploaded_table`.
TABLE_EXTENSIONS: tuple[str, ...] = ("csv", "xlsx")

# Range of numeric values accepted as calendar years in a timestamp column.
_MIN_YEAR = 1000
_MAX_YEAR = 2999

# Encodings tried, in order, before falling back to Latin-1 (which accepts any
# byte sequence). Windows-1252 covers files saved by Excel on Windows.
_TEXT_ENCODINGS = ("utf-8-sig", "cp1252")
_CSV_DELIMITERS = ",;\t|"
_SNIFF_SAMPLE_CHARS = 64 * 1024


@lru_cache(maxsize=None)
def _load_spacy_model(language: str) -> spacy.language.Language | None:
    """
    Load and cache a spaCy language model for the given language.

    Args:
        language: The language name.

    Returns:
        The loaded spaCy language model if available, otherwise None.
    """
    model_name = SPACY_MODELS.get(language)
    if not model_name:
        return None
    try:
        return spacy.load(model_name, disable=["parser", "ner"])
    except OSError:
        LOGGER.warning(
            "spaCy model '%s' for %s is not installed. Run 'python -m spacy "
            "download %s' in the host environment. Falling back to basic "
            "regex tokenization for this job.",
            model_name,
            language,
            model_name,
        )
        return None


def decode_text_bytes(data: bytes) -> str:
    """
    Decode uploaded text without silently dropping characters.

    UTF-8 (with or without a byte-order mark) is tried first, then
    Windows-1252, and finally Latin-1, which accepts any byte sequence.

    Args:
        data: The raw file content.

    Returns:
        The decoded text.
    """
    for encoding in _TEXT_ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _sniff_delimiter(text: str) -> str:
    """
    Guess the delimiter of CSV text, defaulting to a comma.

    Args:
        text: The decoded CSV content.

    Returns:
        One of the supported delimiters (comma, semicolon, tab or pipe).
    """
    sample = text[:_SNIFF_SAMPLE_CHARS]
    if len(text) > _SNIFF_SAMPLE_CHARS and "\n" in sample:
        # Only sniff complete lines; a truncated last line skews the guess.
        sample = sample[: sample.rfind("\n")]
    try:
        return csv.Sniffer().sniff(sample, delimiters=_CSV_DELIMITERS).delimiter
    except csv.Error:
        return ","


def read_uploaded_table(filename: str, data: bytes) -> pd.DataFrame:
    """
    Read an uploaded CSV or Excel (.xlsx) file into a pandas DataFrame.

    CSV files may use a comma, semicolon, tab or pipe as delimiter and may be
    encoded in UTF-8, Windows-1252 or Latin-1.

    Args:
        filename: The uploaded file name.
        data: The raw file content.

    Returns:
        The loaded pandas DataFrame.

    Raises:
        ValueError: If the file format is not supported.
    """
    lower_name = filename.lower()
    if lower_name.endswith(".csv"):
        text = decode_text_bytes(data)
        return pd.read_csv(io.StringIO(text), sep=_sniff_delimiter(text))
    if lower_name.endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(data))
    if lower_name.endswith(".xls"):
        raise ValueError(
            "Legacy Excel files (.xls) are not supported. Please save the "
            "file as .xlsx or .csv and upload it again."
        )
    raise ValueError("Unsupported tabular file format.")


def load_zip_texts(zip_bytes: bytes) -> pd.DataFrame:
    """
    Load non-empty text files from a ZIP archive into a DataFrame.

    Files whose basename starts with ``._`` are ignored. Each file is decoded
    with :func:`decode_text_bytes`.

    Args:
        zip_bytes: The ZIP archive content as bytes.

    Returns:
        A pandas DataFrame with columns ``Filename`` and ``Text``.
    """
    data = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as archive:
        for info in upload_safety.safe_zip_members(
            archive, allowed_extensions={".txt"}
        ):
            filename = info.filename
            if os.path.basename(filename).startswith("._"):
                continue
            content = decode_text_bytes(archive.read(info))
            if content.strip():
                data.append({"Filename": filename, "Text": content})
    return pd.DataFrame(data, columns=["Filename", "Text"])


def drop_empty_text_rows(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Remove rows with missing values in the selected text column.

    Args:
        df: The input DataFrame.
        text_column: The column containing the source text.

    Returns:
        A cleaned DataFrame with reset index.

    Raises:
        ValueError: If no valid rows remain after filtering.
    """
    cleaned_df = df.dropna(subset=[text_column]).reset_index(drop=True)

    if cleaned_df.empty:
        raise ValueError("No valid documents remain after removing empty text rows.")

    return cleaned_df


def _parse_timestamp_column(values: pd.Series) -> pd.Series:
    """
    Parse a column of dates, timestamps or years into timezone-naive datetimes.

    Numeric columns are only accepted when they hold whole years (e.g. 2019),
    which are mapped to 1 January of that year. pandas would otherwise read
    numbers as nanoseconds since 1970, collapsing every row into the same
    instant without any error.

    Args:
        values: The raw timestamp column.

    Returns:
        A datetime Series in which unparseable values are ``NaT``.

    Raises:
        ValueError: If the column is numeric but does not contain whole years.
    """
    if pd.api.types.is_numeric_dtype(values):
        numbers = values.dropna()
        is_year = (numbers % 1 == 0) & numbers.between(_MIN_YEAR, _MAX_YEAR)
        if not is_year.all():
            raise ValueError(
                "The timestamp column contains numbers that are not years. "
                "Use a column with dates (e.g. 2019-05-31) or whole years "
                "(e.g. 2019)."
            )
        years = values.astype("Int64").astype("string")
        return pd.to_datetime(years, format="%Y", errors="coerce")

    return pd.to_datetime(values, errors="coerce", utc=True).dt.tz_localize(None)


def prepare_timestamps(
    df: pd.DataFrame,
    date_column: str,
) -> tuple[pd.DataFrame, list[pd.Timestamp], int]:
    """
    Parse and validate timestamps from a selected date column.

    Dates, date-times and whole years (numeric or text) are supported.

    Args:
        df: The input DataFrame.
        date_column: The column containing timestamps, dates or years.

    Returns:
        A tuple containing:
            - The filtered and sorted DataFrame
            - A list of parsed timestamps
            - The number of dropped rows

    Raises:
        ValueError: If the column is numeric but does not hold years, or if
            no valid timestamps remain after parsing.
    """
    parsed = _parse_timestamp_column(df[date_column])
    valid_mask = parsed.notna()
    dropped = int((~valid_mask).sum())

    filtered_df = df.loc[valid_mask].copy()
    filtered_df[date_column] = parsed.loc[valid_mask]
    filtered_df = filtered_df.sort_values(date_column).reset_index(drop=True)

    if filtered_df.empty:
        raise ValueError(
            "No valid timestamps remained after parsing the selected timestamp column."
        )

    return filtered_df, filtered_df[date_column].tolist(), dropped


def resolve_time_bins(timestamps: list[Any], requested_bins: int) -> int | None:
    """
    Decide how many intervals the topics-over-time analysis should use.

    Args:
        timestamps: The parsed document timestamps.
        requested_bins: The number of intervals selected by the user.

    Returns:
        ``requested_bins`` if there are more distinct timestamps than that,
        otherwise ``None`` so that every distinct timestamp is kept as its own
        point instead of being spread over mostly empty intervals.
    """
    return requested_bins if len(set(timestamps)) > requested_bins else None


def validate_minimum_documents(texts: list[str], minimum_docs: int = 5) -> None:
    """
    Validate that the dataset contains a minimum number of documents.

    Args:
        texts: The raw text documents.
        minimum_docs: The minimum number of required documents.

    Raises:
        ValueError: If too few documents are provided.
    """
    if len(texts) < minimum_docs:
        raise ValueError(
            f"A minimum of {minimum_docs} valid documents is required to "
            "perform topic modeling."
        )


def build_topic_table(
    topics: list[TopicKeywords],
    with_counts: bool = True,
) -> pd.DataFrame:
    """
    Build the topic table shown on the page and exported as CSV.

    Args:
        topics: The topics, in display order.
        with_counts: Whether to include the ``Count`` column.

    Returns:
        A DataFrame with the columns ``Topic``, ``Count`` (optional) and
        ``Keywords`` (comma-separated).
    """
    columns = ["Topic", "Count", "Keywords"] if with_counts else ["Topic", "Keywords"]
    rows = [
        {
            "Topic": topic.topic,
            "Count": topic.count,
            "Keywords": ", ".join(topic.keywords),
        }
        for topic in topics
    ]
    return pd.DataFrame(rows, columns=columns)


def get_embedding_model_name(config: TopicModelingConfig) -> str:
    """
    Resolve the human-readable embedding model name for the selected configuration.

    Args:
        config: The topic modeling configuration.

    Returns:
        The resolved embedding model name.
    """
    if config.algorithm == Algorithm.BERTOPIC:
        return resolve_bertopic_embedding_model_id(config)

    if config.algorithm == Algorithm.TOP2VEC:
        if config.top2vec_backend == "transformer":
            return (
                DEFAULT_EMBEDDING_MODEL_ENGLISH
                if config.language == "English"
                else DEFAULT_EMBEDDING_MODEL_MULTILINGUAL
            )
        return "Doc2Vec (Trained from scratch)"

    return "N/A"


def resolve_bertopic_embedding_model_id(config: TopicModelingConfig) -> str:
    """
    Resolve the HuggingFace sentence-transformer model ID used by BERTopic.

    Args:
        config: The topic modeling configuration.

    Returns:
        The HuggingFace model ID (e.g. "all-MiniLM-L6-v2").
    """
    if config.embedding_model_id and config.embedding_model_id.strip():
        return config.embedding_model_id.strip()

    return (
        DEFAULT_EMBEDDING_MODEL_ENGLISH
        if config.language == "English"
        else DEFAULT_EMBEDDING_MODEL_MULTILINGUAL
    )


def is_shared_embedding_model(model_id: str) -> bool:
    """
    Check whether a model is expected in the shared read-only cache.

    Non-shared models must be downloaded to the calling user's home cache.

    Args:
        model_id: The HuggingFace model ID.

    Returns:
        True if the model is one of the pre-downloaded shared models.
    """
    return model_id in SHARED_EMBEDDING_MODELS


def get_user_hf_cache_dir() -> str:
    """
    Return the calling user's writable HuggingFace cache, creating it if needed.

    This is used for user-selected custom embedding models so they do not need
    write access to the shared model cache.

    Returns:
        The absolute path of the cache directory.
    """
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError as exc:
        LOGGER.warning("Could not create user HF cache dir '%s': %s", cache_dir, exc)
    return cache_dir


def load_sentence_transformer(model_id: str, trust_remote_code: bool = False) -> Any:
    """
    Instantiate a SentenceTransformer for the given model ID.

    Shared/curated models resolve through the container-level HF_HOME (the
    read-only shared cache). Custom models are directed to the user's own
    HuggingFace cache directory so they can be downloaded without write access
    to the shared cache.

    Args:
        model_id: The HuggingFace sentence-transformer model ID.
        trust_remote_code: Whether a custom model may run Python code shipped
            in its repository (needed by some long-context models such as
            Jina or Nomic). Ignored for the shared models.

    Returns:
        A ready-to-use SentenceTransformer instance.
    """
    from sentence_transformers import SentenceTransformer

    if is_shared_embedding_model(model_id):
        return SentenceTransformer(model_id)

    return SentenceTransformer(
        model_id,
        cache_folder=get_user_hf_cache_dir(),
        trust_remote_code=trust_remote_code,
    )


def count_docs_exceeding_context(
    embedding_model: Any,
    texts: list[str],
) -> tuple[int, int, int]:
    """
    Count how many documents would be truncated by the embedding model.

    The tokenizer associated with the sentence-transformer is used to count
    WordPiece/BPE tokens for each document. Documents whose token count exceeds
    the model's ``max_seq_length`` will be silently truncated by the encoder
    and only their leading portion will contribute to the topic embedding.

    Args:
        embedding_model: A SentenceTransformer (or compatible) instance.
        texts: The documents to inspect.

    Returns:
        A tuple ``(over_count, total_count, max_seq_length)``.
    """
    if not texts:
        return 0, 0, 0

    max_seq_length = int(getattr(embedding_model, "max_seq_length", 0) or 0)
    tokenizer = getattr(embedding_model, "tokenizer", None)
    if not max_seq_length or tokenizer is None:
        return 0, len(texts), max_seq_length

    over = 0
    for text in texts:
        try:
            n_tokens = len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
        except Exception:
            # Fall back to a coarse whitespace heuristic on tokenizer error.
            n_tokens = len(str(text).split())
        if n_tokens > max_seq_length:
            over += 1

    return over, len(texts), max_seq_length


def _report_header(filename: str, config: TopicModelingConfig) -> list[str]:
    """
    Build the source and core-settings part of the metadata report.

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

    stopword_text = config.custom_stopwords if config.custom_stopwords.strip() else "None"
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
    """
    Build the BERTopic-specific part of the metadata report.

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
        f"N-Gram Range: {config.ngram_range}",
        f"Min Topic Frequency (min_df): {config.min_df}",
        f"Reduce Frequent Words (ClassTfidfTransformer): {config.reduce_frequent}",
        "",
        f"--- DIMENSIONALITY REDUCTION ({config.dim_reduction_algo}) ---",
    ]

    if config.dim_reduction_algo != "None":
        lines.append(f"N Components: {config.dim_params.get('n_components')}")
        if config.dim_reduction_algo == "UMAP":
            lines.append(f"N Neighbors: {config.dim_params.get('n_neighbors')}")
            lines.append(f"Min Distance: {config.dim_params.get('min_dist')}")
        lines.append(f"Random State: {config.dim_params.get('random_state', 'None')}")

    lines.extend(["", f"--- CLUSTERING ({config.clustering_algo}) ---"])

    params = config.clustering_params
    if config.clustering_algo == "KMeans":
        lines.append(f"N Clusters: {params.get('n_clusters')}")
    else:
        min_samples = params.get("min_samples", "Default (equals min_cluster_size)")
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
    """
    Compile a formatted metadata report for reproducibility.

    Args:
        filename: The source file name.
        config: The topic modeling configuration.
        embedding_model_name: The resolved embedding model name.
        evaluation_metrics: Optional dictionary of calculated performance metrics.

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
            ]
        )
    else:
        report.extend(_report_bertopic_parameters(config, embedding_model_name))

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
            report.append(f"{metric_name}: {'N/A' if score is None else score}")

    return "\n".join(report)


def build_results_zip(
    metadata_report: str,
    docs_df: pd.DataFrame,
    topic_df: pd.DataFrame,
    dashboard_assets: dict[str, str],
) -> bytes:
    """
    Build a ZIP archive containing modeling outputs.

    Args:
        metadata_report: The run configuration report.
        docs_df: The document-level topic assignments.
        topic_df: The topic keywords table.
        dashboard_assets: HTML dashboard artifacts.

    Returns:
        ZIP archive content as bytes.
    """
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
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


def get_stopword_set(language: str, custom_stopwords_str: str) -> set[str]:
    """
    Build a stopword set from custom, spaCy, and NLTK sources.

    Args:
        language: The language name.
        custom_stopwords_str: A comma-separated string of custom stopwords.

    Returns:
        A set of lowercase stopwords.
    """
    stop_set: set[str] = set()

    if custom_stopwords_str:
        stop_set.update(
            {w.strip().lower() for w in custom_stopwords_str.split(",") if w.strip()}
        )

    nlp = _load_spacy_model(language)
    if nlp is not None:
        stop_set.update(nlp.Defaults.stop_words)
    elif language in NLTK_LANGUAGES:
        try:
            stop_set.update(stopwords.words(NLTK_LANGUAGES[language]))
        except LookupError:
            LOGGER.warning(
                "NLTK stopwords for '%s' are not installed. Only custom "
                "stopwords will be applied.",
                language,
            )
    else:
        LOGGER.info(
            "No default stopword list is bundled for '%s'. Only custom "
            "stopwords will be applied.",
            language,
        )

    return stop_set


def split_long_text(text: str, max_chars: int) -> list[str]:
    """
    Split a text into consecutive chunks of at most ``max_chars`` characters.

    Chunks end at the last space before the limit when possible, so words are
    not cut in half. Joining the chunks gives back the original text.

    Args:
        text: The text to split.
        max_chars: The maximum chunk length; must be positive.

    Returns:
        The chunks, in order. Short texts are returned as a single chunk.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            split_at = text.rfind(" ", start, end)
            if split_at > start:
                end = split_at
        chunks.append(text[start:end])
        start = end
    return chunks


def _regex_tokens(text: str, stop_set: set[str]) -> list[str]:
    """
    Tokenize a text with a simple regex, for languages without a spaCy model.

    Args:
        text: The document to tokenize.
        stop_set: Lowercase stopwords to remove.

    Returns:
        Lowercase, non-numeric tokens of at least three characters.
    """
    words = re.findall(r"\b\w{3,}\b", text.lower())
    return [w for w in words if w not in stop_set and not w.isnumeric()]


def _tokenize_texts(
    texts: list[str],
    language: str,
    custom_stopwords_str: str,
    lemmatize: bool,
) -> list[list[str]]:
    """
    Tokenize documents with spaCy when available, otherwise with a regex.

    spaCy refuses texts longer than ``nlp.max_length`` characters, so long
    documents are processed in chunks and their tokens concatenated.

    Args:
        texts: The documents to tokenize.
        language: The language name.
        custom_stopwords_str: A comma-separated string of custom stopwords.
        lemmatize: Whether to return lemmas (``True``) or surface forms.

    Returns:
        One list of lowercase alphabetic tokens (longer than two characters,
        stopwords removed) per document.
    """
    stop_set = get_stopword_set(language, custom_stopwords_str)
    nlp = _load_spacy_model(language)
    if nlp is None:
        return [_regex_tokens(text, stop_set) for text in texts]

    processed_texts: list[list[str]] = [[] for _ in texts]
    chunks = (
        (chunk, index)
        for index, text in enumerate(texts)
        for chunk in split_long_text(text, nlp.max_length)
    )
    for doc, index in nlp.pipe(chunks, as_tuples=True, batch_size=50):
        for token in doc:
            if not token.is_alpha or len(token) <= 2:
                continue
            form = (token.lemma_ if lemmatize else token.text).lower()
            if form not in stop_set:
                processed_texts[index].append(form)

    return processed_texts


def tokenize_texts_for_coherence(
    texts: list[str],
    language: str,
    custom_stopwords_str: str,
) -> list[list[str]]:
    """
    Tokenize texts into surface-form tokens for coherence evaluation.

    Unlike :func:`preprocess_texts_for_lda`, this function does **not**
    lemmatize tokens. It is intended for evaluating models (BERTopic,
    Top2Vec) whose keywords are drawn from the original surface forms of
    the corpus. Using lemmatized tokens for those models causes most
    keywords to be missing from the coherence dictionary and produces
    artificially low Coherence / C_npmi / U_mass scores.

    Tokens are lowercased, restricted to alphabetic tokens of length > 2,
    and filtered by the stopword set for the given language.

    Args:
        texts: The input documents to tokenize.
        language: The language name.
        custom_stopwords_str: A comma-separated string of custom stopwords.

    Returns:
        A list of tokenized documents, where each document is a list of
        surface-form tokens.
    """
    return _tokenize_texts(texts, language, custom_stopwords_str, lemmatize=False)


def preprocess_texts_for_lda(
    texts: list[str],
    language: str,
    custom_stopwords_str: str,
    use_bigrams: bool,
) -> list[list[str]]:
    """
    Preprocess texts for LDA topic modeling.

    Texts are tokenized, lemmatized when a spaCy model is available, filtered
    by stopwords, and optionally enriched with bigrams.

    Args:
        texts: The input documents to preprocess.
        language: The language name.
        custom_stopwords_str: A comma-separated string of custom stopwords.
        use_bigrams: Whether to generate bigrams with gensim.

    Returns:
        A list of tokenized documents, where each document is a list of tokens.
    """
    import gensim

    processed_texts = _tokenize_texts(
        texts, language, custom_stopwords_str, lemmatize=True
    )

    if use_bigrams and processed_texts:
        bigram = gensim.models.Phrases(processed_texts, min_count=5, threshold=10)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        processed_texts = [list(bigram_mod[doc]) for doc in processed_texts]

    return processed_texts
