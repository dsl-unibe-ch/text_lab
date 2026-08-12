import datetime
import io
import os
import sys
import re
import zipfile
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set

import nltk
import pandas as pd
import spacy
from nltk.corpus import stopwords

from core import upload_safety
from .topic_config import TopicModelingConfig

if "NLTK_DATA" in os.environ:
    nltk.data.path.append(os.environ["NLTK_DATA"])

SUPPORTED_LANGUAGES: List[str] = [
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

SPACY_MODELS: Dict[str, str] = {
    "English": "en_core_web_sm",
    "German": "de_core_news_sm",
    "French": "fr_core_news_sm",
}

NLTK_LANGUAGES: Dict[str, str] = {
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
SHARED_EMBEDDING_MODELS: Set[str] = {
    "all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}

DEFAULT_EMBEDDING_MODEL_ENGLISH: str = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_MODEL_MULTILINGUAL: str = "paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=None)
def _load_spacy_model(language: str) -> Optional[spacy.language.Language]:
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
        print(
            f"WARNING (Text Lab): spaCy model '{model_name}' for {language} is not installed! "
            f"Please run 'python -m spacy download {model_name}' in the host environment. "
            f"Falling back to basic regex tokenization for this job.",
            file=sys.stderr
        )
        return None


def read_uploaded_table(filename: str, uploaded_file: Any) -> pd.DataFrame:
    """
    Read an uploaded CSV or Excel file into a pandas DataFrame.

    Args:
        filename: The uploaded file name.
        uploaded_file: The uploaded file object.

    Returns:
        The loaded pandas DataFrame.

    Raises:
        ValueError: If the file format is not supported.
    """
    lower_name = filename.lower()
    if lower_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if lower_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported tabular file format.")


def load_zip_texts(zip_bytes: bytes) -> pd.DataFrame:
    """
    Load non-empty text files from a ZIP archive into a DataFrame.

    Files whose basename starts with ``._`` are ignored.

    Args:
        zip_bytes: The ZIP archive content as bytes.

    Returns:
        A pandas DataFrame with columns ``Filename`` and ``Text``.
    """
    data = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        for info in upload_safety.safe_zip_members(z, allowed_extensions={".txt"}):
            filename = info.filename
            lower_name = filename.lower()
            if lower_name.endswith(".txt") and not os.path.basename(
                filename
            ).startswith("._"):
                content = z.read(info).decode("utf-8", errors="ignore")
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


def prepare_timestamps(
    df: pd.DataFrame,
    date_column: str,
) -> tuple[pd.DataFrame, list[pd.Timestamp], int]:
    """
    Parse and validate timestamps from a selected date column.

    Args:
        df: The input DataFrame.
        date_column: The column containing timestamps or dates.

    Returns:
        A tuple containing:
            - The filtered and sorted DataFrame
            - A list of parsed timestamps
            - The number of dropped rows

    Raises:
        ValueError: If no valid timestamps remain after parsing.
    """
    parsed = pd.to_datetime(df[date_column], errors="coerce", utc=True).dt.tz_localize(
        None
    )
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


def validate_minimum_documents(texts: List[str], minimum_docs: int = 5) -> None:
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


def get_embedding_model_name(config: TopicModelingConfig) -> str:
    """
    Resolve the human-readable embedding model name for the selected configuration.

    Args:
        config: The topic modeling configuration.

    Returns:
        The resolved embedding model name.
    """
    if "BERTopic" in config.algorithm:
        return resolve_bertopic_embedding_model_id(config)

    if "Top2Vec" in config.algorithm:
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
    Resolve the concrete HuggingFace sentence-transformer model ID that BERTopic
    should use, based on the user's configuration.

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
    Return True if the given HuggingFace model ID is expected to be present in
    the shared read-only cache. Non-shared models must be downloaded to the
    calling user's home cache directory.
    """
    return model_id in SHARED_EMBEDDING_MODELS


def get_user_hf_cache_dir() -> str:
    """
    Return the path to the calling user's writable HuggingFace cache directory,
    creating it if necessary.

    This is used for user-selected custom embedding models so they do not need
    write access to the shared model cache.
    """
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError as e:
        print(
            f"WARNING (Text Lab): Could not create user HF cache dir '{cache_dir}': {e}",
            file=sys.stderr,
        )
    return cache_dir


def load_sentence_transformer(model_id: str) -> Any:
    """
    Instantiate a SentenceTransformer for the given model ID.

    Shared/curated models resolve through the container-level HF_HOME (the
    read-only shared cache). Custom models are directed to the user's own
    HuggingFace cache directory so they can be downloaded without write access
    to the shared cache.

    Args:
        model_id: The HuggingFace sentence-transformer model ID.

    Returns:
        A ready-to-use SentenceTransformer instance.
    """
    from sentence_transformers import SentenceTransformer

    if is_shared_embedding_model(model_id):
        return SentenceTransformer(model_id)

    # Custom user-selected model — download into the user's own cache and allow
    # remote code so common long-context models (Jina, Nomic, …) work.
    return SentenceTransformer(
        model_id,
        cache_folder=get_user_hf_cache_dir(),
        trust_remote_code=True,
    )


def count_docs_exceeding_context(
    embedding_model: Any,
    texts: List[str],
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


def generate_metadata_report(
    filename: str,
    config: TopicModelingConfig,
    embedding_model_name: str,
    evaluation_metrics: dict[str, float] | None = None
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
    report = [
        "=========================================",
        " TEXT LAB - TOPIC MODELING CONFIGURATION ",
        "=========================================",
        f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Source File: {filename}",
        f"Target Text Column: {config.text_column}",
    ]

    if config.enable_dtm and config.date_column:
        report.append(f"Timestamp Column (DTM): {config.date_column}")

    report.extend(
        [
            "",
            "--- CORE SETTINGS ---",
            f"Framework: {config.algorithm}",
            f"Primary Language: {config.language}",
            (
                "Custom Stopwords: "
                f"{config.custom_stopwords if config.custom_stopwords.strip() else 'None'}"
            ),
            "",
        ]
    )

    if "LDA" in config.algorithm:
        report.extend(
            [
                "--- LDA PARAMETERS ---",
                f"Number of Topics: {config.num_topics}",
                f"Training Passes: {config.passes}",
                f"Extract Bigrams: {config.use_bigrams}",
            ]
        )

    elif "Top2Vec" in config.algorithm:
        report.extend(
            [
                "--- TOP2VEC PARAMETERS ---",
                f"Target Topics: {config.num_topics}",
                f"Embedding Backend: {config.top2vec_backend_label}",
                f"Embedding Model: {embedding_model_name}",
                f"Training Speed: {config.top2vec_speed}",
            ]
        )

    else:
        report.extend(
            [
                "--- BERTOPIC PARAMETERS ---",
                f"Target Topics: {config.bertopic_nr_topics}",
                f"Embedding Model: {embedding_model_name}",
                f"N-Gram Range: {config.ngram_range}",
                f"Min Document Frequency (min_df): {config.min_df}",
                (
                    "Reduce Frequent Words (ClassTfidfTransformer): "
                    f"{config.reduce_frequent}"
                ),
                "",
                f"--- DIMENSIONALITY REDUCTION ({config.dim_reduction_algo}) ---",
            ]
        )

        if config.dim_reduction_algo != "None":
            report.append(f"N Components: {config.dim_params.get('n_components')}")
            if config.dim_reduction_algo == "UMAP":
                report.append(f"N Neighbors: {config.dim_params.get('n_neighbors')}")
                report.append(f"Min Distance: {config.dim_params.get('min_dist')}")
            report.append(
                f"Random State: {config.dim_params.get('random_state', 'None')}"
            )

        report.extend(
            [
                "",
                f"--- CLUSTERING ({config.clustering_algo}) ---",
            ]
        )

        if config.clustering_algo == "KMeans":
            report.append(f"N Clusters: {config.clustering_params.get('n_clusters')}")
        else:
            report.extend(
                [
                    (
                        "Min Cluster Size: "
                        f"{config.clustering_params.get('min_cluster_size')}"
                    ),
                    (
                        "Min Samples: "
                        f"{config.clustering_params.get('min_samples', 'Default (equals min_cluster_size)')}"
                    ),
                    f"Force-reduce Outliers: {config.reduce_outliers}",
                ]
            )

    if evaluation_metrics:
        report.extend([
            "",
            "=========================================",
            " MODEL EVALUATION METRICS                ",
            "========================================="
        ])
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


def get_stopword_set(language: str, custom_stopwords_str: str) -> Set[str]:
    """
    Build a stopword set from custom, spaCy, and NLTK sources.

    Args:
        language: The language name.
        custom_stopwords_str: A comma-separated string of custom stopwords.

    Returns:
        A set of lowercase stopwords.
    """
    stop_set: Set[str] = set()

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
            print(
                f"WARNING (Text Lab): NLTK stopwords for '{language}' are not "
                "installed. Only custom stopwords will be applied.",
                file=sys.stderr,
            )
    else:
        print(
            f"NOTICE (Text Lab): No default stopword list is bundled for "
            f"'{language}'. Only custom stopwords will be applied.",
            file=sys.stderr,
        )

    return stop_set


def tokenize_texts_for_coherence(
    texts: List[str],
    language: str,
    custom_stopwords_str: str,
) -> List[List[str]]:
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
    stop_set = get_stopword_set(language, custom_stopwords_str)
    processed_texts: List[List[str]] = []

    nlp = _load_spacy_model(language)
    if nlp is not None:
        for doc in nlp.pipe(texts, batch_size=50):
            tokens = [
                token.text.lower()
                for token in doc
                if token.is_alpha
                and len(token) > 2
                and token.text.lower() not in stop_set
            ]
            processed_texts.append(tokens)
    else:
        for text in texts:
            words = re.findall(r"\b\w{3,}\b", text.lower())
            tokens = [w for w in words if w not in stop_set and not w.isnumeric()]
            processed_texts.append(tokens)

    return processed_texts


def preprocess_texts_for_lda(
    texts: List[str],
    language: str,
    custom_stopwords_str: str,
    use_bigrams: bool,
) -> List[List[str]]:
    """
    Preprocess texts for LDA topic modeling.

    Texts are tokenized, normalized, filtered by stopwords, and optionally
    enriched with bigrams.

    Args:
        texts: The input documents to preprocess.
        language: The language name.
        custom_stopwords_str: A comma-separated string of custom stopwords.
        use_bigrams: Whether to generate bigrams with gensim.

    Returns:
        A list of tokenized documents, where each document is a list of tokens.
    """
    import gensim

    stop_set = get_stopword_set(language, custom_stopwords_str)
    processed_texts: List[List[str]] = []

    nlp = _load_spacy_model(language)
    if nlp is not None:
        for doc in nlp.pipe(texts, batch_size=50):
            tokens = [
                token.lemma_.lower()
                for token in doc
                if token.is_alpha
                and len(token) > 2
                and token.lemma_.lower() not in stop_set
            ]
            processed_texts.append(tokens)
    else:
        for text in texts:
            words = re.findall(r"\b\w{3,}\b", text.lower())
            tokens = [w for w in words if w not in stop_set and not w.isnumeric()]
            processed_texts.append(tokens)

    if use_bigrams and processed_texts:
        bigram = gensim.models.Phrases(processed_texts, min_count=5, threshold=10)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        processed_texts = [list(bigram_mod[doc]) for doc in processed_texts]

    return processed_texts