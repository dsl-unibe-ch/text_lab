"""Streamlit page for topic modeling with BERTopic, Top2Vec and LDA."""

import logging
import traceback
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit.delta_generator import DeltaGenerator

from textlab.ui.streamlit.auth import check_token
from textlab.common import gpu_manager
from textlab.ui.streamlit.components.gpu import free_gpu_for
from textlab.features.topic_modeling.evaluation import evaluate_run
from textlab.features.topic_modeling.topic_config import (
    DEFAULT_TIME_BINS,
    TOP2VEC_BACKEND_LABELS,
    Algorithm,
    TopicModelingConfig,
)
from textlab.features.topic_modeling.topic_pipeline import (
    STABILITY_RUNS,
    evaluate_topic_stability,
    run_topic_modeling_pipeline,
)
from textlab.features.topic_modeling.topic_utils import (
    DEFAULT_EMBEDDING_MODEL_ENGLISH,
    DEFAULT_EMBEDDING_MODEL_MULTILINGUAL,
    SUPPORTED_LANGUAGES,
    TABLE_EXTENSIONS,
    build_results_zip,
    count_docs_exceeding_context,
    drop_empty_text_rows,
    embed_documents,
    generate_metadata_report,
    get_embedding_model_name,
    load_sentence_transformer,
    load_zip_texts,
    prepare_timestamps,
    read_uploaded_table,
    resolve_bertopic_embedding_model_id,
)

LOGGER = logging.getLogger(__name__)

_TABULAR_MODE = "Tabular Data (CSV / Excel)"
_ZIP_MODE = "ZIP Archive (Text files)"

_BASE_METRIC_KEYS = (
    "Topic Diversity",
    "Coherence (C_v)",
    "Coherence (C_npmi)",
    "Coherence (U_mass)",
)

# Share of documents above the embedding context window that triggers a notice.
_TRUNCATION_WARNING_RATIO = 0.10

_METRICS_EXPLANATION = """
**Topic Diversity** — the proportion of unique words across
the top-10 keywords of every topic. It is computed directly
from the *Topic Dictionary* below; no re-processing of the
corpus is involved. Higher values mean topics share fewer
keywords.

**Coherence (C_v, C_npmi, U_mass)** — computed with
Gensim's `CoherenceModel`. To do this, the raw corpus is
re-tokenized into a reference vocabulary that co-occurrence
statistics are read from:

- For **LDA**, the same lemmatized tokens used to train
  the model are reused, so keywords (which are lemmas)
  match the reference dictionary exactly.
- For **BERTopic** and **Top2Vec**, the corpus is
  tokenized to *surface forms* (lowercased, alphabetic,
  stopword-filtered, no lemmatization) because those
  models' keywords are surface tokens from the original
  text. Keywords not present in the reference dictionary
  (e.g. multi-word phrases when n-grams are enabled) are
  skipped, and topics with fewer than 2 remaining
  keywords do not contribute to the score.

Because coherence values depend on the reference
tokenization, they are best used to compare runs on the
**same dataset and language**, not as absolute quality
scores.

**LDA Perplexity** (LDA only) — how well the fitted model
predicts the documents it was trained on
(`2^(−log_perplexity)`). It is not measured on held-out
documents, so it does not show how well the model
generalizes. Lower is better; useful only for comparing LDA
runs on the same data.

**Topic Stability** (when enabled) — the average Jaccard
similarity of the best-matching topics across three
independent runs with unlocked random seeds. Embeddings
(BERTopic) are computed once and reused, so only the
dimensionality-reduction / clustering variance is
measured.
"""


@dataclass
class _DataSelection:
    """The uploaded data and the columns selected for the analysis."""

    filename: str
    df: pd.DataFrame
    text_column: str
    date_column: str | None = None
    time_bins: int = DEFAULT_TIME_BINS


SettingsRenderer = Callable[[DeltaGenerator, DeltaGenerator, str], dict[str, Any]]


@st.cache_data(show_spinner=False, max_entries=2)
def _load_table(filename: str, data: bytes) -> pd.DataFrame:
    """
    Parse an uploaded table once per file, dropping fully empty rows.

    Args:
        filename: The uploaded file name, used to pick the parser.
        data: The raw file content.

    Returns:
        The loaded DataFrame.
    """
    return read_uploaded_table(filename, data).dropna(how="all")


@st.cache_data(show_spinner=False, max_entries=2)
def _load_archive(data: bytes) -> pd.DataFrame:
    """
    Extract the text files of an uploaded ZIP archive once per file.

    Args:
        data: The raw archive content.

    Returns:
        A DataFrame with the columns ``Filename`` and ``Text``.
    """
    return load_zip_texts(data)


def _format_metric(value: float | None, as_percent: bool = False) -> str:
    """
    Format an evaluation metric for display.

    Args:
        value: The metric value, or None if it could not be computed.
        as_percent: Whether to format the value as a percentage.

    Returns:
        The formatted value, or "N/A" for missing values.
    """
    if value is None:
        return "N/A"
    return f"{value:.2%}" if as_percent else f"{value:.4f}"


def _render_data_source_section() -> _DataSelection | None:
    """
    Render the data source section and load the uploaded data.

    Returns:
        The loaded data and selected columns, or None if nothing is uploaded.
    """
    st.header("1. Data Source", divider="gray")

    data_mode = st.radio(
        "Data Format",
        [_TABULAR_MODE, _ZIP_MODE],
        horizontal=True,
        help=(
            "Tabular data is recommended to keep metadata attached to the "
            "extracted topics."
        ),
    )
    is_tabular = data_mode == _TABULAR_MODE

    uploaded_file = st.file_uploader(
        "Upload your file",
        type=list(TABLE_EXTENSIONS) if is_tabular else ["zip"],
    )
    if uploaded_file is None:
        return None

    try:
        if is_tabular:
            df = _load_table(uploaded_file.name, uploaded_file.getvalue())
            if df.empty or len(df.columns) == 0:
                raise ValueError(
                    "The uploaded file does not contain any usable rows or columns."
                )
        else:
            df = _load_archive(uploaded_file.getvalue())
            if df.empty:
                raise ValueError(
                    "The ZIP archive does not contain any non-empty .txt files."
                )
    except Exception as exc:
        st.error(f"Error loading file: {exc}")
        st.stop()

    if is_tabular:
        return _render_column_selection(uploaded_file.name, df)

    st.success(f"Successfully loaded {len(df)} documents from archive.")
    return _DataSelection(filename=uploaded_file.name, df=df, text_column="Text")


def _render_column_selection(filename: str, df: pd.DataFrame) -> _DataSelection:
    """
    Let the user pick the text column and, optionally, a timestamp column.

    Args:
        filename: The uploaded file name.
        df: The loaded table.

    Returns:
        The data selection.
    """
    col_text, col_time = st.columns(2)

    with col_text:
        text_column = st.selectbox(
            "Target Text Column",
            options=df.columns.tolist(),
            help="Select the column containing the raw text you wish to analyze.",
        )

    date_column = None
    time_bins = DEFAULT_TIME_BINS
    with col_time:
        enable_dtm = st.checkbox(
            "Analyze Topics Over Time",
            help=(
                "Used only for BERTopic. Requires a column with "
                "dates, timestamps or whole years."
            ),
        )
        if enable_dtm:
            date_options = [column for column in df.columns if column != text_column]
            if date_options:
                date_column = st.selectbox("Timestamp Column", options=date_options)
                time_bins = int(
                    st.number_input(
                        "Number of Time Bins",
                        min_value=2,
                        max_value=100,
                        value=DEFAULT_TIME_BINS,
                        step=1,
                        help=(
                            "Timestamps are grouped into this many equal-width "
                            "intervals. If there are fewer distinct timestamps, "
                            "each one is kept as its own point."
                        ),
                    )
                )
            else:
                st.info("The table has no other column to use as timestamps.")

    return _DataSelection(
        filename=filename,
        df=df,
        text_column=text_column,
        date_column=date_column,
        time_bins=time_bins,
    )


def _render_custom_stopwords() -> str:
    """
    Render the custom stopword input shared by BERTopic and LDA.

    Returns:
        The comma-separated stopwords entered by the user.
    """
    return st.text_area(
        "Custom Stopwords",
        placeholder="patient, report, conclusion, dataset",
        help=(
            "Comma-separated list of domain-specific words you want "
            "the model to ignore."
        ),
    )


def _render_auto_topic_count(
    max_topics: int = 100,
    show_auto_notice: bool = False,
) -> int | str:
    """
    Render the "auto-detect" checkbox with a manual topic-count fallback.

    Args:
        max_topics: The upper bound of the manual topic-count slider.
        show_auto_notice: Whether to explain what auto-detection does.

    Returns:
        ``"auto"`` or the selected number of topics.
    """
    if st.checkbox("Auto-detect optimal number of topics", value=True):
        if show_auto_notice:
            st.info(
                "The algorithm will dynamically determine the best number "
                "of topics using density-based clustering."
            )
        return "auto"
    return st.slider(
        "Target Number of Topics",
        min_value=2,
        max_value=max_topics,
        value=10,
        step=1,
    )


def _render_bertopic_clustering() -> dict[str, Any]:
    """
    Render the BERTopic clustering engine and topic-count settings.

    Returns:
        The ``clustering_algo``, ``clustering_params`` and ``num_topics``
        configuration fields.
    """
    clustering_choice = st.radio(
        "Clustering Engine",
        [
            "HDBSCAN (Density-based, handles noise)",
            "KMeans (Centroid-based, strict groups)",
        ],
        help=(
            "HDBSCAN dynamically finds clusters and isolates outliers. "
            "KMeans forces every document into a strict number of topics."
        ),
    )

    if "KMeans" in clustering_choice:
        n_clusters = st.slider(
            "Exact Number of Clusters (K)",
            min_value=2,
            max_value=100,
            value=10,
            step=1,
        )
        return {
            "clustering_algo": "KMeans",
            "clustering_params": {"n_clusters": n_clusters},
            "num_topics": None,
        }

    num_topics = _render_auto_topic_count()
    min_cluster_size = st.number_input(
        "Minimum Documents per Topic",
        min_value=3,
        max_value=500,
        value=10,
        step=1,
    )
    return {
        "clustering_algo": "HDBSCAN",
        "clustering_params": {"min_cluster_size": int(min_cluster_size)},
        "num_topics": num_topics,
    }


def _render_embedding_model_choice(language: str) -> dict[str, Any]:
    """
    Render the BERTopic embedding model selection.

    Args:
        language: The selected primary language.

    Returns:
        The ``embedding_model_id``, ``trust_remote_code`` and
        ``chunk_long_documents`` configuration fields.
    """
    embedding_model_id = None
    trust_remote_code = False

    with st.expander("Embedding Model", expanded=False):
        default_model = (
            DEFAULT_EMBEDDING_MODEL_ENGLISH
            if language == "English"
            else DEFAULT_EMBEDDING_MODEL_MULTILINGUAL
        )
        embedding_choice = st.radio(
            "Which sentence-transformer should embed documents?",
            [
                f"Default for language ({default_model})",
                f"English only – {DEFAULT_EMBEDDING_MODEL_ENGLISH} (256 tokens)",
                f"Multilingual – {DEFAULT_EMBEDDING_MODEL_MULTILINGUAL} (128 tokens)",
                "Custom HuggingFace model",
            ],
            help=(
                "The default MiniLM models are already cached on the "
                "cluster. Custom models are downloaded to your own "
                "HuggingFace cache under ~/.cache/huggingface. "
                "For long documents pick a model with a larger "
                "context window (e.g. jinaai/jina-embeddings-v2-base-en, "
                "BAAI/bge-m3, nomic-ai/nomic-embed-text-v1.5)."
            ),
        )
        if embedding_choice.startswith("English only"):
            embedding_model_id = DEFAULT_EMBEDDING_MODEL_ENGLISH
        elif embedding_choice.startswith("Multilingual"):
            embedding_model_id = DEFAULT_EMBEDDING_MODEL_MULTILINGUAL
        elif embedding_choice.startswith("Custom"):
            custom_id = st.text_input(
                "HuggingFace model ID",
                placeholder="e.g. jinaai/jina-embeddings-v2-base-en",
                help=(
                    "First-time use will download the model into "
                    "your home cache. Requires network access from "
                    "the compute node."
                ),
            )
            embedding_model_id = custom_id.strip() or None
            trust_remote_code = st.checkbox(
                "Allow custom code from this model repository",
                value=False,
                help=(
                    "Some long-context models (e.g. Jina, Nomic) ship their own "
                    "Python code. Enable this only for repositories you trust: "
                    "the code runs on the compute node under your account."
                ),
            )

        chunk_long_documents = st.checkbox(
            "Embed long documents in chunks",
            value=False,
            help=(
                "Documents longer than the model's context window are split "
                "into pieces that fit, each piece is embedded, and the "
                "results are averaged, so the whole document shapes its "
                "topic instead of only its beginning. Encoding takes longer "
                "for long documents."
            ),
        )

    return {
        "embedding_model_id": embedding_model_id,
        "trust_remote_code": trust_remote_code,
        "chunk_long_documents": chunk_long_documents,
    }


def _render_vocabulary_settings() -> dict[str, Any]:
    """
    Render the BERTopic keyword extraction settings.

    Returns:
        The ``ngram_range``, ``min_df`` and ``reduce_frequent`` configuration
        fields.
    """
    with st.expander("Text Extraction & Vocabulary", expanded=True):
        extract_phrases = st.checkbox("Extract Phrases (N-grams)", value=False)
        min_df = st.number_input(
            "Minimum Topic Frequency (min_df)",
            min_value=1,
            max_value=500,
            value=1,
            help=(
                "A word is only used as a keyword if it appears "
                "in at least this many topics. BERTopic applies "
                "this to topics, not documents, so it must not "
                "exceed the number of topics found. Higher values "
                "reduce memory on very large vocabularies but "
                "remove topic-specific words; keep 1 unless "
                "memory is an issue."
            ),
        )
        reduce_frequent = st.checkbox(
            "Auto-penalize frequent words",
            value=True,
            help=(
                "Uses ClassTfidfTransformer to reduce the impact "
                "of common filler words without explicitly "
                "deleting them."
            ),
        )
    return {
        "ngram_range": (1, 2) if extract_phrases else (1, 1),
        "min_df": min_df,
        "reduce_frequent": reduce_frequent,
    }


def _render_dim_reduction_settings() -> dict[str, Any]:
    """
    Render the BERTopic dimensionality reduction settings.

    Returns:
        The ``dim_reduction_algo`` and ``dim_params`` configuration fields.
    """
    dim_mapping = {
        "UMAP (Recommended)": "UMAP",
        "PCA (Fast & Linear)": "PCA",
        "Truncated SVD": "Truncated SVD",
        "None (Skip reduction)": "None",
    }
    dim_params: dict[str, Any] = {}

    with st.expander("Dimensionality Reduction"):
        dim_reduction_raw = st.radio(
            "Algorithm",
            options=list(dim_mapping.keys()),
            help=(
                "UMAP preserves global and local structure. "
                "PCA/SVD are faster but linear. "
                "'None' skips this step entirely."
            ),
        )
        dim_reduction_algo = dim_mapping[dim_reduction_raw]

        if dim_reduction_algo != "None":
            dim_params["n_components"] = st.slider("Target Dimensions", 2, 50, 5)

            if dim_reduction_algo == "UMAP":
                dim_params["n_neighbors"] = st.slider(
                    "Neighbors (Local vs Global)",
                    2,
                    200,
                    15,
                    help=(
                        "Balances local vs global structure. "
                        "High values preserve global structure."
                    ),
                )
                dim_params["min_dist"] = st.slider(
                    "Minimum Distance (Tightness)",
                    0.0,
                    1.0,
                    0.0,
                    step=0.01,
                )

            lock_seed = st.checkbox("Lock Seed for Reproducibility", value=True)
            dim_params["random_state"] = 42 if lock_seed else None

    return {"dim_reduction_algo": dim_reduction_algo, "dim_params": dim_params}


def _render_outlier_settings(
    clustering_algo: str,
    min_cluster_size: int,
) -> tuple[int | None, bool]:
    """
    Render the HDBSCAN outlier settings.

    Args:
        clustering_algo: The selected clustering engine.
        min_cluster_size: The selected minimum topic size, used as the
            default outlier sensitivity.

    Returns:
        A tuple ``(min_samples, reduce_outliers)``; ``min_samples`` is None
        when the settings do not apply (KMeans).
    """
    with st.expander("Clustering & Outliers (HDBSCAN)"):
        if clustering_algo != "HDBSCAN":
            st.info("Outlier settings only apply to HDBSCAN.")
            return None, False

        min_samples = st.number_input(
            "Outlier Sensitivity (min_samples)",
            min_value=1,
            max_value=500,
            value=min_cluster_size,
            help="Lower values reduce noise/outliers.",
        )
        reduce_outliers = st.checkbox(
            "Force-assign outliers (-1) to nearest topics",
            value=False,
        )
    return min_samples, reduce_outliers


def _render_bertopic_settings(
    col_basic: DeltaGenerator,
    col_adv: DeltaGenerator,
    language: str,
) -> dict[str, Any]:
    """
    Render all BERTopic settings.

    Args:
        col_basic: The column for core settings.
        col_adv: The column for advanced settings.
        language: The selected primary language.

    Returns:
        The BERTopic-specific configuration fields.
    """
    with col_basic:
        clustering = _render_bertopic_clustering()

    with col_adv:
        settings = {
            **clustering,
            "custom_stopwords": _render_custom_stopwords(),
            **_render_embedding_model_choice(language),
            **_render_vocabulary_settings(),
            **_render_dim_reduction_settings(),
        }
        min_samples, reduce_outliers = _render_outlier_settings(
            clustering["clustering_algo"],
            clustering["clustering_params"].get("min_cluster_size", 10),
        )

    if min_samples is not None:
        settings["clustering_params"]["min_samples"] = min_samples
    settings["reduce_outliers"] = reduce_outliers
    return settings


def _render_top2vec_settings(
    col_basic: DeltaGenerator,
    col_adv: DeltaGenerator,
    language: str,
) -> dict[str, Any]:
    """
    Render all Top2Vec settings.

    Args:
        col_basic: The column for core settings.
        col_adv: The column for advanced settings.
        language: The selected primary language (unused; Top2Vec picks its
            embedding model from the language in the engine).

    Returns:
        The Top2Vec-specific configuration fields.
    """
    with col_basic:
        num_topics = _render_auto_topic_count(show_auto_notice=True)

    with col_adv:
        st.info(
            "Top2Vec relies on the natural, raw structure of sentences "
            "to generate joint embeddings. Custom stopwords are disabled "
            "for this algorithm."
        )
        backend_by_label = {
            label: backend for backend, label in TOP2VEC_BACKEND_LABELS.items()
        }
        backend_label = st.radio(
            "Embedding Backend",
            [TOP2VEC_BACKEND_LABELS["transformer"], TOP2VEC_BACKEND_LABELS["doc2vec"]],
            help=(
                "Transformers are faster and understand general language. "
                "Doc2Vec trains specifically on your data."
            ),
        )
        speed = st.select_slider(
            "Training Depth",
            options=["fast-learn", "learn", "deep-learn"],
            value="learn",
        )
        min_count = st.number_input(
            "Minimum Word Count (min_count)",
            min_value=1,
            max_value=500,
            value=10,
            step=1,
            help=(
                "Words that appear fewer times than this in the whole "
                "collection are ignored. Lower it for small collections or "
                "if Top2Vec finds no topics; raise it to ignore rare words "
                "in large collections."
            ),
        )

    return {
        "num_topics": num_topics,
        "top2vec_backend": backend_by_label[backend_label],
        "top2vec_speed": speed,
        "top2vec_min_count": int(min_count),
    }


def _render_lda_settings(
    col_basic: DeltaGenerator,
    col_adv: DeltaGenerator,
    language: str,
) -> dict[str, Any]:
    """
    Render all LDA settings.

    Args:
        col_basic: The column for core settings.
        col_adv: The column for advanced settings.
        language: The selected primary language (unused; LDA preprocessing
            reads it from the configuration).

    Returns:
        The LDA-specific configuration fields.
    """
    with col_basic:
        num_topics = st.slider(
            "Number of Topics",
            min_value=2,
            max_value=50,
            value=10,
            step=1,
        )

    with col_adv:
        custom_stopwords = _render_custom_stopwords()
        use_bigrams = st.checkbox("Extract Phrases (Bigrams)", value=False)
        passes = st.slider(
            "Training Passes (Iterations)",
            min_value=5,
            max_value=50,
            value=10,
            step=5,
        )

    return {
        "num_topics": num_topics,
        "custom_stopwords": custom_stopwords,
        "use_bigrams": use_bigrams,
        "passes": passes,
    }


_SETTINGS_RENDERERS: dict[Algorithm, SettingsRenderer] = {
    Algorithm.BERTOPIC: _render_bertopic_settings,
    Algorithm.TOP2VEC: _render_top2vec_settings,
    Algorithm.LDA: _render_lda_settings,
}


def _render_model_configuration(
    data: _DataSelection | None,
) -> tuple[TopicModelingConfig, bool]:
    """
    Render the model configuration section and collect user selections.

    Args:
        data: The uploaded data selection, or None if nothing is uploaded.

    Returns:
        A tuple containing:
            - TopicModelingConfig instance with the selected options.
            - Boolean indicating if Topic Stability execution is requested.
    """
    st.header("2. Model Configuration", divider="gray")

    algorithm = st.radio(
        "Select Algorithm",
        options=list(Algorithm),
        format_func=lambda option: option.label,
        help=(
            "BERTopic and Top2Vec use modern AI semantic models. "
            "LDA uses traditional statistical term frequencies."
        ),
    )

    date_column = data.date_column if data is not None else None
    if date_column and algorithm != Algorithm.BERTOPIC:
        st.info("Topics-over-time analysis is only applied for BERTopic in this page.")

    col_basic, col_adv = st.columns(2)

    with col_basic:
        st.subheader("Core Settings")
        language = st.selectbox(
            "Primary Language",
            SUPPORTED_LANGUAGES,
            help="Determines the underlying embedding model used for the analysis.",
        )

    with col_adv:
        st.subheader("Advanced Processing")
        with st.expander("Academic Evaluation Metrics", expanded=False):
            run_stability = st.checkbox(
                "Evaluate Topic Stability / Reproducibility",
                value=False,
                help=(
                    f"Runs the model {STABILITY_RUNS} times and calculates "
                    "Jaccard Similarity to prove stability. WARNING: Triples "
                    "execution time."
                ),
            )

    settings = _SETTINGS_RENDERERS[algorithm](col_basic, col_adv, language)

    config = TopicModelingConfig(
        algorithm=algorithm,
        language=language,
        text_column=data.text_column if data is not None else "",
        enable_dtm=date_column is not None,
        date_column=date_column,
        time_bins=data.time_bins if data is not None else DEFAULT_TIME_BINS,
        **settings,
    )
    return config, run_stability


def _report_long_documents(
    embedding_model: Any,
    model_id: str,
    raw_texts: list[str],
    chunk_long_documents: bool,
) -> None:
    """
    Tell the user how documents above the context window will be handled.

    Nothing is shown when few documents are affected.

    Args:
        embedding_model: The loaded SentenceTransformer.
        model_id: The model ID shown in the message.
        raw_texts: The documents to check.
        chunk_long_documents: Whether long documents are embedded in chunks.
    """
    over_count, total_count, max_seq_length = count_docs_exceeding_context(
        embedding_model, raw_texts
    )
    if not total_count or over_count / total_count <= _TRUNCATION_WARNING_RATIO:
        return

    summary = (
        f"**{over_count} of {total_count} documents "
        f"({over_count / total_count:.0%}) exceed the "
        f"{max_seq_length}-token context window of `{model_id}`.**"
    )
    larger_models = (
        "`jinaai/jina-embeddings-v2-base-en` (8192 tokens), "
        "`BAAI/bge-m3` (8192 tokens) or "
        "`nomic-ai/nomic-embed-text-v1.5` (8192 tokens)"
    )

    if chunk_long_documents:
        st.info(
            f"{summary} They will be split into chunks that fit the window "
            "and the chunk embeddings averaged, so encoding takes longer. "
            "A model with a larger context window needs fewer chunks, for "
            f"example {larger_models}."
        )
        return

    st.warning(
        f"{summary} Only the leading portion of each of those documents "
        "will be used to generate the topic embedding, which may bias topic "
        "assignments toward document openings.\n\n"
        "To capture the full content of long documents, either:\n\n"
        "- enable *Embed long documents in chunks* in the *Embedding Model* "
        "section, which embeds each document piece by piece and averages "
        "the results, or\n"
        "- pick a sentence-transformer with a larger context window from the "
        f"same section, for example {larger_models}. Custom models are "
        "downloaded to your own home cache (`~/.cache/huggingface`)."
    )


def _encode_documents(
    config: TopicModelingConfig,
    raw_texts: list[str],
) -> tuple[Any, np.ndarray]:
    """
    Load the BERTopic embedding model and embed the documents once.

    The embeddings are reused by the optional stability runs.

    Args:
        config: The topic modeling configuration.
        raw_texts: The documents to embed.

    Returns:
        A tuple ``(embedding_model, embeddings)``.

    Raises:
        ValueError: If the embedding model cannot be loaded.
    """
    model_id = resolve_bertopic_embedding_model_id(config)
    try:
        with st.spinner(f"Loading embedding model '{model_id}'..."):
            embedding_model = load_sentence_transformer(
                model_id,
                trust_remote_code=config.trust_remote_code,
            )
    except Exception as exc:
        message = f"Failed to load embedding model '{model_id}': {exc}"
        if "trust_remote_code" in str(exc) and not config.trust_remote_code:
            message += (
                "\n\nThis model ships its own code. If you trust the "
                "repository, enable 'Allow custom code from this model "
                "repository' in the Embedding Model section and run again."
            )
        raise ValueError(message) from exc

    _report_long_documents(
        embedding_model, model_id, raw_texts, config.chunk_long_documents
    )

    with st.spinner("Encoding documents with the embedding model..."):
        embeddings = embed_documents(
            embedding_model,
            raw_texts,
            chunk_long_documents=config.chunk_long_documents,
        )
    return embedding_model, embeddings


def _run_analysis(
    data: _DataSelection,
    config: TopicModelingConfig,
    run_stability: bool,
) -> dict[str, Any]:
    """
    Train the selected model, evaluate it and package the results.

    Args:
        data: The uploaded data selection.
        config: The topic modeling configuration.
        run_stability: Whether to run the topic stability evaluation.

    Returns:
        The results stored in the session state for rendering.
    """
    prepared_df = drop_empty_text_rows(data.df, config.text_column)

    timestamps = None
    if config.date_column and config.algorithm == Algorithm.BERTOPIC:
        prepared_df, timestamps, dropped = prepare_timestamps(
            prepared_df,
            config.date_column,
        )
        if dropped:
            st.warning(
                f"{dropped} rows were skipped because "
                f"'{config.date_column}' could not be parsed as a date/time."
            )

    raw_texts = prepared_df[config.text_column].astype(str).tolist()

    embedding_model = None
    embeddings = None
    if config.algorithm == Algorithm.BERTOPIC:
        embedding_model, embeddings = _encode_documents(config, raw_texts)

    spinner_msg = (
        f"Running topic extraction (Run 1/{STABILITY_RUNS})..."
        if run_stability
        else "Running topic extraction..."
    )
    with st.spinner(spinner_msg):
        run_result = run_topic_modeling_pipeline(
            prepared_df,
            config,
            timestamps=timestamps,
            embedding_model=embedding_model,
            precomputed_embeddings=embeddings,
        )

    with st.spinner("Calculating Topic Coherence and Diversity..."):
        evaluation_metrics = evaluate_run(run_result, raw_texts, config)

    if run_stability:
        with st.spinner(
            f"Running stability iterations 2-{STABILITY_RUNS} (unlocked seeds) "
            "and comparing topics..."
        ):
            evaluation_metrics["Topic Stability"] = evaluate_topic_stability(
                prepared_df,
                config,
                run_result["topic_keywords"],
                embedding_model=embedding_model,
                precomputed_embeddings=embeddings,
            )

    metadata_report = generate_metadata_report(
        filename=data.filename,
        config=config,
        embedding_model_name=get_embedding_model_name(config),
        evaluation_metrics=evaluation_metrics,
    )
    zip_bytes = build_results_zip(
        metadata_report=metadata_report,
        docs_df=run_result["docs_df"],
        topic_df=run_result["topic_df"],
        dashboard_assets=run_result["dashboard_assets"],
    )

    return {
        "topic_df": run_result["topic_df"],
        "dashboard_assets": run_result["dashboard_assets"],
        "evaluation_metrics": evaluation_metrics,
        "algorithm": config.algorithm,
        "enable_dtm": config.enable_dtm,
        "zip_bytes": zip_bytes,
    }


def _render_evaluation_metrics(metrics: dict[str, float | None]) -> None:
    """
    Render the evaluation metric cards and their explanation.

    Args:
        metrics: The evaluation metrics of the run.
    """
    st.subheader("Model Evaluation Metrics")
    st.caption("Quantitative metrics to compare hyperparameter performance.")

    with st.expander("How are these scores computed?", expanded=False):
        st.markdown(_METRICS_EXPLANATION)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Topic Diversity",
        _format_metric(metrics.get("Topic Diversity"), as_percent=True),
        help="Percentage of unique words across all topics (Higher is better).",
    )
    col2.metric(
        "Coherence (C_v)",
        _format_metric(metrics.get("Coherence (C_v)")),
        help=(
            "Highly correlated with human interpretability. "
            "Range 0 to 1 (Higher is better)."
        ),
    )
    col3.metric(
        "Coherence (C_npmi)",
        _format_metric(metrics.get("Coherence (C_npmi)")),
        help=(
            "Normalized Pointwise Mutual Information. "
            "Typically -1 to 1 (Higher is better)."
        ),
    )
    col4.metric(
        "Coherence (U_mass)",
        _format_metric(metrics.get("Coherence (U_mass)")),
        help=(
            "Measures word co-occurrence within the corpus. "
            "Typically negative (Closer to 0 is better)."
        ),
    )
    if any(metrics.get(key) is None for key in _BASE_METRIC_KEYS):
        st.caption(
            "N/A: the metric could not be computed for this run, for "
            "example because too few topic keywords occur in the corpus."
        )

    # Dynamic Extra Metrics (Perplexity / Stability)
    extra_keys = [key for key in metrics if key not in _BASE_METRIC_KEYS]
    if extra_keys:
        extra_cols = st.columns(len(extra_keys))
        for col, key in zip(extra_cols, extra_keys):
            if "Stability" in key:
                col.metric(
                    key,
                    _format_metric(metrics[key], as_percent=True),
                    help=(
                        f"Jaccard Similarity across {STABILITY_RUNS} runs. "
                        "100% means perfectly reproducible."
                    ),
                )
            else:
                col.metric(
                    key,
                    _format_metric(metrics[key]),
                    help=(
                        "Statistical measure of prediction accuracy. "
                        "Lower is better."
                    ),
                )

    st.divider()


def _render_html_asset(
    assets: dict[str, str],
    name: str,
    width: int = 1000,
    height: int = 600,
    scrolling: bool = False,
) -> None:
    """
    Embed one HTML dashboard asset.

    Args:
        assets: The dashboard assets of the run.
        name: The asset file name.
        width: The frame width in pixels.
        height: The frame height in pixels.
        scrolling: Whether the frame may scroll.
    """
    components.html(
        assets.get(name, ""),
        width=width,
        height=height,
        scrolling=scrolling,
    )


def _render_dashboards(res: dict[str, Any]) -> None:
    """
    Render the interactive dashboards of the run.

    Args:
        res: The results stored in the session state.
    """
    assets = res["dashboard_assets"]

    if res["algorithm"] == Algorithm.LDA:
        st.subheader("Interactive Topic Dashboard")
        _render_html_asset(assets, "lda_dashboard.html", width=1300, height=800)
        return

    if res["algorithm"] == Algorithm.TOP2VEC:
        st.subheader("Interactive Topic Dashboard")
        _render_html_asset(
            assets, "top2vec_barchart.html", height=800, scrolling=True
        )
        return

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Intertopic Distance",
            "Word Scores",
            "Similarity Heatmap",
            "Topics Over Time",
        ]
    )

    with tab1:
        st.caption("Maps the semantic distance between discovered topics.")
        _render_html_asset(assets, "intertopic_distance.html")

    with tab2:
        st.caption("Displays the highest frequency terms for the top topics.")
        _render_html_asset(assets, "topic_barchart.html")

    with tab3:
        st.caption(
            "Shows how semantically similar the generated topics are to each other."
        )
        _render_html_asset(assets, "similarity_heatmap.html")

    with tab4:
        if res["enable_dtm"] and "topics_over_time.html" in assets:
            st.caption(
                "Visualizes topic frequency evolution over the provided timestamps."
            )
            _render_html_asset(assets, "topics_over_time.html")
        else:
            st.info("Dynamic Topic Modeling was not enabled during configuration.")


def _render_results(res: dict[str, Any]) -> None:
    """
    Render the results section from session state.

    Args:
        res: The results stored in the session state.
    """
    st.header("Results Analysis", divider="gray")

    if res["evaluation_metrics"]:
        _render_evaluation_metrics(res["evaluation_metrics"])

    st.subheader("Topic Dictionary")
    if res["algorithm"] in (Algorithm.BERTOPIC, Algorithm.TOP2VEC):
        st.caption(
            "Note: Density-based algorithms automatically classify outlier "
            "documents into an 'Outlier' category."
        )
    st.dataframe(res["topic_df"], use_container_width=True, hide_index=True)

    _render_dashboards(res)

    st.subheader("Export Artifacts")
    st.write(
        "Download your original dataset augmented with topic classifications, "
        "alongside the standalone interactive HTML dashboards and metadata "
        "report."
    )
    st.download_button(
        label="Download Extraction Package (.zip)",
        data=res["zip_bytes"],
        file_name="Topic_Modeling_Artifacts.zip",
        mime="application/zip",
        type="primary",
    )


def main() -> None:
    """
    Render and run the Topic Modeling Streamlit page.
    """
    st.set_page_config(page_title="Topic Modeling", layout="wide")
    check_token()

    if "topic_results" not in st.session_state:
        st.session_state.topic_results = None

    st.title("Topic Modeling")
    st.markdown(
        "Discover hidden themes in large text datasets automatically. "
        "Upload your data to generate an interactive topic distribution map."
    )

    data = _render_data_source_section()
    config, run_stability = _render_model_configuration(data)

    st.header("3. Execution", divider="gray")

    if st.button("Run Topic Extraction", type="primary", disabled=data is None):
        free_gpu_for(gpu_manager.TOPIC_MODELING)
        try:
            st.session_state.topic_results = _run_analysis(data, config, run_stability)
            st.success("Topic Modeling execution complete.")

        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        except Exception:
            st.error("Topic modeling failed. See technical details below:")
            st.code(traceback.format_exc())
            LOGGER.exception("Topic modeling failed.")
            st.stop()

    if st.session_state.topic_results is not None:
        _render_results(st.session_state.topic_results)


if __name__ == "__main__":
    main()
