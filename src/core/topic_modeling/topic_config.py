from dataclasses import dataclass, field
from typing import Any, TypedDict, NotRequired

import pandas as pd


@dataclass
class TopicModelingConfig:
    """
    Store topic modeling configuration selected from the UI.
    """

    algorithm: str
    language: str
    text_column: str
    enable_dtm: bool = False
    date_column: str | None = None
    custom_stopwords: str = ""
    num_topics: int | str = 10
    bertopic_nr_topics: int | str | None = None
    dim_reduction_algo: str = "UMAP"
    dim_params: dict[str, Any] = field(default_factory=dict)
    clustering_algo: str = "HDBSCAN"
    clustering_params: dict[str, Any] = field(default_factory=dict)
    ngram_range: tuple[int, int] = (1, 1)
    min_df: int = 1
    reduce_frequent: bool = True
    reduce_outliers: bool = False
    top2vec_backend: str = "doc2vec"
    top2vec_backend_label: str = "Doc2Vec (Train from scratch)"
    top2vec_speed: str = "learn"
    use_bigrams: bool = False
    passes: int = 10
    random_state: int | None = 42
    # Optional HuggingFace sentence-transformer model ID for BERTopic embeddings.
    # None → use the language-based default (MiniLM). Custom IDs are downloaded
    # to the user's home cache rather than the shared read-only model cache.
    embedding_model_id: str | None = None


class TopicModelingRunResult(TypedDict):
    """
    Store topic modeling outputs returned by the execution pipeline.
    """

    topic_df: pd.DataFrame
    docs_df: pd.DataFrame
    dashboard_assets: dict[str, str]

    # Optional keys returned only by the LDA pipeline for Perplexity evaluation
    lda_model: NotRequired[Any]
    corpus: NotRequired[Any]
    # Pre-tokenized corpus, if the pipeline already tokenized for training —
    # allows :func:`evaluate_topic_quality` to skip re-tokenizing.
    tokenized_texts: NotRequired[list[list[str]]]