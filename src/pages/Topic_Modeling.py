import os
import sys
import traceback
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Ensure absolute imports resolve correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from auth import check_token
from core.topic_modeling.topic_config import TopicModelingConfig
from core.topic_modeling.topic_pipeline import run_topic_modeling_pipeline
from core.topic_modeling.evaluation import evaluate_topic_quality
from core.topic_modeling.topic_utils import (
    SUPPORTED_LANGUAGES,
    build_results_zip,
    drop_empty_text_rows,
    generate_metadata_report,
    get_embedding_model_name,
    load_zip_texts,
    prepare_timestamps,
    read_uploaded_table,
)


def _render_data_source_section(
) -> tuple[UploadedFile | None, pd.DataFrame | None, str | None, str | None, bool]:
    """
    Render the data source section and load the uploaded data.

    Returns:
        A tuple containing:
            - The uploaded file object
            - The loaded DataFrame
            - The selected text column
            - The selected date column
            - Whether dynamic topic modeling is enabled
    """
    st.header("1. Data Source", divider="gray")

    data_mode = st.radio(
        "Data Format",
        ["Tabular Data (CSV / Excel)", "ZIP Archive (Text files)"],
        horizontal=True,
        help=(
            "Tabular data is recommended to keep metadata attached to the "
            "extracted topics."
        ),
    )

    uploaded_file = st.file_uploader(
        "Upload your file",
        type=["csv", "xlsx", "xls"] if "Tabular" in data_mode else ["zip"],
    )

    df = None
    text_column = None
    date_column = None
    enable_dtm = False

    if uploaded_file:
        try:
            if "Tabular" in data_mode:
                df = read_uploaded_table(uploaded_file.name, uploaded_file)
                df = df.dropna(how="all")

                if df.empty or len(df.columns) == 0:
                    raise ValueError(
                        "The uploaded file does not contain any usable rows or columns."
                    )

                col_data1, col_data2 = st.columns(2)

                with col_data1:
                    text_column = st.selectbox(
                        "Target Text Column",
                        options=df.columns.tolist(),
                        help=(
                            "Select the column containing the raw text you "
                            "wish to analyze."
                        ),
                    )

                with col_data2:
                    enable_dtm = st.checkbox(
                        "Analyze Topics Over Time",
                        help=(
                            "Used only for BERTopic. Requires a column with "
                            "parseable dates or timestamps."
                        ),
                    )
                    if enable_dtm:
                        date_column = st.selectbox(
                            "Timestamp Column",
                            options=df.columns.tolist(),
                        )
            else:
                df = load_zip_texts(uploaded_file.getvalue())
                text_column = "Text"

                if df.empty:
                    raise ValueError(
                        "The ZIP archive does not contain any non-empty .txt files."
                    )

                st.success(f"Successfully loaded {len(df)} documents from archive.")

        except Exception as exc:
            st.error(f"Error loading file: {exc}")
            st.stop()

    return uploaded_file, df, text_column, date_column, enable_dtm


def _render_model_configuration(
    text_column: str | None,
    date_column: str | None,
    enable_dtm: bool,
) -> TopicModelingConfig:
    """
    Render the model configuration section and collect user selections.

    Args:
        text_column: The selected text column.
        date_column: The selected date column.
        enable_dtm: Whether dynamic topic modeling is enabled.

    Returns:
        A TopicModelingConfig instance containing the selected options.
    """
    st.header("2. Model Configuration", divider="gray")

    algorithm = st.radio(
        "Select Algorithm",
        [
            "BERTopic (Transformer Embeddings)",
            "Top2Vec (Joint Semantic Embeddings)",
            "Latent Dirichlet Allocation (LDA)",
        ],
        help=(
            "BERTopic and Top2Vec use modern AI semantic models. "
            "LDA uses traditional statistical term frequencies."
        ),
    )

    if enable_dtm and "BERTopic" not in algorithm:
        st.info("Topics-over-time analysis is only applied for BERTopic in this page.")

    col_basic, col_adv = st.columns(2)

    clustering_params: dict[str, Any] = {}
    bertopic_nr_topics: int | str | None = None
    clustering_algo_clean = "HDBSCAN"
    ngram_range = (1, 1)

    dim_reduction_clean = "UMAP"
    dim_params: dict[str, Any] = {}

    min_df = 1
    reduce_frequent = True
    reduce_outliers = False

    backend_clean = "doc2vec"
    top2vec_backend_raw = "Doc2Vec (Train from scratch)"
    top2vec_speed = "learn"
    use_bigrams = False
    passes = 10
    custom_stopwords = ""
    num_topics: int | str = 10

    with col_basic:
        st.subheader("Core Settings")

        language = st.selectbox(
            "Primary Language",
            SUPPORTED_LANGUAGES,
            help=(
                "Determines the underlying embedding model used for the analysis."
            ),
        )

        if "BERTopic" in algorithm:
            clustering_algo = st.radio(
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

            if "KMeans" in clustering_algo:
                n_clusters = st.slider(
                    "Exact Number of Clusters (K)",
                    min_value=2,
                    max_value=100,
                    value=10,
                    step=1,
                )
                clustering_params["n_clusters"] = n_clusters
                clustering_algo_clean = "KMeans"
                bertopic_nr_topics = None
            else:
                auto_topics = st.checkbox(
                    "Auto-detect optimal number of topics",
                    value=True,
                )
                bertopic_nr_topics = (
                    "auto"
                    if auto_topics
                    else st.slider(
                        "Target Number of Topics",
                        min_value=2,
                        max_value=100,
                        value=10,
                        step=1,
                    )
                )
                min_cluster_size = st.number_input(
                    "Minimum Documents per Topic",
                    min_value=3,
                    max_value=500,
                    value=10,
                    step=1,
                )
                clustering_params["min_cluster_size"] = int(min_cluster_size)
                clustering_algo_clean = "HDBSCAN"

        elif "Top2Vec" in algorithm:
            auto_topics = st.checkbox(
                "Auto-detect optimal number of topics",
                value=True,
            )
            if auto_topics:
                num_topics = "auto"
                st.info(
                    "The algorithm will dynamically determine the best number "
                    "of topics using density-based clustering."
                )
            else:
                num_topics = st.slider(
                    "Target Number of Topics",
                    min_value=2,
                    max_value=100,
                    value=10,
                    step=1,
                )

        else:
            num_topics = st.slider(
                "Number of Topics",
                min_value=2,
                max_value=50,
                value=10,
                step=1,
            )

    with col_adv:
        st.subheader("Advanced Processing")

        if "Top2Vec" in algorithm:
            st.info(
                "Top2Vec relies on the natural, raw structure of sentences "
                "to generate joint embeddings. Custom stopwords are disabled "
                "for this algorithm."
            )

            top2vec_backend_raw = st.radio(
                "Embedding Backend",
                ["Transformer (Pre-trained)", "Doc2Vec (Train from scratch)"],
                help=(
                    "Transformers are faster and understand general language. "
                    "Doc2Vec trains specifically on your data."
                ),
            )
            backend_clean = (
                "transformer"
                if "Transformer" in top2vec_backend_raw
                else "doc2vec"
            )

            top2vec_speed = st.select_slider(
                "Training Depth",
                options=["fast-learn", "learn", "deep-learn"],
                value="learn",
            )

        else:
            custom_stopwords = st.text_area(
                "Custom Stopwords",
                placeholder="patient, report, conclusion, dataset",
                help=(
                    "Comma-separated list of domain-specific words you want "
                    "the model to ignore."
                ),
            )

            if "BERTopic" in algorithm:
                with st.expander("Text Extraction & Vocabulary", expanded=True):
                    extract_phrases = st.checkbox(
                        "Extract Phrases (N-grams)",
                        value=False,
                    )
                    ngram_range = (1, 2) if extract_phrases else (1, 1)
                    min_df = st.number_input(
                        "Minimum Document Frequency (min_df)",
                        min_value=1,
                        max_value=500,
                        value=1,
                        help=(
                            "Higher values reduce memory usage by ignoring "
                            "extremely rare words."
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

                with st.expander("Dimensionality Reduction"):
                    dim_mapping = {
                        "UMAP (Recommended)": "UMAP",
                        "PCA (Fast & Linear)": "PCA",
                        "Truncated SVD": "Truncated SVD",
                        "None (Skip reduction)": "None",
                    }
                    dim_reduction_raw = st.radio(
                        "Algorithm",
                        options=list(dim_mapping.keys()),
                        help=(
                            "UMAP preserves global and local structure. "
                            "PCA/SVD are faster but linear. "
                            "'None' skips this step entirely."
                        ),
                    )
                    dim_reduction_clean = dim_mapping[dim_reduction_raw]

                    if dim_reduction_clean != "None":
                        n_components = st.slider("Target Dimensions", 2, 50, 5)
                        dim_params["n_components"] = n_components

                        if dim_reduction_clean == "UMAP":
                            n_neighbors = st.slider(
                                "Neighbors (Local vs Global)",
                                2,
                                200,
                                15,
                                help=(
                                    "Balances local vs global structure. "
                                    "High values preserve global structure."
                                ),
                            )
                            min_dist = st.slider(
                                "Minimum Distance (Tightness)",
                                0.0,
                                1.0,
                                0.0,
                                step=0.01,
                            )
                            dim_params["n_neighbors"] = n_neighbors
                            dim_params["min_dist"] = min_dist

                        random_state = st.checkbox(
                            "Lock Seed for Reproducibility",
                            value=True,
                        )
                        dim_params["random_state"] = 42 if random_state else None

                with st.expander("Clustering & Outliers (HDBSCAN)"):
                    if clustering_algo_clean == "HDBSCAN":
                        min_samples = st.number_input(
                            "Outlier Sensitivity (min_samples)",
                            min_value=1,
                            max_value=500,
                            value=int(clustering_params.get("min_cluster_size", 10)),
                            help="Lower values reduce noise/outliers.",
                        )
                        clustering_params["min_samples"] = min_samples
                        reduce_outliers = st.checkbox(
                            "Force-assign outliers (-1) to nearest topics",
                            value=False,
                        )
                    else:
                        st.info("Outlier settings only apply to HDBSCAN.")

            elif "LDA" in algorithm:
                use_bigrams = st.checkbox("Extract Phrases (Bigrams)", value=False)
                passes = st.slider(
                    "Training Passes (Iterations)",
                    min_value=5,
                    max_value=50,
                    value=10,
                    step=5,
                )

    return TopicModelingConfig(
        algorithm=algorithm,
        language=language,
        text_column=text_column or "",
        enable_dtm=enable_dtm,
        date_column=date_column,
        custom_stopwords=custom_stopwords,
        num_topics=num_topics,
        bertopic_nr_topics=bertopic_nr_topics,
        dim_reduction_algo=dim_reduction_clean,
        dim_params=dim_params,
        clustering_algo=clustering_algo_clean,
        clustering_params=clustering_params,
        ngram_range=ngram_range,
        min_df=min_df,
        reduce_frequent=reduce_frequent,
        reduce_outliers=reduce_outliers,
        top2vec_backend=backend_clean,
        top2vec_backend_label=top2vec_backend_raw,
        top2vec_speed=top2vec_speed,
        use_bigrams=use_bigrams,
        passes=passes,
    )


def _render_results(res: dict[str, Any]) -> None:
    """
    Render the results section from session state.
    """
    st.header("Results Analysis", divider="gray")

    # Render Evaluation Metrics
    if "evaluation_metrics" in res and res["evaluation_metrics"]:
        st.subheader("Model Evaluation Metrics")
        st.caption("Quantitative metrics to compare hyperparameter performance.")
        
        metrics = res["evaluation_metrics"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Topic Diversity", f"{metrics.get('Topic Diversity', 0.0):.2%}", help="Percentage of unique words across all topics (Higher is better).")
        col2.metric("Coherence (C_v)", metrics.get("Coherence (C_v)", 0.0), help="Highly correlated with human interpretability. Range 0 to 1 (Higher is better).")
        col3.metric("Coherence (C_npmi)", metrics.get("Coherence (C_npmi)", 0.0), help="Normalized Pointwise Mutual Information. Typically -1 to 1 (Higher is better).")
        col4.metric("Coherence (U_mass)", metrics.get("Coherence (U_mass)", 0.0), help="Measures word co-occurrence within the corpus. Typically negative (Closer to 0 is better).")
        
        st.divider()

    st.subheader("Topic Dictionary")
    if "BERTopic" in res["algorithm"] or "Top2Vec" in res["algorithm"]:
        st.caption(
            "Note: Density-based algorithms automatically classify outlier "
            "documents into an 'Outlier' category."
        )
    st.dataframe(res["topic_df"], use_container_width=True, hide_index=True)

    if "LDA" in res["algorithm"]:
        st.subheader("Interactive Topic Dashboard")
        components.html(
            res["dashboard_assets"]["lda_dashboard.html"],
            width=1300,
            height=800,
            scrolling=False,
        )

    elif "Top2Vec" in res["algorithm"]:
        st.subheader("Interactive Topic Dashboard")
        components.html(
            res["dashboard_assets"]["top2vec_barchart.html"],
            width=1000,
            height=800,
            scrolling=True,
        )

    else:
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
            components.html(
                res["dashboard_assets"].get("intertopic_distance.html", ""),
                width=1000,
                height=600,
                scrolling=False,
            )

        with tab2:
            st.caption("Displays the highest frequency terms for the top topics.")
            components.html(
                res["dashboard_assets"].get("topic_barchart.html", ""),
                width=1000,
                height=600,
                scrolling=False,
            )

        with tab3:
            st.caption(
                "Shows how semantically similar the generated topics are to each other."
            )
            components.html(
                res["dashboard_assets"].get("similarity_heatmap.html", ""),
                width=1000,
                height=600,
                scrolling=False,
            )

        with tab4:
            if res["enable_dtm"] and "topics_over_time.html" in res["dashboard_assets"]:
                st.caption(
                    "Visualizes topic frequency evolution over the provided timestamps."
                )
                components.html(
                    res["dashboard_assets"]["topics_over_time.html"],
                    width=1000,
                    height=600,
                    scrolling=False,
                )
            else:
                st.info("Dynamic Topic Modeling was not enabled during configuration.")

    st.subheader("Export Artifacts")
    st.write(
        "Download your original dataset augmented with topic classifications, "
        "alongside the standalone interactive HTML dashboards and metadata "
        "report."
    )

    zip_bytes = build_results_zip(
        metadata_report=res["metadata_report"],
        docs_df=res["docs_df"],
        topic_df=res["topic_df"],
        dashboard_assets=res["dashboard_assets"],
    )

    st.download_button(
        label="Download Extraction Package (.zip)",
        data=zip_bytes,
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

    uploaded_file, df, text_column, date_column, enable_dtm = (
        _render_data_source_section()
    )
    config = _render_model_configuration(text_column, date_column, enable_dtm)

    st.header("3. Execution", divider="gray")

    if st.button("Run Topic Extraction", type="primary", disabled=(df is None)):
        try:
            prepared_df = drop_empty_text_rows(df, config.text_column)

            timestamps = None
            if config.enable_dtm and "BERTopic" in config.algorithm and config.date_column:
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
            embedding_model_name = get_embedding_model_name(config)

            with st.spinner("Running topic extraction..."):
                run_result = run_topic_modeling_pipeline(
                    prepared_df,
                    config,
                    timestamps=timestamps,
                )

            # Run Mathematical Evaluation
            with st.spinner("Calculating Topic Coherence and Diversity..."):
                # Extract comma-separated keywords back into lists for evaluation
                topic_keywords = [
                    [word.strip() for word in keywords.split(",")] 
                    for keywords in run_result["topic_df"]["Keywords"].tolist()
                ]
                
                evaluation_metrics = evaluate_topic_quality(
                    topic_keywords=topic_keywords,
                    raw_texts=raw_texts,
                    language=config.language,
                    custom_stopwords_str=config.custom_stopwords
                )

            # Generate Metadata Report with metrics appended
            metadata_report = generate_metadata_report(
                filename=uploaded_file.name,
                config=config,
                embedding_model_name=embedding_model_name,
                evaluation_metrics=evaluation_metrics
            )

            st.session_state.topic_results = {
                "topic_df": run_result["topic_df"],
                "docs_df": run_result["docs_df"],
                "dashboard_assets": run_result["dashboard_assets"],
                "metadata_report": metadata_report,
                "evaluation_metrics": evaluation_metrics,
                "algorithm": config.algorithm,
                "enable_dtm": config.enable_dtm,
            }

            st.success("Topic Modeling execution complete.")

        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        except Exception:
            st.error("Topic modeling failed. See technical details below:")
            st.code(traceback.format_exc())
            print(
                f"--- Topic Modeling Error ---\n{traceback.format_exc()}",
                file=sys.stderr,
            )
            st.stop()

    if st.session_state.topic_results is not None:
        _render_results(st.session_state.topic_results)


if __name__ == "__main__":
    main()