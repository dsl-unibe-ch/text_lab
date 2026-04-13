import os
import sys
import io
import zipfile
import traceback
import datetime

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
from core.topic_modeling.topic_utils import load_zip_texts, preprocess_texts_for_lda, get_stopword_set
from core.topic_modeling.lda_engine import (
    train_lda_model,
    generate_lda_keywords_df,
    generate_lda_document_topics_df,
    generate_lda_html
)
from core.topic_modeling.bertopic_engine import (
    train_bertopic_model,
    generate_bertopic_keywords_df,
    generate_bertopic_document_topics_df,
    generate_bertopic_visualizations,
    generate_topics_over_time_html
)
from core.topic_modeling.top2vec_engine import (
    train_top2vec_model,
    generate_top2vec_keywords_df,
    generate_top2vec_document_topics_df,
    generate_top2vec_barchart_html
)

SUPPORTED_LANGUAGES = [
    "English", "German", "French",
    "Spanish", "Italian", "Dutch",
    "Portuguese", "Russian", "Arabic",
    "Chinese", "Other / Mixed"
]


def _read_uploaded_table(uploaded_file: UploadedFile) -> pd.DataFrame:
    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported tabular file format.")


def _prepare_timestamps(df: pd.DataFrame, date_column: str) -> tuple[pd.DataFrame, list]:
    parsed = pd.to_datetime(df[date_column], errors="coerce", utc=True).dt.tz_localize(None)
    valid_mask = parsed.notna()

    dropped = int((~valid_mask).sum())
    if dropped:
        st.warning(f"{dropped} rows were skipped because '{date_column}' could not be parsed as a date/time.")

    df = df.loc[valid_mask].copy()
    df[date_column] = parsed.loc[valid_mask]
    df = df.sort_values(date_column).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid timestamps remained after parsing the selected timestamp column.")

    return df, df[date_column].tolist()



def _generate_metadata_report(
    filename: str,
    text_column: str,
    enable_dtm: bool,
    date_column: str | None,
    algorithm: str,
    language: str,
    embedding_model_name: str,
    custom_stopwords: str,
    num_topics: int | str,
    bertopic_nr_topics: int | str | None,
    dim_reduction_algo: str,
    dim_params: dict,
    clustering_algo_clean: str,
    clustering_params: dict,
    ngram_range: tuple,
    min_df: int,
    reduce_frequent: bool,
    reduce_outliers: bool,
    top2vec_backend: str,
    top2vec_speed: str,
    use_bigrams: bool,
    passes: int
) -> str:
    """Compiles all execution parameters into a formatted string for reproducibility."""
    report = [
        "=========================================",
        " TEXT LAB - TOPIC MODELING CONFIGURATION ",
        "=========================================",
        f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Source File: {filename}",
        f"Target Text Column: {text_column}"
    ]
    
    if enable_dtm and date_column:
        report.append(f"Timestamp Column (DTM): {date_column}")
        
    report.extend([
        "",
        "--- CORE SETTINGS ---",
        f"Framework: {algorithm}",
        f"Primary Language: {language}",
        f"Custom Stopwords: {custom_stopwords if custom_stopwords.strip() else 'None'}",
        ""
    ])

    if "LDA" in algorithm:
        report.extend([
            "--- LDA PARAMETERS ---",
            f"Number of Topics: {num_topics}",
            f"Training Passes: {passes}",
            f"Extract Bigrams: {use_bigrams}"
        ])

    elif "Top2Vec" in algorithm:
        report.extend([
            "--- TOP2VEC PARAMETERS ---",
            f"Target Topics: {num_topics}",
            f"Embedding Backend: {top2vec_backend}",
            f"Embedding Model: {embedding_model_name}",
            f"Training Speed: {top2vec_speed}"
        ])

    else:  # BERTopic
        report.extend([
            "--- BERTOPIC PARAMETERS ---",
            f"Target Topics: {bertopic_nr_topics}",
            f"Embedding Model: {embedding_model_name}",
            f"N-Gram Range: {ngram_range}",
            f"Min Document Frequency (min_df): {min_df}",
            f"Reduce Frequent Words (ClassTfidfTransformer): {reduce_frequent}",
            "",
            f"--- DIMENSIONALITY REDUCTION ({dim_reduction_algo}) ---"
        ])
        
        if dim_reduction_algo != "None":
            report.append(f"N Components: {dim_params.get('n_components')}")
            if dim_reduction_algo == "UMAP":
                report.append(f"N Neighbors: {dim_params.get('n_neighbors')}")
                report.append(f"Min Distance: {dim_params.get('min_dist')}")
            report.append(f"Random State: {dim_params.get('random_state', 'None')}")

        report.extend([
            "",
            f"--- CLUSTERING ({clustering_algo_clean}) ---"
        ])
        
        if clustering_algo_clean == "KMeans":
            report.append(f"N Clusters: {clustering_params.get('n_clusters')}")
        else:
            report.extend([
                f"Min Cluster Size: {clustering_params.get('min_cluster_size')}",
                f"Min Samples: {clustering_params.get('min_samples', 'Default (equals min_cluster_size)')}",
                f"Force-reduce Outliers: {reduce_outliers}"
            ])

    return "\n".join(report)


def main() -> None:
    st.set_page_config(page_title="Topic Modeling", layout="wide")
    check_token()

    if "topic_results" not in st.session_state:
        st.session_state.topic_results = None

    st.title("Topic Modeling")
    st.markdown(
        "Discover hidden themes in large text datasets automatically. "
        "Upload your data to generate an interactive topic distribution map."
    )

    # --- 1. Data Input Section ---
    st.header("1. Data Source", divider="gray")
    data_mode = st.radio(
        "Data Format",
        ["Tabular Data (CSV / Excel)", "ZIP Archive (Text files)"],
        horizontal=True,
        help="Tabular data is recommended to keep metadata attached to the extracted topics."
    )

    uploaded_file = st.file_uploader(
        "Upload your file",
        type=["csv", "xlsx", "xls"] if "Tabular" in data_mode else ["zip"]
    )

    df = None
    text_column = None
    date_column = None
    enable_dtm = False

    if uploaded_file:
        try:
            if "Tabular" in data_mode:
                df = _read_uploaded_table(uploaded_file)
                df = df.dropna(how="all")

                if df.empty or len(df.columns) == 0:
                    raise ValueError("The uploaded file does not contain any usable rows or columns.")

                col_data1, col_data2 = st.columns(2)
                with col_data1:
                    text_column = st.selectbox(
                        "Target Text Column",
                        options=df.columns.tolist(),
                        help="Select the column containing the raw text you wish to analyze."
                    )
                with col_data2:
                    enable_dtm = st.checkbox(
                        "Analyze Topics Over Time",
                        help="Used only for BERTopic. Requires a column with parseable dates or timestamps."
                    )
                    if enable_dtm:
                        date_column = st.selectbox(
                            "Timestamp Column",
                            options=df.columns.tolist()
                        )
            else:
                df = load_zip_texts(uploaded_file.getvalue())
                text_column = "Text"
                if df.empty:
                    raise ValueError("The ZIP archive does not contain any non-empty .txt files.")
                st.success(f"Successfully loaded {len(df)} documents from archive.")

        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.stop()

    # --- 2. Algorithm & Parameter Configuration ---
    st.header("2. Model Configuration", divider="gray")

    algorithm = st.radio(
        "Select Algorithm",
        [
            "BERTopic (Transformer Embeddings)",
            "Top2Vec (Joint Semantic Embeddings)",
            "Latent Dirichlet Allocation (LDA)"
        ],
        help="BERTopic and Top2Vec use modern AI semantic models. LDA uses traditional statistical term frequencies."
    )

    if enable_dtm and "BERTopic" not in algorithm:
        st.info("Topics-over-time analysis is only applied for BERTopic in this page.")

    col_basic, col_adv = st.columns(2)

    # Safe defaults to prevent UnboundLocalError
    clustering_params = {}
    bertopic_nr_topics = None
    clustering_algo_clean = "HDBSCAN"
    ngram_range = (1, 1)
    
    dim_reduction_clean = "UMAP"
    dim_params = {}
    
    min_df = 1
    reduce_frequent = True
    reduce_outliers = False
    
    backend_clean = "doc2vec"
    top2vec_backend_raw = "doc2vec"
    top2vec_speed = "learn"
    use_bigrams = False
    passes = 10
    custom_stopwords = ""
    num_topics = 10

    with col_basic:
        st.subheader("Core Settings")

        language = st.selectbox(
            "Primary Language",
            SUPPORTED_LANGUAGES,
            help="Determines the underlying embedding model used for the analysis."
        )

        if "BERTopic" in algorithm:
            clustering_algo = st.radio(
                "Clustering Engine",
                ["HDBSCAN (Density-based, handles noise)", "KMeans (Centroid-based, strict groups)"],
                help="HDBSCAN dynamically finds clusters and isolates outliers. KMeans forces every document into a strict number of topics."
            )

            if "KMeans" in clustering_algo:
                n_clusters = st.slider("Exact Number of Clusters (K)", min_value=2, max_value=100, value=10, step=1)
                clustering_params["n_clusters"] = n_clusters
                clustering_algo_clean = "KMeans"
                bertopic_nr_topics = None
            else:
                auto_topics = st.checkbox("Auto-detect optimal number of topics", value=True)
                bertopic_nr_topics = "auto" if auto_topics else st.slider("Target Number of Topics", min_value=2, max_value=100, value=10, step=1)
                min_cluster_size = st.number_input("Minimum Documents per Topic", min_value=3, max_value=500, value=10, step=1)
                clustering_params["min_cluster_size"] = int(min_cluster_size)
                clustering_algo_clean = "HDBSCAN"

        elif "Top2Vec" in algorithm:
            auto_topics = st.checkbox("Auto-detect optimal number of topics", value=True)
            if auto_topics:
                num_topics = "auto"
                st.info("The algorithm will dynamically determine the best number of topics using density-based clustering.")
            else:
                num_topics = st.slider("Target Number of Topics", min_value=2, max_value=100, value=10, step=1)

        else:  # LDA
            num_topics = st.slider("Number of Topics", min_value=2, max_value=50, value=10, step=1)

    with col_adv:
        st.subheader("Advanced Processing")

        if "Top2Vec" in algorithm:
            st.info("Top2Vec relies on the natural, raw structure of sentences to generate joint embeddings. Custom stopwords are disabled for this algorithm.")

            top2vec_backend_raw = st.radio(
                "Embedding Backend",
                ["Transformer (Pre-trained)", "Doc2Vec (Train from scratch)"],
                help="Transformers are faster and understand general language. Doc2Vec trains specifically on your data."
            )
            backend_clean = "transformer" if "Transformer" in top2vec_backend_raw else "doc2vec"

            top2vec_speed = st.select_slider(
                "Training Depth",
                options=["fast-learn", "learn", "deep-learn"],
                value="learn"
            )

        else:
            custom_stopwords = st.text_area(
                "Custom Stopwords",
                placeholder="patient, report, conclusion, dataset",
                help="Comma-separated list of domain-specific words you want the model to ignore."
            )

            if "BERTopic" in algorithm:
                with st.expander("Text Extraction & Vocabulary", expanded=True):
                    extract_phrases = st.checkbox("Extract Phrases (N-grams)", value=False)
                    ngram_range = (1, 2) if extract_phrases else (1, 1)
                    min_df = st.number_input("Minimum Document Frequency (min_df)", min_value=1, max_value=500, value=1, help="Higher values reduce memory usage by ignoring extremely rare words.")
                    reduce_frequent = st.checkbox("Auto-penalize frequent words", value=True, help="Uses ClassTfidfTransformer to reduce the impact of common filler words without explicitly deleting them.")
                
                with st.expander("Dimensionality Reduction"):
                    dim_mapping = {
                        "UMAP (Recommended)": "UMAP",
                        "PCA (Fast & Linear)": "PCA",
                        "Truncated SVD": "Truncated SVD",
                        "None (Skip reduction)": "None"
                    }
                    dim_reduction_raw = st.radio(
                        "Algorithm",
                        options=list(dim_mapping.keys()),
                        help="UMAP preserves global and local structure. PCA/SVD are faster but linear. 'None' skips this step entirely."
                    )
                    dim_reduction_clean = dim_mapping[dim_reduction_raw]

                    if dim_reduction_clean != "None":
                        n_components = st.slider("Target Dimensions", 2, 50, 5)
                        dim_params["n_components"] = n_components
                        
                        if dim_reduction_clean == "UMAP":
                            n_neighbors = st.slider("Neighbors (Local vs Global)", 2, 200, 15, help="Balances local vs global structure. High values preserve global structure.")
                            min_dist = st.slider("Minimum Distance (Tightness)", 0.0, 1.0, 0.0, step=0.01)
                            dim_params["n_neighbors"] = n_neighbors
                            dim_params["min_dist"] = min_dist
                            
                        random_state = st.checkbox("Lock Seed for Reproducibility", value=True)
                        dim_params["random_state"] = 42 if random_state else None

                with st.expander("Clustering & Outliers (HDBSCAN)"):
                    if clustering_algo_clean == "HDBSCAN":
                        min_samples = st.number_input("Outlier Sensitivity (min_samples)", min_value=1, max_value=500, value=int(clustering_params.get("min_cluster_size", 10)), help="Lower values reduce noise/outliers.")
                        clustering_params["min_samples"] = min_samples
                        reduce_outliers = st.checkbox("Force-assign outliers (-1) to nearest topics", value=False)
                    else:
                        st.info("Outlier settings only apply to HDBSCAN.")

            elif "LDA" in algorithm:
                use_bigrams = st.checkbox("Extract Phrases (Bigrams)", value=False)
                passes = st.slider("Training Passes (Iterations)", min_value=5, max_value=50, value=10, step=5)

    # --- 3. Execution & Results ---
    st.header("3. Execution", divider="gray")

    if st.button("Run Topic Extraction", type="primary", disabled=(df is None)):

        df = df.dropna(subset=[text_column]).reset_index(drop=True)

        if df.empty:
            st.error("No valid documents remain after removing empty text rows.")
            st.stop()

        timestamps = None
        if enable_dtm and "BERTopic" in algorithm:
            try:
                df, timestamps = _prepare_timestamps(df, date_column)
            except Exception as e:
                st.error(str(e))
                st.stop()

        raw_texts = df[text_column].astype(str).tolist()

        if len(raw_texts) < 5:
            st.error("A minimum of 5 valid documents is required to perform topic modeling.")
            st.stop()

        embedding_model_name = "N/A"
        if "BERTopic" in algorithm:
            embedding_model_name = "all-MiniLM-L6-v2" if language == "English" else "paraphrase-multilingual-MiniLM-L12-v2"
        elif "Top2Vec" in algorithm:
            if backend_clean == "transformer":
                embedding_model_name = "all-MiniLM-L6-v2" if language == "English" else "paraphrase-multilingual-MiniLM-L12-v2"
            else:
                embedding_model_name = "Doc2Vec (Trained from scratch)"

        metadata_report = _generate_metadata_report(
            filename=uploaded_file.name,
            text_column=text_column,
            enable_dtm=enable_dtm,
            date_column=date_column,
            algorithm=algorithm,
            language=language,
            embedding_model_name=embedding_model_name,
            custom_stopwords=custom_stopwords,
            num_topics=num_topics,
            bertopic_nr_topics=bertopic_nr_topics,
            dim_reduction_algo=dim_reduction_clean,
            dim_params=dim_params,
            clustering_algo_clean=clustering_algo_clean,
            clustering_params=clustering_params,
            ngram_range=ngram_range,
            min_df=min_df,
            reduce_frequent=reduce_frequent,
            reduce_outliers=reduce_outliers,
            top2vec_backend=top2vec_backend_raw,
            top2vec_speed=top2vec_speed,
            use_bigrams=use_bigrams,
            passes=passes
        )

        dashboard_assets = {}

        if "LDA" in algorithm:
            with st.spinner("Step 1/3: Cleaning and tokenizing text corpus..."):
                try:
                    processed_texts = preprocess_texts_for_lda(raw_texts, language, custom_stopwords, use_bigrams)
                except Exception:
                    st.error("Failed to process text corpus. See technical details below:")
                    st.code(traceback.format_exc())
                    print(f"--- LDA Preprocessing Error ---\n{traceback.format_exc()}", file=sys.stderr)
                    st.stop()

            with st.spinner(f"Step 2/3: Training LDA Model for {num_topics} topics..."):
                try:
                    lda_model, corpus, id2word = train_lda_model(processed_texts, num_topics, passes)
                except Exception:
                    st.error("Failed to train LDA model. See technical details below:")
                    st.code(traceback.format_exc())
                    print(f"--- LDA Training Error ---\n{traceback.format_exc()}", file=sys.stderr)
                    st.stop()

            with st.spinner("Step 3/3: Generating visualization dashboard..."):
                try:
                    topic_df = generate_lda_keywords_df(lda_model, num_topics)
                    docs_df = generate_lda_document_topics_df(lda_model, corpus, df)
                    html_string = generate_lda_html(lda_model, corpus, id2word)
                    dashboard_assets["lda_dashboard.html"] = html_string
                except Exception:
                    st.error("Failed to generate dashboard artifacts. See technical details below:")
                    st.code(traceback.format_exc())
                    print(f"--- LDA Dashboard Error ---\n{traceback.format_exc()}", file=sys.stderr)
                    st.stop()

        elif "Top2Vec" in algorithm:
            with st.spinner("Step 1/3: Training Joint Semantic Embeddings (This may take several minutes)..."):
                try:
                    topic_model = train_top2vec_model(
                        texts=raw_texts,
                        language=language,
                        embedding_backend=backend_clean,
                        speed=top2vec_speed,
                        target_topics=num_topics
                    )
                except Exception:
                    st.error("Failed to train Top2Vec model. See technical details below:")
                    st.code(traceback.format_exc())
                    print(f"--- Top2Vec Training Error ---\n{traceback.format_exc()}", file=sys.stderr)
                    st.stop()

            with st.spinner("Step 2/3: Extracting document classifications..."):
                try:
                    topic_df = generate_top2vec_keywords_df(topic_model, num_topics)
                    docs_df = generate_top2vec_document_topics_df(topic_model, df, num_topics)
                except Exception:
                    st.error("Failed to extract classifications. See technical details below:")
                    st.code(traceback.format_exc())
                    print(f"--- Top2Vec Extraction Error ---\n{traceback.format_exc()}", file=sys.stderr)
                    st.stop()

            with st.spinner("Step 3/3: Generating visualization suites..."):
                try:
                    html_string = generate_top2vec_barchart_html(topic_model, num_topics)
                    dashboard_assets["top2vec_barchart.html"] = html_string
                except Exception:
                    st.error("Failed to generate dashboard artifacts. See technical details below:")
                    st.code(traceback.format_exc())
                    print(f"--- Top2Vec Dashboard Error ---\n{traceback.format_exc()}", file=sys.stderr)
                    st.stop()

        else:  # BERTopic
            with st.spinner("Step 1/3: Generating embeddings and training BERTopic model (This may take a moment)..."):
                try:
                    stop_set = get_stopword_set(language, custom_stopwords)
                    topic_model, topics = train_bertopic_model(
                        texts=raw_texts,
                        language=language,
                        num_topics=bertopic_nr_topics,
                        stop_words_set=stop_set,
                        dim_reduction_algo=dim_reduction_clean,
                        dim_params=dim_params,
                        clustering_algo=clustering_algo_clean,
                        clustering_params=clustering_params,
                        ngram_range=ngram_range,
                        min_df=min_df,
                        reduce_outliers=reduce_outliers,
                        reduce_frequent_words=reduce_frequent
                    )
                except Exception:
                    st.error("Failed to train BERTopic model. See technical details below:")
                    st.code(traceback.format_exc())
                    print(f"--- BERTopic Training Error ---\n{traceback.format_exc()}", file=sys.stderr)
                    st.stop()

            with st.spinner("Step 2/3: Extracting document classifications..."):
                try:
                    topic_df = generate_bertopic_keywords_df(topic_model)
                    docs_df = generate_bertopic_document_topics_df(topics, df)
                except Exception:
                    st.error("Failed to extract classifications. See technical details below:")
                    st.code(traceback.format_exc())
                    print(f"--- BERTopic Extraction Error ---\n{traceback.format_exc()}", file=sys.stderr)
                    st.stop()

            with st.spinner("Step 3/3: Generating visualization suites..."):
                try:
                    visualizations = generate_bertopic_visualizations(topic_model)
                    dashboard_assets["intertopic_distance.html"] = visualizations.get("distance_map", "")
                    dashboard_assets["topic_barchart.html"] = visualizations.get("barchart", "")
                    dashboard_assets["similarity_heatmap.html"] = visualizations.get("heatmap", "")

                    if enable_dtm and timestamps is not None:
                        dtm_html = generate_topics_over_time_html(topic_model, raw_texts, timestamps)
                        dashboard_assets["topics_over_time.html"] = dtm_html

                except Exception:
                    st.error("Failed to generate dashboard artifacts. See technical details below:")
                    st.code(traceback.format_exc())
                    print(f"--- BERTopic Dashboard Error ---\n{traceback.format_exc()}", file=sys.stderr)
                    st.stop()

        st.success("Topic Modeling execution complete.")
        
        st.session_state.topic_results = {
            "topic_df": topic_df,
            "docs_df": docs_df,
            "dashboard_assets": dashboard_assets,
            "metadata_report": metadata_report,
            "algorithm": algorithm,
            "enable_dtm": enable_dtm
        }

    # --- 4. Display Results (Reads from Session State) ---
    if st.session_state.topic_results is not None:
        res = st.session_state.topic_results
        
        st.header("Results Analysis", divider="gray")

        st.subheader("Topic Dictionary")
        if "BERTopic" in res["algorithm"] or "Top2Vec" in res["algorithm"]:
            st.caption("Note: Density-based algorithms automatically classify outlier documents into an 'Outlier' category.")
        st.dataframe(res["topic_df"], use_container_width=True, hide_index=True)

        if "LDA" in res["algorithm"]:
            st.subheader("Interactive Topic Dashboard")
            components.html(res["dashboard_assets"]["lda_dashboard.html"], width=1300, height=800, scrolling=False)

        elif "Top2Vec" in res["algorithm"]:
            st.subheader("Interactive Topic Dashboard")
            components.html(res["dashboard_assets"]["top2vec_barchart.html"], width=1000, height=800, scrolling=True)

        else:
            tab1, tab2, tab3, tab4 = st.tabs(["Intertopic Distance", "Word Scores", "Similarity Heatmap", "Topics Over Time"])

            with tab1:
                st.caption("Maps the semantic distance between discovered topics.")
                components.html(res["dashboard_assets"].get("intertopic_distance.html", ""), width=1000, height=600, scrolling=False)
            with tab2:
                st.caption("Displays the highest frequency terms for the top topics.")
                components.html(res["dashboard_assets"].get("topic_barchart.html", ""), width=1000, height=600, scrolling=False)
            with tab3:
                st.caption("Shows how semantically similar the generated topics are to each other.")
                components.html(res["dashboard_assets"].get("similarity_heatmap.html", ""), width=1000, height=600, scrolling=False)
            with tab4:
                if res["enable_dtm"] and "topics_over_time.html" in res["dashboard_assets"]:
                    st.caption("Visualizes topic frequency evolution over the provided timestamps.")
                    components.html(res["dashboard_assets"]["topics_over_time.html"], width=1000, height=600, scrolling=False)
                else:
                    st.info("Dynamic Topic Modeling was not enabled during configuration.")

        st.subheader("Export Artifacts")
        st.write("Download your original dataset augmented with topic classifications, alongside the standalone interactive HTML dashboards and metadata report.")

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("run_configuration.txt", res["metadata_report"].encode("utf-8"))
            zf.writestr("document_topics.csv", res["docs_df"].to_csv(index=False).encode("utf-8"))
            zf.writestr("topic_keywords.csv", res["topic_df"].to_csv(index=False).encode("utf-8"))
            for filename, html_data in res["dashboard_assets"].items():
                if html_data:
                    zf.writestr(filename, html_data.encode("utf-8"))

        st.download_button(
            label="Download Extraction Package (.zip)",
            data=zip_buffer.getvalue(),
            file_name="Topic_Modeling_Artifacts.zip",
            mime="application/zip",
            type="primary"
        )


if __name__ == "__main__":
    main()