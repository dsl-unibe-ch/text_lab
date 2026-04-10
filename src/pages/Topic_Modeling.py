import os
import sys
import io
import zipfile
import traceback

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Ensure absolute imports resolve correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))

from auth import check_token
from core.topic_utils import load_zip_texts, preprocess_texts_for_lda, get_stopword_set
from core.lda_engine import (
    train_lda_model,
    generate_lda_keywords_df,
    generate_lda_document_topics_df,
    generate_lda_html
)
from core.bertopic_engine import (
    train_bertopic_model,
    generate_bertopic_keywords_df,
    generate_bertopic_document_topics_df,
    generate_bertopic_visualizations,
    generate_topics_over_time_html
)
from core.top2vec_engine import (
    train_top2vec_model,
    generate_top2vec_keywords_df,
    generate_top2vec_document_topics_df,
    generate_top2vec_barchart_html
)

check_token()

SUPPORTED_LANGUAGES = [
    "English", "German", "French", 
    "Spanish", "Italian", "Dutch", 
    "Portuguese", "Russian", "Arabic", 
    "Other / Mixed"
]


def main() -> None:
    st.set_page_config(page_title="Topic Modeling", layout="wide")
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
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                df = df.dropna(how='all')
                
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
                        help="Requires a column with dates or timestamps to track topic evolution."
                    )
                    if enable_dtm:
                        date_column = st.selectbox(
                            "Timestamp Column", 
                            options=df.columns.tolist()
                        )
            else:
                df = load_zip_texts(uploaded_file.getvalue())
                text_column = "Text"
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
    
    col_basic, col_adv = st.columns(2)
    
    # CRITICAL FIX: Initialize all variables with safe defaults to prevent UnboundLocalError
    clustering_params = {}
    bertopic_nr_topics = None
    clustering_algo_clean = "HDBSCAN"
    ngram_range = (1, 1)
    
    backend_clean = "doc2vec"
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
                bertopic_nr_topics = None # KMeans handles the exact topic count naturally
            else:
                auto_topics = st.checkbox("Auto-detect optimal number of topics", value=True)
                bertopic_nr_topics = "auto" if auto_topics else st.slider("Target Number of Topics", min_value=2, max_value=100, value=10, step=1)
                min_cluster_size = st.number_input("Minimum Documents per Topic", min_value=3, max_value=500, value=10, step=1)
                clustering_params["min_cluster_size"] = min_cluster_size
                clustering_algo_clean = "HDBSCAN"

        elif "Top2Vec" in algorithm:
            auto_topics = st.checkbox("Auto-detect optimal number of topics", value=True)
            if auto_topics:
                num_topics = "auto"
                st.info("The algorithm will dynamically determine the best number of topics using density-based clustering.")
            else:
                num_topics = st.slider("Target Number of Topics", min_value=2, max_value=100, value=10, step=1)
        
        else: # LDA
            num_topics = st.slider("Number of Topics", min_value=2, max_value=50, value=10, step=1)

    with col_adv:
        st.subheader("Advanced Processing")
        
        if "Top2Vec" in algorithm:
            st.info("Top2Vec relies on the natural, raw structure of sentences to generate joint embeddings. Custom stopwords are disabled for this algorithm.")
            
            top2vec_backend = st.radio(
                "Embedding Backend",
                ["Transformer (Pre-trained)", "Doc2Vec (Train from scratch)"],
                help="Transformers are faster and understand general language. Doc2Vec trains specifically on your data."
            )
            backend_clean = "transformer" if "Transformer" in top2vec_backend else "doc2vec"
            
            top2vec_speed = st.select_slider(
                "Training Depth (Doc2Vec only)",
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
                extract_phrases = st.checkbox("Extract Phrases (N-grams)", value=False)
                ngram_range = (1, 2) if extract_phrases else (1, 1)
                
            elif "LDA" in algorithm:
                use_bigrams = st.checkbox("Extract Phrases (Bigrams)", value=False)
                passes = st.slider("Training Passes (Iterations)", min_value=5, max_value=50, value=10, step=5)

    # --- 3. Execution & Results ---
    st.header("3. Execution", divider="gray")
    
    if st.button("Run Topic Extraction", type="primary", disabled=(df is None)):
        
        df = df.dropna(subset=[text_column])
        
        if enable_dtm and "BERTopic" in algorithm:
            df = df.dropna(subset=[date_column])
            timestamps = df[date_column].tolist()
            
        raw_texts = df[text_column].astype(str).tolist()
        
        if len(raw_texts) < 5:
            st.error("A minimum of 5 valid documents is required to perform topic modeling.")
            st.stop()

        dashboard_assets = {}

        # --- Pipeline Routing ---
        if "LDA" in algorithm:
            with st.spinner("Step 1/3: Cleaning and tokenizing text corpus..."):
                try:
                    processed_texts = preprocess_texts_for_lda(raw_texts, language, custom_stopwords, use_bigrams)
                except Exception as e:
                    st.error("Failed to process text corpus. See technical details below:")
                    st.code(traceback.format_exc())
                    print(f"--- LDA Preprocessing Error ---\n{traceback.format_exc()}", file=sys.stderr)
                    st.stop()

            with st.spinner(f"Step 2/3: Training LDA Model for {num_topics} topics..."):
                try:
                    lda_model, corpus, id2word = train_lda_model(processed_texts, num_topics, passes)
                except Exception as e:
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
                except Exception as e:
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
                except Exception as e:
                    st.error("Failed to train Top2Vec model. See technical details below:")
                    st.code(traceback.format_exc())
                    print(f"--- Top2Vec Training Error ---\n{traceback.format_exc()}", file=sys.stderr)
                    st.stop()

            with st.spinner("Step 2/3: Extracting document classifications..."):
                try:
                    topic_df = generate_top2vec_keywords_df(topic_model, num_topics)
                    docs_df = generate_top2vec_document_topics_df(topic_model, df, num_topics)
                except Exception as e:
                    st.error("Failed to extract classifications. See technical details below:")
                    st.code(traceback.format_exc())
                    print(f"--- Top2Vec Extraction Error ---\n{traceback.format_exc()}", file=sys.stderr)
                    st.stop()

            with st.spinner("Step 3/3: Generating visualization suites..."):
                try:
                    html_string = generate_top2vec_barchart_html(topic_model, num_topics)
                    dashboard_assets["top2vec_barchart.html"] = html_string
                except Exception as e:
                    st.error("Failed to generate dashboard artifacts. See technical details below:")
                    st.code(traceback.format_exc())
                    print(f"--- Top2Vec Dashboard Error ---\n{traceback.format_exc()}", file=sys.stderr)
                    st.stop()

        else: # BERTopic
            with st.spinner("Step 1/3: Generating embeddings and training BERTopic model (This may take a moment)..."):
                try:
                    stop_set = get_stopword_set(language, custom_stopwords)
                    topic_model, topics = train_bertopic_model(
                        texts=raw_texts, 
                        language=language, 
                        num_topics=bertopic_nr_topics, 
                        stop_words_set=stop_set,
                        clustering_algo=clustering_algo_clean,
                        clustering_params=clustering_params,
                        ngram_range=ngram_range
                    )
                except Exception as e:
                    st.error("Failed to train BERTopic model. See technical details below:")
                    st.code(traceback.format_exc())
                    print(f"--- BERTopic Training Error ---\n{traceback.format_exc()}", file=sys.stderr)
                    st.stop()

            with st.spinner("Step 2/3: Extracting document classifications..."):
                try:
                    topic_df = generate_bertopic_keywords_df(topic_model)
                    docs_df = generate_bertopic_document_topics_df(topics, df)
                except Exception as e:
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
                    
                    if enable_dtm:
                        dtm_html = generate_topics_over_time_html(topic_model, raw_texts, timestamps)
                        dashboard_assets["topics_over_time.html"] = dtm_html
                        
                except Exception as e:
                    st.error("Failed to generate dashboard artifacts. See technical details below:")
                    st.code(traceback.format_exc())
                    print(f"--- BERTopic Dashboard Error ---\n{traceback.format_exc()}", file=sys.stderr)
                    st.stop()

        st.success("Topic Modeling execution complete.")

        # --- 4. Display Results ---
        st.header("Results Analysis", divider="gray")
        
        st.subheader("Topic Dictionary")
        if "BERTopic" in algorithm or "Top2Vec" in algorithm:
            st.caption("Note: Density-based algorithms automatically classify outlier documents into an 'Outlier' category.")
        st.dataframe(topic_df, use_container_width=True, hide_index=True)

        if "LDA" in algorithm:
            st.subheader("Interactive Topic Dashboard")
            components.html(dashboard_assets["lda_dashboard.html"], width=1300, height=800, scrolling=False)
            
        elif "Top2Vec" in algorithm:
            st.subheader("Interactive Topic Dashboard")
            components.html(dashboard_assets["top2vec_barchart.html"], width=1000, height=800, scrolling=True)
            
        else:
            tab1, tab2, tab3, tab4 = st.tabs(["Intertopic Distance", "Word Scores", "Similarity Heatmap", "Topics Over Time"])
            
            with tab1:
                st.caption("Maps the semantic distance between discovered topics.")
                components.html(dashboard_assets.get("intertopic_distance.html", ""), width=1000, height=600, scrolling=False)
            with tab2:
                st.caption("Displays the highest frequency terms for the top topics.")
                components.html(dashboard_assets.get("topic_barchart.html", ""), width=1000, height=600, scrolling=False)
            with tab3:
                st.caption("Shows how semantically similar the generated topics are to each other.")
                components.html(dashboard_assets.get("similarity_heatmap.html", ""), width=1000, height=600, scrolling=False)
            with tab4:
                if enable_dtm:
                    st.caption("Visualizes topic frequency evolution over the provided timestamps.")
                    components.html(dashboard_assets.get("topics_over_time.html", ""), width=1000, height=600, scrolling=False)
                else:
                    st.info("Dynamic Topic Modeling was not enabled during configuration.")

        st.subheader("Export Artifacts")
        st.write("Download your original dataset augmented with topic classifications, alongside the standalone interactive HTML dashboards.")
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("document_topics.csv", docs_df.to_csv(index=False).encode('utf-8'))
            zf.writestr("topic_keywords.csv", topic_df.to_csv(index=False).encode('utf-8'))
            for filename, html_data in dashboard_assets.items():
                if html_data:
                    zf.writestr(filename, html_data.encode('utf-8'))
        
        st.download_button(
            label="Download Extraction Package (.zip)",
            data=zip_buffer.getvalue(),
            file_name="Topic_Modeling_Artifacts.zip",
            mime="application/zip",
            type="primary"
        )

if __name__ == "__main__":
    main()