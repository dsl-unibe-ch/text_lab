import sys
import traceback
from typing import List, Tuple, Set, Union, Dict, Any

import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from hdbscan import HDBSCAN


def train_bertopic_model(
    texts: List[str], 
    language: str, 
    num_topics: Union[int, str, None], 
    stop_words_set: Set[str],
    clustering_algo: str = "HDBSCAN",
    clustering_params: Dict[str, Any] = None,
    ngram_range: Tuple[int, int] = (1, 1)
) -> Tuple[BERTopic, List[int]]:
    """
    Train a BERTopic model using Transformer embeddings and customizable clustering sub-models.
    
    Args:
        texts: List of raw text documents.
        language: UI language selection to determine the embedding model.
        num_topics: Target number of topics (int, "auto", or None).
        stop_words_set: Stopwords to ignore in the representation phase.
        clustering_algo: The clustering algorithm to use ("HDBSCAN" or "KMeans").
        clustering_params: Dictionary of parameters specific to the chosen clustering algorithm.
        ngram_range: Tuple indicating whether to extract unigrams, bigrams, etc.
        
    Returns:
        tuple: (Trained BERTopic model, List of topic assignments per document)
    """
    if clustering_params is None:
        clustering_params = {}

    embedding_model = "english" if language == "English" else "multilingual"
    
    vectorizer_model = CountVectorizer(
        stop_words=list(stop_words_set) if stop_words_set else None,
        ngram_range=ngram_range
    )
    
    # Configure the chosen clustering sub-model
    if clustering_algo == "KMeans":
        n_clusters = clustering_params.get("n_clusters", 10)
        cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
        # KMeans strictly dictates topic count; disable BERTopic's secondary reduction
        bertopic_nr_topics = None
    else:
        min_cluster_size = clustering_params.get("min_cluster_size", 10)
        # HDBSCAN requires prediction_data=True for BERTopic compatibility
        cluster_model = HDBSCAN(
            min_cluster_size=min_cluster_size, 
            metric='euclidean', 
            cluster_selection_method='eom', 
            prediction_data=True
        )
        bertopic_nr_topics = num_topics
    
    topic_model = BERTopic(
        language=embedding_model,
        vectorizer_model=vectorizer_model,
        hdbscan_model=cluster_model,
        nr_topics=bertopic_nr_topics,
        calculate_probabilities=False
    )
    
    topics, _ = topic_model.fit_transform(texts)
    return topic_model, topics


def generate_bertopic_keywords_df(topic_model: BERTopic) -> pd.DataFrame:
    """
    Extract the top keywords for each BERTopic topic into a clean DataFrame.
    """
    topic_info = topic_model.get_topic_info()
    topic_data = []
    
    for _, row in topic_info.iterrows():
        topic_id = int(row['Topic'])
        
        if topic_id == -1:
            continue  
            
        words = topic_model.get_topic(topic_id)
        if words:
            topic_keywords = ", ".join([word for word, score in words[:10]])
            topic_data.append({
                "Topic": topic_id + 1, 
                "Count": int(row['Count']), 
                "Keywords": topic_keywords
            })
            
    return pd.DataFrame(topic_data)


def generate_bertopic_document_topics_df(
    topics: List[int], 
    original_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Append dominant BERTopic classifications to the original dataset.
    """
    result_df = original_df.copy()
    
    # Map outlier topics (-1) to a readable string, shift others by +1
    formatted_topics = [t + 1 if t != -1 else "Outlier" for t in topics]
    result_df['Dominant_Topic'] = formatted_topics
    
    cols = result_df.columns.tolist()
    cols = [cols[-1]] + cols[:-1]
    return result_df[cols]


def generate_bertopic_visualizations(topic_model: BERTopic) -> Dict[str, str]:
    """
    Generate a suite of interactive Plotly HTML dashboards for BERTopic.
    
    Returns:
        Dict[str, str]: A dictionary containing HTML strings for various plots.
    """
    visualizations = {}
    topic_info = topic_model.get_topic_info()
    valid_topics = topic_info[topic_info['Topic'] != -1]
    
    if len(valid_topics) < 2:
        error_html = (
            "<div style='padding: 30px; font-family: sans-serif; color: #555; background: #f9f9f9; border-radius: 8px;'>"
            "<h3>Visualization Notice</h3>"
            "<p>The visual maps require at least <strong>2 distinct topics</strong>. "
            f"The model only identified {len(valid_topics)} valid topic in this dataset.</p>"
            "</div>"
        )
        visualizations["distance_map"] = error_html
        visualizations["barchart"] = error_html
        visualizations["heatmap"] = error_html
        return visualizations
        
    try:
        fig_distance = topic_model.visualize_topics()
        visualizations["distance_map"] = fig_distance.to_html(full_html=False, include_plotlyjs='cdn')
    except Exception:
        pass

    try:
        fig_barchart = topic_model.visualize_barchart(top_n_topics=12)
        visualizations["barchart"] = fig_barchart.to_html(full_html=False, include_plotlyjs='cdn')
    except Exception:
        pass

    try:
        fig_heatmap = topic_model.visualize_heatmap()
        visualizations["heatmap"] = fig_heatmap.to_html(full_html=False, include_plotlyjs='cdn')
    except Exception:
        pass

    return visualizations


def generate_topics_over_time_html(
    topic_model: BERTopic, 
    texts: List[str], 
    timestamps: List[Any]
) -> str:
    """
    Perform Dynamic Topic Modeling to visualize how topics evolve over time.
    """
    try:
        topics_over_time = topic_model.topics_over_time(texts, timestamps)
        fig = topic_model.visualize_topics_over_time(topics_over_time)
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    except Exception as e:
        print(f"--- Topics Over Time Error ---\n{traceback.format_exc()}", file=sys.stderr)
        return f"<div style='padding:20px; color:red;'>Failed to generate topics over time: {str(e)}</div>"