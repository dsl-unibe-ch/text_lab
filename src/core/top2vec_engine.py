import sys
import traceback
from typing import List, Tuple, Dict, Union

import numpy as np
import pandas as pd
from top2vec import Top2Vec
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def train_top2vec_model(
    texts: List[str], 
    language: str, 
    embedding_backend: str = "doc2vec",
    speed: str = "learn",
    min_count: int = 10,
    target_topics: Union[int, str] = "auto"
) -> Top2Vec:
    """
    Train a Top2Vec model using joint document and word semantic embeddings.
    """
    # Map backend to the correct string for Top2Vec
    if embedding_backend == "doc2vec":
        embed_model = "doc2vec"
    else:
        # Leverage the locally cached models downloaded previously
        embed_model = "all-MiniLM-L6-v2" if language == "English" else "paraphrase-multilingual-MiniLM-L12-v2"
        
    topic_model = Top2Vec(
        documents=texts,
        speed=speed,
        min_count=min_count,
        embedding_model=embed_model,
        workers=8
    )
    
    # Top2Vec finds topics automatically. If the user requested a specific number,
    # and the model found more than that, we hierarchically reduce them.
    if target_topics != "auto" and isinstance(target_topics, int):
        num_found = topic_model.get_num_topics()
        if target_topics < num_found:
            topic_model.hierarchical_topic_reduction(num_topics=target_topics)
            
    return topic_model


def generate_top2vec_keywords_df(topic_model: Top2Vec, target_topics: Union[int, str]) -> pd.DataFrame:
    """
    Extract the top keywords for each Top2Vec topic into a clean DataFrame.
    """
    is_reduced = target_topics != "auto" and topic_model.get_num_topics(reduced=True) < topic_model.get_num_topics(reduced=False)
    
    topic_words, word_scores, topic_nums = topic_model.get_topics(reduced=is_reduced)
    topic_sizes, _ = topic_model.get_topic_sizes(reduced=is_reduced)
    
    topic_data = []
    for idx, t_num in enumerate(topic_nums):
        keywords = ", ".join(topic_words[idx][:10])
        topic_data.append({
            "Topic": int(t_num) + 1,  # 1-based indexing for UI
            "Count": int(topic_sizes[idx]),
            "Keywords": keywords
        })
        
    return pd.DataFrame(topic_data)


def generate_top2vec_document_topics_df(
    topic_model: Top2Vec, 
    original_df: pd.DataFrame,
    target_topics: Union[int, str]
) -> pd.DataFrame:
    """
    Append dominant Top2Vec classifications and similarity scores to the dataset.
    """
    is_reduced = target_topics != "auto" and topic_model.get_num_topics(reduced=True) < topic_model.get_num_topics(reduced=False)
    
    doc_topics, doc_dist, _, _ = topic_model.get_documents_topics(
        doc_ids=list(range(len(original_df))),
        reduced=is_reduced,
        num_topics=1
    )
    
    # Safely flatten arrays to handle varying Top2Vec return shapes
    doc_topics_flat = np.array(doc_topics).flatten()
    doc_dist_flat = np.array(doc_dist).flatten()
    
    result_df = original_df.copy()
    
    # CRITICAL: Notice how it is just 't' and 'd' here, absolutely no t[0] or d[0]
    result_df['Dominant_Topic'] = [int(t) + 1 for t in doc_topics_flat]
    result_df['Topic_Confidence'] = [round(float(d), 4) for d in doc_dist_flat]
    
    cols = result_df.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    return result_df[cols]


def generate_top2vec_barchart_html(topic_model: Top2Vec, target_topics: Union[int, str]) -> str:
    """
    Generate a custom Plotly interactive bar chart mimicking BERTopic's dashboard.
    """
    is_reduced = target_topics != "auto" and topic_model.get_num_topics(reduced=True) < topic_model.get_num_topics(reduced=False)
    
    num_topics = topic_model.get_num_topics(reduced=is_reduced)
    display_topics = min(num_topics, 12)  # Cap at 12 topics for visual clarity
    
    topic_words, word_scores, topic_nums = topic_model.get_topics(num_topics=display_topics, reduced=is_reduced)
    
    cols = 4
    rows = (display_topics + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, cols=cols, 
        subplot_titles=[f"Topic {t_num + 1}" for t_num in topic_nums],
        horizontal_spacing=0.1,
        vertical_spacing=0.1
    )
    
    for i in range(display_topics):
        row = (i // cols) + 1
        col = (i % cols) + 1
        
        # Take top 8 words and reverse for bottom-to-top rendering in Plotly
        words = topic_words[i][:8][::-1]
        scores = word_scores[i][:8][::-1]
        
        fig.add_trace(
            go.Bar(x=scores, y=words, orientation='h', marker_color="#4A90E2", showlegend=False),
            row=row, col=col
        )
        
        fig.update_xaxes(showticklabels=False, row=row, col=col)
        fig.update_yaxes(tickfont=dict(size=11), row=row, col=col)
        
    fig.update_layout(
        height=300 * rows,
        title_text="Top2Vec Word Similarity Scores",
        template="plotly_white",
        margin=dict(t=50, b=20, l=20, r=20)
    )
    
    return fig.to_html(full_html=False, include_plotlyjs='cdn')