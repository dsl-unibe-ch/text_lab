from typing import List, Tuple
import pandas as pd
import gensim
import gensim.corpora as corpora
import pyLDAvis
import pyLDAvis.gensim_models


def train_lda_model(
    processed_texts: List[List[str]], 
    num_topics: int, 
    passes: int
) -> Tuple[gensim.models.LdaModel, List[List[Tuple[int, int]]], corpora.Dictionary]:
    """
    Train a Gensim Latent Dirichlet Allocation (LDA) model.
    """
    id2word = corpora.Dictionary(processed_texts)
    id2word.filter_extremes(no_below=2, no_above=0.9)
    corpus = [id2word.doc2bow(text) for text in processed_texts]

    lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        random_state=42,
        update_every=1,
        passes=passes,
        alpha='auto',
        per_word_topics=True
    )
    
    return lda_model, corpus, id2word


def generate_lda_keywords_df(lda_model: gensim.models.LdaModel, num_topics: int) -> pd.DataFrame:
    """
    Extract the top keywords for each topic into a DataFrame.
    """
    topic_data = []
    for i in range(num_topics):
        word_probs = lda_model.show_topic(i, topn=10)
        topic_keywords = ", ".join([word for word, prop in word_probs])
        topic_data.append({"Topic": i + 1, "Keywords": topic_keywords})
    return pd.DataFrame(topic_data)


def generate_lda_document_topics_df(
    lda_model: gensim.models.LdaModel, 
    corpus: List[List[Tuple[int, int]]], 
    original_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Append dominant LDA topic information to the original user dataset.
    """
    dominant_topics = []
    topic_probs = []

    for row_list in lda_model[corpus]:
        row = row_list[0] if lda_model.per_word_topics else row_list
        if row:
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            dominant_topics.append(row[0][0] + 1)
            topic_probs.append(round(row[0][1], 4))
        else:
            dominant_topics.append(None)
            topic_probs.append(None)

    result_df = original_df.copy()
    result_df['Dominant_Topic'] = dominant_topics
    result_df['Topic_Confidence'] = topic_probs
    
    cols = result_df.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    return result_df[cols]


def generate_lda_html(
    lda_model: gensim.models.LdaModel, 
    corpus: List[List[Tuple[int, int]]], 
    id2word: corpora.Dictionary
) -> str:
    """
    Generate an interactive pyLDAvis HTML dashboard string.
    """
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word, mds='mmds')
    return pyLDAvis.prepared_data_to_html(vis)