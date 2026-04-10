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
    if not processed_texts:
        raise ValueError("No processed texts were provided to the LDA trainer.")

    non_empty_docs = [doc for doc in processed_texts if doc]
    if len(non_empty_docs) < 5:
        raise ValueError(
            "Too few non-empty documents remained after preprocessing. "
            "Try removing some stopwords or uploading more text."
        )

    id2word = corpora.Dictionary(non_empty_docs)
    id2word.filter_extremes(no_below=2, no_above=0.9)

    if len(id2word) == 0:
        raise ValueError(
            "The LDA dictionary is empty after filtering extremes. "
            "Try reducing stopword removal, disabling bigrams, or using more diverse text."
        )

    corpus = [id2word.doc2bow(text) for text in processed_texts]

    if not any(len(bow) > 0 for bow in corpus):
        raise ValueError("All bag-of-words documents are empty after preprocessing.")

    lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        random_state=42,
        update_every=1,
        passes=passes,
        alpha="auto",
        per_word_topics=True
    )

    return lda_model, corpus, id2word


def generate_lda_keywords_df(lda_model: gensim.models.LdaModel, num_topics: int) -> pd.DataFrame:
    topic_data = []
    for i in range(num_topics):
        word_probs = lda_model.show_topic(i, topn=10)
        topic_keywords = ", ".join(word for word, _ in word_probs)
        topic_data.append({"Topic": i + 1, "Keywords": topic_keywords})
    return pd.DataFrame(topic_data)


def generate_lda_document_topics_df(
    lda_model: gensim.models.LdaModel,
    corpus: List[List[Tuple[int, int]]],
    original_df: pd.DataFrame
) -> pd.DataFrame:
    dominant_topics = []
    topic_probs = []

    for row_list in lda_model[corpus]:
        row = row_list[0] if lda_model.per_word_topics else row_list
        if row:
            row = sorted(row, key=lambda x: x[1], reverse=True)
            dominant_topics.append(row[0][0] + 1)
            topic_probs.append(round(float(row[0][1]), 4))
        else:
            dominant_topics.append(None)
            topic_probs.append(None)

    if len(dominant_topics) != len(original_df):
        raise ValueError(
            f"LDA output length ({len(dominant_topics)}) does not match dataframe length ({len(original_df)})."
        )

    result_df = original_df.copy()
    result_df["Dominant_Topic"] = dominant_topics
    result_df["Topic_Confidence"] = topic_probs

    cols = result_df.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    return result_df[cols]


def generate_lda_html(
    lda_model: gensim.models.LdaModel,
    corpus: List[List[Tuple[int, int]]],
    id2word: corpora.Dictionary
) -> str:
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word, mds="mmds")
    return pyLDAvis.prepared_data_to_html(vis)