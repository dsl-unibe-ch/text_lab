import sys
import traceback

from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

from core.topic_modeling.topic_utils import preprocess_texts_for_lda


def evaluate_topic_quality(
    topic_keywords: list[list[str]],
    raw_texts: list[str],
    language: str,
    custom_stopwords_str: str
) -> dict[str, float]:
    """
    Calculate Topic Diversity and Gensim Coherence metrics (C_v, C_npmi, U_mass).

    Args:
        topic_keywords: A list of topics, where each topic is a list of top words.
        raw_texts: The raw string documents from the dataset.
        language: The primary language of the texts.
        custom_stopwords_str: Comma-separated custom stopwords to ignore.

    Returns:
        A dictionary containing the calculated evaluation metrics.
    """
    metrics = {
        "Topic Diversity": 0.0,
        "Coherence (C_v)": 0.0,
        "Coherence (C_npmi)": 0.0,
        "Coherence (U_mass)": 0.0
    }

    if not topic_keywords or not raw_texts:
        return metrics

    # 1. Calculate Topic Diversity (Percentage of unique words across all topics)
    all_words = [word for topic in topic_keywords for word in topic]
    unique_words = set(all_words)
    metrics["Topic Diversity"] = round(len(unique_words) / len(all_words), 4) if all_words else 0.0

    # 2. Tokenize texts to create a strict Gensim Dictionary and Corpus
    tokenized_texts = preprocess_texts_for_lda(
        texts=raw_texts, 
        language=language, 
        custom_stopwords_str=custom_stopwords_str, 
        use_bigrams=False
    )
    
    dictionary = Dictionary(tokenized_texts)
    
    # Filter out out-of-vocabulary words to prevent Gensim KeyErrors
    safe_topics = []
    for topic in topic_keywords:
        safe_topic = [w for w in topic if w in dictionary.token2id]
        if safe_topic:
            safe_topics.append(safe_topic)

    if not safe_topics:
        return metrics

    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

    # 3. Calculate Coherence Metrics safely
    try:
        cm_cv = CoherenceModel(
            topics=safe_topics, texts=tokenized_texts, dictionary=dictionary, coherence="c_v"
        )
        metrics["Coherence (C_v)"] = round(cm_cv.get_coherence(), 4)
    except Exception:
        print(f"--- C_v Coherence Error ---\n{traceback.format_exc()}", file=sys.stderr)

    try:
        cm_npmi = CoherenceModel(
            topics=safe_topics, texts=tokenized_texts, dictionary=dictionary, coherence="c_npmi"
        )
        metrics["Coherence (C_npmi)"] = round(cm_npmi.get_coherence(), 4)
    except Exception:
        print(f"--- C_npmi Coherence Error ---\n{traceback.format_exc()}", file=sys.stderr)

    try:
        cm_umass = CoherenceModel(
            topics=safe_topics, corpus=corpus, dictionary=dictionary, coherence="u_mass"
        )
        metrics["Coherence (U_mass)"] = round(cm_umass.get_coherence(), 4)
    except Exception:
        print(f"--- U_mass Coherence Error ---\n{traceback.format_exc()}", file=sys.stderr)

    return metrics