import sys
import traceback
from typing import Any

import numpy as np
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

from core.topic_modeling.topic_utils import (
    preprocess_texts_for_lda,
    tokenize_texts_for_coherence,
)


def evaluate_topic_quality(
    topic_keywords: list[list[str]],
    raw_texts: list[str],
    language: str,
    custom_stopwords_str: str,
    tokenized_texts: list[list[str]] | None = None,
    use_lemmatization: bool = True,
) -> dict[str, float]:
    """
    Calculate Topic Diversity and Gensim Coherence metrics (C_v, C_npmi, U_mass).

    Args:
        topic_keywords: A list of topics, where each topic is a list of top words.
        raw_texts: The raw string documents from the dataset.
        language: The primary language of the texts.
        custom_stopwords_str: Comma-separated custom stopwords to ignore.
        tokenized_texts: Optional pre-computed tokenized corpus (as returned by
            :func:`preprocess_texts_for_lda`). Providing this avoids
            re-tokenizing the corpus for the coherence metrics and takes
            precedence over ``use_lemmatization``.
        use_lemmatization: When ``tokenized_texts`` is not provided, controls
            whether the corpus is lemmatized (``True``, appropriate for LDA,
            which trains on lemmas) or tokenized to surface forms (``False``,
            required for BERTopic / Top2Vec, whose keywords are surface forms
            from the original text). Using the wrong mode causes most
            keywords to be filtered out of the coherence dictionary and
            produces artificially low scores.

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

    # 2. Tokenize texts (or reuse a pre-computed tokenization) for Gensim.
    if tokenized_texts is None:
        if use_lemmatization:
            tokenized_texts = preprocess_texts_for_lda(
                texts=raw_texts,
                language=language,
                custom_stopwords_str=custom_stopwords_str,
                use_bigrams=False,
            )
        else:
            tokenized_texts = tokenize_texts_for_coherence(
                texts=raw_texts,
                language=language,
                custom_stopwords_str=custom_stopwords_str,
            )
    
    dictionary = Dictionary(tokenized_texts)
    
    # Filter out out-of-vocabulary words to prevent Gensim KeyErrors
    safe_topics = []
    for topic in topic_keywords:
        safe_topic = [w for w in topic if w in dictionary.token2id]
        if len(safe_topic) >= 2: # Need at least 2 words to measure co-occurrence
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


def calculate_lda_perplexity(lda_model: Any, corpus: list[list[tuple[int, int]]]) -> float | None:
    """
    Calculate the perplexity of a trained Gensim LDA model.

    Perplexity is a statistical measure of how well a probability model predicts
    a sample. Lower perplexity indicates better generalization performance.

    Args:
        lda_model: A trained gensim.models.LdaModel instance.
        corpus: The bag-of-words corpus used to train or evaluate the model.

    Returns:
        The calculated perplexity score as a float, rounded to 4 decimal places,
        or ``None`` if the calculation failed or returned a non-finite value.
    """
    try:
        # Gensim returns the bound (log perplexity). We exponentiate it for the standard metric.
        log_perplexity = lda_model.log_perplexity(corpus)
        perplexity = float(np.exp2(-log_perplexity))
        if not np.isfinite(perplexity):
            return None
        return round(perplexity, 4)
    except Exception as e:
        print(f"--- LDA Perplexity Error ---\n{e}", file=sys.stderr)
        return None


def calculate_jaccard_stability(
    run_1_topics: list[list[str]], 
    run_2_topics: list[list[str]]
) -> float:
    """
    Calculate the Topic Stability between two independent model runs.
    
    This function uses Jaccard Similarity to compare topic keywords. It finds 
    the best-matching topic in Run 2 for every topic in Run 1 and averages 
    the maximum similarity scores. A score of 1.0 means perfectly identical 
    topics; 0.0 means completely different.

    Args:
        run_1_topics: A list of topics from the first run (each topic is a list of words).
        run_2_topics: A list of topics from the second run (each topic is a list of words).

    Returns:
        The average Jaccard stability score across all topics as a float, 
        rounded to 4 decimal places.
    """
    if not run_1_topics or not run_2_topics:
        return 0.0

    total_similarity = 0.0
    for topic1 in run_1_topics:
        set1 = set(topic1)
        max_sim = 0.0
        
        for topic2 in run_2_topics:
            set2 = set(topic2)
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            sim = float(intersection / union) if union > 0 else 0.0
            if sim > max_sim:
                max_sim = sim
                
        total_similarity += max_sim

    return round(total_similarity / len(run_1_topics), 4)