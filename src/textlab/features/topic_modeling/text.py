"""Tokenizing documents: stopwords, spaCy lemmas and regex tokens.

LDA trains on the lemmatized tokens (:func:`preprocess_texts_for_lda`);
coherence is measured on surface forms for the other algorithms
(:func:`tokenize_texts_for_coherence`).
"""

from __future__ import annotations

import logging
import os
import re
from functools import cache

import nltk
import spacy
from nltk.corpus import stopwords

LOGGER = logging.getLogger(__name__)

if "NLTK_DATA" in os.environ:
    nltk.data.path.append(os.environ["NLTK_DATA"])

SPACY_MODELS: dict[str, str] = {
    "English": "en_core_web_sm",
    "German": "de_core_news_sm",
    "French": "fr_core_news_sm",
}

NLTK_LANGUAGES: dict[str, str] = {
    "Spanish": "spanish",
    "Italian": "italian",
    "Dutch": "dutch",
    "Portuguese": "portuguese",
    "Russian": "russian",
    "Arabic": "arabic",
}


@cache
def _load_spacy_model(language: str) -> spacy.language.Language | None:
    """Load and cache a spaCy language model for the given language.

    Args:
        language: The language name.

    Returns:
        The loaded spaCy language model if available, otherwise None.
    """
    model_name = SPACY_MODELS.get(language)
    if not model_name:
        return None
    try:
        return spacy.load(model_name, disable=["parser", "ner"])
    except OSError:
        LOGGER.warning(
            "spaCy model '%s' for %s is not installed. Run 'python -m spacy "
            "download %s' in the host environment. Falling back to basic "
            "regex tokenization for this job.",
            model_name,
            language,
            model_name,
        )
        return None


def get_stopword_set(language: str, custom_stopwords_str: str) -> set[str]:
    """Build a stopword set from custom, spaCy, and NLTK sources.

    Args:
        language: The language name.
        custom_stopwords_str: A comma-separated string of custom stopwords.

    Returns:
        A set of lowercase stopwords.
    """
    stop_set: set[str] = set()

    if custom_stopwords_str:
        stop_set.update(
            {
                w.strip().lower()
                for w in custom_stopwords_str.split(",")
                if w.strip()
            }
        )

    nlp = _load_spacy_model(language)
    if nlp is not None:
        stop_set.update(nlp.Defaults.stop_words)
    elif language in NLTK_LANGUAGES:
        try:
            stop_set.update(stopwords.words(NLTK_LANGUAGES[language]))
        except LookupError:
            LOGGER.warning(
                "NLTK stopwords for '%s' are not installed. Only custom "
                "stopwords will be applied.",
                language,
            )
    else:
        LOGGER.info(
            "No default stopword list is bundled for '%s'. Only custom "
            "stopwords will be applied.",
            language,
        )

    return stop_set


def split_long_text(text: str, max_chars: int) -> list[str]:
    """Split a text into consecutive chunks of at most ``max_chars``.

    Chunks end at the last space before the limit when possible, so words are
    not cut in half. Joining the chunks gives back the original text.

    Args:
        text: The text to split.
        max_chars: The maximum chunk length; must be positive.

    Returns:
        The chunks, in order. Short texts are returned as a single chunk.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            split_at = text.rfind(" ", start, end)
            if split_at > start:
                end = split_at
        chunks.append(text[start:end])
        start = end
    return chunks


def _regex_tokens(text: str, stop_set: set[str]) -> list[str]:
    """Tokenize a text with a regex, for languages without a spaCy model.

    Args:
        text: The document to tokenize.
        stop_set: Lowercase stopwords to remove.

    Returns:
        Lowercase, non-numeric tokens of at least three characters.
    """
    words = re.findall(r"\b\w{3,}\b", text.lower())
    return [w for w in words if w not in stop_set and not w.isnumeric()]


def _tokenize_texts(
    texts: list[str],
    language: str,
    custom_stopwords_str: str,
    lemmatize: bool,
) -> list[list[str]]:
    """Tokenize documents with spaCy when available, otherwise with a regex.

    spaCy refuses texts longer than ``nlp.max_length`` characters, so long
    documents are processed in chunks and their tokens concatenated.

    Args:
        texts: The documents to tokenize.
        language: The language name.
        custom_stopwords_str: A comma-separated string of custom stopwords.
        lemmatize: Whether to return lemmas (``True``) or surface forms.

    Returns:
        One list of lowercase alphabetic tokens (longer than two characters,
        stopwords removed) per document.
    """
    stop_set = get_stopword_set(language, custom_stopwords_str)
    nlp = _load_spacy_model(language)
    if nlp is None:
        return [_regex_tokens(text, stop_set) for text in texts]

    processed_texts: list[list[str]] = [[] for _ in texts]
    chunks = (
        (chunk, index)
        for index, text in enumerate(texts)
        for chunk in split_long_text(text, nlp.max_length)
    )
    for doc, index in nlp.pipe(chunks, as_tuples=True, batch_size=50):
        for token in doc:
            if not token.is_alpha or len(token) <= 2:
                continue
            form = (token.lemma_ if lemmatize else token.text).lower()
            if form not in stop_set:
                processed_texts[index].append(form)

    return processed_texts


def tokenize_texts_for_coherence(
    texts: list[str],
    language: str,
    custom_stopwords_str: str,
) -> list[list[str]]:
    """Tokenize texts into surface-form tokens for coherence evaluation.

    Unlike :func:`preprocess_texts_for_lda`, this function does **not**
    lemmatize tokens. It is intended for evaluating models (BERTopic,
    Top2Vec) whose keywords are drawn from the original surface forms of
    the corpus. Using lemmatized tokens for those models causes most
    keywords to be missing from the coherence dictionary and produces
    artificially low Coherence / C_npmi / U_mass scores.

    Tokens are lowercased, restricted to alphabetic tokens of length > 2,
    and filtered by the stopword set for the given language.

    Args:
        texts: The input documents to tokenize.
        language: The language name.
        custom_stopwords_str: A comma-separated string of custom stopwords.

    Returns:
        A list of tokenized documents, where each document is a list of
        surface-form tokens.
    """
    return _tokenize_texts(
        texts, language, custom_stopwords_str, lemmatize=False
    )


def preprocess_texts_for_lda(
    texts: list[str],
    language: str,
    custom_stopwords_str: str,
    use_bigrams: bool,
) -> list[list[str]]:
    """Preprocess texts for LDA topic modeling.

    Texts are tokenized, lemmatized when a spaCy model is available, filtered
    by stopwords, and optionally enriched with bigrams.

    Args:
        texts: The input documents to preprocess.
        language: The language name.
        custom_stopwords_str: A comma-separated string of custom stopwords.
        use_bigrams: Whether to generate bigrams with gensim.

    Returns:
        A list of tokenized documents, where each document is a list of tokens.
    """
    import gensim

    processed_texts = _tokenize_texts(
        texts, language, custom_stopwords_str, lemmatize=True
    )

    if use_bigrams and processed_texts:
        bigram = gensim.models.Phrases(
            processed_texts, min_count=5, threshold=10
        )
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        processed_texts = [list(bigram_mod[doc]) for doc in processed_texts]

    return processed_texts
