import io
import os
import re
import zipfile
from functools import lru_cache
from typing import Dict, List, Optional, Set

import nltk
import pandas as pd
import spacy
from nltk.corpus import stopwords

if "NLTK_DATA" in os.environ:
    nltk.data.path.append(os.environ["NLTK_DATA"])

SPACY_MODELS: Dict[str, str] = {
    "English": "en_core_web_sm",
    "German": "de_core_news_sm",
    "French": "fr_core_news_sm",
}

NLTK_LANGUAGES: Dict[str, str] = {
    "Spanish": "spanish",
    "Italian": "italian",
    "Dutch": "dutch",
    "Portuguese": "portuguese",
    "Russian": "russian",
    "Arabic": "arabic",
}


@lru_cache(maxsize=None)
def _load_spacy_model(language: str) -> Optional[spacy.language.Language]:
    """
    Load and cache a spaCy language model for the given language.

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
        return None


def load_zip_texts(zip_bytes: bytes) -> pd.DataFrame:
    """
    Load non-empty text files from a ZIP archive into a DataFrame.

    Files whose basename starts with ``._`` are ignored.

    Args:
        zip_bytes: The ZIP archive content as bytes.

    Returns:
        A pandas DataFrame with columns ``Filename`` and ``Text``.
    """
    data = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        for filename in z.namelist():
            lower_name = filename.lower()
            if lower_name.endswith(".txt") and not os.path.basename(
                filename
            ).startswith("._"):
                content = z.read(filename).decode("utf-8", errors="ignore")
                if content.strip():
                    data.append({"Filename": filename, "Text": content})
    return pd.DataFrame(data, columns=["Filename", "Text"])


def get_stopword_set(language: str, custom_stopwords_str: str) -> Set[str]:
    """
    Build a stopword set from custom, spaCy, and NLTK sources.

    Args:
        language: The language name.
        custom_stopwords_str: A comma-separated string of custom stopwords.

    Returns:
        A set of lowercase stopwords.
    """
    stop_set: Set[str] = set()

    if custom_stopwords_str:
        stop_set.update(
            {w.strip().lower() for w in custom_stopwords_str.split(",") if w.strip()}
        )

    nlp = _load_spacy_model(language)
    if nlp is not None:
        stop_set.update(nlp.Defaults.stop_words)
    elif language in NLTK_LANGUAGES:
        try:
            stop_set.update(stopwords.words(NLTK_LANGUAGES[language]))
        except LookupError:
            pass

    return stop_set


def preprocess_texts_for_lda(
    texts: List[str],
    language: str,
    custom_stopwords_str: str,
    use_bigrams: bool,
) -> List[List[str]]:
    """
    Preprocess texts for LDA topic modeling.

    Texts are tokenized, normalized, filtered by stopwords, and optionally
    enriched with bigrams.

    Args:
        texts: The input documents to preprocess.
        language: The language name.
        custom_stopwords_str: A comma-separated string of custom stopwords.
        use_bigrams: Whether to generate bigrams with gensim.

    Returns:
        A list of tokenized documents, where each document is a list of tokens.
    """
    import gensim

    stop_set = get_stopword_set(language, custom_stopwords_str)
    processed_texts: List[List[str]] = []

    nlp = _load_spacy_model(language)
    if nlp is not None:
        for doc in nlp.pipe(texts, batch_size=50):
            tokens = [
                token.lemma_.lower()
                for token in doc
                if token.is_alpha
                and len(token) > 2
                and token.lemma_.lower() not in stop_set
            ]
            processed_texts.append(tokens)
    else:
        for text in texts:
            words = re.findall(r"\b\w{3,}\b", text.lower())
            tokens = [w for w in words if w not in stop_set and not w.isnumeric()]
            processed_texts.append(tokens)

    if use_bigrams and processed_texts:
        bigram = gensim.models.Phrases(processed_texts, min_count=5, threshold=10)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        processed_texts = [list(bigram_mod[doc]) for doc in processed_texts]

    return processed_texts