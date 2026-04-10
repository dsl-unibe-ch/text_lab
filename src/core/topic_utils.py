import io
import re
import zipfile
from typing import List, Set, Dict

import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords

# Ensure NLTK data path is recognized by the container environment
import os
if "NLTK_DATA" in os.environ:
    nltk.data.path.append(os.environ["NLTK_DATA"])

SPACY_MODELS: Dict[str, str] = {
    "English": "en_core_web_sm",
    "German": "de_core_news_sm",
    "French": "fr_core_news_sm"
}

NLTK_LANGUAGES: Dict[str, str] = {
    "Spanish": "spanish",
    "Italian": "italian",
    "Dutch": "dutch",
    "Portuguese": "portuguese",
    "Russian": "russian",
    "Arabic": "arabic"
}


def load_zip_texts(zip_bytes: bytes) -> pd.DataFrame:
    """
    Extract text files from a ZIP archive into a pandas DataFrame.

    Args:
        zip_bytes (bytes): The raw bytes of the uploaded ZIP file.

    Returns:
        pd.DataFrame: A dataframe containing 'Filename' and 'Text' columns.
    """
    data = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as z:
        for filename in z.namelist():
            if filename.endswith('.txt') and not filename.startswith('._'):
                content = z.read(filename).decode('utf-8', errors='ignore')
                if content.strip():
                    data.append({"Filename": filename, "Text": content})
    return pd.DataFrame(data)


def get_stopword_set(language: str, custom_stopwords_str: str) -> Set[str]:
    """
    Compile a comprehensive set of stopwords based on language and user input.

    Args:
        language (str): The language selected in the UI.
        custom_stopwords_str (str): Comma-separated custom stopwords.

    Returns:
        Set[str]: A set of lowercase stopword strings.
    """
    stop_set = set()
    
    if custom_stopwords_str:
        stop_set.update({w.strip().lower() for w in custom_stopwords_str.split(',') if w.strip()})

    if language in SPACY_MODELS:
        nlp = spacy.load(SPACY_MODELS[language], disable=['parser', 'ner'])
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
    use_bigrams: bool
) -> List[List[str]]:
    """
    Clean and tokenize texts specifically for LDA, which requires heavy preprocessing.

    Args:
        texts (list): List of raw text documents.
        language (str): The language selected in the UI.
        custom_stopwords_str (str): Comma-separated custom stopwords.
        use_bigrams (bool): Whether to generate n-grams (phrases).

    Returns:
        list: A list of tokenized documents.
    """
    import gensim
    
    stop_set = get_stopword_set(language, custom_stopwords_str)
    processed_texts: List[List[str]] = []

    if language in SPACY_MODELS:
        nlp = spacy.load(SPACY_MODELS[language], disable=['parser', 'ner'])
        for doc in nlp.pipe(texts, batch_size=50):
            tokens = [
                token.lemma_.lower() for token in doc 
                if token.is_alpha 
                and len(token) > 2
                and token.lemma_.lower() not in stop_set
            ]
            processed_texts.append(tokens)
    else:
        for text in texts:
            words = re.findall(r'\b\w{3,}\b', text.lower())
            tokens = [w for w in words if w not in stop_set and not w.isnumeric()]
            processed_texts.append(tokens)

    if use_bigrams:
        bigram = gensim.models.Phrases(processed_texts, min_count=5, threshold=10)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        processed_texts = [bigram_mod[doc] for doc in processed_texts]

    return processed_texts