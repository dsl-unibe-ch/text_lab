"""Which Whisper model to use for a language, and where it is.

Most languages use the standard WhisperX model, which WhisperX finds in the
mounted model store. Swiss German uses models stored in the site's custom
Whisper folder (``TEXT_LAB_CUSTOM_WHISPER_DIR`` in the site configuration).
"""

from __future__ import annotations

import os

from textlab.common.config import get_settings

#: Model used for every language without a dedicated model.
DEFAULT_MODEL = "large-v3-turbo"

#: Swiss German Whisper model, a folder in the custom Whisper folder.
SWHISPER_MODEL = "swhisper-large-1.1"
#: CTranslate2 conversion of Flurin17/whisper-large-v3-turbo-swiss-german,
#: built once with ``ct2-transformers-converter`` so WhisperX can load it.
FLURIN_SWISS_MODEL = "flurin-swiss-german-turbo-ct2"

#: Text Lab language codes with a dedicated model in the custom folder.
_CUSTOM_MODELS = {
    "ch_de": SWHISPER_MODEL,
    "ch_de_flurin": FLURIN_SWISS_MODEL,
}

#: Language codes that WhisperX aligns as another language.
_ALIGN_AS = {"ch_de": "de", "ch_de_flurin": "de"}


def alignment_language(language: str) -> str:
    """Return the language WhisperX transcribes and aligns ``language`` as.

    Args:
        language: A Text Lab language code.

    Returns:
        The code WhisperX understands; Swiss German becomes ``"de"``.
    """
    return _ALIGN_AS.get(language, language)


def default_model(language: str | None) -> str:
    """Return the model to use for a language.

    Args:
        language: A Text Lab language code, or ``None`` when the language
            is detected per file.

    Returns:
        The model name, or the path of a model in the custom Whisper folder.

    Raises:
        MissingSettingError: If the language needs a custom model and
            TEXT_LAB_CUSTOM_WHISPER_DIR is not set.
    """
    if language in _CUSTOM_MODELS:
        return custom_whisper_model_path(_CUSTOM_MODELS[language])
    return DEFAULT_MODEL


def custom_whisper_model_path(model_name: str) -> str:
    """Return the path of a model in the site's custom Whisper folder.

    Args:
        model_name: Folder name of the model, e.g. :data:`SWHISPER_MODEL`.

    Returns:
        The path as a string, as ``whisperx.load_model`` expects.

    Raises:
        MissingSettingError: If TEXT_LAB_CUSTOM_WHISPER_DIR is not set.
    """
    return str(get_settings().require("custom_whisper_dir") / model_name)


def converted_model_missing(language: str | None, model: str) -> bool:
    """Return True if the Flurin model is chosen but not converted yet.

    The Flurin model must be converted to CTranslate2 format once before
    WhisperX can load it; until then its folder does not exist.

    Args:
        language: The chosen Text Lab language code.
        model: The model path that will be used.

    Returns:
        True if the language needs the converted Flurin model and its
        folder is missing.
    """
    return language == "ch_de_flurin" and not os.path.isdir(model)


def model_label(model: str) -> str:
    """Return a model's name for messages, without revealing storage paths.

    Args:
        model: A model name or path.

    Returns:
        The last path component for paths, otherwise the name itself.
    """
    return os.path.basename(model) if os.sep in model else model
