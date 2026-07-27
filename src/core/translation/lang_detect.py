"""
Lightweight language detection for the Translate page.

Uses ``papluca/xlm-roberta-base-language-detection`` (a 278 MB
XLM-Roberta-base classifier fine-tuned for 20 common languages) via the
Hugging Face ``transformers`` pipeline. The model file is cached under
``$HF_HOME`` (bind-mounted from research storage in the OOD container), so
the first call downloads once and every subsequent call is a cache hit.

Public API
----------

    detect_language(text) -> DetectionResult

The result carries:

* ``iso639_1``      -- e.g. ``"de"``
* ``flores_code``   -- e.g. ``"deu_Latn"`` (mapped for use with NLLB)
* ``display_name``  -- e.g. ``"German"`` (mapped for the UI dropdown)
* ``confidence``    -- float in [0, 1]

If detection fails or the language is outside the classifier's 20-language
coverage, ``flores_code`` and ``display_name`` may be ``None`` while the
raw ISO code is still returned. Callers should fall back to a manual
source-language pick in that case.

The module is Streamlit-free and safe to import from MCP or CLI code.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Optional


# ISO 639-1 codes returned by the classifier -> FLORES-200 codes used by NLLB.
_ISO2_TO_FLORES: dict[str, str] = {
    "ar": "arb_Arab",   # Arabic (MSA)
    "bg": "bul_Cyrl",   # Bulgarian
    "de": "deu_Latn",   # German
    "el": "ell_Grek",   # Greek
    "en": "eng_Latn",   # English
    "es": "spa_Latn",   # Spanish
    "fr": "fra_Latn",   # French
    "hi": "hin_Deva",   # Hindi
    "it": "ita_Latn",   # Italian
    "ja": "jpn_Jpan",   # Japanese
    "nl": "nld_Latn",   # Dutch
    "pl": "pol_Latn",   # Polish
    "pt": "por_Latn",   # Portuguese
    "ru": "rus_Cyrl",   # Russian
    "sw": "swh_Latn",   # Swahili
    "th": "tha_Thai",   # Thai
    "tr": "tur_Latn",   # Turkish
    "ur": "urd_Arab",   # Urdu
    "vi": "vie_Latn",   # Vietnamese
    "zh": "zho_Hans",   # Chinese (Simplified) - default; user can flip to Traditional
}

_MODEL_ID = "papluca/xlm-roberta-base-language-detection"
_MAX_CHARS_FOR_DETECT = 2_000  # more than enough; keeps inference fast


@dataclass(frozen=True)
class DetectionResult:
    """Result of a language-detection call."""

    iso639_1: str
    confidence: float
    flores_code: Optional[str]
    display_name: Optional[str]

    @property
    def is_confident(self) -> bool:
        return self.confidence >= 0.60


@functools.lru_cache(maxsize=1)
def _load_pipeline():
    """Lazy-load the classifier once and reuse it across calls."""
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "text-classification",
        model=_MODEL_ID,
        top_k=1,
        device=device,
        truncation=True,
        max_length=256,
    )


def _flores_and_name_from_iso2(iso2: str) -> tuple[Optional[str], Optional[str]]:
    """Map an ISO 639-1 code to its FLORES-200 code and UI display name."""
    from language_mappings import TRANSLATE_LANGUAGE_CODE_TO_NAME

    flores = _ISO2_TO_FLORES.get(iso2)
    if flores is None:
        return None, None
    return flores, TRANSLATE_LANGUAGE_CODE_TO_NAME.get(flores)


def detect_language(text: str) -> Optional[DetectionResult]:
    """
    Detect the language of ``text``.

    Returns ``None`` if the input is empty or the classifier errors out
    (network problem, corrupt cache, etc). The caller should treat ``None``
    as "unknown -- please pick the source language manually".
    """
    if not text or not text.strip():
        return None

    sample = text.strip()[:_MAX_CHARS_FOR_DETECT]

    try:
        clf = _load_pipeline()
        raw = clf(sample)
    except Exception:
        return None

    # ``top_k=1`` returns [[{'label': 'de', 'score': 0.99}]] for a single input.
    if not raw:
        return None
    first = raw[0]
    if isinstance(first, list):
        first = first[0] if first else None
    if not first or "label" not in first:
        return None

    iso2 = str(first["label"]).lower()
    score = float(first.get("score", 0.0))
    flores, name = _flores_and_name_from_iso2(iso2)

    return DetectionResult(
        iso639_1=iso2,
        confidence=score,
        flores_code=flores,
        display_name=name,
    )


def supported_iso639_1_codes() -> list[str]:
    """Return the sorted list of ISO 639-1 codes the classifier can predict."""
    return sorted(_ISO2_TO_FLORES.keys())
