"""Lightweight language detection for the Translate page.

Uses ``papluca/xlm-roberta-base-language-detection`` (a 278 MB
XLM-Roberta-base classifier fine-tuned for 20 common languages) via the
Hugging Face ``transformers`` pipeline. The model is read from
``$HF_HOME``, the model store the launch script mounts, and loaded once per
session.

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

# ISO 639-1 codes returned by the classifier -> FLORES-200 codes used by NLLB.
_ISO2_TO_FLORES: dict[str, str] = {
    "ar": "arb_Arab",  # Arabic (MSA)
    "bg": "bul_Cyrl",  # Bulgarian
    "de": "deu_Latn",  # German
    "el": "ell_Grek",  # Greek
    "en": "eng_Latn",  # English
    "es": "spa_Latn",  # Spanish
    "fr": "fra_Latn",  # French
    "hi": "hin_Deva",  # Hindi
    "it": "ita_Latn",  # Italian
    "ja": "jpn_Jpan",  # Japanese
    "nl": "nld_Latn",  # Dutch
    "pl": "pol_Latn",  # Polish
    "pt": "por_Latn",  # Portuguese
    "ru": "rus_Cyrl",  # Russian
    "sw": "swh_Latn",  # Swahili
    "th": "tha_Thai",  # Thai
    "tr": "tur_Latn",  # Turkish
    "ur": "urd_Arab",  # Urdu
    "vi": "vie_Latn",  # Vietnamese
    # Chinese (Simplified) - default; user can flip to Traditional
    "zh": "zho_Hans",
}

_MODEL_ID = "papluca/xlm-roberta-base-language-detection"
_MAX_CHARS_FOR_DETECT = 2_000  # more than enough; keeps inference fast


@dataclass(frozen=True)
class DetectionResult:
    """Result of a language-detection call."""

    iso639_1: str
    confidence: float
    flores_code: str | None
    display_name: str | None

    @property
    def is_confident(self) -> bool:
        """Whether the confidence is high enough to use the result."""
        return self.confidence >= 0.60


@functools.lru_cache(maxsize=1)
def _load_pipeline():
    """Lazy-load the classifier once and reuse it across calls."""
    from transformers import pipeline

    device = -1
    return pipeline(
        "text-classification",
        model=_MODEL_ID,
        top_k=1,
        device=device,
        truncation=True,
        max_length=256,
    )


def _flores_and_name_from_iso2(iso2: str) -> tuple[str | None, str | None]:
    """Map an ISO 639-1 code to its FLORES-200 code and UI display name."""
    from textlab.common.language_mappings import (
        TRANSLATE_LANGUAGE_CODE_TO_NAME,
    )

    flores = _ISO2_TO_FLORES.get(iso2)
    if flores is None:
        return None, None
    return flores, TRANSLATE_LANGUAGE_CODE_TO_NAME.get(flores)


def detect_language(text: str) -> DetectionResult | None:
    """Detect the language of ``text``.

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

    # With top_k=1, a single input can still return a nested result list.
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


def detect_document_language(
    name: str,
    data: bytes,
    *,
    samples: int = 5,
) -> DetectionResult | None:
    """Detect a document's language from paragraphs spread across it.

    A title page, an English abstract or a reference list would mislead a
    detector that only reads the beginning, so several paragraphs are
    classified and their confidences summed per language. Returns ``None``
    for unsupported formats, too little text, or an uncertain result.
    """
    try:
        text = read_text_from_upload(name, data)
    except Exception:
        return None
    paragraphs = [
        " ".join(part.split())
        for part in text.split("\n\n")
        if sum(char.isalpha() for char in part) >= 60
    ]
    if not paragraphs:
        return None
    step = max(1, len(paragraphs) // samples)
    picked = paragraphs[::step][:samples]
    votes: dict[str, float] = {}
    results: dict[str, DetectionResult] = {}
    for paragraph in picked:
        result = detect_language(paragraph)
        if result is None or result.flores_code is None:
            continue
        votes[result.flores_code] = (
            votes.get(result.flores_code, 0.0) + result.confidence
        )
        results.setdefault(result.flores_code, result)
    if not votes:
        return None
    best = max(votes, key=votes.__getitem__)
    confidence = votes[best] / len(picked)
    if confidence < 0.60:
        return None
    winner = results[best]
    return DetectionResult(
        iso639_1=winner.iso639_1,
        confidence=confidence,
        flores_code=winner.flores_code,
        display_name=winner.display_name,
    )


def read_text_from_upload(name: str, data: bytes) -> str:
    """Extract plain text from a supported upload.

    - .txt / .csv / .tsv / .md -> utf-8 decode with replacement
    - .pdf                     -> pymupdf text extraction
    - .docx                    -> python-docx if available, else raise

    Kept intentionally lightweight; the UI decides which extensions to allow.
    """
    lower = name.lower()
    if lower.endswith((".txt", ".md", ".csv", ".tsv", ".srt", ".vtt")):
        return data.decode("utf-8", errors="replace")
    if lower.endswith(".pdf"):
        import fitz  # pymupdf

        doc = fitz.open(stream=data, filetype="pdf")
        try:
            return "\n\n".join(page.get_text() for page in doc)
        finally:
            doc.close()
    if lower.endswith(".docx"):
        try:
            import docx  # python-docx, optional
        except ImportError as exc:
            raise RuntimeError(
                "python-docx is not installed in this container; "
                "upload .txt or .pdf instead."
            ) from exc
        import io as _io

        d = docx.Document(_io.BytesIO(data))
        return "\n\n".join(p.text for p in d.paragraphs)
    raise ValueError(f"Unsupported file type: {name}")
