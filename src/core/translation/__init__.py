"""
Machine-translation subpackage for Text Lab.

Layout
------

* :mod:`.engine`       -- model loading, language adapters, dispatch, and
                          the :func:`make_translate_fn` factory.
* :mod:`.chunking`     -- lossless boundaries, budgets, and limit errors.
* :mod:`.hf_backend`   -- tokenizer-aware batching and bounded recovery.
* :mod:`.ollama_backend` -- conservative context budgets and stop checks.
* :mod:`.gpu_memory`   -- serialized lifecycle and CUDA cleanup.
* :mod:`.gpu_profile`  -- allocated-device and live free-memory budgets.
* :mod:`.format`       -- document translation and checked reconstruction.
* :mod:`.pdf_checks`   -- per-page routing and extraction coverage checks.
* :mod:`.pdf_extract`  -- native/blank extraction and the isolated OCR subset.
* :mod:`.pdf_workflow` -- independent PDF/Markdown outputs and failure reports.
* :mod:`.shield`       -- pre/post-translation shielding for URLs,
                          math, code, HTML, placeholders, plus glossary
                          / term-lock support.
* :mod:`.lang_detect`  -- HF-based source-language detection with
                          confidence.
* :mod:`.quality`      -- CometKiwi reference-free quality estimation.

Everything the UI (or an MCP tool / CLI) needs is re-exported here so
consumers can write ``from core.translation import ...`` without knowing
the internal file layout.
"""

from __future__ import annotations

from .chunking import (
    InputTooLongError,
    OutputTruncatedError,
    TranslationLimitError,
)
from .engine import (
    FORMALITY_CAPABLE_BACKENDS,
    FORMALITY_CHOICES,
    MADLAD_MODEL_IDS,
    NLLB_MODEL_IDS,
    TRANSLATION_BACKENDS,
    backend_is_loaded,
    backend_load_signature,
    chunk_text_for_translation,
    flores_to_iso2,
    free_translation_vram,
    make_translate_fn,
    preload_backend,
    read_text_from_upload,
    split_into_sentences,
    translate,
    translate_madlad,
    translate_many,
    translate_nllb,
    translate_ollama,
    translate_opus_mt,
)
from .format import (
    reflow_soft_wraps,
    translate_docx,
    translate_markdown,
    translate_pdf,
    translate_pptx,
    translate_xlsx,
    detect_pdf_is_scanned,
    pdf_needs_ocr,
    pdf_to_markdown_bundle,
    translate_pdf_to_markdown,
    pack_markdown_bundle,
)
from .lang_detect import (
    DetectionResult,
    detect_language,
    supported_iso639_1_codes,
)
from .gpu_profile import (
    GpuProfile,
    detect_gpu_profile,
    ocr_with_translation_allowed,
    resolve_batch_size,
    sequential_ocr_allowed,
)
from .gpu_memory import translation_session
from .quality import (
    SCORE_UNAVAILABLE,
    estimate_quality,
    is_available,
    quality_badge,
)
from .pdf_checks import PDFIntegrityError
from .pdf_workflow import PDFTranslationResult, translate_pdf_outputs
from .shield import (
    ProtectedContentError,
    shield,
    shielded_translate,
    shielded_translate_many,
    unshield,
)

__all__ = [
    "InputTooLongError",
    "OutputTruncatedError",
    "TranslationLimitError",
    "ProtectedContentError",
    "PDFIntegrityError",
    "PDFTranslationResult",
    "translate_pdf_outputs",
    # Engine
    "FORMALITY_CAPABLE_BACKENDS",
    "FORMALITY_CHOICES",
    "MADLAD_MODEL_IDS",
    "NLLB_MODEL_IDS",
    "TRANSLATION_BACKENDS",
    "backend_is_loaded",
    "backend_load_signature",
    "chunk_text_for_translation",
    "flores_to_iso2",
    "free_translation_vram",
    "make_translate_fn",
    "preload_backend",
    "read_text_from_upload",
    "split_into_sentences",
    "translate",
    "translate_madlad",
    "translate_many",
    "translate_nllb",
    "translate_ollama",
    "translate_opus_mt",
    # Format
    "reflow_soft_wraps",
    "translate_docx",
    "translate_markdown",
    "translate_pdf",
    "translate_pptx",
    "translate_xlsx",
    "detect_pdf_is_scanned",
    "pdf_needs_ocr",
    "pdf_to_markdown_bundle",
    "translate_pdf_to_markdown",
    "pack_markdown_bundle",
    # Language detection
    "DetectionResult",
    "detect_language",
    "supported_iso639_1_codes",
    # GPU profile
    "GpuProfile",
    "detect_gpu_profile",
    "ocr_with_translation_allowed",
    "resolve_batch_size",
    "sequential_ocr_allowed",
    "translation_session",
    # Quality estimation
    "SCORE_UNAVAILABLE",
    "estimate_quality",
    "is_available",
    "quality_badge",
    # Shielding
    "shield",
    "shielded_translate",
    "shielded_translate_many",
    "unshield",
]
