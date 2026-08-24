"""
Machine-translation subpackage for Text Lab.

Layout
------

* :mod:`.engine`       -- backend registry, NLLB / MADLAD / OPUS-MT /
                          Ollama inference, dispatch, and the
                          :func:`make_translate_fn` factory.
* :mod:`.format`       -- format-preserving document translation for
                          Markdown, DOCX, PDF, XLSX, PPTX.
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

from .engine import (
    FORMALITY_CAPABLE_BACKENDS,
    FORMALITY_CHOICES,
    MADLAD_MODEL_IDS,
    NLLB_MODEL_IDS,
    TRANSLATION_BACKENDS,
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
from .lang_detect import DetectionResult, detect_language, supported_iso639_1_codes
from .gpu_profile import (
    GpuProfile,
    detect_gpu_profile,
    ocr_with_translation_allowed,
    resolve_batch_size,
)
from .quality import SCORE_UNAVAILABLE, estimate_quality, is_available, quality_badge
from .shield import shield, shielded_translate, shielded_translate_many, unshield

__all__ = [
    # Engine
    "FORMALITY_CAPABLE_BACKENDS",
    "FORMALITY_CHOICES",
    "MADLAD_MODEL_IDS",
    "NLLB_MODEL_IDS",
    "TRANSLATION_BACKENDS",
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
