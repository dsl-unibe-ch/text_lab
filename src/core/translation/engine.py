"""
Core machine-translation engine for Text Lab.

Provides multiple selectable backends:

- ``nllb``       : facebook/nllb-200-distilled-600M (default, 200+ languages,
                   good CPU/GPU trade-off, ships in transformers).
- ``nllb-large`` : facebook/nllb-200-3.3B (higher quality, needs more VRAM).
- ``opus-mt``    : Helsinki-NLP/opus-mt-<src>-<tgt> bilingual MarianMT models
                   (small, fast, per-pair; auto-resolved when a pair exists).
- ``ollama``     : LLM prompt-based translation, useful as a fallback and
                   for dialects such as Swiss German.

The module is intentionally free of Streamlit calls. All state / progress
reporting is done via ``progress_cb`` callbacks so it can be reused by the
Streamlit page, MCP tools, or CLI scripts.

Language codes exchanged by callers are the FLORES-200 codes defined in
``language_mappings.TRANSLATE_LANGUAGE_MAPPING`` (e.g. ``deu_Latn``).
Backend-specific adapters convert them internally.

NOTE: The first call for a given (backend, model) pair downloads the model
into ``HF_HOME`` (``/opt/huggingface``), which is bind-mounted from research
storage. Subsequent calls are cache hits.
"""

from __future__ import annotations

import functools
import re
import time
from typing import Callable, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

# Public backend identifier -> user-facing label.
TRANSLATION_BACKENDS: Dict[str, str] = {
    "nllb": "NLLB-200 Distilled (600M, fast, 200 languages)",
    "nllb-large": "NLLB-200 (3.3B, higher quality, needs bigger GPU)",
    "madlad-3b": "MADLAD-400 (3B, strong on low-resource languages)",
    "opus-mt": "OPUS-MT / MarianMT (small, bilingual per pair)",
    "ollama": "LLM (Ollama) - prompt-based, good for dialects",
}

NLLB_MODEL_IDS: Dict[str, str] = {
    "nllb": "facebook/nllb-200-distilled-600M",
    "nllb-large": "facebook/nllb-200-3.3B",
}

MADLAD_MODEL_IDS: Dict[str, str] = {
    "madlad-3b": "google/madlad400-3b-mt",
}

# Backends that meaningfully honour a formality preference. Others silently
# ignore the parameter so a UI toggle can be shown/hidden accordingly.
FORMALITY_CAPABLE_BACKENDS = frozenset({"ollama"})

# Allowed formality values exchanged with the UI / MCP layer.
FORMALITY_CHOICES = ("default", "formal", "informal")

# FLORES-200 -> ISO 639-1 for OPUS-MT (subset; extended lazily).
_FLORES_TO_ISO2: Dict[str, str] = {
    "eng_Latn": "en", "deu_Latn": "de", "fra_Latn": "fr", "ita_Latn": "it",
    "spa_Latn": "es", "por_Latn": "pt", "nld_Latn": "nl", "dan_Latn": "da",
    "swe_Latn": "sv", "nob_Latn": "no", "fin_Latn": "fi", "pol_Latn": "pl",
    "ces_Latn": "cs", "slk_Latn": "sk", "slv_Latn": "sl", "hrv_Latn": "hr",
    "bul_Cyrl": "bg", "ron_Latn": "ro", "hun_Latn": "hu", "ell_Grek": "el",
    "tur_Latn": "tr", "rus_Cyrl": "ru", "ukr_Cyrl": "uk", "arb_Arab": "ar",
    "heb_Hebr": "he", "pes_Arab": "fa", "urd_Arab": "ur", "hin_Deva": "hi",
    "zho_Hans": "zh", "jpn_Jpan": "ja", "kor_Hang": "ko", "vie_Latn": "vi",
    "tha_Thai": "th", "ind_Latn": "id", "swh_Latn": "sw", "cat_Latn": "ca",
    "eus_Latn": "eu",
}


def flores_to_iso2(code: str) -> Optional[str]:
    """Return an ISO 639-1 code for the given FLORES-200 code, or None."""
    return _FLORES_TO_ISO2.get(code)


# ---------------------------------------------------------------------------
# Text chunking (translation models have a hard token limit; ~512 tokens for
# NLLB / MarianMT). We chunk by sentence to avoid mid-sentence cuts.
# ---------------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\?\u3002\uFF01\uFF1F])\s+")


def split_into_sentences(text: str) -> List[str]:
    """Simple regex-based sentence split. Good enough for chunking."""
    text = text.strip()
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text_for_translation(
    text: str,
    max_chars: int = 1200,
) -> List[str]:
    """
    Split text into ~max_chars chunks on sentence boundaries where possible.

    Paragraphs (double newline) are preserved. Very long sentences are
    hard-split so no chunk exceeds ``max_chars`` characters.
    """
    if not text:
        return []

    chunks: List[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        sentences = split_into_sentences(paragraph)
        if not sentences:
            continue

        buf = ""
        for sent in sentences:
            if len(sent) > max_chars:
                if buf:
                    chunks.append(buf)
                    buf = ""
                for i in range(0, len(sent), max_chars):
                    chunks.append(sent[i : i + max_chars])
                continue
            if buf and len(buf) + 1 + len(sent) > max_chars:
                chunks.append(buf)
                buf = sent
            else:
                buf = f"{buf} {sent}".strip() if buf else sent
        if buf:
            chunks.append(buf)
    return chunks


# ---------------------------------------------------------------------------
# NLLB backend
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=2)
def _load_nllb(model_id: str, device: str, dtype_name: str):
    """Lazy-load and cache a NLLB tokenizer/model pair."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    dtype = getattr(torch, dtype_name) if dtype_name != "auto" else "auto"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        torch_dtype=dtype if dtype != "auto" else None,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def translate_nllb(
    text: str,
    src_lang: str,
    tgt_lang: str,
    backend: str = "nllb",
    device: Optional[str] = None,
    max_new_tokens: int = 512,
    num_beams: int = 2,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> str:
    """
    Translate ``text`` from ``src_lang`` to ``tgt_lang`` using NLLB-200.

    ``src_lang`` / ``tgt_lang`` are FLORES-200 codes (e.g. ``deu_Latn``).
    """
    import torch

    if backend not in NLLB_MODEL_IDS:
        raise ValueError(f"Unknown NLLB backend: {backend}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_name = "float16" if device == "cuda" else "float32"

    tokenizer, model = _load_nllb(NLLB_MODEL_IDS[backend], device, dtype_name)

    chunks = chunk_text_for_translation(text)
    total = len(chunks)
    outputs: List[str] = []

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    if forced_bos_token_id is None or forced_bos_token_id == tokenizer.unk_token_id:
        raise ValueError(f"Target language {tgt_lang} is not supported by NLLB.")

    for i, chunk in enumerate(chunks, start=1):
        tokenizer.src_lang = src_lang
        enc = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            gen = model.generate(
                **enc,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
        outputs.append(tokenizer.batch_decode(gen, skip_special_tokens=True)[0])
        if progress_cb is not None:
            progress_cb(i, total)

    return "\n\n".join(outputs)


# ---------------------------------------------------------------------------
# OPUS-MT / MarianMT backend
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=8)
def _load_marian(model_id: str, device: str):
    from transformers import MarianMTModel, MarianTokenizer

    tokenizer = MarianTokenizer.from_pretrained(model_id)
    model = MarianMTModel.from_pretrained(model_id).to(device).eval()
    return tokenizer, model


def opus_mt_model_for(src_iso2: str, tgt_iso2: str) -> str:
    """Return the canonical Helsinki-NLP model ID for a pair."""
    return f"Helsinki-NLP/opus-mt-{src_iso2}-{tgt_iso2}"


def translate_opus_mt(
    text: str,
    src_lang: str,
    tgt_lang: str,
    device: Optional[str] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> str:
    """
    Translate using an appropriate MarianMT bilingual model.

    Falls back gracefully with a clear error if no direct pair exists (many
    Helsinki-NLP pairs exist but not all - e.g. de<->fr is direct, but exotic
    pairs may need pivoting through English, which is not implemented here).
    """
    import torch

    src_iso = flores_to_iso2(src_lang)
    tgt_iso = flores_to_iso2(tgt_lang)
    if not src_iso or not tgt_iso:
        raise ValueError(
            f"OPUS-MT backend does not have an ISO-2 mapping for "
            f"{src_lang} -> {tgt_lang}. Try the NLLB backend."
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = opus_mt_model_for(src_iso, tgt_iso)
    try:
        tokenizer, model = _load_marian(model_id, device)
    except Exception as exc:
        raise RuntimeError(
            f"No OPUS-MT model available for {src_iso}->{tgt_iso} ({model_id}). "
            "Try the NLLB backend instead."
        ) from exc

    chunks = chunk_text_for_translation(text)
    total = len(chunks)
    outputs: List[str] = []
    for i, chunk in enumerate(chunks, start=1):
        enc = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            gen = model.generate(**enc, num_beams=2, max_new_tokens=512)
        outputs.append(tokenizer.batch_decode(gen, skip_special_tokens=True)[0])
        if progress_cb is not None:
            progress_cb(i, total)
    return "\n\n".join(outputs)


# ---------------------------------------------------------------------------
# Ollama LLM backend (prompt-based, useful for dialects / low-resource cases)
# ---------------------------------------------------------------------------


def translate_ollama(
    text: str,
    src_lang_name: str,
    tgt_lang_name: str,
    model_name: str,
    formality: str = "default",
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> str:
    """
    Translate by prompting a chat LLM served via Ollama.

    ``src_lang_name`` and ``tgt_lang_name`` are the human-readable names
    from ``TRANSLATE_LANGUAGE_MAPPING`` (e.g. ``"German"``).

    ``formality`` is one of :data:`FORMALITY_CHOICES`. When not ``default``
    a matching register instruction is appended to the system prompt.
    """
    import ollama

    formality_instr = {
        "formal": (
            " Use a formal, professional register throughout — polite"
            " pronouns, complete sentences, no colloquialisms."
        ),
        "informal": (
            " Use a casual, conversational register — everyday vocabulary,"
            " contractions where natural, informal pronouns."
        ),
    }.get(formality, "")

    system = (
        "You are a professional translator. Translate the user's text from "
        f"{src_lang_name} into {tgt_lang_name}. "
        "Preserve meaning, tone, formatting (paragraphs, lists) and named "
        f"entities.{formality_instr} Do not add commentary. Return ONLY the translated text."
    )

    chunks = chunk_text_for_translation(text, max_chars=3000)
    total = len(chunks)
    outputs: List[str] = []
    for i, chunk in enumerate(chunks, start=1):
        resp = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": chunk},
            ],
            options={"temperature": 0.2},
        )
        outputs.append(resp["message"]["content"].strip())
        if progress_cb is not None:
            progress_cb(i, total)
    return "\n\n".join(outputs)


# ---------------------------------------------------------------------------
# MADLAD-400 backend (T5-style; source-language-agnostic, uses <2xx> prefix)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=2)
def _load_madlad(model_id: str, device: str, dtype_name: str):
    """Lazy-load and cache a MADLAD-400 tokenizer/model pair."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    dtype = getattr(torch, dtype_name) if dtype_name != "auto" else "auto"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        torch_dtype=dtype if dtype != "auto" else None,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def translate_madlad(
    text: str,
    src_lang: str,
    tgt_lang: str,
    backend: str = "madlad-3b",
    device: Optional[str] = None,
    max_new_tokens: int = 512,
    num_beams: int = 2,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> str:
    """
    Translate ``text`` to ``tgt_lang`` using MADLAD-400.

    MADLAD is source-language-agnostic: it detects the source and takes
    the target language as a ``<2xx>`` prefix (2-letter ISO 639-1). The
    ``src_lang`` argument is accepted for API symmetry but ignored.
    """
    import torch

    if backend not in MADLAD_MODEL_IDS:
        raise ValueError(f"Unknown MADLAD backend: {backend}")

    tgt_iso = flores_to_iso2(tgt_lang)
    if not tgt_iso:
        raise ValueError(
            f"MADLAD needs an ISO 639-1 target code; {tgt_lang} is unmapped. "
            "Try the NLLB backend for this language."
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_name = "float16" if device == "cuda" else "float32"

    tokenizer, model = _load_madlad(MADLAD_MODEL_IDS[backend], device, dtype_name)

    chunks = chunk_text_for_translation(text)
    total = len(chunks)
    outputs: List[str] = []
    prefix = f"<2{tgt_iso}> "

    for i, chunk in enumerate(chunks, start=1):
        enc = tokenizer(prefix + chunk, return_tensors="pt", truncation=True, max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
        outputs.append(tokenizer.batch_decode(gen, skip_special_tokens=True)[0])
        if progress_cb is not None:
            progress_cb(i, total)

    return "\n\n".join(outputs)


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


def translate(
    text: str,
    src_lang: str,
    tgt_lang: str,
    backend: str = "nllb",
    ollama_model: Optional[str] = None,
    src_lang_name: Optional[str] = None,
    tgt_lang_name: Optional[str] = None,
    formality: str = "default",
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> str:
    """Dispatch to the requested backend."""
    if not text or not text.strip():
        return ""

    if backend in NLLB_MODEL_IDS:
        return translate_nllb(text, src_lang, tgt_lang, backend=backend, progress_cb=progress_cb)
    if backend in MADLAD_MODEL_IDS:
        return translate_madlad(text, src_lang, tgt_lang, backend=backend, progress_cb=progress_cb)
    if backend == "opus-mt":
        return translate_opus_mt(text, src_lang, tgt_lang, progress_cb=progress_cb)
    if backend == "ollama":
        if not ollama_model:
            raise ValueError("ollama backend requires ollama_model")
        return translate_ollama(
            text,
            src_lang_name or src_lang,
            tgt_lang_name or tgt_lang,
            model_name=ollama_model,
            formality=formality,
            progress_cb=progress_cb,
        )
    raise ValueError(f"Unknown translation backend: {backend}")


def make_translate_fn(
    src_lang: str,
    tgt_lang: str,
    backend: str = "nllb",
    ollama_model: Optional[str] = None,
    src_lang_name: Optional[str] = None,
    tgt_lang_name: Optional[str] = None,
    formality: str = "default",
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Callable[[str], str]:
    """
    Return a single-argument ``str -> str`` translator with all backend
    parameters baked in. Handy for feeding to format-preserving pipelines
    (:mod:`core.translation.format`) or the shielding wrapper.
    """
    def _fn(text: str) -> str:
        return translate(
            text,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            backend=backend,
            ollama_model=ollama_model,
            src_lang_name=src_lang_name,
            tgt_lang_name=tgt_lang_name,
            formality=formality,
            progress_cb=progress_cb,
        )
    return _fn


# ---------------------------------------------------------------------------
# Helpers for batch / file handling used by the UI
# ---------------------------------------------------------------------------


def read_text_from_upload(name: str, data: bytes) -> str:
    """
    Extract plain text from a supported upload.

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
                "python-docx is not installed in this container; upload .txt or .pdf instead."
            ) from exc
        import io as _io

        d = docx.Document(_io.BytesIO(data))
        return "\n\n".join(p.text for p in d.paragraphs)
    raise ValueError(f"Unsupported file type: {name}")


# ---------------------------------------------------------------------------
# Explicit preload / warm-up
#
# Text Lab runs on Slurm-allocated GPU nodes where the first inference for
# any given (backend, model) pair pays the full cost of downloading weights
# (if not cached) and moving them to VRAM — anywhere from a few seconds to
# a couple of minutes. Users interacting with the split-screen editor
# expect near-instant responses, so the UI exposes an explicit "Load model"
# gate that calls into these helpers.
# ---------------------------------------------------------------------------


def backend_load_signature(
    backend: str,
    src_lang: str = "",
    tgt_lang: str = "",
    ollama_model: Optional[str] = None,
) -> tuple:
    """
    Return an opaque tuple identifying what needs to be resident in VRAM
    for a given (backend, language-pair, ollama-model) combination.

    * NLLB / MADLAD only depend on the backend: one model handles every
      pair.
    * OPUS-MT is per-pair, so the language codes participate in the
      signature.
    * Ollama depends on the selected LLM name.

    The UI compares a stored "loaded" signature against the current one to
    decide whether a fresh warm-up is required.
    """
    if backend in NLLB_MODEL_IDS or backend in MADLAD_MODEL_IDS:
        return ("hf", backend)
    if backend == "opus-mt":
        return ("hf", backend, src_lang, tgt_lang)
    if backend == "ollama":
        return ("ollama", ollama_model or "")
    return (backend,)


def preload_backend(
    backend: str,
    src_lang: str = "",
    tgt_lang: str = "",
    ollama_model: Optional[str] = None,
    src_lang_name: Optional[str] = None,
    tgt_lang_name: Optional[str] = None,
) -> None:
    """
    Warm up the given backend so the next :func:`translate` call is fast.

    * For NLLB / MADLAD / OPUS-MT this triggers the transformers download
      (if the wheels aren't already in ``HF_HOME``) and moves the weights
      to the GPU.
    * For Ollama this sends a one-token chat request so Ollama loads the
      model into VRAM and keeps it there.

    Idempotent — calling twice with the same arguments is essentially a
    no-op because the underlying loaders are ``lru_cache``-d.
    """
    import torch

    if backend in NLLB_MODEL_IDS:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype_name = "float16" if device == "cuda" else "float32"
        _load_nllb(NLLB_MODEL_IDS[backend], device, dtype_name)
        return

    if backend in MADLAD_MODEL_IDS:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype_name = "float16" if device == "cuda" else "float32"
        _load_madlad(MADLAD_MODEL_IDS[backend], device, dtype_name)
        return

    if backend == "opus-mt":
        src_iso = flores_to_iso2(src_lang)
        tgt_iso = flores_to_iso2(tgt_lang)
        if not src_iso or not tgt_iso:
            raise ValueError(
                f"OPUS-MT has no direct ISO-2 mapping for {src_lang} → "
                f"{tgt_lang}. Choose the NLLB backend for this pair."
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            _load_marian(opus_mt_model_for(src_iso, tgt_iso), device)
        except Exception as exc:
            raise RuntimeError(
                f"No OPUS-MT model available for {src_iso}→{tgt_iso}. "
                "Try the NLLB backend instead."
            ) from exc
        return

    if backend == "ollama":
        if not ollama_model:
            raise ValueError("The Ollama backend requires an ollama_model.")
        import ollama

        # A one-token request is enough for Ollama to page the model into
        # VRAM. It stays resident for a while afterwards.
        ollama.chat(
            model=ollama_model,
            messages=[{"role": "user", "content": "hi"}],
            options={"num_predict": 1, "temperature": 0.0},
        )
        return

    raise ValueError(f"Unknown backend: {backend}")
