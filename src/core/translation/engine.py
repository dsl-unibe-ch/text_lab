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
storage. Only the active HF model remains cached. All model lifecycle and
inference work shares the reentrant ``translation_session()`` guard; OCR
callers hold that guard across eviction, OCR, and subsequent translation.
"""

from __future__ import annotations

import functools
from typing import Callable, Dict, List, Optional

from .chunking import (
    chunk_text_for_translation,
    split_into_sentences,
    translate_lines,
)
from .gpu_memory import (
    clear_cuda_cache,
    discard_exception_tensors,
    is_cuda_device,
    is_cuda_oom,
    serialized,
    translation_session,
)
from .gpu_profile import cap_batch_size, resolve_batch_size
from .hf_backend import (
    DEFAULT_NUM_BEAMS,
    generate_translations,
)
from .ollama_backend import (
    ollama_model_is_loaded,
    prepare_ollama_model,
    release_ollama_model,
    translate_ollama,
)

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


_ACTIVE_HF_SIGNATURE = None
_ACTIVE_HF_DEVICE = None


def _resolve_device(device: Optional[str]) -> str:
    import torch

    if device is not None and not is_cuda_device(device):
        return device
    if not torch.cuda.is_available():
        return "cpu"
    if device is None or device == "cuda":
        return f"cuda:{torch.cuda.current_device()}"
    return device


def _free_hf_cache(device=None) -> None:
    """Evict only translation loaders; caller holds translation_session."""
    global _ACTIVE_HF_SIGNATURE, _ACTIVE_HF_DEVICE

    old_device = _ACTIVE_HF_DEVICE
    _ACTIVE_HF_SIGNATURE = None
    _ACTIVE_HF_DEVICE = None
    for loader in (_load_nllb, _load_marian, _load_madlad):
        clear = getattr(loader, "cache_clear", None)
        if clear is not None:
            clear()
    clear_cuda_cache(old_device or device)


def _load_hf(backend, src_lang, tgt_lang, device):
    """Keep one active HF model, evicting before a backend/pair/device swap."""
    global _ACTIVE_HF_SIGNATURE, _ACTIVE_HF_DEVICE

    signature = backend_load_signature(backend, src_lang, tgt_lang)
    dtype = "float16" if is_cuda_device(device) else "float32"
    if backend in NLLB_MODEL_IDS:
        loader = _load_nllb
        args = (NLLB_MODEL_IDS[backend], device, dtype)
    elif backend in MADLAD_MODEL_IDS:
        loader = _load_madlad
        args = (MADLAD_MODEL_IDS[backend], device, dtype)
    elif backend == "opus-mt":
        src_iso = flores_to_iso2(src_lang)
        tgt_iso = flores_to_iso2(tgt_lang)
        if not src_iso or not tgt_iso:
            raise ValueError(
                f"OPUS-MT has no ISO-2 mapping for {src_lang} -> {tgt_lang}. "
                "Try the NLLB backend."
            )
        loader = _load_marian
        args = (opus_mt_model_for(src_iso, tgt_iso), device)
    else:
        raise ValueError(f"Not an HF seq2seq backend: {backend}")

    release_ollama_model()
    if (_ACTIVE_HF_SIGNATURE != signature or _ACTIVE_HF_DEVICE != device):
        _free_hf_cache()
    loaded = None
    try:
        loaded = loader(*args)
    except OSError as error:
        if backend != "opus-mt":
            raise
        raise RuntimeError(
            f"No OPUS-MT model available for {src_lang} -> {tgt_lang}. "
            "Try the NLLB backend instead."
        ) from error
    except RuntimeError as error:
        import torch

        if not is_cuda_oom(error, device, torch):
            raise
        discard_exception_tensors(error)
    if loaded is None:
        _free_hf_cache(device)
        raise RuntimeError(
            "CUDA out of memory while loading the translation model. "
            "Reducing batches cannot fit model weights; free GPU memory "
            "or choose a smaller/CPU backend."
        ) from None
    _ACTIVE_HF_SIGNATURE = signature
    _ACTIVE_HF_DEVICE = device
    return loaded


@serialized
def _translate_chunks_hf(
    chunks: List[str],
    src_lang: str,
    tgt_lang: str,
    backend: str,
    *,
    device: Optional[str] = None,
    num_beams: int = DEFAULT_NUM_BEAMS,
    max_new_tokens: int = 512,
    batch_size: Optional[int] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    status_cb: Optional[Callable[[str], None]] = None,
) -> List[str]:
    """Load a backend, then split and batch inputs with its real tokenizer.

    Return one complete translation per input. Language prefixes, special
    tokens, and model limits are included in the generation layer's budget.
    """
    if not chunks:
        return []

    if batch_size is not None and batch_size <= 0:
        raise ValueError("Batch size must be positive.")
    device = _resolve_device(device)
    tokenizer, model = _load_hf(backend, src_lang, tgt_lang, device)
    # Keep the legacy one-argument resolver seam for integrations/fakes.
    if batch_size is None:
        batch_size = resolve_batch_size(backend)
    batch_size = cap_batch_size(
        backend, batch_size, device, num_beams=num_beams,
    )
    source_prefix = ""

    if backend in NLLB_MODEL_IDS:
        tokenizer.src_lang = src_lang
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
        if (forced_bos_token_id is None
                or forced_bos_token_id == tokenizer.unk_token_id):
            raise ValueError(
                f"Target language {tgt_lang} is not supported by NLLB.")
    elif backend in MADLAD_MODEL_IDS:
        tgt_iso = flores_to_iso2(tgt_lang)
        if not tgt_iso:
            raise ValueError(
                "MADLAD needs an ISO 639-1 target code; "
                f"{tgt_lang} is unmapped. "
                "Try the NLLB backend for this language."
            )
        forced_bos_token_id = None
        source_prefix = f"<2{tgt_iso}> "
    elif backend == "opus-mt":
        forced_bos_token_id = None
    else:
        raise ValueError(f"Not an HF seq2seq backend: {backend}")

    return generate_translations(
        model,
        tokenizer,
        chunks,
        device,
        source_prefix=source_prefix,
        forced_bos_token_id=forced_bos_token_id,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        progress_cb=progress_cb,
        status_cb=status_cb,
    )


def _translate_hf_texts(
    texts: List[str],
    src_lang: str,
    tgt_lang: str,
    backend: str,
    *,
    device: Optional[str] = None,
    num_beams: int = DEFAULT_NUM_BEAMS,
    max_new_tokens: int = 512,
    batch_size: Optional[int] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    status_cb: Optional[Callable[[str], None]] = None,
) -> List[str]:
    """Use the same layout and generation path for single and batch inputs."""
    translator = functools.partial(
        _translate_chunks_hf,
        src_lang=src_lang, tgt_lang=tgt_lang, backend=backend,
        device=device, num_beams=num_beams, max_new_tokens=max_new_tokens,
        batch_size=batch_size, progress_cb=progress_cb, status_cb=status_cb,
    )
    return translate_lines(texts, translator)


# ---------------------------------------------------------------------------
# NLLB backend
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
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
    num_beams: int = DEFAULT_NUM_BEAMS,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    *,
    status_cb: Optional[Callable[[str], None]] = None,
) -> str:
    """Translate with NLLB using FLORES-200 language codes."""
    if backend not in NLLB_MODEL_IDS:
        raise ValueError(f"Unknown NLLB backend: {backend}")
    return _translate_hf_texts(
        [text], src_lang, tgt_lang, backend,
        device=device, num_beams=num_beams,
        max_new_tokens=max_new_tokens, progress_cb=progress_cb,
        status_cb=status_cb,
    )[0]


# ---------------------------------------------------------------------------
# OPUS-MT / MarianMT backend
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
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
    *,
    status_cb: Optional[Callable[[str], None]] = None,
) -> str:
    """Translate with a direct MarianMT pair, or report an unavailable pair."""
    return _translate_hf_texts(
        [text], src_lang, tgt_lang, "opus-mt", device=device,
        progress_cb=progress_cb, status_cb=status_cb,
    )[0]


# ---------------------------------------------------------------------------
# MADLAD-400 backend (T5-style; source-language-agnostic, uses <2xx> prefix)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
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
    num_beams: int = DEFAULT_NUM_BEAMS,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    *,
    status_cb: Optional[Callable[[str], None]] = None,
) -> str:
    """Translate with MADLAD's target-language prefix (source is detected)."""
    if backend not in MADLAD_MODEL_IDS:
        raise ValueError(f"Unknown MADLAD backend: {backend}")
    return _translate_hf_texts(
        [text], src_lang, tgt_lang, backend,
        device=device, num_beams=num_beams,
        max_new_tokens=max_new_tokens, progress_cb=progress_cb,
        status_cb=status_cb,
    )[0]


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
    *,
    status_cb: Optional[Callable[[str], None]] = None,
) -> str:
    """Dispatch to a backend; limit failures propagate to the caller."""
    if not text or not text.strip():
        return ""

    if (backend in NLLB_MODEL_IDS or backend in MADLAD_MODEL_IDS
            or backend == "opus-mt"):
        return _translate_hf_texts(
            [text], src_lang, tgt_lang, backend,
            progress_cb=progress_cb, status_cb=status_cb,
        )[0]
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
            status_cb=status_cb,
        )
    raise ValueError(f"Unknown translation backend: {backend}")


@serialized
def translate_many(
    texts: List[str],
    src_lang: str,
    tgt_lang: str,
    backend: str = "nllb",
    ollama_model: Optional[str] = None,
    src_lang_name: Optional[str] = None,
    tgt_lang_name: Optional[str] = None,
    formality: str = "default",
    batch_size: Optional[int] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    *,
    status_cb: Optional[Callable[[str], None]] = None,
) -> List[str]:
    """
    Translate a list of independent texts, returning one output per input.

    For the HuggingFace seq2seq backends this flattens every text into its
    chunks, translates all chunks in padded mini-batches (one shared GPU
    pass instead of one call per text), then reassembles per input. This is
    the fast path used by the format-preserving document pipelines, where a
    document may contain hundreds of short paragraphs.

    ``batch_size`` defaults to the allocated GPU profile and is capped by
    current free memory after model loading, including explicit overrides.
    Ollama has no batched API, so it falls back to a per-text loop.
    """
    if not texts:
        return []

    # Empty / whitespace-only inputs pass through untouched.
    to_do = [i for i, t in enumerate(texts) if t and t.strip()]
    result: List[str] = list(texts)
    if not to_do:
        return result

    if (backend in NLLB_MODEL_IDS or backend in MADLAD_MODEL_IDS
            or backend == "opus-mt"):
        return _translate_hf_texts(
            texts, src_lang, tgt_lang, backend, batch_size=batch_size,
            progress_cb=progress_cb, status_cb=status_cb,
        )

    # Ollama (and any other non-batchable backend): per-text loop.
    total = len(to_do)
    for done, idx in enumerate(to_do, start=1):
        result[idx] = translate(
            texts[idx],
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            backend=backend,
            ollama_model=ollama_model,
            src_lang_name=src_lang_name,
            tgt_lang_name=tgt_lang_name,
            formality=formality,
            status_cb=status_cb,
        )
        if progress_cb is not None:
            progress_cb(done, total)
    return result


def make_translate_fn(
    src_lang: str,
    tgt_lang: str,
    backend: str = "nllb",
    ollama_model: Optional[str] = None,
    src_lang_name: Optional[str] = None,
    tgt_lang_name: Optional[str] = None,
    formality: str = "default",
    progress_cb: Optional[Callable[[int, int], None]] = None,
    *,
    status_cb: Optional[Callable[[str], None]] = None,
) -> Callable[[str], str]:
    """
    Return a single-argument ``str -> str`` translator with all backend
    parameters baked in. Handy for feeding to format-preserving pipelines
    (:mod:`core.translation.format`) or the shielding wrapper.

    The returned callable also exposes a ``.many`` attribute: a
    ``List[str] -> List[str]`` batched translator with the same parameters
    baked in. Format-preserving pipelines use it to translate all units of
    a document in a few padded GPU passes instead of one call per unit.
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
            status_cb=status_cb,
        )

    def _many(texts: List[str]) -> List[str]:
        return translate_many(
            texts,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            backend=backend,
            ollama_model=ollama_model,
            src_lang_name=src_lang_name,
            tgt_lang_name=tgt_lang_name,
            formality=formality,
            progress_cb=progress_cb,
            status_cb=status_cb,
        )

    _fn.many = _many  # type: ignore[attr-defined]
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
                "python-docx is not installed in this container; "
                "upload .txt or .pdf instead."
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


@serialized
def free_translation_vram() -> None:
    """Evict HF and any translation-owned Ollama model under the shared lock.

    For sequential OCR, hold ``with translation_session():`` across this
    call, OCR, and subsequent translation. Nested calls are safe. HF reloads
    on the next translation; UI residency is invalidated immediately.
    Unrelated or unverified Ollama models are never unloaded. Failure to
    release an owned Ollama model propagates rather than claiming it is free.
    """
    _free_hf_cache()
    release_ollama_model()


@serialized
def backend_is_loaded(
    backend: str,
    src_lang: str = "",
    tgt_lang: str = "",
    ollama_model: Optional[str] = None,
) -> bool:
    """Validate residency, rather than trusting a stale UI load signature.

    HF state is process-local and includes the selected CUDA logical device.
    Ollama residency requires a successful current ``ps`` check; legacy
    clients without that API cannot prove residency and return False.
    """
    if backend == "ollama":
        return bool(ollama_model) and ollama_model_is_loaded(ollama_model)
    signature = backend_load_signature(backend, src_lang, tgt_lang)
    return (
        _ACTIVE_HF_SIGNATURE == signature
        and _ACTIVE_HF_DEVICE == _resolve_device(None)
    )


@serialized
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
    if (backend in NLLB_MODEL_IDS or backend in MADLAD_MODEL_IDS
            or backend == "opus-mt"):
        _load_hf(backend, src_lang, tgt_lang, _resolve_device(None))
        return

    if backend == "ollama":
        if not ollama_model:
            raise ValueError("The Ollama backend requires an ollama_model.")
        import ollama

        prepare_ollama_model(ollama_model)
        ollama.chat(
            model=ollama_model,
            messages=[{"role": "user", "content": "hi"}],
            options={"num_predict": 1, "temperature": 0.0},
        )
        return

    raise ValueError(f"Unknown backend: {backend}")
