"""
Reference-free machine-translation quality estimation.

Wraps CometKiwi (``Unbabel/wmt22-cometkiwi-da``) into a single
callable that returns a float score in [0, 1]. Higher = better.

Notes
-----
* The model is *gated* on Hugging Face and requires the user (or the
  bind-mounted HF cache) to have accepted its licence and provided a
  token. First call attempts a download; if that fails the module
  degrades gracefully to :data:`SCORE_UNAVAILABLE` and callers can
  hide the badge.
* Loading is lazy and memoised. The first call takes several seconds
  (model download + GPU transfer); subsequent calls are fast.
* Only meant for short, interactive text — running CometKiwi over an
  entire document is prohibitively slow. Document-mode translation
  therefore does *not* invoke this module.

Public API
----------

    is_available() -> bool
    estimate_quality(src, translated) -> float
    quality_badge(score) -> (level: str, emoji: str, css_color: str)
"""

from __future__ import annotations

import functools
from typing import Tuple


_COMET_MODEL_ID = "Unbabel/wmt22-cometkiwi-da"

# Sentinel returned when the model could not be loaded (gated repo /
# missing token / offline). Positive scores are always in [0, 1].
SCORE_UNAVAILABLE = -1.0


@functools.lru_cache(maxsize=1)
def _try_load_model():
    """
    Try to load CometKiwi. Returns the model on success, ``None`` on any
    failure (auth, network, missing package). Result is cached so we do
    not retry on every call.
    """
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        return None

    try:
        model_path = download_model(_COMET_MODEL_ID)
        return load_from_checkpoint(model_path)
    except Exception:
        # Gated repo, missing HF token, no network, corrupt cache, etc.
        return None


def is_available() -> bool:
    """Whether CometKiwi can be used in this environment. Loads on first call."""
    return _try_load_model() is not None


def estimate_quality(src: str, translated: str) -> float:
    """
    Return a CometKiwi reference-free quality score in [0, 1].

    Returns :data:`SCORE_UNAVAILABLE` if the model is not usable or the
    inputs are empty. Errors during prediction are swallowed so the UI
    never crashes because of a QE failure.
    """
    if not src or not src.strip() or not translated or not translated.strip():
        return SCORE_UNAVAILABLE

    model = _try_load_model()
    if model is None:
        return SCORE_UNAVAILABLE

    try:
        import torch

        gpus = 1 if torch.cuda.is_available() else 0
        data = [{"src": src, "mt": translated}]
        out = model.predict(data, batch_size=1, gpus=gpus, progress_bar=False)
        # ``out.system_score`` is a single float on modern comet releases;
        # fall back to averaging ``out.scores`` for older versions.
        score = getattr(out, "system_score", None)
        if score is None:
            scores = getattr(out, "scores", None) or []
            if not scores:
                return SCORE_UNAVAILABLE
            score = sum(scores) / len(scores)
        # Clamp to [0, 1] just in case a slightly-out-of-range score sneaks in.
        return max(0.0, min(1.0, float(score)))
    except Exception:
        return SCORE_UNAVAILABLE


def quality_badge(score: float) -> Tuple[str, str, str]:
    """
    Map a numeric quality score to a (level, emoji, css_color) tuple
    suitable for a small UI badge.

    Thresholds follow the CometKiwi paper's rough rule of thumb:
    * >= 0.80 -- high confidence
    * 0.60 - 0.79 -- review suggested
    * < 0.60 -- low confidence
    """
    if score < 0:
        return ("unavailable", "\u26AA", "#9e9e9e")
    if score >= 0.80:
        return ("high", "\U0001F7E2", "#2e7d32")
    if score >= 0.60:
        return ("medium", "\U0001F7E1", "#f9a825")
    return ("low", "\U0001F534", "#c62828")
