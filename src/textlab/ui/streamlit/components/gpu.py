"""Freeing the GPU before a feature runs, with feedback for the user."""

from collections.abc import Iterable

import streamlit as st

from textlab.common.gpu_manager import prepare_gpu


def free_gpu_for(
    feature: str,
    *,
    ollama_model: str | None = None,
    keep: Iterable[str] = (),
) -> list[str]:
    """Free the GPU for ``feature`` and tell the user what was released.

    Wraps :func:`textlab.common.gpu_manager.prepare_gpu` with a spinner and a
    toast. Call it from the Streamlit script thread, not a background thread.

    Args:
        feature: The feature about to run, one of the ``gpu_manager``
            feature constants (e.g. ``gpu_manager.OCR``).
        ollama_model: The Ollama model the feature is about to use; it is
            kept loaded.
        keep: Further features whose GPU memory must not be released.

    Returns:
        Notes describing what was freed, e.g. ``"Unloaded LLM qwen3:8b"``.
    """
    with st.spinner("Preparing the GPU..."):
        freed = prepare_gpu(feature, ollama_model=ollama_model, keep=keep)
    if freed:
        st.toast("Freed GPU memory: " + "; ".join(freed))
    return freed
