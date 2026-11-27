"""The language models a site offers, by the session's GPU.

The site's model list is ``models.json`` next to this module, with local
changes in ``models.local.json``: small models for every GPU, large ones
for GPUs with much memory.
"""

from __future__ import annotations

import dataclasses
import functools
import json
from pathlib import Path
from typing import Any

_CONFIG_DIR = Path(__file__).resolve().parent

DEFAULT_CONFIG_PATH = _CONFIG_DIR / "models.json"
LOCAL_CONFIG_PATH = _CONFIG_DIR / "models.local.json"


def _load_json(path: Path) -> dict[str, Any]:
    """Return a JSON file's content, or an empty dict if it is missing."""
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _merge_config(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    """Merge the local model list into the default one.

    Lists in ``models.local.json`` replace the default lists.
    """
    merged = base.copy()
    merged.update(override)
    return merged


def load_model_config() -> dict[str, Any]:
    """Return the site's model list, with its local changes."""
    base_config = _load_json(DEFAULT_CONFIG_PATH)
    local_config = _load_json(LOCAL_CONFIG_PATH)

    return _merge_config(base_config, local_config)


def is_high_memory_gpu(gpu_name: str) -> bool:
    """Return whether a GPU has the memory for the large models.

    Args:
        gpu_name: The GPU, as ``nvidia-smi`` names it.

    Returns:
        True if its name contains one of the list's ``high_memory_gpus``.
    """
    config = load_model_config()
    high_memory_gpus = config.get("high_memory_gpus", ["A100", "H100", "H200"])

    return any(gpu in gpu_name for gpu in high_memory_gpus)


def get_available_models(gpu_name: str) -> list[str]:
    """Return the models offered on a GPU, the default first.

    Args:
        gpu_name: The GPU, as ``nvidia-smi`` names it.

    Returns:
        The large models on GPUs with much memory, else the small ones.
    """
    config = load_model_config()

    small_models = config.get("small_models", [])
    large_models = config.get("large_models", [])

    if is_high_memory_gpu(gpu_name):
        return large_models

    return small_models


@dataclasses.dataclass(frozen=True)
class ModelOptions:
    """The language models a session can choose from.

    Attributes:
        models: The Ollama model names of the site's model list for the
            session's GPU; the first is the default.
        gpu: The session's GPU, as ``nvidia-smi`` names it.
        large: Whether the large models are offered, which only GPUs with
            much memory get.
    """

    models: tuple[str, ...]
    gpu: str
    large: bool


@functools.cache
def session_gpu() -> str:
    """Return the session's GPU, asked once (it does not change)."""
    from textlab.common.gpu_manager import get_gpu_name

    return get_gpu_name()


def model_options() -> ModelOptions:
    """Return the models to choose from, for this session's GPU.

    Returns:
        The models; large ones only on GPUs with much memory.
    """
    gpu = session_gpu()
    return ModelOptions(
        models=tuple(get_available_models(gpu)),
        gpu=gpu,
        large=is_high_memory_gpu(gpu),
    )
