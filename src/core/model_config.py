import json
from pathlib import Path
from typing import Any


_SRC_DIR = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _SRC_DIR / "config"

DEFAULT_CONFIG_PATH = _CONFIG_DIR / "models.json"
LOCAL_CONFIG_PATH = _CONFIG_DIR / "models.local.json"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _merge_config(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    """
    Merge local config into base config.

    For this use case, lists in models.local.json replace the default lists.
    """
    merged = base.copy()
    merged.update(override)
    return merged


def load_model_config() -> dict[str, Any]:
    base_config = _load_json(DEFAULT_CONFIG_PATH)
    local_config = _load_json(LOCAL_CONFIG_PATH)

    return _merge_config(base_config, local_config)


def is_high_memory_gpu(gpu_name: str) -> bool:
    config = load_model_config()
    high_memory_gpus = config.get("high_memory_gpus", ["A100", "H100", "H200"])

    return any(gpu in gpu_name for gpu in high_memory_gpus)


def get_available_models(gpu_name: str) -> list[str]:
    config = load_model_config()

    small_models = config.get("small_models", [])
    large_models = config.get("large_models", [])

    if is_high_memory_gpu(gpu_name):
        return small_models + large_models

    return small_models