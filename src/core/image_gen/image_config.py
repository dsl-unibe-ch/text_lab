"""
Configuration, type definitions, and helpers for the Ideogram-4 image-generation
feature exposed in the chat page.

The heavy `ideogram4` package lives in the isolated `imagegen_backend` conda env
(not the Streamlit env), so this module keeps its top-level imports lightweight —
nothing here imports torch or ideogram4 at module load time.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
from typing import Literal, TypedDict

# Gated Hugging Face repo (nf4 = bitsandbytes 4-bit, CUDA only). Downloaded once
# into the bind-mounted HF cache; read locally from then on.
MODEL_REPO: str = "ideogram-ai/ideogram-4-nf4"

# Absolute path to the shared HF cache on research storage. It is reachable both
# on the host and inside the container (via the /storage bind), so it works as a
# universal fallback when HF_HOME is not already set in the environment.
DEFAULT_HF_HOME: str = (
    "/storage/research/dsl_shared/solutions/ondemand/text_lab/container/models/huggingface"
)


def get_hf_home() -> str:
    """Resolve the HF cache root used for image-generation weights.

    Respects an explicit HF_HOME (e.g. ``/opt/huggingface`` inside the container)
    and otherwise falls back to the shared research-storage cache, which holds the
    pre-downloaded NF4 model.
    """
    return os.environ.get("HF_HOME") or DEFAULT_HF_HOME


def get_hf_token() -> str | None:
    """Resolve a Hugging Face access token for the gated ideogram-4-nf4 repo.

    Loading the gated repo online requires a token; without one, HEAD requests to
    the repo 401 (not 404), which breaks the pipeline's file-resolution fallback.
    Resolution order (first hit wins):

    1. ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` env vars.
    2. A token file pointed to by ``TEXT_LAB_HF_TOKEN_FILE`` (admin-provided,
       shared-deployment friendly).
    3. ``<HF_HOME>/token`` (the in-container HF token location).
    4. ``~/.cache/huggingface/token`` (the user's local `huggingface-cli login`).

    Returns the token string, or None if none is available (offline mode).
    """
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok and tok.strip():
        return tok.strip()

    candidates: list[str] = []
    file_env = os.environ.get("TEXT_LAB_HF_TOKEN_FILE")
    if file_env:
        candidates.append(file_env)
    candidates.append(os.path.join(get_hf_home(), "token"))
    candidates.append(os.path.expanduser("~/.cache/huggingface/token"))

    for path in candidates:
        try:
            if path and os.path.isfile(path):
                with open(path, encoding="utf-8") as fh:
                    value = fh.read().strip()
                if value:
                    return value
        except OSError:
            continue
    return None

# UI-facing quality labels -> ideogram4.sampler_configs.PRESETS keys.
PRESET_LABELS: dict[str, str] = {
    "Fast (12 steps)": "V4_TURBO_12",
    "Balanced (20 steps)": "V4_DEFAULT_20",
    "Quality (48 steps)": "V4_QUALITY_48",
}
DEFAULT_PRESET: str = "V4_DEFAULT_20"

# Ideogram 4 supports up to 2048 px; dimensions must be multiples of 16.
SIZE_OPTIONS: dict[str, tuple[int, int]] = {
    "Square 1024×1024": (1024, 1024),
    "Portrait 832×1216": (832, 1216),
    "Landscape 1216×832": (1216, 832),
    "Large square 1536×1536": (1536, 1536),
}
DEFAULT_SIZE: str = "Square 1024×1024"

# Generous timeout: the very first generation also pays the one-time model load.
IMAGE_TIMEOUT_SECONDS: int = 900


class ImageArtifact(TypedDict):
    """A single generated image plus the parameters that produced it."""
    path: str
    prompt: str
    preset: str
    seed: int
    width: int
    height: int


class ImageGenResult(TypedDict):
    """The complete result of an image-generation run."""
    summary: str
    images: list[ImageArtifact]
    logs: list[tuple[Literal["info", "warning", "error"], str]]


def get_artifacts_dir() -> str:
    """Directory where generated PNGs are written. Overridable via env var so the
    MCP subprocess and the Streamlit process always agree on the location."""
    env_dir = os.environ.get("IMAGE_GEN_ARTIFACTS_DIR")
    if env_dir:
        return env_dir
    # src/core/image_gen/image_config.py -> parents[2] == src
    return str(pathlib.Path(__file__).resolve().parents[2] / "mcp_artifacts" / "images")


def is_image_gen_installed() -> bool:
    """True if the Ideogram-4 image-generation backend is available.

    The `ideogram4` package (with transformers 5.x) lives in the isolated
    `imagegen_backend` conda env, reached via ``TEXT_LAB_IMAGEGEN_PYTHON``, not in
    the Streamlit env. So we consider the backend installed when that interpreter
    exists, and otherwise fall back to a find_spec check in the current env (dev
    single-env setups). find_spec avoids importing torch/ideogram4 (heavy).
    """
    backend_py = os.environ.get("TEXT_LAB_IMAGEGEN_PYTHON")
    if backend_py and os.path.isfile(backend_py):
        return True
    try:
        return importlib.util.find_spec("ideogram4") is not None
    except (ImportError, ValueError):
        return False


# Minimum total GPU VRAM (MiB) required to run the NF4 (4-bit) pipeline.
# The quantized weights are ~16 GB; a 24 GB card (e.g. RTX 4090) has enough
# headroom once the chat LLM is evicted during generation.
MIN_IMAGE_GEN_VRAM_MB: int = 20000


def get_gpu_vram_mb() -> int:
    """Return the largest single-GPU total VRAM in MiB, or 0 if undetectable."""
    import subprocess

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        )
        values = [int(line.strip()) for line in out.splitlines() if line.strip()]
        return max(values) if values else 0
    except Exception:
        return 0


def is_image_gen_supported(gpu_name: str) -> bool:
    """True if the GPU has enough VRAM to run the NF4 (4-bit) pipeline.

    The quantized model fits in ~16 GB, so any card with >= 20 GB total VRAM
    (RTX 4090 and up) qualifies. On lower-memory cards the chat LLM is evicted
    for the generation phase and reloaded afterwards.
    """
    from core.model_config import is_high_memory_gpu

    if is_high_memory_gpu(gpu_name):
        return True
    return get_gpu_vram_mb() >= MIN_IMAGE_GEN_VRAM_MB


# Image-generation intent is decided by the model's generate_image tool
# (chat_engine.stream_chat_with_image_tool), NOT by keyword matching here: a regex
# cannot tell a request ("draw Zoro") from a question about one ("which prompt would
# you use to generate an image of him?"). The explicit "Generate image" toggle is
# the deterministic override.
