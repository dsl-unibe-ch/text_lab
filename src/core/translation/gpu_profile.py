"""
GPU capability profiling for the translation pipeline.

Text Lab runs on Slurm-allocated nodes that may carry very different GPUs:

* **RTX 4090** (24 GB) — what most users have access to.
* **A100** (40 / 80 GB), **H100** (80 GB), **H200** (141 GB) — high-memory
  cards available to fewer users.

Two things scale with the card:

1. **Translation batch size.** Batching many chunks into one padded
   ``model.generate`` pass is the main speed lever; a bigger card can hold a
   bigger batch and therefore run faster.
2. **Whether OCR and translation may be resident at the same time.** The
   PaddleOCR-VL worker needs ~8-9 GB. On a 24 GB card it does not co-exist
   reliably with a resident translation model, so scanned PDFs (which require
   OCR) are refused and the user is asked to relaunch on an A100 or better.
   On >= 32 GB cards both fit, so everything is allowed.

This module is Streamlit-free and dependency-light: it only shells out to
``nvidia-smi`` once (cached) and exposes a small immutable profile.
"""

from __future__ import annotations

import functools
import subprocess
from dataclasses import dataclass
from typing import Tuple

# ---------------------------------------------------------------------------
# Thresholds (MiB of total VRAM on the largest visible GPU)
# ---------------------------------------------------------------------------

# Minimum VRAM for the OCR worker (~8-9 GB) to co-exist with a resident
# translation model. 24 GB (RTX 4090) is below this; 40 GB (A100) and up
# clear it comfortably.
OCR_COEXIST_MIN_MB = 32_000

# Batch-size tiers by total VRAM. Larger cards -> larger batches -> faster.
_H200_MIN_MB = 120_000   # H200 (~141 GB)
_80GB_MIN_MB = 60_000    # A100-80 / H100 (~80 GB)
_A100_40_MIN_MB = 32_000  # A100-40 (~40 GB)

# 3B-parameter backends use much more activation memory per sample, so their
# batch is scaled down relative to the small (600M) default.
_LARGE_BACKENDS = frozenset({"nllb-large", "madlad-3b"})


@dataclass(frozen=True)
class GpuProfile:
    """Immutable description of the current GPU's translation capabilities."""

    name: str
    vram_mb: int
    tier: str                 # "cpu" | "standard" | "high"
    batch_size: int           # base translation mini-batch (small models)
    ocr_with_translation: bool  # may OCR + translation be resident together?

    @property
    def is_high_memory(self) -> bool:
        return self.tier == "high"


@functools.lru_cache(maxsize=1)
def _query_gpu() -> Tuple[str, int]:
    """Return (name, total_vram_mb) of the largest visible GPU, or ("", 0)."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return ("", 0)

    best_name, best_vram = "", 0
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        name = parts[0]
        try:
            vram = int(parts[1])
        except ValueError:
            continue
        if vram > best_vram:
            best_name, best_vram = name, vram
    return (best_name, best_vram)


@functools.lru_cache(maxsize=1)
def detect_gpu_profile() -> GpuProfile:
    """Detect the current GPU and derive the translation profile (cached)."""
    name, vram = _query_gpu()

    if vram <= 0:
        return GpuProfile(
            name=name or "CPU",
            vram_mb=0,
            tier="cpu",
            batch_size=4,
            ocr_with_translation=False,
        )

    if vram >= _H200_MIN_MB:
        batch = 64
    elif vram >= _80GB_MIN_MB:
        batch = 48
    elif vram >= _A100_40_MIN_MB:
        batch = 32
    else:
        batch = 16

    ocr_ok = vram >= OCR_COEXIST_MIN_MB
    return GpuProfile(
        name=name,
        vram_mb=vram,
        tier="high" if ocr_ok else "standard",
        batch_size=batch,
        ocr_with_translation=ocr_ok,
    )


def resolve_batch_size(backend: str) -> int:
    """
    Return the translation mini-batch size to use for ``backend`` on the
    current GPU. 3B-parameter backends are scaled down to stay within VRAM.
    """
    profile = detect_gpu_profile()
    batch = profile.batch_size
    if backend in _LARGE_BACKENDS:
        batch = max(2, batch // 2)
    return batch


def ocr_with_translation_allowed() -> bool:
    """
    True if the OCR worker may run while a translation model is resident.

    False on the RTX 4090 (24 GB) tier: scanned PDFs (which need OCR) are
    refused there and the user is asked to relaunch on an A100 / H100 / H200.
    """
    return detect_gpu_profile().ocr_with_translation
