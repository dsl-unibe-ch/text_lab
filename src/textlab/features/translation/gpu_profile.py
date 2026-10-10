"""Profile the allocated CUDA logical device, not the largest host GPU.

Torch respects CUDA_VISIBLE_DEVICES and Slurm allocation. Detection is lazy
and uncached so CPU fallback, device changes and free memory remain honest.
Batch limits are conservative heuristics, not a guarantee against OOM.
Sequential OCR requires translation eviction under its lifecycle guard.
"""

from __future__ import annotations

from dataclasses import dataclass

from .gpu_memory import is_cuda_device

# Thresholds refer to the allocated logical CUDA device, in MiB.

# Minimum VRAM for the OCR worker (~8-9 GB) to co-exist with a resident
# translation model. 24 GB (RTX 4090) is below this; 40 GB (A100) and up
# clear it comfortably.
OCR_COEXIST_MIN_MB = 32_000

# Batch-size tiers by total VRAM. Larger cards -> larger batches -> faster.
_H200_MIN_MB = 120_000  # H200 (~141 GB)
_80GB_MIN_MB = 60_000  # A100-80 / H100 (~80 GB)
_A100_40_MIN_MB = 32_000  # A100-40 (~40 GB)

# 3B-parameter backends use much more activation memory per sample, so their
# batch is scaled down relative to the small (600M) default.
_LARGE_BACKENDS = frozenset({"nllb-large", "madlad-3b"})


@dataclass(frozen=True)
class GpuProfile:
    """Immutable description of the current GPU's translation capabilities."""

    name: str
    vram_mb: int
    tier: str  # "cpu" | "standard" | "high"
    batch_size: int  # base translation mini-batch (small models)
    ocr_with_translation: bool  # may OCR + translation be resident together?
    device: str = "cpu"
    free_mb: int | None = None

    @property
    def is_high_memory(self) -> bool:
        """Whether the GPU belongs to the high-memory tier."""
        return self.tier == "high"


def detect_gpu_profile(device: str | None = None) -> GpuProfile:
    """Read the current (or explicit) CUDA logical device through torch.

    Missing CUDA/properties gives a CPU profile. Missing free-memory data
    retains the device identity but forces a conservative batch of one.
    """
    cpu = GpuProfile("CPU", 0, "cpu", 4, False)
    if device is not None and not is_cuda_device(device):
        return cpu
    try:
        import torch

        if not torch.cuda.is_available():
            return cpu
        index = (
            int(str(device).split(":", 1)[1])
            if device is not None and ":" in str(device)
            else torch.cuda.current_device()
        )
        properties = torch.cuda.get_device_properties(index)
        name = properties.name
        vram = int(properties.total_memory) // (1024 * 1024)
    except (ImportError, AttributeError, RuntimeError, ValueError):
        return cpu
    if vram <= 0:
        return cpu
    try:
        free, _ = torch.cuda.mem_get_info(index)
        free_mb = max(0, int(free) // (1024 * 1024))
    except (AttributeError, RuntimeError, ValueError):
        free_mb = None

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
        device=f"cuda:{index}",
        free_mb=free_mb,
    )


def cap_batch_size(
    backend: str,
    requested: int,
    device: str | None = None,
    *,
    num_beams: int = 1,
) -> int:
    """Cap a requested batch against *current* post-load free memory.

    Reserve 2 GiB plus 768 MiB/sample (small models) or 1536 MiB/sample
    (3B models), scaled by beam count. OOM recovery is still necessary as
    sequence lengths, model implementations and other GPU jobs vary.
    """
    if requested <= 0:
        raise ValueError("Batch size must be positive.")
    profile = detect_gpu_profile(device)
    if profile.tier == "cpu":
        # Unknown CUDA properties must not be mistaken for ample free VRAM.
        ceiling = 1 if is_cuda_device(device) else profile.batch_size
        return min(requested, ceiling)
    if profile.free_mb is None:
        return 1
    per_sample = 1536 if backend in _LARGE_BACKENDS else 768
    available = max(0, profile.free_mb - 2048)
    memory_cap = max(1, available // (per_sample * max(1, num_beams)))
    return min(requested, memory_cap)


def resolve_batch_size(backend: str, device: str | None = None) -> int:
    """Resolve a live memory-aware batch; the one-argument API is retained."""
    profile = detect_gpu_profile(device)
    batch = profile.batch_size
    if backend in _LARGE_BACKENDS:
        batch = max(1, batch // 2)
    return cap_batch_size(backend, batch, device)


def ocr_with_translation_allowed() -> bool:
    """Compatibility advisory for simultaneous residency (not a guarantee)."""
    return detect_gpu_profile().ocr_with_translation


def sequential_ocr_allowed(
    *,
    min_free_mb: int | None = None,
    device: str | None = None,
) -> bool:
    """Check allocated GPU capacity and optionally current free VRAM (MiB).

    With no arguments this is a >=24 GB capacity advisory, usable before
    eviction. For execution, hold ``translation_session()``, evict only if
    OCR is needed, then pass the OCR worker's required free-memory budget.
    Missing free-memory data fails that check closed. Neither check reserves
    memory against other processes or implies hardware-tested OCR support.
    Pass ``device='cuda:0'`` for Paddle's first-visible-device worker, which
    does not inherit torch's in-process current-device selection.
    """
    if min_free_mb is not None and min_free_mb < 0:
        raise ValueError("The free-memory budget must be nonnegative.")
    profile = (
        detect_gpu_profile(device)
        if device is not None
        else detect_gpu_profile()
    )
    return profile.vram_mb >= 24_000 and (
        min_free_mb is None
        or (profile.free_mb is not None and profile.free_mb >= min_free_mb)
    )
