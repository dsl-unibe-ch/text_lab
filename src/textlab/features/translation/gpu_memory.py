"""Process-local translation lifecycle guard and CUDA cleanup helpers.

OCR callers must hold ``translation_session()`` across eviction, OCR and
subsequent translation. The guard is reentrant; it does not coordinate
other processes or unrelated applications using the same GPU.
"""

from __future__ import annotations

import functools
import gc
import threading
import traceback


_LOCK = threading.RLock()


def translation_session():
    """Return the shared RLock, usable as a reentrant context manager."""
    return _LOCK


def serialized(function):
    """Serialize model mutation and inference, including nested calls."""
    @functools.wraps(function)
    def guarded(*args, **kwargs):
        with translation_session():
            return function(*args, **kwargs)
    return guarded


def is_cuda_device(device) -> bool:
    return str(device).split(":", 1)[0] == "cuda"


def clear_cuda_cache(device=None) -> None:
    """Collect dead tensors before releasing this process's cached blocks.

    This never unloads other applications' models. Cleanup is best effort
    when CUDA is unavailable; it must not replace the original failure.
    """
    gc.collect()
    if device is not None and not is_cuda_device(device):
        return
    try:
        import torch

        if torch.cuda.is_available():
            if device is not None:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            else:
                torch.cuda.empty_cache()
    except (ImportError, AttributeError, RuntimeError):
        pass


def discard_exception_tensors(error: BaseException) -> None:
    """Detach failed frames/chains before leaving an OOM exception handler."""
    seen = set()
    pending = [error]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        pending.extend(
            linked for linked in (current.__cause__, current.__context__)
            if linked is not None
        )
        traceback.clear_frames(current.__traceback__)
        current.__traceback__ = None
        current.__context__ = None
        current.__cause__ = None


def is_cuda_oom(error: RuntimeError, device, torch) -> bool:
    """Recognize CUDA OOM only, never arbitrary generation failures."""
    if not is_cuda_device(device):
        return False
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if isinstance(oom_type, type) and isinstance(error, oom_type):
        return True
    message = str(error).lower()
    return message.startswith((
        "cuda out of memory", "cuda error: out of memory",
    ))
