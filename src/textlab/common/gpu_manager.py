"""One place that frees the GPU before a feature runs.

Text Lab runs every feature in one Slurm job on one GPU (usually a 24 GB
RTX 4090). Each feature loads its model on first use and keeps it, so using
Chat, then OCR, then Transcribe in one session runs out of memory. Pages
therefore call :func:`prepare_gpu` when the user presses a feature's run
button: every *other* feature's models are released, the requested
feature's own models stay warm.

GPU memory lives in three places, each released differently:

* **Ollama** (a separate server) -- models are unloaded through its API,
  except the one the feature is about to use. The server keeps running.
* **Helper processes** this app starts (PaddleOCR workers, the
  transcription worker) -- stopped through their owner's
  release hook; as a last resort a leftover worker still holding GPU memory
  is terminated. Only descendants of this process are ever touched.
* **This Streamlit process** (translation models, EasyOCR, Whisper language
  detection, WhisperX during Transcribe) -- caches are cleared through
  registered hooks, then PyTorch's allocator cache is emptied.

Features register a release hook for each owner name with :func:`register`.
"""

from __future__ import annotations

import gc
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Callable, Dict, Iterable, List, Optional

LOGGER = logging.getLogger(__name__)

#: Owner names used by the features. A feature keeps its own owner (and any
#: extra ``keep`` owners) and releases all others.
OCR = "ocr"
TRANSCRIBE = "transcribe"
TRANSLATION = "translation"
TOPIC_MODELING = "topic_modeling"
#: Pure LLM features (summaries, knowledge graph, visualisation) own nothing
#: beyond their Ollama model.
LLM = "llm"

#: Helper scripts that hold GPU memory in their own process, by owner. Used
#: only for the last-resort cleanup of leftovers (e.g. a run interrupted
#: before its ``finally`` block ran).
_WORKER_SCRIPTS = {
    "paddle_vl_worker.py": OCR,
    "paddle_ocr_worker.py": OCR,
    "transcribe_worker.py": TRANSCRIBE,
}

_RELEASERS: Dict[str, Dict[str, Callable[[], object]]] = {}
_LOCK = threading.RLock()


def register(owner: str, name: str, release: Callable[[], object]) -> None:
    """Register ``release`` to free one of ``owner``'s GPU resources.

    ``name`` identifies the hook, so re-registering (Streamlit re-executes
    pages on every rerun) replaces it instead of adding a duplicate.
    """
    with _LOCK:
        _RELEASERS.setdefault(owner, {})[name] = release


def prepare_gpu(
    feature: str,
    *,
    ollama_model: Optional[str] = None,
    keep: Iterable[str] = (),
) -> List[str]:
    """Release every GPU consumer except ``feature``'s; return what was freed.

    ``ollama_model`` is the Ollama model the feature is about to use; it is
    the only one left loaded. The returned list holds short, user-readable
    notes such as "Unloaded LLM qwen3:8b" for an optional toast.
    """
    kept = {feature, *keep}
    freed: List[str] = []
    with _LOCK:
        for owner, hooks in list(_RELEASERS.items()):
            if owner in kept:
                continue
            for name, release in list(hooks.items()):
                try:
                    if release():
                        freed.append(name)
                except Exception:
                    LOGGER.exception("Releasing %s/%s failed", owner, name)
        freed += [f"Unloaded LLM {name}"
                  for name in _unload_ollama(ollama_model)]
        freed += _stop_leftover_workers(kept)
        _empty_torch_cache()
    if freed:
        LOGGER.info("GPU prepared for %s: %s", feature, "; ".join(freed))
    return freed


def _normalize(model: str) -> str:
    return model if ":" in model else model + ":latest"


def _unload_ollama(keep_model: Optional[str]) -> List[str]:
    try:
        from textlab.features.ocr import vision_enrich
    except ImportError:  # pragma: no cover - standalone imports
        import vision_enrich  # type: ignore
    keep = {_normalize(keep_model)} if keep_model else set()
    try:
        return vision_enrich.free_gpu(keep=keep)
    except Exception:
        LOGGER.exception("Unloading Ollama models failed")
        return []


def _empty_torch_cache() -> None:
    """Return freed PyTorch blocks to the driver so other processes can use
    them. Never initializes CUDA in a process that has not used it."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        LOGGER.exception("Emptying the PyTorch CUDA cache failed")


def _gpu_processes() -> List[int]:
    """PIDs of processes holding GPU memory, or [] if nvidia-smi is missing."""
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except Exception:
        return []
    return [int(line) for line in output.split() if line.strip().isdigit()]


def _stat(pid: int) -> Optional[List[str]]:
    """``/proc/<pid>/stat`` fields after the command name (state, ppid...)."""
    try:
        with open(f"/proc/{pid}/stat") as handle:
            # The command name may contain spaces; fields resume after ")".
            return handle.read().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return None


def _parent(pid: int) -> Optional[int]:
    fields = _stat(pid)
    try:
        return int(fields[1]) if fields else None
    except (ValueError, IndexError):
        return None


def _running(pid: int) -> bool:
    """Alive and not a zombie (a zombie has already released its memory)."""
    fields = _stat(pid)
    return bool(fields) and fields[0] not in ("Z", "X")


def _is_descendant(pid: int, ancestor: int) -> bool:
    for _ in range(64):
        pid = _parent(pid)
        if pid is None or pid <= 1:
            return False
        if pid == ancestor:
            return True
    return False


def _command_line(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            return handle.read().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def _stop_leftover_workers(kept: set) -> List[str]:
    """Terminate this app's own GPU worker processes of released features."""
    me = os.getpid()
    stopped = []
    for pid in _gpu_processes():
        if pid == me or not _is_descendant(pid, me):
            continue
        command = _command_line(pid)
        owner = next((owner for script, owner in _WORKER_SCRIPTS.items()
                      if script in command), None)
        if owner is None or owner in kept:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and _running(pid):
                time.sleep(0.2)
            if _running(pid):
                os.kill(pid, signal.SIGKILL)
            stopped.append(f"Stopped leftover {owner} worker")
        except ProcessLookupError:
            continue
        except OSError:
            LOGGER.exception("Stopping GPU worker %s failed", pid)
    return stopped


def gpu_memory_mb() -> Optional[tuple]:
    """``(used, total)`` MiB of the first visible GPU, or ``None``."""
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.splitlines()
        used, total = (int(value) for value in output[0].split(","))
        return used, total
    except Exception:
        return None
