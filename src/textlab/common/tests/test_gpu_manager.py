"""GPU hand-over between features, with Ollama, nvidia-smi and /proc faked."""


import os
import signal

import pytest

from textlab.common import gpu_manager

#: The real function, before the fixture below replaces it.
UNLOAD_OLLAMA = gpu_manager._unload_ollama


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setattr(gpu_manager, "_RELEASERS", {})
    monkeypatch.setattr(gpu_manager, "_unload_ollama", lambda keep: [])
    monkeypatch.setattr(gpu_manager, "_gpu_processes", lambda: [])


def test_other_features_are_released_and_the_requested_one_kept():
    calls = []
    gpu_manager.register("ocr", "Stopped OCR worker",
                         lambda: calls.append("ocr") or True)
    gpu_manager.register("translation", "Unloaded translation model",
                         lambda: calls.append("translation") or False)
    gpu_manager.register("transcribe", "Unloaded audio language detector",
                         lambda: calls.append("transcribe") or True)
    freed = gpu_manager.prepare_gpu("translation", keep=("transcribe",))
    assert calls == ["ocr"]
    assert freed == ["Stopped OCR worker"]


def test_reregistering_a_hook_replaces_it_and_failures_do_not_stop_others():
    calls = []
    gpu_manager.register("ocr", "hook", lambda: calls.append("old"))
    gpu_manager.register("ocr", "hook", lambda: calls.append("new"))

    def broken():
        raise RuntimeError("release failed")

    gpu_manager.register("transcribe", "broken", broken)
    gpu_manager.register("transcribe", "fine", lambda: calls.append("fine"))
    gpu_manager.prepare_gpu("llm")
    assert calls == ["new", "fine"]


def test_ollama_keeps_only_the_requested_model(monkeypatch):
    seen = []
    monkeypatch.setattr(gpu_manager, "_unload_ollama",
                        lambda keep: seen.append(keep) or ["chat:8b"])
    assert gpu_manager.prepare_gpu("llm", ollama_model="qwen3") == [
        "Unloaded LLM chat:8b",
    ]
    assert seen == ["qwen3"]


def test_ollama_release_spares_the_kept_model(monkeypatch):
    seen = []

    def release_models(keep):
        seen.append(keep)
        return ["chat:8b"]

    monkeypatch.setattr(gpu_manager, "release_models", release_models)
    assert UNLOAD_OLLAMA("qwen3") == ["chat:8b"]
    assert UNLOAD_OLLAMA(None) == ["chat:8b"]
    assert seen == [{"qwen3:latest"}, set()]


def test_only_our_own_workers_of_released_features_are_stopped(monkeypatch):
    me = os.getpid()
    processes = {
        # pid: (parent, command line)
        101: (me, "python -m textlab.features.ocr.paddle_vl_worker --serve"),
        102: (me, "python -m textlab.features.transcription.worker /job"),
        103: (1, "/usr/local/lib/ollama/llama-server"),       # Ollama
        104: (999, "python paddle_ocr_worker.py --lang en"),  # not ours
        105: (me, "python something_else.py"),                # unknown
        999: (1, "other user's process"),
    }
    alive = set(processes)
    killed = []
    monkeypatch.setattr(gpu_manager, "_gpu_processes",
                        lambda: [me, 101, 102, 103, 104, 105])
    monkeypatch.setattr(
        gpu_manager, "_stat",
        lambda pid: ["S" if pid in alive else "Z", str(processes[pid][0])]
        if pid in processes else None,
    )
    monkeypatch.setattr(
        gpu_manager, "_command_line", lambda pid: processes[pid][1],
    )

    def fake_kill(pid, sig):
        killed.append((pid, sig))
        alive.discard(pid)

    monkeypatch.setattr(gpu_manager.os, "kill", fake_kill)
    freed = gpu_manager.prepare_gpu("llm", keep=("transcribe",))
    assert killed == [(101, signal.SIGTERM)]
    assert freed == ["Stopped leftover ocr worker"]


def test_torch_cache_is_not_initialized_by_the_manager(monkeypatch):
    class FakeCuda:
        emptied = False

        @staticmethod
        def is_initialized():
            return False

        @classmethod
        def empty_cache(cls):
            cls.emptied = True

    fake_torch = type("torch", (), {"cuda": FakeCuda})
    monkeypatch.setitem(gpu_manager.sys.modules, "torch", fake_torch)
    gpu_manager.prepare_gpu("ocr")
    assert not FakeCuda.emptied
