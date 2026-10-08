"""GPU hand-over between features, with Ollama, nvidia-smi and /proc faked."""

import conftest_path  # noqa: F401

import os
import signal

import pytest

from core import gpu_manager, vision_enrich


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


def test_free_gpu_spares_kept_ollama_models(monkeypatch):
    unloaded = []
    resident = ["chat:8b", "qwen3:latest"]

    def fake_request(base_url, path, payload=None, timeout=60.0):
        if path == "/api/generate":
            unloaded.append(payload["model"])
            resident.remove(payload["model"])
            return {}
        return {"models": [{"model": name} for name in resident]}

    monkeypatch.setattr(vision_enrich, "_ollama_request", fake_request)
    assert vision_enrich.free_gpu(keep={"qwen3:latest"}) == ["chat:8b"]
    assert unloaded == ["chat:8b"]
    assert gpu_manager._normalize("qwen3") == "qwen3:latest"
    assert gpu_manager._normalize("qwen3:8b") == "qwen3:8b"


def test_only_our_own_workers_of_released_features_are_stopped(monkeypatch):
    me = os.getpid()
    processes = {
        # pid: (parent, command line)
        101: (me, "python paddle_vl_worker.py --serve"),
        102: (me, "python transcribe_worker.py"),
        103: (1, "/usr/local/lib/ollama/llama-server"),       # Ollama
        104: (999, "python paddle_vl_worker.py --serve"),     # not ours
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
