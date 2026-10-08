"""Offline GPU allocation, OOM recovery and lifecycle contracts."""

import conftest_path  # noqa: F401

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
import functools
import subprocess
import sys
import threading
from types import SimpleNamespace
import weakref

import pytest

from core.translation import engine, gpu_profile, hf_backend
from core.translation import ollama_backend as ollama
from core.translation.chunking import MAX_SPLIT_RETRIES, OutputTruncatedError
from test_translation_limits import Model, Tensor, Tokenizer


MIB = 1024 * 1024


class Cuda:
    class OutOfMemoryError(RuntimeError):
        pass

    def __init__(self):
        self.available = True
        self.current = 1
        self.total = {0: 141_000, 1: 24_576}
        self.free = {0: 130_000, 1: 20_000}
        self.queries = []
        self.cleanups = []
        self.on_cleanup = lambda: None

    def is_available(self):
        return self.available

    def current_device(self):
        return self.current

    def get_device_properties(self, index):
        self.queries.append(("properties", index))
        return SimpleNamespace(
            name=f"allocated-{index}", total_memory=self.total[index] * MIB,
        )

    def mem_get_info(self, index):
        self.queries.append(("free", index))
        return self.free[index] * MIB, self.total[index] * MIB

    @contextmanager
    def device(self, device):
        previous = self.current
        self.current = int(str(device).split(":", 1)[1])
        try:
            yield
        finally:
            self.current = previous

    def empty_cache(self):
        self.on_cleanup()
        self.cleanups.append(self.current)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    cuda = Cuda()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=cuda, inference_mode=nullcontext,
    ))
    monkeypatch.setitem(sys.modules, "transformers", None)
    monkeypatch.setitem(sys.modules, "ollama", None)
    monkeypatch.setattr(ollama, "_ACTIVE_OLLAMA", None)
    monkeypatch.setattr(ollama, "_OWNED_OLLAMA", None)
    monkeypatch.setattr(engine, "_ACTIVE_HF_SIGNATURE", None)
    monkeypatch.setattr(engine, "_ACTIVE_HF_DEVICE", None)

    def forbidden(*args, **kwargs):
        raise AssertionError("No host GPU probes or services in offline tests")

    monkeypatch.setattr(subprocess, "check_output", forbidden)
    yield cuda
    for loader in (engine._load_nllb, engine._load_madlad,
                   engine._load_marian):
        clear = getattr(loader, "cache_clear", None)
        if clear is not None:
            clear()


def test_profile_uses_allocated_logical_device_and_no_host_probe(offline):
    profile = gpu_profile.detect_gpu_profile()
    assert profile.name == "allocated-1"
    assert profile.device == "cuda:1"
    assert profile.vram_mb == 24_576
    assert profile.free_mb == 20_000
    assert not profile.ocr_with_translation
    assert gpu_profile.sequential_ocr_allowed()
    assert all(index == 1 for _, index in offline.queries)
    offline.current = 0
    assert gpu_profile.detect_gpu_profile().vram_mb == 141_000
    assert gpu_profile.ocr_with_translation_allowed()
    assert gpu_profile.detect_gpu_profile("cuda:1").device == "cuda:1"


def test_sequential_ocr_free_budget_is_explicit_and_live(monkeypatch, offline):
    offline.free[1] = 8_000
    assert gpu_profile.sequential_ocr_allowed()
    assert not gpu_profile.sequential_ocr_allowed(min_free_mb=12_000)
    offline.free[1] = 12_000
    assert gpu_profile.sequential_ocr_allowed(min_free_mb=12_000)
    offline.total[1] = 16_000
    assert not gpu_profile.sequential_ocr_allowed(min_free_mb=12_000)
    offline.total[1] = 24_576

    def unknown(index):
        raise RuntimeError("free memory unavailable")

    monkeypatch.setattr(offline, "mem_get_info", unknown)
    assert gpu_profile.sequential_ocr_allowed()
    assert not gpu_profile.sequential_ocr_allowed(min_free_mb=12_000)
    with pytest.raises(ValueError, match="nonnegative"):
        gpu_profile.sequential_ocr_allowed(min_free_mb=-1)


def test_no_cuda_and_explicit_cpu_report_honest_cpu(offline):
    assert gpu_profile.detect_gpu_profile("cpu").tier == "cpu"
    offline.available = False
    assert gpu_profile.detect_gpu_profile().name == "CPU"
    assert gpu_profile.resolve_batch_size("nllb") == 4
    assert not gpu_profile.sequential_ocr_allowed()
    assert not gpu_profile.ocr_with_translation_allowed()
    assert engine._resolve_device("cuda:1") == "cpu"
    assert not offline.queries


def test_profile_fields_are_backwards_compatible():
    profile = gpu_profile.GpuProfile("old", 24_000, "standard", 16, False)
    assert profile.batch_size == 16
    assert profile.free_mb is None


def test_live_free_memory_caps_batch_and_beams(offline):
    offline.free[1] = 2048 + 4 * 768
    assert gpu_profile.resolve_batch_size("nllb") == 4
    assert gpu_profile.resolve_batch_size("madlad-3b") == 2
    assert gpu_profile.cap_batch_size("nllb", 16, num_beams=2) == 2
    offline.free[1] = 100
    assert gpu_profile.resolve_batch_size("nllb") == 1
    assert gpu_profile.cap_batch_size("nllb", 100, "cpu") == 4


def test_unknown_free_memory_does_not_assume_large_capacity(
    monkeypatch, offline,
):
    def unavailable(index):
        raise RuntimeError("memory information unavailable")

    monkeypatch.setattr(offline, "mem_get_info", unavailable)
    assert gpu_profile.detect_gpu_profile().device == "cuda:1"
    assert gpu_profile.resolve_batch_size("nllb") == 1
    monkeypatch.delattr(Cuda, "get_device_properties")
    assert gpu_profile.detect_gpu_profile().tier == "cpu"
    assert gpu_profile.cap_batch_size("nllb", 16, "cuda:1") == 1


def test_engine_caps_default_and_explicit_batch_after_model_load(
    monkeypatch, offline,
):
    tokenizer, model = Tokenizer(), Model()
    events = []

    def load(*args):
        events.append(args)
        offline.free[1] = 2048 + 2 * 768
        return tokenizer, model

    monkeypatch.setattr(engine, "_load_nllb", load)
    for requested in (None, 32):
        offline.free[1] = 20_000
        model.calls.clear()
        assert engine.translate_many(
            ["one", "two", "three", "four", "five"], "deu_Latn",
            "fra_Latn", batch_size=requested,
        ) == ["ONE", "TWO", "THREE", "FOUR", "FIVE"]
        assert [len(texts) for texts, _ in model.calls] == [2, 2, 1]
    assert all(args[1:] == ("cuda:1", "float16") for args in events)


def test_explicit_cpu_uses_float32_and_small_batch(monkeypatch, offline):
    tokenizer, model = Tokenizer(), Model()
    calls = []

    def load(*args):
        calls.append(args)
        return tokenizer, model

    monkeypatch.setattr(engine, "_load_nllb", load)
    engine._translate_chunks_hf(
        ["one"] * 9, "deu_Latn", "fra_Latn", "nllb", device="cpu",
        batch_size=64,
    )
    assert calls[0][1:] == ("cpu", "float32")
    assert [len(texts) for texts, _ in model.calls] == [4, 4, 1]


class OomModel(Model):
    def __init__(self, cuda, allowed=2, finish=None):
        super().__init__(finish)
        self.cuda = cuda
        self.allowed = allowed
        self.attempts = []
        self.failed_tensors = []

    def generate(self, input_ids, **kwargs):
        self.attempts.append((list(input_ids.texts), kwargs.copy()))
        if len(input_ids.texts) > self.allowed:
            self.failed_tensors.append(weakref.ref(input_ids))
            raise self.cuda.OutOfMemoryError("CUDA out of memory.")
        return super().generate(input_ids, **kwargs)


def test_oom_retries_are_bounded_lossless_ordered_and_release_tensors(
    offline,
):
    model = OomModel(offline)
    tokenizer = Tokenizer()
    statuses, progress = [], []

    def cleanup():
        assert all(ref() is None for ref in model.failed_tensors)

    offline.on_cleanup = cleanup
    sources = ["one", "two", "three", "four", "five", "six", "seven"]
    outputs = hf_backend.generate_translations(
        model, tokenizer, sources, "cuda:1", batch_size=6,
        max_new_tokens=17, num_beams=2, status_cb=statuses.append,
        progress_cb=lambda *args: progress.append(args),
    )
    assert outputs == [text.upper() for text in sources]
    assert [len(texts) for texts, _ in model.attempts] == [6, 3] + [1] * 7
    assert [text for batch, _ in model.calls for text in batch] == sources
    assert all(options["max_new_tokens"] == 17
               and options["num_beams"] == 2
               for _, options in model.attempts)
    assert len(statuses) == 2
    assert all("Token budgets unchanged" in status for status in statuses)
    assert offline.cleanups == [1, 1]
    assert progress == [(6, 7), (7, 7)]


def test_tensor_transfer_oom_releases_failed_allocations(offline):
    references = []

    class TransferTensor(Tensor):
        def to(self, device):
            if len(self.texts) > 1:
                references.append(weakref.ref(self))
                raise RuntimeError("CUDA error: out of memory")
            return self

    class TransferTokenizer(Tokenizer):
        def __call__(self, value, **kwargs):
            encoded = super().__call__(value, **kwargs)
            if isinstance(value, list):
                tensor = encoded["input_ids"]
                encoded["input_ids"] = TransferTensor(tensor.rows, value)
            return encoded

    def cleanup():
        assert all(ref() is None for ref in references)

    offline.on_cleanup = cleanup
    assert hf_backend.generate_translations(
        Model(), TransferTokenizer(), ["one", "two"], "cuda:1",
    ) == ["ONE", "TWO"]
    assert len(offline.cleanups) == 1


def test_output_retries_do_not_retain_generated_tensors(offline):
    references = []

    class NoRetainedTensorModel(Model):
        def generate(self, input_ids, **kwargs):
            assert all(ref() is None for ref in references)
            tensor = super().generate(input_ids, **kwargs)
            references.append(weakref.ref(tensor))
            return tensor

    model = NoRetainedTensorModel(finish=lambda text: len(text.split()) == 1)
    assert hf_backend.generate_translations(
        model, Tokenizer(), ["one two", "three"], "cuda:1",
    ) == ["ONE TWO", "THREE"]
    assert all(ref() is None for ref in references)


def test_oom_batch_one_fails_clearly_without_retaining_traceback(offline):
    model = OomModel(offline, allowed=0)
    statuses = []
    with pytest.raises(RuntimeError, match="microbatch 1") as caught:
        hf_backend.generate_translations(
            model, Tokenizer(), ["one", "two"], "cuda:1", batch_size=2,
            status_cb=statuses.append,
        )
    assert [len(batch) for batch, _ in model.attempts] == [2, 1]
    assert caught.value.__context__ is None
    assert all(ref() is None for ref in model.failed_tensors)
    assert "No partial" in statuses[-1]
    assert len(offline.cleanups) == 2


@pytest.mark.parametrize("device,message", [
    ("cuda:1", "unrelated runtime error"),
    ("cuda:1", "CUDA error: device-side assert triggered"),
    ("cpu", "CUDA out of memory."),
])
def test_unrelated_runtime_errors_are_not_caught(offline, device, message):
    error = RuntimeError(message)

    class Broken(Model):
        def generate(self, **kwargs):
            raise error

    with pytest.raises(RuntimeError) as caught:
        hf_backend.generate_translations(
            Broken(), Tokenizer(), ["one", "two"], device,
        )
    assert caught.value is error
    assert not offline.cleanups


def test_legacy_cuda_oom_message_recovers(offline):
    class Legacy(Model):
        def generate(self, input_ids, **kwargs):
            if len(input_ids.texts) > 1:
                raise RuntimeError("CUDA out of memory. Tried to allocate")
            return super().generate(input_ids, **kwargs)

    assert hf_backend.generate_translations(
        Legacy(), Tokenizer(), ["one", "two"], "cuda:1",
    ) == ["ONE", "TWO"]
    assert len(offline.cleanups) == 1


def test_oom_does_not_consume_output_retry_budget(offline):
    model = OomModel(
        offline, allowed=1, finish=lambda text: len(text.split()) == 1,
    )
    statuses = []
    assert hf_backend.generate_translations(
        model, Tokenizer(), ["one two", "three four"], "cuda:1",
        batch_size=4, max_new_tokens=3, status_cb=statuses.append,
    ) == ["ONE TWO", "THREE FOUR"]
    assert sum("CUDA" in message for message in statuses) == 1
    assert sum("attempt 1/" in message for message in statuses) == 2
    assert all(kwargs["max_new_tokens"] == 3 for _, kwargs in model.attempts)


def test_output_retry_bound_is_preserved_after_oom(offline):
    tokenizer = Tokenizer()
    tokenizer.model_max_length = 512
    model = OomModel(offline, allowed=1, finish=lambda text: False)
    model.config.max_position_embeddings = 512
    statuses = []
    with pytest.raises(OutputTruncatedError, match="No partial"):
        hf_backend.generate_translations(
            model, tokenizer, ["word " * 64, "fine"], "cuda:1",
            max_new_tokens=2, status_cb=statuses.append,
        )
    token_retries = [s for s in statuses if "output-token budget" in s]
    assert len(token_retries) == MAX_SPLIT_RETRIES


@pytest.fixture
def cached_loaders(monkeypatch):
    loads, references = [], []

    def make_loader(label):
        @functools.lru_cache(maxsize=1)
        def load(*args):
            assert all(ref() is None for ref in references)
            tokenizer, model = Tokenizer(), Model()
            loads.append((label, args))
            references.extend([weakref.ref(tokenizer), weakref.ref(model)])
            return tokenizer, model
        return load

    loaders = {}
    for label in ("nllb", "madlad", "marian"):
        loader = make_loader(label)
        loaders[label] = loader
        monkeypatch.setattr(engine, "_load_" + label, loader)
    return SimpleNamespace(loads=loads, refs=references, loaders=loaders)


def test_only_one_cached_model_survives_backend_and_pair_switches(
    cached_loaders,
):
    engine.preload_backend("nllb")
    engine.preload_backend("nllb")
    assert len(cached_loaders.loads) == 1
    assert engine.backend_is_loaded("nllb")
    engine.preload_backend("madlad-3b")
    assert not engine.backend_is_loaded("nllb")
    assert engine.backend_is_loaded("madlad-3b")
    engine.preload_backend("opus-mt", "deu_Latn", "fra_Latn")
    engine.preload_backend("opus-mt", "eng_Latn", "deu_Latn")
    assert len(cached_loaders.loads) == 4
    assert not engine.backend_is_loaded("opus-mt", "deu_Latn", "fra_Latn")
    assert engine.backend_is_loaded("opus-mt", "eng_Latn", "deu_Latn")
    assert sum(loader.cache_info().currsize
               for loader in cached_loaders.loaders.values()) == 1
    engine.free_translation_vram()
    assert not engine.backend_is_loaded("opus-mt", "eng_Latn", "deu_Latn")
    assert all(ref() is None for ref in cached_loaders.refs)


def test_ocr_guard_is_reentrant_and_translation_reloads(cached_loaders):
    engine.preload_backend("nllb")
    with engine.translation_session():
        engine.free_translation_vram()
        assert not engine.backend_is_loaded("nllb")
        assert all(ref() is None for ref in cached_loaders.refs)
        assert engine.translate("one", "deu_Latn", "fra_Latn") == "ONE"
        assert engine.backend_is_loaded("nllb")
    assert len(cached_loaders.loads) == 2


def test_device_change_invalidates_ui_and_reloads(cached_loaders, offline):
    engine.preload_backend("nllb")
    offline.current = 0
    assert not engine.backend_is_loaded("nllb")
    engine.preload_backend("nllb")
    assert len(cached_loaders.loads) == 2
    assert engine.backend_is_loaded("nllb")
    assert offline.cleanups[-1] == 1


def test_load_oom_cleans_failed_model_and_does_not_retry(
    monkeypatch, offline,
):
    references, calls = [], []

    def load(*args):
        model = Model()
        references.append(weakref.ref(model))
        calls.append(args)
        raise offline.OutOfMemoryError("CUDA out of memory.")

    def cleanup():
        assert all(ref() is None for ref in references)

    offline.on_cleanup = cleanup
    monkeypatch.setattr(engine, "_load_nllb", load)
    with pytest.raises(RuntimeError, match="while loading") as caught:
        engine.preload_backend("nllb")
    assert len(calls) == 1
    assert caught.value.__context__ is None
    assert not engine.backend_is_loaded("nllb")
    assert all(ref() is None for ref in references)


def test_failed_load_never_marks_backend_loaded(monkeypatch):
    def fail(*args):
        raise RuntimeError("weights unavailable")

    monkeypatch.setattr(engine, "_load_marian", fail)
    with pytest.raises(RuntimeError, match="weights unavailable"):
        engine.preload_backend("opus-mt", "deu_Latn", "fra_Latn")
    assert not engine.backend_is_loaded("opus-mt", "deu_Latn", "fra_Latn")


def test_inference_serializes_language_mutation_and_eviction(monkeypatch):
    tokenizer = Tokenizer()
    entered, release, second_started, eviction_started = (
        threading.Event() for _ in range(4)
    )
    languages = []

    class SlowModel(Model):
        def generate(self, input_ids, **kwargs):
            languages.append(tokenizer.src_lang)
            if not entered.is_set():
                entered.set()
                assert release.wait(5)
                assert tokenizer.src_lang == "deu_Latn"
            return super().generate(input_ids, **kwargs)

    model = SlowModel()
    monkeypatch.setattr(engine, "_load_nllb", lambda *args: (tokenizer, model))

    def second():
        second_started.set()
        return engine.translate("two", "eng_Latn", "fra_Latn")

    def evict():
        eviction_started.set()
        engine.free_translation_vram()

    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(engine.translate, "one", "deu_Latn", "fra_Latn")
        try:
            assert entered.wait(5)
            later = pool.submit(second)
            eviction = pool.submit(evict)
            assert second_started.wait(5)
            assert eviction_started.wait(5)
            assert not later.done()
            assert not eviction.done()
        finally:
            release.set()
        assert first.result(5) == "ONE"
        assert later.result(5) == "TWO"
        eviction.result(5)
    assert languages == ["deu_Latn", "eng_Latn"]


@pytest.fixture
def fake_service(monkeypatch):
    state = SimpleNamespace(running={"unrelated:latest"}, calls=[])

    def ps():
        return {"models": [{"model": name} for name in state.running]}

    def chat(**request):
        state.calls.append(("chat", request))
        state.running.add(ollama._canonical_model(request["model"]))
        return {
            "done_reason": "stop",
            "message": {"content": request["messages"][-1]["content"]},
        }

    def generate(**request):
        state.calls.append(("unload", request))
        assert request["keep_alive"] == 0
        state.running.discard(ollama._canonical_model(request["model"]))

    state.client = SimpleNamespace(ps=ps, chat=chat, generate=generate)
    monkeypatch.setitem(sys.modules, "ollama", state.client)
    return state


def test_hf_ollama_transitions_unload_only_owned_model(
    cached_loaders, fake_service,
):
    engine.preload_backend("nllb")
    assert engine.translate(
        "one", "deu_Latn", "fra_Latn", backend="ollama",
        ollama_model="translator",
    ) == "one"
    assert all(ref() is None for ref in cached_loaders.refs)
    assert not engine.backend_is_loaded("nllb")
    assert engine.backend_is_loaded("ollama", ollama_model="translator")
    engine.preload_backend("madlad-3b")
    assert fake_service.running == {"unrelated:latest"}
    assert fake_service.calls[-1] == (
        "unload", {"model": "translator", "keep_alive": 0},
    )
    assert not engine.backend_is_loaded("ollama", ollama_model="translator")


def test_preexisting_ollama_model_is_borrowed_not_unloaded(
    fake_service, cached_loaders,
):
    fake_service.running.add("translator:latest")
    engine.preload_backend("ollama", ollama_model="translator")
    engine.preload_backend("nllb")
    engine.free_translation_vram()
    assert fake_service.running == {"translator:latest", "unrelated:latest"}
    assert all(kind == "chat" for kind, _ in fake_service.calls)


def test_ollama_model_switch_and_ocr_release_are_scoped(fake_service):
    engine.preload_backend("ollama", ollama_model="first")
    engine.preload_backend("ollama", ollama_model="second")
    assert fake_service.running == {"second:latest", "unrelated:latest"}
    with engine.translation_session():
        engine.free_translation_vram()
    assert fake_service.running == {"unrelated:latest"}
    assert [call[1]["model"] for call in fake_service.calls
            if call[0] == "unload"] == ["first", "second"]


def test_ollama_ui_state_checks_server_expiry(fake_service):
    engine.preload_backend("ollama", ollama_model="translator")
    assert engine.backend_is_loaded("ollama", ollama_model="translator")
    fake_service.running.remove("translator:latest")
    assert not engine.backend_is_loaded("ollama", ollama_model="translator")
    engine.free_translation_vram()
    assert all(kind != "unload" for kind, _ in fake_service.calls)


def test_legacy_ollama_client_has_no_unproven_ownership(
    monkeypatch, fake_service,
):
    monkeypatch.delattr(fake_service.client, "ps")
    engine.preload_backend("ollama", ollama_model="translator")
    assert not engine.backend_is_loaded("ollama", ollama_model="translator")
    engine.free_translation_vram()
    assert all(kind != "unload" for kind, _ in fake_service.calls)


def test_unverified_ollama_release_stops_handoff(
    monkeypatch, fake_service, cached_loaders,
):
    engine.preload_backend("ollama", ollama_model="translator")
    monkeypatch.delattr(fake_service.client, "ps")
    with pytest.raises(RuntimeError, match="Cannot verify"):
        engine.preload_backend("nllb")
    assert not cached_loaders.loads
    assert all(kind != "unload" for kind, _ in fake_service.calls)


def test_failed_ollama_unload_does_not_claim_memory_was_freed(
    monkeypatch, fake_service, cached_loaders,
):
    engine.preload_backend("ollama", ollama_model="translator")
    monkeypatch.setattr(fake_service.client, "generate", lambda **kwargs: {})
    with pytest.raises(RuntimeError, match="has not released"):
        engine.preload_backend("nllb")
    assert not cached_loaders.loads
    assert engine.backend_is_loaded("ollama", ollama_model="translator")
