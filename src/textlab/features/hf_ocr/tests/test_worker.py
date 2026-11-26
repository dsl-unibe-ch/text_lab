"""The worker: what it imports, the model checks and reading a page.

Transformers, torch and the Hub are faked: these tests check the worker's
logic, not a model.
"""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image

from textlab.common import gpu_manager
from textlab.features.hf_ocr import service, worker

PACKAGE = Path(__file__).resolve().parents[3]

#: Modules the worker loads from ``textlab``, all of which must run in the
#: Hugging Face OCR environment, with only the standard library.
WORKER_TEXTLAB = {
    "textlab.common.jobs": "common/jobs.py",
    "textlab.common.progress": "common/progress.py",
    "textlab.common.storage": "common/storage.py",
    "textlab.common.config": "common/config.py",
}


def imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            yield node.module


def test_the_worker_imports_only_what_its_environment_has():
    files = {
        "features/hf_ocr/worker.py": {
            "huggingface_hub",
            "PIL",
            "torch",
            "transformers",
        },
        **{file: set() for file in WORKER_TEXTLAB.values()},
    }
    for file, third_party in files.items():
        for module in imports(PACKAGE / file):
            top = module.split(".")[0]
            if top == "textlab":
                assert module in WORKER_TEXTLAB or module == "textlab", (
                    file,
                    module,
                )
            elif top not in sys.stdlib_module_names:
                assert top in third_party, (file, module)


def test_the_gpu_manager_knows_the_worker():
    assert gpu_manager._WORKER_SCRIPTS[service.WORKER] == gpu_manager.HF_OCR


def test_a_model_too_large_for_the_gpu_is_refused():
    gb = 1024**3
    worker.check_size("org/m", 4_000_000_000, 20 * gb)  # 8 GB fits
    worker.check_size("org/m", None, 20 * gb)  # size unknown: try it
    worker.check_size("org/m", 4_000_000_000, None)  # no GPU: no check
    with pytest.raises(worker.OcrModelError, match="7.0 billion"):
        worker.check_size("org/m", 7_000_000_000, 12 * gb)


@pytest.fixture
def hub(monkeypatch):
    """A fake Hub whose answer the test sets."""

    class RepositoryNotFoundError(Exception):
        pass

    class GatedRepoError(RepositoryNotFoundError):
        pass

    state = {"answer": None}

    class HfApi:
        def model_info(self, repo_id, token):
            assert token is False  # no token is ever sent
            answer = state["answer"]
            if isinstance(answer, Exception):
                raise answer
            return answer

    hub_module = ModuleType("huggingface_hub")
    hub_module.HfApi = HfApi
    errors = ModuleType("huggingface_hub.errors")
    errors.GatedRepoError = GatedRepoError
    errors.RepositoryNotFoundError = RepositoryNotFoundError
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_module)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    state["errors"] = errors
    return state


def info(gated=False, parameters=None, files=("config.json",)):
    safetensors = SimpleNamespace(total=parameters) if parameters else None
    siblings = [SimpleNamespace(rfilename=name) for name in files]
    return SimpleNamespace(
        gated=gated, safetensors=safetensors, sha="c0ffee", siblings=siblings
    )


def test_a_public_model_passes_and_gives_its_revision(hub):
    hub["answer"] = info(parameters=1_000_000)
    assert worker.check_model("org/m", 10 * 1024**3) == "c0ffee"


@pytest.mark.parametrize("flag", [True, "auto", "manual"])
def test_a_gated_model_is_refused(hub, flag):
    hub["answer"] = info(gated=flag)
    with pytest.raises(worker.OcrModelError, match="gated"):
        worker.check_model("org/m", None)
    hub["answer"] = hub["errors"].GatedRepoError("gated")
    with pytest.raises(worker.OcrModelError, match="gated"):
        worker.check_model("org/m", None)


def test_an_adapter_without_its_model_is_refused(hub):
    hub["answer"] = info(files=("adapter_config.json", "adapter.safetensors"))
    with pytest.raises(worker.OcrModelError, match="adapter"):
        worker.check_model("org/lora", None)
    # A full model that also ships its adapter is fine.
    hub["answer"] = info(files=("adapter_config.json", "config.json"))
    assert worker.check_model("org/merged", None) == "c0ffee"


def test_a_missing_model_is_refused(hub):
    hub["answer"] = hub["errors"].RepositoryNotFoundError("404")
    with pytest.raises(worker.OcrModelError, match="no public model"):
        worker.check_model("org/missing", None)


class FakeTensor:
    """Just enough of a tensor: a shape and slicing of the second axis."""

    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def __getitem__(self, index):
        _rows, columns = index
        return FakeTensor([row[columns] for row in self.rows])


class FakeInputs(dict):
    """The processor's output: moved and cast like a ``BatchFeature``."""

    def to(self, target):
        self.setdefault("moved_to", []).append(target)
        return self


class FakeProcessor:
    def __init__(self, chat_template):
        self.chat_template = chat_template
        self.calls = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.calls.append(("template", messages))
        return "<chat>"

    def __call__(self, **kwargs):
        self.calls.append(("process", kwargs))
        return FakeInputs(input_ids=FakeTensor([[1, 2, 3]]))

    def batch_decode(self, output, skip_special_tokens):
        return [" ".join(map(str, output.rows[0])) + "  "]


class FakeModel:
    device = "cuda"
    dtype = "bfloat16"

    def __init__(self, encoder_decoder=False):
        self.config = SimpleNamespace(is_encoder_decoder=encoder_decoder)
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return FakeTensor([[1, 2, 3, 7, 8]])


FAKE_TORCH = SimpleNamespace(inference_mode=lambda: _NoContext())


class _NoContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_a_chat_model_gets_the_page_and_prompt_as_one_message():
    processor, model = FakeProcessor("{{ template }}"), FakeModel()
    image = Image.new("RGB", (10, 10))
    text = worker.read_page(model, processor, image, "Read.", 99, FAKE_TORCH)
    # Only the generated tokens are returned, stripped.
    assert text == "7 8"
    (_, messages), (_, processed) = processor.calls
    assert messages == [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": "Read."}],
        }
    ]
    assert processed["text"] == ["<chat>"] and processed["images"] == [image]
    assert model.generate_kwargs["max_new_tokens"] == 99
    assert model.generate_kwargs["do_sample"] is False
    assert model.generate_kwargs["moved_to"] == ["cuda", "bfloat16"]


def test_a_model_without_chat_template_gets_image_and_prompt_directly():
    processor = FakeProcessor(None)
    image = Image.new("RGB", (10, 10))
    worker.read_page(FakeModel(), processor, image, "", 50, FAKE_TORCH)
    ((kind, processed),) = processor.calls
    assert kind == "process" and processed["images"] is image
    assert "text" not in processed  # no prompt given, none sent


def test_an_encoder_decoder_output_is_not_cut():
    processor, model = FakeProcessor(None), FakeModel(encoder_decoder=True)
    image = Image.new("RGB", (10, 10))
    text = worker.read_page(model, processor, image, "", 50, FAKE_TORCH)
    assert text == "1 2 3 7 8"


def test_large_pages_are_scaled_down(tmp_path):
    path = tmp_path / "page.png"
    Image.new("L", (worker.MAX_PAGE_SIDE * 2, 100)).save(path)
    page = worker.page_image(str(path))
    assert page.mode == "RGB"
    assert page.size == (worker.MAX_PAGE_SIDE, 50)
