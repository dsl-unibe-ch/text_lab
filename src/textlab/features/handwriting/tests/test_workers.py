"""The two workers: what they import, and their requests and results."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from textlab.common import gpu_manager, jobs
from textlab.common.storage import Workspace
from textlab.features.handwriting import detect_worker, service, worker
from textlab.features.handwriting.models import HandwritingModel

PACKAGE = Path(__file__).resolve().parents[3]

#: Modules the line-detection worker loads from ``textlab``, all of which
#: must run in the kraken environment, with only the standard library.
DETECT_TEXTLAB = {
    "textlab.common.jobs": "common/jobs.py",
    "textlab.common.progress": "common/progress.py",
    "textlab.common.storage": "common/storage.py",
    "textlab.common.config": "common/config.py",
}

#: Fake kraken 7 modules: the segmenter finds one line on every page.
FAKE_KRAKEN = {
    "__init__.py": "",
    "configs.py": """
class SegmentationInferenceConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
""",
    "tasks.py": """
from types import SimpleNamespace


class SegmentationTaskModel:
    @classmethod
    def load_model(cls, path=None):
        return cls()

    def predict(self, im, config):
        line = SimpleNamespace(boundary=[(0, 0), (9, 0), (9, 4), (0, 4)])
        return SimpleNamespace(lines=[line])
""",
}

FAKE_TORCH = """
from types import SimpleNamespace

cuda = SimpleNamespace(is_available=lambda: False)
"""


def imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            yield node.module


def test_the_detection_worker_imports_only_what_its_environment_has():
    files = {
        "features/handwriting/detect_worker.py": {"kraken", "PIL", "torch"},
        **{file: set() for file in DETECT_TEXTLAB.values()},
    }
    for file, third_party in files.items():
        for module in imports(PACKAGE / file):
            top = module.split(".")[0]
            if top == "textlab":
                assert module in DETECT_TEXTLAB or module == "textlab", (
                    file,
                    module,
                )
            elif top not in sys.stdlib_module_names:
                assert top in third_party, (file, module)


def test_the_detection_worker_runs_with_the_jobs_protocol(
    tmp_path, monkeypatch
):
    fakes = tmp_path / "fakes"
    (fakes / "kraken").mkdir(parents=True)
    for name, source in FAKE_KRAKEN.items():
        (fakes / "kraken" / name).write_text(source)
    (fakes / "torch.py").write_text(FAKE_TORCH)
    monkeypatch.setenv("PYTHONPATH", str(fakes))
    workspace = Workspace(tmp_path / "workspace")
    monkeypatch.setattr(jobs, "get_workspace", lambda: workspace)
    image = tmp_path / "page.png"
    Image.new("RGB", (10, 10)).save(image)

    updates = []
    polygons = service.detect_lines(
        [image, image],
        on_progress=updates.append,
        backend_python=sys.executable,
    )
    assert polygons == [[[[0, 0], [9, 0], [9, 4], [0, 4]]]] * 2
    assert updates[-1].fraction == 1.0


def test_line_boundaries_keep_their_order_and_skip_missing_ones():
    lines = [
        SimpleNamespace(boundary=[(5, 6), (7, 8)]),
        SimpleNamespace(boundary=None),
        SimpleNamespace(baseline=[(0, 0)]),
        SimpleNamespace(boundary=[(1, 2), (3, 4)]),
    ]
    assert detect_worker.boundaries(lines) == [
        [[5.0, 6.0], [7.0, 8.0]],
        [[1.0, 2.0], [3.0, 4.0]],
    ]


def test_the_recognition_worker_reads_every_crop(tmp_path, monkeypatch):
    spec = HandwritingModel("k", "Label", "", "org/base", None, "Read.")
    crops = []
    for index in range(3):
        crops.append(tmp_path / f"{index}.png")
        Image.new("L", (20 + index, 10)).save(crops[-1])

    class Reader:
        def read(self, images, on_done):
            on_done(len(images))
            return [f"{image.mode} {image.width}" for image in images]

    loaded = []
    monkeypatch.setattr(
        worker,
        "load_recognizer",
        lambda model: loaded.append(model) or Reader(),
    )
    monkeypatch.setattr(worker, "CHUNK", 2)
    updates = []
    result = worker.handle(
        {"model": spec.to_dict(), "crops": [str(c) for c in crops]},
        updates.append,
    )
    assert result == {"texts": ["RGB 20", "RGB 21", "RGB 22"]}
    assert loaded == [spec]
    assert [u.fraction for u in updates] == [0.0, 2 / 3, 1.0]


def test_the_gpu_manager_knows_the_workers():
    assert gpu_manager.HANDWRITING == "handwriting"
    for module in (service.DETECT_WORKER, service.RECOGNIZE_WORKER):
        assert gpu_manager._WORKER_SCRIPTS[module] == gpu_manager.HANDWRITING
