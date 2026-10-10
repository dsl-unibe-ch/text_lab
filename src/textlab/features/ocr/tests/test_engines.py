"""The manual engine adapters, without their models.

EasyOCR, GLM-OCR and OlmOCR are replaced by fakes at the point where they
would run a model; PaddleOCR runs a stub worker that prints a result in the
real worker's format.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from textlab.common import gpu_manager
from textlab.common.progress import Progress
from textlab.features.ocr.engines import (
    base,
    easy_ocr,
    glm_ocr,
    olm_ocr,
    paddle_ocr,
    payloads,
)

ENGINES_DIR = Path(base.__file__).parent


def write_png(path, width=40, height=20):
    import cv2

    cv2.imwrite(str(path), np.full((height, width, 3), 255, np.uint8))
    return path


# --- payloads ------------------------------------------------------------


def test_numpy_values_become_json_types():
    value = {
        "box": np.array([[1, 2], [3, 4]]),
        "score": np.float32(0.5),
        "items": (np.int64(3), {"x"}),
    }
    result = payloads.make_json_serializable(value)
    assert result == {
        "box": [[1, 2], [3, 4]],
        "score": 0.5,
        "items": [3, ["x"]],
    }
    json.dumps(result)


def test_texts_are_found_at_any_depth():
    payload = {
        "res": {"rec_texts": ["  first ", "", "second"]},
        "pages": [{"text": "third"}, {"text": {"rec_texts": "fourth"}}],
    }
    assert payloads.extract_texts(payload) == [
        "first",
        "second",
        "third",
        "fourth",
    ]


def test_a_four_number_box_becomes_a_rectangle():
    points = payloads.polygon_points([0, 0, 10, 5])
    assert points.tolist() == [[0, 0], [10, 0], [10, 5], [0, 5]]
    assert payloads.polygon_points([1, 2]) is None


def test_polygons_are_found_in_flat_and_nested_form():
    payload = {
        "rec_polys": [[[0, 0], [4, 0], [4, 4], [0, 4]]],
        "boxes": [[0, 0, 2, 2]],
    }
    polygons = payloads.extract_polygons(payload)
    assert [p.shape for p in polygons] == [(4, 2), (4, 2)]


def test_a_paddle_prediction_is_compacted():
    pred = SimpleNamespace(
        json={
            "input_path": "page.png",
            "other": "dropped",
            "res": {"rec_texts": ["a", "b", "a"], "rec_scores": [0.9]},
        }
    )
    compact = payloads.compact_paddle_prediction(pred)
    assert compact == {
        "input_path": "page.png",
        "rec_texts": ["a", "b"],
        "rec_scores": [0.9],
    }


def test_the_rendered_result_image_is_preferred():
    pred = SimpleNamespace(img={"other": b"x", "ocr_res_img": b"result"})
    assert payloads.rendered_png(pred) == b"result"
    assert payloads.rendered_png(SimpleNamespace()) is None


def test_the_worker_imports_only_what_its_environment_has():
    """The worker and payloads run in the PaddleOCR environment."""
    allowed_textlab = {"textlab.features.ocr.engines.payloads"}
    for name, allowed_third_party in (
        ("paddle_ocr_worker.py", {"numpy", "paddle", "paddleocr"}),
        ("payloads.py", {"numpy", "cv2"}),
    ):
        tree = ast.parse((ENGINES_DIR / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                modules = [node.module]
            else:
                continue
            for module in modules:
                top = module.split(".")[0]
                if top == "textlab":
                    assert module in allowed_textlab, (name, module)
                elif top not in sys.stdlib_module_names:
                    assert top in allowed_third_party, (name, module)


def test_the_worker_and_the_adapter_agree_on_the_result_marker():
    tree = ast.parse(
        (ENGINES_DIR / "paddle_ocr_worker.py").read_text(encoding="utf-8")
    )
    markers = [
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and node.targets[0].id == "RESULT_MARKER"
    ]
    assert markers == [paddle_ocr.RESULT_MARKER]


# --- base ----------------------------------------------------------------


def test_an_image_is_its_own_page(tmp_path):
    image = write_png(tmp_path / "scan.png")
    assert base.page_images(image, tmp_path / "pages") == [image]


def test_html_tables_are_detected():
    assert base.contains_html_table("<table><tr><td>1</td></tr></table>")
    assert not base.contains_html_table("no table here")
    assert base.html_table("no table here") is None


def test_an_html_table_becomes_a_data_frame():
    pytest.importorskip("lxml")
    html = "<table><tr><th>a</th></tr><tr><td>1</td></tr></table>"
    assert base.html_table(html)["a"].tolist() == [1]


def test_engine_output_joins_page_texts():
    output = base.EngineOutput(
        pages=[base.PageText(1, "one"), base.PageText(2, "two")]
    )
    assert output.text == "one\n\ntwo"


# --- EasyOCR -------------------------------------------------------------


class FakeReader:
    def readtext(self, path, detail, paragraph):
        assert (detail, paragraph) == (1, True)
        box = [[0, 0], [10, 0], [10, 5], [0, 5]]
        return [(box, f"text of {Path(path).name}"), (box, "more")]


def test_easyocr_reads_each_page(tmp_path, monkeypatch):
    monkeypatch.setattr(easy_ocr, "get_reader", lambda language: FakeReader())
    images = [write_png(tmp_path / "a.png"), write_png(tmp_path / "b.png")]
    updates = []
    output = easy_ocr.EasyOcr().recognize_pages(
        images, base.EngineOptions(previews=True), updates.append
    )
    assert [page.text for page in output.pages] == [
        "text of a.png\nmore",
        "text of b.png\nmore",
    ]
    assert len(output.previews) == 2
    assert all(p.layout is not None for p in output.previews)
    assert updates[-1] == Progress("Running EasyOCR... page 2/2", 1.0)


def test_easyocr_readers_are_released_by_the_gpu_manager():
    easy_ocr._READERS["xx"] = object()
    release = gpu_manager._RELEASERS[gpu_manager.OCR]["Unloaded EasyOCR model"]
    release()
    assert easy_ocr._READERS == {}


# --- PaddleOCR -----------------------------------------------------------

STUB_WORKER = """
import argparse, json, os, sys
parser = argparse.ArgumentParser()
parser.add_argument("--lang")
parser.add_argument("images", nargs="+")
args = parser.parse_args()
if os.environ.get("STUB_FAIL"):
    print("crashed", file=sys.stderr)
    sys.exit(3)
if os.environ.get("STUB_SILENT"):
    sys.exit(0)
pages = [
    {"page": i, "image": image, "text": f"{args.lang} {i}", "raw": [],
     "rendered_png_b64": "cmVuZGVyZWQ="}
    for i, image in enumerate(args.images, start=1)
]
print("log line")
print("TEXTLAB_PADDLEOCR_RESULT_JSON=" + json.dumps({"pages": pages}))
"""


@pytest.fixture
def stub_worker(tmp_path):
    path = tmp_path / "stub_worker.py"
    path.write_text(STUB_WORKER, encoding="utf-8")
    return str(path)


def test_paddleocr_worker_result_is_read(stub_worker):
    pages = paddle_ocr.run_worker(
        [Path("p1.png"), Path("p2.png")],
        "german",
        backend_python=sys.executable,
        worker_path=stub_worker,
    )
    assert [page["text"] for page in pages] == ["german 1", "german 2"]


@pytest.mark.parametrize(
    "variable,message",
    [("STUB_FAIL", "failed"), ("STUB_SILENT", "did not return JSON")],
)
def test_paddleocr_worker_failures_are_reported(
    stub_worker, monkeypatch, variable, message
):
    monkeypatch.setenv(variable, "1")
    with pytest.raises(base.EngineError, match=message) as error:
        paddle_ocr.run_worker(
            [Path("p1.png")],
            "en",
            backend_python=sys.executable,
            worker_path=stub_worker,
        )
    assert "stdout:" in error.value.details


def test_paddleocr_adapter_uses_the_rendered_preview(tmp_path, monkeypatch):
    monkeypatch.setattr(
        paddle_ocr,
        "run_worker",
        lambda images, language: [
            {"text": "x", "raw": [], "rendered_png_b64": "cmVuZGVyZWQ="}
        ],
    )
    output = paddle_ocr.PaddleOcr().recognize_pages(
        [tmp_path / "p.png"], base.EngineOptions(), lambda update: None
    )
    assert output.previews == [base.Preview(b"rendered")]


def test_paddleocr_worker_runs_as_a_module_in_its_environment(monkeypatch):
    calls = []

    def fake_run(command, env):
        calls.append((command, env))
        return subprocess.CompletedProcess(
            command, 0, 'TEXTLAB_PADDLEOCR_RESULT_JSON={"pages": []}\n', ""
        )

    monkeypatch.delenv("PADDLE_BACKEND_PYTHON", raising=False)
    monkeypatch.setattr(paddle_ocr, "run_process", fake_run)
    paddle_ocr.run_worker([Path("p.png")], "en")
    command, env = calls[0]
    assert command[:3] == [
        "/opt/conda/envs/paddle_backend/bin/python",
        "-m",
        paddle_ocr.WORKER_MODULE,
    ]
    assert env["PATH"].startswith("/opt/conda/envs/paddle_backend/bin")


# --- OlmOCR --------------------------------------------------------------


def fake_olmocr(returncode=0, record=None):
    calls = []

    def run(command, env):
        calls.append(command)
        workspace = Path(command[3])
        if record is not None:
            results = workspace / "results"
            results.mkdir(parents=True)
            (results / "output_1.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )
        return subprocess.CompletedProcess(command, returncode, "out", "err")

    return run, calls


def test_olmocr_returns_its_record(tmp_path, monkeypatch):
    run, calls = fake_olmocr(record={"text": "# Title", "id": "x"})
    monkeypatch.setattr(olm_ocr, "run_process", run)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    output = olm_ocr.OlmOcr().recognize(
        pdf, tmp_path / "work", base.EngineOptions()
    )
    assert output.text == "# Title"
    assert json.loads(output.record) == {"text": "# Title", "id": "x"}
    assert calls[0][1:3] == ["-m", "olmocr.pipeline"]
    assert calls[0][calls[0].index("--pdfs") + 1] == str(pdf)


def test_olmocr_reads_an_image_as_a_pdf(tmp_path, monkeypatch):
    run, calls = fake_olmocr(record={"text": "t"})
    monkeypatch.setattr(olm_ocr, "run_process", run)
    image = write_png(tmp_path / "scan.png")
    olm_ocr.OlmOcr().recognize(image, tmp_path / "work", base.EngineOptions())
    pdf = Path(calls[0][calls[0].index("--pdfs") + 1])
    assert pdf.suffix == ".pdf"
    assert pdf.read_bytes().startswith(b"%PDF")


@pytest.mark.parametrize(
    "returncode,record,message",
    [(1, None, "Code: 1"), (0, None, "No .jsonl output")],
)
def test_olmocr_failures_are_reported(
    tmp_path, monkeypatch, returncode, record, message
):
    run, _calls = fake_olmocr(returncode, record)
    monkeypatch.setattr(olm_ocr, "run_process", run)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    with pytest.raises(base.EngineError, match=message) as error:
        olm_ocr.OlmOcr().recognize(
            pdf, tmp_path / "work", base.EngineOptions()
        )
    assert "stderr:\nerr" in error.value.details


# --- GLM-OCR -------------------------------------------------------------


class FakeOllama:
    def __init__(self, models, pull_error=None):
        self.models = models
        self.pulled = []
        self.pull_error = pull_error

    def list(self):
        return {"models": [{"name": name} for name in self.models]}

    def pull(self, model):
        if self.pull_error:
            raise self.pull_error
        self.pulled.append(model)


def test_glm_model_is_pulled_only_when_missing():
    updates = []
    present = FakeOllama(["glm-ocr"])
    glm_ocr.ensure_model(updates.append, "glm-ocr:latest", present)
    assert present.pulled == [] and updates == []

    missing = FakeOllama(["other:latest"])
    glm_ocr.ensure_model(updates.append, "glm-ocr:latest", missing)
    assert missing.pulled == ["glm-ocr:latest"]
    assert updates == [Progress("Pulling model 'glm-ocr:latest'...")]


def test_glm_pull_failure_is_reported():
    client = FakeOllama([], pull_error=RuntimeError("offline"))
    with pytest.raises(base.EngineError, match="offline"):
        glm_ocr.ensure_model(lambda update: None, "glm-ocr:latest", client)


def test_glm_pages_are_scaled_down(tmp_path):
    import cv2

    image = write_png(tmp_path / "wide.png", width=4096, height=100)
    png = glm_ocr.page_png(image)
    decoded = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (50, 2048)
    assert glm_ocr.page_png(tmp_path / "missing.png") is None


def test_glm_reads_each_page_with_the_chosen_mode(tmp_path, monkeypatch):
    prompts = []

    def chat(model, messages, options):
        prompts.append(messages[0]["content"])
        if len(prompts) == 2:
            raise RuntimeError("timeout")
        return {"message": {"content": "<table></table>"}}

    monkeypatch.setattr(glm_ocr.ollama, "chat", chat)
    images = [write_png(tmp_path / "a.png"), write_png(tmp_path / "b.png")]
    output = glm_ocr.GlmOcr().recognize_pages(
        images, base.EngineOptions(mode="Table Recognition"), lambda u: None
    )
    assert prompts == ["Table Recognition", "Table Recognition"]
    assert output.pages[0].text == "<table></table>"
    assert output.pages[1].text == "[Error processing page 2: timeout]"
    assert len(output.previews) == 2
