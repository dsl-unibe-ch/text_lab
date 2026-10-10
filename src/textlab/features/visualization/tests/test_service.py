"""The Visualization service: previews, background runs and downloads.

The agent (``viz_agent.run_analysis``) is replaced by a fake that writes
chart files like the real tools do, so no model or MCP server is needed.
"""

import asyncio
import io
import json
import os
import time
import zipfile

import pytest

from textlab.common.storage import Workspace
from textlab.features.visualization import runs, service, viz_agent

FIGURE = json.dumps({"data": [{"type": "bar", "x": [1], "y": [2]}]})


def wait_for(run, timeout=10):
    deadline = time.monotonic() + timeout
    while run.running and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not run.running, "the run did not finish"
    return run


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    workspace = Workspace(tmp_path / "workspace")
    monkeypatch.setattr(runs, "get_workspace", lambda: workspace)
    monkeypatch.setattr(runs, "model_installed", lambda model: True)
    return workspace


@pytest.fixture
def agent(monkeypatch):
    """A fake agent that draws one interactive and one static chart."""
    calls = []

    async def run_analysis(messages, data_file_path, model_name, **kwargs):
        calls.append((messages, data_file_path, model_name, kwargs))
        kwargs["log_callback"]("info", "Supervisor planned 1 task(s).")
        folder = os.path.dirname(data_file_path)
        chart = os.path.join(folder, "bar.json")
        with open(chart, "w", encoding="utf-8") as file:
            file.write(FIGURE)
        image = os.path.join(folder, "hist.png")
        with open(image, "wb") as file:
            file.write(b"\x89PNG")
        return {
            "summary": "Two charts.",
            "plots": [
                {
                    "path": chart,
                    "code": "fig = 1",
                    "r_code": "p <- 1",
                    "tool_name": "plot_interactive_barchart",
                },
                {
                    "path": image,
                    "code": "plt.hist()",
                    "r_code": "",
                    "tool_name": "plot_static_histogram",
                },
                {"path": os.path.join(folder, "gone.png"), "code": ""},
            ],
            "stats": [
                {
                    "title": "t-test",
                    "result": "| p | 0.1 |",
                    "code": "print(1)",
                    "r_code": "",
                }
            ],
            "logs": [],
        }

    monkeypatch.setattr(viz_agent, "run_analysis", run_analysis)
    return calls


CSV = b"age,income,group\n30,1000,a\n40,2000,b\n"


def test_a_run_reads_the_charts_and_cleans_up(workspace, agent):
    request = service.AnalysisRequest(
        model="m", columns=("age", "unknown"), file_name="data.csv"
    )
    run = wait_for(service.start_analysis(request, data=CSV))

    assert run.status == service.DONE, run.error
    ((messages, data_path, model, kwargs),) = agent
    assert model == "m"
    assert messages[0]["content"].startswith(
        f"User Request: {service.DEFAULT_PROMPT}"
    )
    assert (
        "Focus ONLY on these columns chosen by the user: age"
        in (messages[0]["content"])
    )
    assert "unknown" not in messages[0]["content"]
    result = run.result
    assert result.run_id == run.run_id and run.run_id.startswith("ds-")
    assert [a.filename for a in result.artifacts] == ["bar.json", "hist.png"]
    assert result.artifacts[0].figure_json == FIGURE
    assert result.artifacts[1].figure_json is None
    assert ("info", "Starting Supervisor Agent...") in run.logs
    assert any("Could not find plot" in msg for _, msg in run.logs)
    # The upload and the charts are gone once the run ends.
    assert list(workspace.dir("visualization").iterdir()) == []


def test_a_run_on_a_kept_file_leaves_it(workspace, agent, tmp_path):
    path = tmp_path / "uploaded_data.csv"
    path.write_bytes(CSV)
    request = service.AnalysisRequest(model="m", prompt="Plot age")
    run = wait_for(
        service.start_analysis(request, data_path=str(path), run_prefix="x")
    )
    assert run.status == service.DONE
    assert run.run_id.startswith("x-")
    assert agent[0][0][0]["content"] == "User Request: Plot age"
    assert path.exists()


def test_a_slow_run_times_out(workspace, monkeypatch):
    async def slow(*args, **kwargs):
        await asyncio.sleep(5)

    monkeypatch.setattr(viz_agent, "run_analysis", slow)
    request = service.AnalysisRequest(model="m", file_name="data.csv")
    run = wait_for(service.start_analysis(request, data=CSV, timeout=0.2))
    assert run.status == service.TIMEOUT


def test_a_failed_model_download_is_reported(workspace, monkeypatch, agent):
    monkeypatch.setattr(runs, "model_installed", lambda model: False)

    def fail(model):
        raise RuntimeError("offline")

    monkeypatch.setattr(runs.ollama, "pull", fail)
    request = service.AnalysisRequest(model="m", file_name="data.csv")
    run = wait_for(service.start_analysis(request, data=CSV))
    assert run.status == service.ERROR
    assert run.error == "Failed to pull model 'm': offline"
    assert agent == []


def test_a_run_needs_exactly_one_table():
    request = service.AnalysisRequest(model="m")
    with pytest.raises(ValueError):
        service.start_analysis(request)
    with pytest.raises(ValueError):
        service.start_analysis(request, data=b"x", data_path="x.csv")


def test_a_cancelled_run_says_so(workspace, monkeypatch):
    async def cooperative(messages, path, model, cancel_event, **kwargs):
        while not cancel_event.is_set():
            await asyncio.sleep(0.02)
        return {"summary": "", "plots": [], "stats": [], "logs": []}

    monkeypatch.setattr(viz_agent, "run_analysis", cooperative)
    request = service.AnalysisRequest(model="m", file_name="data.csv")
    run = service.start_analysis(request, data=CSV)
    run.cancel()
    assert wait_for(run).status == service.CANCELLED


# --- previews ------------------------------------------------------------


@pytest.mark.parametrize(
    "name,data",
    [
        ("data.csv", CSV),
        ("data.tsv", CSV.replace(b",", b"\t")),
        ("data.json", b'{"age": 30}\n{"age": 40}\n'),
    ],
)
def test_previews_read_the_supported_formats(name, data):
    assert service.read_preview(name, data)["age"].tolist()[:2] == [30, 40]


def test_an_unreadable_preview_is_none():
    assert service.read_preview("data.xlsx", b"not excel") is None
    assert service.read_preview("data.txt", b"x") is None


def test_the_column_profile_describes_each_column():
    table = service.read_preview("data.csv", CSV)
    profile = service.column_profile(table).set_index("Column")
    assert profile.loc["age", "Range (min / mean / max)"] == "30 / 35 / 40"
    assert profile.loc["group", "Top Value"] in {"a", "b"}
    assert profile.loc["income", "Non-Null %"] == "100.0%"


# --- downloads -----------------------------------------------------------


def finished_result():
    return service.AnalysisResult(
        run_id="ds-1",
        summary="Two charts.",
        artifacts=[
            service.Artifact(
                "bar.json",
                FIGURE.encode(),
                "fig = 1",
                "p <- 1",
                "plot_interactive_barchart",
                FIGURE,
            ),
            service.Artifact("hist.png", b"\x89PNG", "plt.hist()"),
        ],
        stats=[
            {
                "title": "t-test",
                "result": "| p |",
                "code": "print(1)",
                "r_code": "t.test()",
            }
        ],
    )


@pytest.mark.parametrize("include_r_code", [False, True])
def test_the_zip_has_charts_code_and_report(include_r_code):
    request = service.AnalysisRequest(
        model="m", include_r_code=include_r_code, file_name="data.csv"
    )
    archive = zipfile.ZipFile(
        io.BytesIO(service.results_zip(finished_result(), request))
    )
    names = set(archive.namelist())
    assert {"bar.html", "bar.py", "hist.png", "hist.py", "report.html"} <= (
        names
    )
    assert ("bar.R" in names) is include_r_code
    report = archive.read("report.html").decode()
    assert "data.csv" in report and "ds-1" in report
    assert ("t.test()" in report) is include_r_code


def test_a_chat_payload_keeps_the_stored_keys():
    payload = finished_result().to_payload()
    assert payload["run_id"] == "ds-1"
    assert payload["artifacts"][0]["fig_json"] == FIGURE
    assert payload["artifacts"][1]["bytes"] == b"\x89PNG"


def test_a_table_is_saved_and_summarized(tmp_path):
    path, summary = service.prepare_data_file("data.csv", CSV, str(tmp_path))
    assert os.path.basename(path) == "uploaded_data.csv"
    assert "age" in summary
    assert service.is_table("Data.XLSX") and not service.is_table("a.pdf")
