"""The Visualization service: previews, background runs and downloads.

The agent (``viz_agent.run_analysis``) is replaced by a fake that writes
chart files like the real tools do, so no model or MCP server is needed.
"""

import asyncio
import io
import json
import os
import threading
import time
import zipfile

import ollama
import pandas as pd
import pytest

from textlab.common.progress import Progress
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

    def fail(model, stream):
        raise RuntimeError("offline")

    monkeypatch.setattr(ollama, "pull", fail)
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


def test_an_analysis_runs_in_the_callers_thread(workspace, agent, tmp_path):
    path = tmp_path / "data.csv"
    path.write_bytes(CSV)
    logged = []
    request = service.AnalysisRequest(model="m", prompt="Plot age")
    run = service.analyze(
        request,
        str(path),
        on_log=lambda level, message: logged.append((level, message)),
        run_prefix="web",
    )
    assert run.status == service.DONE, run.error
    assert run.run_id.startswith("web-")
    assert logged == run.logs
    assert ("info", "Supervisor planned 1 task(s).") in logged
    assert [a.filename for a in run.result.artifacts] == [
        "bar.json",
        "hist.png",
    ]
    # The table is the caller's: it stays.
    assert path.exists()


def test_an_analysis_stops_when_its_event_is_set(
    workspace, monkeypatch, tmp_path
):
    async def cooperative(messages, path, model, cancel_event, **kwargs):
        while not cancel_event.is_set():
            await asyncio.sleep(0.02)
        return {"summary": "", "plots": [], "stats": [], "logs": []}

    monkeypatch.setattr(viz_agent, "run_analysis", cooperative)
    path = tmp_path / "data.csv"
    path.write_bytes(CSV)
    cancel = threading.Event()
    threading.Timer(0.1, cancel.set).start()
    run = service.analyze(
        service.AnalysisRequest(model="m"), str(path), cancel=cancel
    )
    assert run.status == service.CANCELLED


def test_a_missing_model_is_downloaded_with_progress(
    workspace, monkeypatch, agent, tmp_path
):
    monkeypatch.setattr(runs, "model_installed", lambda model: False)

    def pull(model, stream):
        yield {"status": "pulling", "total": 4, "completed": 2}

    monkeypatch.setattr(ollama, "pull", pull)
    path = tmp_path / "data.csv"
    path.write_bytes(CSV)
    updates = []
    run = service.analyze(
        service.AnalysisRequest(model="m"),
        str(path),
        on_progress=updates.append,
    )
    assert run.status == service.DONE
    assert updates == [
        Progress("Downloading m"),
        Progress("Downloading m", 0.5),
    ]
    assert ("info", "Pulling model 'm'...") in run.logs


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


def test_the_preview_drops_empty_unnamed_columns():
    table = service.read_preview("data.csv", b"age,group,\n30,a,\n40,b,\n")
    assert list(table.columns) == ["age", "group"]


def test_an_unreadable_preview_is_none():
    assert service.read_preview("data.xlsx", b"not excel") is None
    assert service.read_preview("data.txt", b"x") is None


def test_the_column_profile_describes_each_column():
    table = service.read_preview("data.csv", CSV)
    profile = service.column_profile(table).set_index("Column")
    assert profile.loc["age", "Range (min / mean / max)"] == "30 / 35 / 40"
    assert profile.loc["group", "Top Value"] in {"a", "b"}
    assert profile.loc["income", "Non-Null %"] == "100.0%"


# --- table previews ------------------------------------------------------

TABLE = (
    "id,age,income,sex,region,comment,\n"
    f"1,30,1000,f,north,{'a' * 70},\n"
    f"2,40,2000,m,south,{'b ' * 40},\n"
    f"3,,6000,f,north,{'c' * 70},\n"
)


def table_file(tmp_path, name, data):
    path = tmp_path / name
    path.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)
    return path


def test_a_preview_describes_the_columns_as_the_agents_see_them(tmp_path):
    preview = service.preview_table(
        "data.csv", table_file(tmp_path, "upload.csv", TABLE)
    )
    columns = {column.name: column for column in preview.columns}
    # The empty column a trailing comma makes is left out, as the agents
    # leave it out.
    assert preview.names == ("id", "age", "income", "sex", "region", "comment")
    assert columns["age"].kind == "number"
    assert columns["age"].filled == pytest.approx(2 / 3)
    assert columns["age"].values == "30 to 40, mean 35"
    assert columns["income"].values == "1,000 to 6,000, mean 3,000"
    assert (columns["sex"].kind, columns["sex"].distinct) == ("category", 2)
    assert columns["sex"].values == "f, m"
    assert columns["comment"].kind == "text"
    assert columns["comment"].values == "73 characters on average"
    assert preview.rows[2][:3] == ("3", "", "6000")
    assert preview.rows[2][5].endswith("...") and len(preview.rows[2][5]) == 60
    assert (preview.rows_read, preview.complete) == (3, True)


def test_a_preview_reads_what_the_agents_read(tmp_path):
    latin = table_file(
        tmp_path, "a.csv", "name,city\nZoë,Zürich\n".encode("latin-1")
    )
    assert service.preview_table("a.csv", latin).rows == (("Zoë", "Zürich"),)
    excel = tmp_path / "b.xlsx"
    pd.DataFrame(
        {"age": [30, 40], "day": pd.to_datetime(["2024-01-02", "2024-03-04"])}
    ).to_excel(excel, index=False)
    preview = service.preview_table("b.xlsx", excel)
    assert [column.kind for column in preview.columns] == ["number", "date"]
    assert preview.columns[1].values == "2024-01-02 to 2024-03-04"
    long = table_file(tmp_path, "c.csv", "x\n" + "1\n" * 5)
    preview = service.preview_table("c.csv", long, nrows=3)
    assert (preview.rows_read, preview.complete) == (3, False)


@pytest.mark.parametrize(
    "name,data,message",
    [
        ("data.xlsx", b"not excel", "could not be read as a table"),
        ("data.csv", b"age,income\n", "holds no data"),
    ],
)
def test_a_preview_says_what_is_wrong(tmp_path, name, data, message):
    with pytest.raises(ValueError, match=message):
        service.preview_table(name, table_file(tmp_path, name, data))


def test_examples_name_the_columns(tmp_path):
    preview = service.preview_table(
        "data.csv", table_file(tmp_path, "data.csv", TABLE)
    )
    # The identifier and the number with only two values are left out.
    assert service.example_requests(preview) == (
        "Plot the distribution of income.",
        "Is income different between the groups of sex? Run a test and "
        "show box plots.",
        "Is sex associated with region?",
        "Make a word cloud of comment.",
    )
    assert len(service.example_requests(preview, limit=2)) == 2


def test_no_examples_without_numbers_groups_or_texts(tmp_path):
    preview = service.preview_table(
        "data.csv",
        table_file(tmp_path, "data.csv", "id,code\n1,a\n2,b\n3,c\n"),
    )
    assert service.example_requests(preview) == ()


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


def test_a_saved_analysis_has_its_downloads_and_charts(tmp_path):
    request = service.AnalysisRequest(
        model="m", include_r_code=True, file_name="data.csv"
    )
    saved = service.save_result(
        finished_result(), request, tmp_path, name="data"
    )
    files = {item.name: item for item in saved.files}
    assert [name for name, item in files.items() if item.listed] == [
        "data-report.html",
        "data-analysis.zip",
    ]
    assert files["chart-1.json"].media_type == "application/json"
    assert files["chart-2.png"].media_type == "image/png"
    assert (tmp_path / "chart-1.json").read_text() == FIGURE
    assert zipfile.is_zipfile(tmp_path / "data-analysis.zip")
    bar, hist = saved.charts
    assert bar.label == service.tool_label("plot_interactive_barchart")
    assert (bar.file, bar.interactive, bar.r_code) == (
        "chart-1.json",
        True,
        "p <- 1",
    )
    assert (hist.file, hist.interactive, hist.label) == (
        "chart-2.png",
        False,
        "Chart 2",
    )
    assert (
        saved.summary == "Two charts." and saved.stats[0]["title"] == "t-test"
    )


def test_an_answer_without_charts_or_statistics_has_no_downloads(tmp_path):
    result = service.AnalysisResult("ds-2", "No numbers here.", [], [])
    saved = service.save_result(
        result, service.AnalysisRequest(model="m"), tmp_path
    )
    assert saved.files == () and saved.charts == ()
    assert list(tmp_path.iterdir()) == []
