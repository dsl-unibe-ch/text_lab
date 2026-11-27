"""The Visualize Data page, with the agents faked.

``service.analyze`` logs two steps, can be held at a gate (to test the
steps shown while it works, and cancelling) and ends as the test says;
the rest runs for real: reading and describing the table, the draft job it
waits in, reading the request, saving the report, the ZIP and the charts,
and drawing the result.
"""

import io
import json
import re
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from textlab.features.visualization import service
from textlab.ui.web.jobs import JobState
from textlab.ui.web.pages import visualize_data as page
from textlab.ui.web.tests.conftest import BASE, wait_for
from textlab.ui.web.tests.test_job_pages import htmx

MODELS = service.ModelOptions(
    ("small:8b", "big:30b"), "NVIDIA RTX 4090", False
)

CSV = (
    "id,age,income,sex,region,comment\n"
    f"1,30,1000,f,north,{'taxes are too high ' * 4}\n"
    f"2,40,2000,m,south,{'the club won the cup ' * 4}\n"
    f"3,50,6000,f,north,{'a new budget for us ' * 4}\n"
).encode()

COLUMNS = ("id", "age", "income", "sex", "region", "comment")

FIGURE = json.dumps({"data": [{"type": "bar", "x": [1], "y": [2]}]})

STEPS = [
    ("info", "Supervisor planned 2 task(s)."),
    ("warning", "Worker 'stats' is retrying.\nThe test had no groups."),
]


def finished_analysis():
    return service.AnalysisResult(
        run_id="viz-1",
        summary="Income **differs** between the groups.",
        artifacts=[
            service.Artifact(
                "bar.json",
                FIGURE.encode(),
                "fig = px.bar(df)",
                "",
                "plot_interactive_barchart",
                FIGURE,
            ),
            service.Artifact(
                "hist.png",
                b"\x89PNG fake",
                "plt.hist(df.age)",
                "hist(df$age)",
                "plot_static_histogram",
            ),
        ],
        stats=[
            {
                "title": "Welch's t-test",
                "result": "| t | p |\n|---|---|\n| 2.1 | 0.04 |",
                "code": "ttest_ind(a, b)",
                "r_code": "t.test(a, b)",
            }
        ],
    )


@pytest.fixture
def agents(monkeypatch):
    """Fake the agents; record what each analysis was asked."""
    calls = SimpleNamespace(
        runs=[], gate=None, status=service.DONE, result=None, error=None
    )

    def analyze(
        request, data_path, *, on_progress, on_log, cancel, run_prefix
    ):
        path = Path(data_path)
        calls.runs.append(
            SimpleNamespace(
                request=request,
                path=path,
                table=path.read_bytes() if path.is_file() else None,
            )
        )
        for level, message in STEPS:
            on_log(level, message)
        status, result = calls.status, calls.result or finished_analysis()
        if calls.gate is not None:
            while not calls.gate.wait(0.01):
                if cancel.is_set():
                    status = service.CANCELLED
                    break
        return SimpleNamespace(
            status=status,
            result=result if status == service.DONE else None,
            logs=list(STEPS),
            error=calls.error,
        )

    monkeypatch.setattr(service, "analyze", analyze)
    monkeypatch.setattr(service, "model_options", lambda: MODELS)
    monkeypatch.setattr(service, "server_ready", lambda attempts=1: True)
    return calls


def upload(client, name="survey.csv", data=CSV, **fields):
    return client.post(
        f"{BASE}/visualize-data/table",
        data=fields,
        files={"file": (name, data, "application/octet-stream")},
        headers=htmx(client),
    )


def table_id(response):
    assert response.status_code == 200, response.text[:500]
    return re.search(r'name="table" value="([0-9a-f]+)"', response.text)[1]


def start(client, table, **fields):
    return client.post(
        f"{BASE}/visualize-data",
        data={"table": table, **fields},
        headers=htmx(client),
    )


def finished(runner, response):
    assert response.status_code == 200, response.text[:500]
    job_id = re.search(r'id="job-([0-9a-f]+)"', response.text).group(1)
    job = runner.get(job_id)
    wait_for(lambda: not job.active)
    return job


def card(client, job):
    return client.get(f"{BASE}/jobs/{job.id}").text


def test_the_page_offers_the_table_the_request_and_the_models(client, agents):
    html = client.get(f"{BASE}/visualize-data").text
    assert 'hx-params="not file"' in html
    assert 'hx-params="file,table"' in html
    assert 'accept=".csv,.tsv,.xls,.xlsx,.json"' in html
    assert "up to the first 300,000 rows" in html
    assert 'id="viz-request" name="request"' in html
    assert '<div class="asks" id="examples" hidden></div>' in html
    assert "<option selected>small:8b</option><option>big:30b</option>" in html
    assert "NVIDIA RTX 4090" in html
    assert 'name="r_code"' in html and "Also write R code" in html
    assert "at most 10 minutes" in html
    assert "Analyze" in html
    assert "does not answer yet" not in html


def test_a_silent_model_server_is_announced(client, agents, monkeypatch):
    monkeypatch.setattr(service, "server_ready", lambda attempts=1: False)
    html = client.get(f"{BASE}/visualize-data").text
    assert "The language model server does not answer yet" in html


def test_a_table_is_described_with_its_columns(client, runner, agents):
    response = upload(client)
    key = table_id(response)
    html = response.text
    assert "survey.csv" in html and "3 rows · 6 columns" in html
    for column in COLUMNS:
        assert f'name="columns" value="{column}" checked' in html
    assert '<span data-check-count="columns">6 of 6</span>' in html
    assert 'data-check="all" data-check-name="columns"' in html
    assert '<td class="dataset__kind">number</td>' in html
    assert '<td class="dataset__kind">category</td>' in html
    assert '<td class="dataset__kind">text</td>' in html
    assert "1,000 to 6,000, mean 3,000" in html
    assert "Described from all its rows." in html
    assert "<td>taxes are too high taxes are too high" in html
    # The requests that name its columns, out of band.
    assert 'id="examples" hx-swap-oob="true"' in html
    assert 'data-fill="viz-request"' in html
    assert "Plot the distribution of age." in html
    assert runner.draft(key) is not None
    assert runner.jobs(page.PAGE) == []


def test_another_table_replaces_the_first(client, runner, agents):
    first = table_id(upload(client))
    second = table_id(upload(client, "b.csv", table=first))
    assert runner.draft(first) is None
    assert runner.draft(second) is not None


def test_only_the_newest_tables_are_kept(client, runner, agents):
    # A page opened again names no table, so its uploads replace none.
    keys = [
        table_id(upload(client, f"t{n}.csv"))
        for n in range(page.MAX_TABLES + 1)
    ]
    assert runner.draft(keys[0]) is None
    assert all(runner.draft(key) is not None for key in keys[1:])


def test_a_table_that_cannot_be_read_keeps_the_one_before(
    client, runner, agents
):
    first = table_id(upload(client))
    response = upload(client, "broken.xlsx", b"not excel", table=first)
    assert response.status_code == 422
    assert "could not be read as a table" in response.text
    assert runner.draft(first) is not None
    response = upload(client, "notes.txt", b"x")
    assert response.status_code == 422
    response = upload(client, "empty.csv", b"age,income\n")
    assert response.status_code == 422
    assert "holds no data" in response.text


def test_an_analysis_runs_and_shows_its_results(
    client, runner, agents, released
):
    key = table_id(upload(client))
    job = finished(
        runner,
        start(
            client,
            key,
            columns=["income", "age"],
            request="Compare income between the groups.",
            model="big:30b",
            r_code="1",
        ),
    )
    assert job.state is JobState.DONE, job.error
    (run,) = agents.runs
    assert run.request == service.AnalysisRequest(
        model="big:30b",
        prompt="Compare income between the groups.",
        columns=("age", "income"),
        include_r_code=True,
        file_name="survey.csv",
    )
    # The agents read the table in the job's own inputs, removed after.
    assert run.table == CSV
    assert run.path.parent == job.inputs
    assert not job.inputs.exists()
    assert released == ["llm:big:30b"]
    assert job.title == "survey.csv: Compare income between the groups."

    html = card(client, job)
    assert f'data-version="{job.version}"' in html
    assert "<dd>Compare income between the groups.</dd>" in html
    assert "<dd>age, income</dd>" in html
    assert "<dd>big:30b, with R code</dd>" in html
    assert "Income <strong>differs</strong> between the groups." in html
    assert '<details class="stat" open>' in html
    assert "Welch&#39;s t-test" in html and "<table>" in html
    assert "t.test(a, b)" in html
    assert f'data-chart="{BASE}/jobs/{job.id}/files/chart-1.json"' in html
    assert f'src="{BASE}/jobs/{job.id}/files/chart-2.png"' in html
    assert "fig = px.bar(df)" in html and "hist(df$age)" in html
    # The interactive chart has no R code, and says why.
    assert service.R_NOT_AVAILABLE in html
    assert "How the agents worked: 2 steps" in html
    assert "survey-report.html" in html and "survey-analysis.zip" in html
    assert "chart-1.json</span>" not in html

    figure = client.get(f"{BASE}/jobs/{job.id}/files/chart-1.json")
    assert figure.json() == json.loads(FIGURE)
    archive = client.get(f"{BASE}/jobs/{job.id}/files/survey-analysis.zip")
    names = zipfile.ZipFile(io.BytesIO(archive.content)).namelist()
    assert "report.html" in names and "bar.html" in names


def test_all_or_no_columns_mean_the_whole_table(client, runner, agents):
    key = table_id(upload(client))
    finished(runner, start(client, key, columns=list(COLUMNS)))
    finished(runner, start(client, key))
    assert [run.request.columns for run in agents.runs] == [(), ()]
    # Without a request, the agents explore the table.
    assert agents.runs[1].request.prompt == ""
    assert runner.jobs(page.PAGE)[1].title == "survey.csv: explore the table"


def test_the_same_table_can_be_asked_again(client, runner, agents):
    key = table_id(upload(client))
    finished(runner, start(client, key, request="Plot age."))
    finished(runner, start(client, key, request="Plot income."))
    assert [run.table for run in agents.runs] == [CSV, CSV]
    assert len(runner.jobs(page.PAGE)) == 2
    assert runner.draft(key) is not None


def test_the_last_model_is_offered_first(client, runner, agents):
    key = table_id(upload(client))
    finished(runner, start(client, key, model="big:30b"))
    html = client.get(f"{BASE}/visualize-data").text
    assert "<option>small:8b</option><option selected>big:30b</option>" in html


def test_the_steps_show_while_the_agents_work(client, runner, agents):
    agents.gate = threading.Event()
    key = table_id(upload(client))
    response = start(client, key)
    job_id = re.search(r'id="job-([0-9a-f]+)"', response.text).group(1)
    job = runner.get(job_id)
    wait_for(lambda: len(job.preview) == len(STEPS))
    html = card(client, job)
    assert "The agents at work: 2 steps" in html
    assert 'class="activity__tail"' in html
    assert "Supervisor planned 2 task(s)." in html
    assert "activity__step--warning" in html
    # The progress is the latest step's first line.
    assert job.progress.message == "Worker 'stats' is retrying."
    client.post(f"{BASE}/jobs/{job.id}/cancel", headers=htmx(client))
    wait_for(lambda: not job.active)
    assert job.state is JobState.CANCELLED
    assert job.downloads == []


@pytest.mark.parametrize(
    "status,error,message",
    [
        (service.ERROR, "Failed to pull model 'small:8b': offline", "offline"),
        (service.TIMEOUT, None, "took longer than 10 minutes"),
    ],
)
def test_a_failed_analysis_says_why(
    client, runner, agents, status, error, message
):
    agents.status, agents.error = status, error
    key = table_id(upload(client))
    job = finished(runner, start(client, key))
    assert job.state is JobState.FAILED
    assert message in job.error
    assert message in card(client, job)


def test_an_answer_without_charts_has_no_downloads(client, runner, agents):
    agents.result = service.AnalysisResult(
        "viz-2", "The table has no numbers to plot.", [], []
    )
    key = table_id(upload(client))
    job = finished(runner, start(client, key))
    assert job.state is JobState.DONE
    html = card(client, job)
    assert "The table has no numbers to plot." in html
    assert job.downloads == []
    assert "Charts" not in html and "Downloads" not in html


@pytest.mark.parametrize(
    "fields,message",
    [
        ({"model": "other:1b"}, "Choose one of the models offered"),
        ({"columns": ["age", "salary"]}, "Choose the columns from the table"),
    ],
)
def test_bad_choices_are_explained(client, agents, fields, message):
    key = table_id(upload(client))
    response = start(client, key, **fields)
    assert response.status_code == 422
    assert message in response.text
    assert agents.runs == []


def test_an_analysis_needs_a_table_and_the_server(client, agents, monkeypatch):
    response = start(client, "unknown")
    assert response.status_code == 422
    assert "Choose the table first" in response.text
    key = table_id(upload(client))
    monkeypatch.setattr(service, "server_ready", lambda attempts=1: False)
    response = start(client, key)
    assert response.status_code == 422
    assert "does not answer" in response.text
    assert agents.runs == []
