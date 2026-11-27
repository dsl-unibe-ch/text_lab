# Visualization

Charts and statistics of an uploaded table, made by LLM agents: a supervisor
plans the user's request and hands it to specialist agents that call
plotting and statistics tools. Every result comes with reproducible Python
code and, if asked, R code. Used by the Visualize Data page and by Chat for
data questions.

## Pipeline

`runs.start_analysis` starts a run in a background thread, which the
Streamlit page and Chat poll; `runs.analyze` runs it in the caller's
thread, for the web UI's jobs, and passes on each entry of the activity
log as it comes. The run:

1. **Gets the model**: pulls it to the Ollama server if it is missing,
   with its progress (it can be cancelled meanwhile).
2. **Saves the table** to a run folder in the workspace (Chat passes a file
   it keeps for the conversation, and the web UI one in the job's folder,
   instead) and reads its first rows to check it and the selected columns.
3. **Runs the agents** (`viz_agent.run_analysis`), at most 10 minutes:
   - The supervisor reads a summary of every column (`plot_data`) and
     plans: an instruction each for the interactive, static and statistics
     agents, or a direct reply.
   - One MCP server process (`python -m textlab.features.visualization.mcp_server`)
     is started for the run. It exposes the tools: interactive Plotly charts
     (`plot_interactive`), static Matplotlib/Seaborn charts and word clouds
     (`plot_static`) and statistical tests (`stats_analysis`), each with its
     R equivalent (`r_code`) when `TEXTLAB_R_CODE` is set. The agents run
     concurrently on it; tools write their charts next to the table.
     When no tool fits, the custom chart tools run code the model writes,
     in the sandbox (see below).
   - Before a tool runs, the agent checks the model's arguments against the
     tool's parameters; a wrong call gets a short error naming them. A
     worker with failed calls retries with a fresh request (the task, the
     calls that succeeded, the errors) instead of the growing chat, so
     requests stay well within the context window. A call repeated with the
     same arguments is skipped, so a result never appears twice.
   - The statistics agent interprets its results (also partial ones), and
     the supervisor writes the summary. A summary cut off by the context
     window is replaced by a short template summary.
4. **Reads the results** (`runs.read_artifacts`): every chart file, its code
   and its R code, before the run folder is removed.

`reports.py` builds the downloads: a self-contained HTML report and a ZIP
with every chart (interactive ones as standalone HTML), its code and the
report. `save_result` writes both into a folder an interface serves, with
each chart as a file of its own (`chart-<n>.json` or an image) for the
interface to show.

Before a run, `preview.preview_table` reads the first 2,000 rows of a
table file with the agents' own readers and describes each column as the
agents will treat it (`plot_data.column_kind`: numbers, dates, categories,
or free text averaging more than 50 characters), and
`preview.example_requests` suggests requests that name the table's
columns.

The heavy work is outside the app process: the models run in the Ollama
server and the tools in the MCP server process, so the run itself is a
thread, not a worker process. The agent and the MCP library are imported
only when a run starts.

## Model-written code

`generate_custom_plotly` and `generate_custom_static_plot` run Python the
model writes. Text in the table can steer the model, so that code is
untrusted (`sandbox.py`):

1. `check_code` parses it and rejects imports outside the data and
   plotting libraries, private attributes (`_x`), and file, process and
   introspection functions (`open`, `eval`, `getattr`, `read_*`, `to_csv`,
   ...). The model gets the reason and rewrites the code.
2. `run_in_sandbox` runs it in a new container made from the app's image
   (`apptainer exec --containall --no-home --net --network none`, started
   with only `PATH` in its environment, since Apptainer's `APPTAINER_BIND`
   would otherwise pass on the app container's bind mounts), seeing only a
   folder with a copy of the table and the read-only source tree.
   There `sandbox_runner` runs the code with reduced builtins and the time
   limit, and saves the chart. The app reads the chart back without
   following links. Starting the sandbox adds a few seconds per custom
   chart.

If the sandbox cannot start (no image, no `apptainer`), custom code is not
run and the model is told so.

## Public API

Interfaces use `service.py`:

| Name | Purpose |
|---|---|
| `preview_table(name, path)` | A table file's columns (`ColumnInfo`: type, filled, distinct, values) and first rows: a `TablePreview` |
| `example_requests(preview)` | Requests that name the table's columns |
| `read_preview(name, data)`, `column_profile(df)` | The same for an upload in memory, as data frames (Streamlit) |
| `model_options()`, `server_ready()` | The models for the session's GPU (`ModelOptions`), and whether the Ollama server answers |
| `AnalysisRequest` | Model, prompt, selected columns, R code, file name |
| `start_analysis(request, data=` or `data_path=)` | Start a run in the background; returns an `AnalysisRun` |
| `analyze(request, data_path, on_log=, cancel=)` | Run in the caller's thread; returns the finished `AnalysisRun` |
| `AnalysisRun` | `status`, `logs`, `result`, `error`, `cancel()` |
| `AnalysisResult` | Summary, charts (`Artifact`), statistics; `to_payload()` for a chat message |
| `save_result(result, request, folder, name=)` | Write the downloads and the charts into a folder: a `SavedAnalysis` with `SavedChart`s and the files |
| `results_zip(result, request)`, `build_html_report(...)` | The downloads, in memory |
| `prepare_data_file(name, data, folder)` | Save a table for repeated runs and summarize its columns (Chat) |
| `tool_label`, `DEFAULT_PROMPT`, `MAX_ROWS`, ... | Re-exported for interfaces |

```python
import time

from textlab.features.visualization import service

request = service.AnalysisRequest(
    model="qwen3:8b", prompt="Compare income by group", file_name="data.csv"
)
run = service.start_analysis(request, data=csv_bytes)
while run.running:
    time.sleep(1)
if run.status == service.DONE:
    archive = service.results_zip(run.result, request)
```

## Layout

```
visualization/
├── service.py          # the API above
├── models.py           # requests and results
├── runs.py             # background runs
├── preview.py          # preview and column profile
├── reports.py          # HTML report and ZIP
├── viz_agent.py        # supervisor and agents
├── viz_config.py       # prompts, tool lists per agent, limits
├── mcp_server.py       # the tools, served over MCP (stdio)
├── plot_data.py        # dataset and column summaries
├── plot_interactive.py # Plotly charts
├── plot_static.py      # Matplotlib/Seaborn charts, word clouds
├── stats_analysis.py   # statistical tests
├── r_code.py           # R equivalents of the tools' code
├── viz_utils.py        # data loading, file names, code formatting
├── sandbox.py          # checking and isolating model-written code
├── sandbox_runner.py   # runs that code inside the sandbox
└── tests/
```

The tools' docstrings in `mcp_server.py` are the descriptions the models
read, and the prompts in `viz_config.py` are model input too; both are kept
exactly as written (see the ruff exceptions in `pyproject.toml`).

## Files written

| What | Where | Removed |
|---|---|---|
| The uploaded table and the charts of a run | `visualization/run-*` in the job workspace | When the run ends |
| The web UI's table, and each analysis's copy with its charts | The draft job's and each job's folder (`web-jobs/`, see the web UI README) | The copies when their analysis ends; the table when another replaces it, or the session ends |
| The web UI's report, ZIP and charts | The job's `outputs/` (`save_result`) | When the job is removed, or the session ends |
| Chat's table and its charts | `chat/conversation-*` (see the Chat README) | When the chat is cleared, or the session ends |

## Configuration

| Setting | Used for |
|---|---|
| `TEXTLAB_AGENT_NUM_CTX` | Context window (tokens) for the agents' model calls: 16384 if unset, `0` for the model's default |
| `TEXTLAB_AGENT_TIMEOUT` | Seconds a single model call may take (300) |
| `TEXTLAB_R_CODE` | Set by the run for the MCP server when R code is requested |

## Tests

In `tests/`, without models or GPU (tests marked `container` need the
image and nested Apptainer, as on a compute node):

- `test_viz_agent.py`: the supervisor and agents with a scripted model and
  an in-memory MCP session: planning, argument checks, retries, column
  fixes, cancellation,
  summaries, the server start.
- `test_viz_plots.py`: every tool on sample data, its code and R code,
  statistics checked against SciPy.
- `test_sandbox.py`: the code check with ordinary and malicious code, the
  runner, reading results back (planted links), and, marked `container`,
  that the real sandbox has no home folder, storage or network.
- `test_service.py`: runs with a fake agent, in the background and in the
  caller's thread (results, cleanup, timeout, cancel, model downloads and
  their errors), previews and their column types, the example requests,
  the ZIP, the report and the saved result.
- `test_visualize_page.py`: the page imports only the service, and neither
  service loads the agent or the MCP library.

## Batch use (planned)

There is no batch command yet; a `textlab visualize` command would call
`runs.start_analysis` and wait for the run.
