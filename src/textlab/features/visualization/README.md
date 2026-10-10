# Visualization

Charts and statistics of an uploaded table, made by LLM agents: a supervisor
plans the user's request and hands it to specialist agents that call
plotting and statistics tools. Every result comes with reproducible Python
code and, if asked, R code. Used by the Visualize Data page and by Chat for
data questions.

## Pipeline

`runs.start_analysis` starts a run in a background thread; the page polls
it. The run:

1. **Gets the model**: pulls it to the Ollama server if it is missing.
2. **Saves the table** to a run folder in the workspace (Chat passes a file
   it keeps for the conversation instead) and reads its first rows to check
   it and the selected columns.
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
   - Statistical results are interpreted by the model, and the supervisor
     writes the summary.
4. **Reads the results** (`runs.read_artifacts`): every chart file, its code
   and its R code, before the run folder is removed.

`reports.py` builds the downloads: a self-contained HTML report and a ZIP
with every chart (interactive ones as standalone HTML), its code and the
report.

The heavy work is outside the app process: the models run in the Ollama
server and the tools in the MCP server process, so the run itself is a
thread, not a worker process. The agent and the MCP library are imported
only when a run starts.

## Public API

Interfaces use `service.py`:

| Name | Purpose |
|---|---|
| `read_preview(name, data)`, `column_profile(df)` | First rows of an upload, and a summary of each column |
| `AnalysisRequest` | Model, prompt, selected columns, R code, file name |
| `start_analysis(request, data=` or `data_path=)` | Start a run; returns an `AnalysisRun` |
| `AnalysisRun` | `status`, `logs`, `result`, `error`, `cancel()` |
| `AnalysisResult` | Summary, charts (`Artifact`), statistics; `to_payload()` for a chat message |
| `results_zip(result, request)`, `build_html_report(...)` | The downloads |
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
└── tests/
```

The tools' docstrings in `mcp_server.py` are the descriptions the models
read, and the prompts in `viz_config.py` are model input too; both are kept
exactly as written (see the ruff exceptions in `pyproject.toml`).

## Files written

| What | Where | Removed |
|---|---|---|
| The uploaded table and the charts of a run | `visualization/run-*` in the job workspace | When the run ends |
| Chat's table and its charts | `chat/conversation-*` (see the Chat README) | When the chat is cleared, or the session ends |

## Configuration

| Setting | Used for |
|---|---|
| `TEXTLAB_AGENT_NUM_CTX` | Context window for the agents' model calls; the model's default if unset |
| `TEXTLAB_AGENT_TIMEOUT` | Seconds a single model call may take (300) |
| `TEXTLAB_R_CODE` | Set by the run for the MCP server when R code is requested |

## Tests

In `tests/`, without models or GPU:

- `test_viz_agent.py`: the supervisor and agents with a scripted model and
  an in-memory MCP session: planning, retries, column fixes, cancellation,
  summaries, the server start.
- `test_viz_plots.py`: every tool on sample data, its code and R code,
  statistics checked against SciPy.
- `test_service.py`: background runs with a fake agent (results, cleanup,
  timeout, cancel, model download errors), previews, the ZIP and report.
- `test_visualize_page.py`: the page imports only the service, and neither
  service loads the agent or the MCP library.

## Batch use (planned)

There is no batch command yet; a `textlab visualize` command would call
`runs.start_analysis` and wait for the run.
