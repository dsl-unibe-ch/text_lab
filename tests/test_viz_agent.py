"""Supervisor/worker control flow of the viz agent, with Ollama and MCP faked.

No model, GPU or MCP subprocess is used: ``chat_no_think`` is replaced by a
scripted fake and the MCP session by an in-memory stub (``fake_server``).
"""

import conftest_path  # noqa: F401

import asyncio
import contextlib
import json
from types import SimpleNamespace

import pytest
from mcp import types

from core.visualization import viz_agent

DATA_PATH = "/tmp/run/uploaded_data.csv"
SCHEMA = "Dataset: 3 rows x 2 columns\nNumeric columns (2): age, income"


def _reply(text="", calls=None):
    """Build an Ollama-style response, optionally with tool calls."""
    message = {"role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = [
            {"function": {"name": name, "arguments": args}} for name, args in calls
        ]
    return {"message": message}


class ScriptedChat:
    """Stand-in for ``chat_no_think`` returning queued responses in order.

    A queued exception instance is raised instead of returned, simulating an
    Ollama error such as an unparseable tool call (HTTP 500).
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, model, messages, tools=None, options=None, timeout=None,
                 json_schema=None):
        self.calls.append({
            "messages": list(messages), "tools": tools, "options": options,
            "timeout": timeout, "json_schema": json_schema,
        })
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeSession:
    """In-memory MCP session exposing every agent tool with a data_file_path arg."""

    def __init__(self, outputs):
        self.outputs = {name: list(values) for name, values in outputs.items()}
        self.calls = []
        self.timeouts = []

    async def list_tools(self):
        schema = {
            "type": "object",
            "properties": {
                "data_file_path": {"type": "string"},
                "column": {"type": "string"},
                "title": {"type": "string"},
            },
            "required": ["data_file_path", "column"],
        }
        names = [n for tools in viz_agent.AGENT_TOOLS.values() for n in tools]
        return SimpleNamespace(tools=[
            SimpleNamespace(name=n, description=n, inputSchema=schema) for n in names
        ])

    async def call_tool(self, name, arguments, read_timeout_seconds=None):
        self.calls.append((name, dict(arguments)))
        self.timeouts.append(read_timeout_seconds)
        text = self.outputs[name].pop(0)
        return SimpleNamespace(
            content=[types.TextContent(type="text", text=text)], isError=False
        )


@pytest.fixture(autouse=True)
def fake_server(monkeypatch):
    """Replace the MCP server start-up; records how often a server was started."""
    starts = []

    @contextlib.asynccontextmanager
    async def fake_session(mcp_server_script):
        starts.append(mcp_server_script)
        yield FakeSession({})

    monkeypatch.setattr(viz_agent, "_mcp_session", fake_session)
    return starts


@pytest.fixture
def logs():
    return []


def _run_loop(role, session, logs, instruction="Plot the age column"):
    plots, stats = [], []
    report = asyncio.run(viz_agent._run_worker_agent(
        session=session,
        agent_role=role,
        task_instruction=instruction,
        data_file_path=DATA_PATH,
        schema=SCHEMA,
        model_name="fake",
        global_plots=plots,
        global_stats=stats,
        log=lambda level, msg: logs.append((level, msg)),
        cancel_event=None,
    ))
    return report, plots, stats


def test_tool_schemas_hide_data_file_path():
    schema = {
        "type": "object",
        "properties": {"data_file_path": {}, "column": {}},
        "required": ["data_file_path", "column"],
    }
    clean = viz_agent._strip_injected_args(schema)
    assert clean["properties"] == {"column": {}}
    assert clean["required"] == ["column"]
    assert "data_file_path" in schema["properties"]  # original left untouched


def test_plot_worker_stops_after_first_successful_round(monkeypatch, logs):
    chat = ScriptedChat([
        _reply(calls=[("plot_interactive_histogram", {"column": "age", "title": "Age"})]),
    ])
    monkeypatch.setattr(viz_agent, "chat_no_think", chat)
    session = FakeSession({"plot_interactive_histogram": ["/tmp/run/plots/h.json|||code"]})

    report, plots, _ = _run_loop("interactive", session, logs)

    assert len(chat.calls) == 1
    assert report["completed"] is True
    assert report["plots"] == ["Interactive Histogram: Age"]
    assert plots == [{
        "path": "/tmp/run/plots/h.json", "code": "code",
        "tool_name": "plot_interactive_histogram",
    }]
    assert session.calls[0][1]["data_file_path"] == DATA_PATH
    assert session.timeouts[0].total_seconds() == viz_agent.TOOL_CALL_TIMEOUT
    for tool in chat.calls[0]["tools"]:
        assert "data_file_path" not in tool["function"]["parameters"]["properties"]
    assert chat.calls[0]["options"]["temperature"] == viz_agent.AGENT_OPTIONS["temperature"]


def test_plot_error_is_logged_with_its_message_and_retried(monkeypatch, logs):
    chat = ScriptedChat([
        _reply(calls=[("generate_custom_plotly", {"python_code": "x", "title": "T"})]),
        _reply(calls=[("generate_custom_plotly", {"python_code": "fig=1", "title": "T"})]),
    ])
    monkeypatch.setattr(viz_agent, "chat_no_think", chat)
    session = FakeSession({"generate_custom_plotly": [
        "Error executing custom plotly code: name 'x' is not defined",
        "/tmp/run/plots/c.json|||code",
    ]})

    report, plots, _ = _run_loop("interactive", session, logs)

    assert report["completed"] is True and report["errors"] == []
    assert len(plots) == 1
    warnings = [msg for level, msg in logs if level == "warning"]
    assert any("name 'x' is not defined" in msg for msg in warnings)
    assert not any("stat tool" in msg for msg in warnings)


def test_unknown_tool_is_rejected_without_calling_mcp(monkeypatch, logs):
    chat = ScriptedChat([
        _reply(calls=[("run_correlation", {"x_column": "age"})]),
        _reply("I cannot do that."),
    ])
    monkeypatch.setattr(viz_agent, "chat_no_think", chat)
    session = FakeSession({})

    report, _, _ = _run_loop("interactive", session, logs)

    assert session.calls == []
    assert report["text"] == "I cannot do that."


def test_stats_worker_gets_one_tool_free_interpretation_turn(monkeypatch, logs):
    chat = ScriptedChat([
        _reply(calls=[("run_correlation", {"x_column": "age", "y_column": "income"})]),
        _reply("Age and income are strongly correlated."),
    ])
    monkeypatch.setattr(viz_agent, "chat_no_think", chat)
    session = FakeSession({"run_correlation": ["| r | 0.9 |\n```python\nprint(1)\n```"]})

    report, _, stats = _run_loop("stats", session, logs)

    assert len(chat.calls) == 2
    assert chat.calls[1]["tools"] is None
    assert report["completed"] is True
    assert report["text"] == "Age and income are strongly correlated."
    assert stats == [{"title": "Correlation Analysis", "result": "| r | 0.9 |", "code": "print(1)"}]


PARSE_ERROR = RuntimeError(
    "XML syntax error on line 16: element <parameter> closed by </function> (status code: 500)"
)


def test_unparseable_tool_call_stops_the_worker_without_retrying(monkeypatch, logs):
    chat = ScriptedChat([PARSE_ERROR])
    monkeypatch.setattr(viz_agent, "chat_no_think", chat)

    report, _, _ = _run_loop("interactive", FakeSession({}), logs)

    assert len(chat.calls) == 1
    assert chat.calls[0]["timeout"] == viz_agent.AGENT_REQUEST_TIMEOUT
    assert report["completed"] is False
    assert "try another model" in report["errors"][0]
    assert any(level == "error" and "try another model" in msg for level, msg in logs)


def test_supervisor_failure_gives_a_clear_summary(monkeypatch):
    chat = ScriptedChat([PARSE_ERROR])
    monkeypatch.setattr(viz_agent, "chat_no_think", chat)

    result = asyncio.run(viz_agent.run_analysis(
        [{"role": "user", "content": "Plot age"}], DATA_PATH, "fake", "server.py"
    ))

    assert len(chat.calls) == 1
    assert result["summary"].startswith("The analysis could not be started.")
    assert "try another model" in result["summary"]
    assert not any("Fatal error" in msg for _, msg in result["logs"])


def test_long_tool_output_is_clipped_for_the_model_only(monkeypatch, logs):
    long_table = "| row |\n" * 2000
    chat = ScriptedChat([
        _reply(calls=[("run_correlation", {"x_column": "age", "y_column": "income"})]),
        _reply("Interpretation."),
    ])
    monkeypatch.setattr(viz_agent, "chat_no_think", chat)
    session = FakeSession({"run_correlation": [long_table]})

    _, _, stats = _run_loop("stats", session, logs)

    tool_messages = [m for m in chat.calls[1]["messages"] if m.get("role") == "tool"]
    assert len(tool_messages[0]["content"]) <= viz_agent.MAX_TOOL_OUTPUT_CHARS + 20
    assert stats[0]["result"] == long_table.strip()


def test_misspelled_tool_name_is_mapped_to_the_allowed_tool(monkeypatch, logs):
    chat = ScriptedChat([
        _reply(calls=[("run_rank_target_correlations", {"target_col": "income"})]),
        _reply("Age is the strongest predictor."),
    ])
    monkeypatch.setattr(viz_agent, "chat_no_think", chat)
    session = FakeSession({"rank_target_correlations": ["| age | 0.9 |"]})

    report, _, stats = _run_loop("stats", session, logs)

    assert session.calls[0][0] == "rank_target_correlations"
    assert report["completed"] is True
    assert len(stats) == 1
    assert ("info", "Worker 'stats' called 'run_rank_target_correlations'; "
            "using 'rank_target_correlations'.") in logs


def test_repeated_identical_failure_stops_early_with_partial_results(monkeypatch, logs):
    good = ("plot_interactive_histogram", {"column": "age", "title": "Age"})
    bad = ("plot_interactive_boxplot", {"column": "nope", "title": "Box"})
    chat = ScriptedChat([_reply(calls=[good, bad]), _reply(calls=[bad])])
    monkeypatch.setattr(viz_agent, "chat_no_think", chat)
    session = FakeSession({
        "plot_interactive_histogram": ["/tmp/run/plots/h.json|||code"],
        "plot_interactive_boxplot": ["Error: Columns 'nope' not found."] * 2,
    })

    report, plots, _ = _run_loop("interactive", session, logs)

    assert len(chat.calls) == 2
    assert report["completed"] is False
    assert report["plots"] == ["Interactive Histogram: Age"]
    assert len(plots) == 1
    assert "not found" in report["errors"][0]
    assert ("error", "Worker 'interactive' repeated the same failing call; stopping.") in logs


def test_iteration_limit_returns_partial_results(monkeypatch, logs):
    good = ("plot_interactive_histogram", {"column": "age", "title": "Age"})
    rounds = viz_agent.WORKER_MAX_ITERATIONS
    bad_calls = [
        ("plot_interactive_boxplot", {"column": f"col{i}", "title": "Box"}) for i in range(rounds)
    ]
    responses = [_reply(calls=[good, bad_calls[0]])] + [
        _reply(calls=[call]) for call in bad_calls[1:]
    ]
    monkeypatch.setattr(viz_agent, "chat_no_think", ScriptedChat(responses))
    session = FakeSession({
        "plot_interactive_histogram": ["/tmp/run/plots/h.json|||code"],
        # A different error each round, so the repeat guard never triggers.
        "plot_interactive_boxplot": [f"Error: Column 'col{i}' not found." for i in range(rounds)],
    })

    report, plots, _ = _run_loop("interactive", session, logs)

    assert report["completed"] is False
    assert report["plots"] == ["Interactive Histogram: Age"]
    assert len(plots) == 1
    assert f"col{rounds - 1}" in report["errors"][0]
    assert ("warning", "Worker 'interactive' reached the iteration limit; "
            "returning partial results.") in logs


def _plan(interactive="", static="", stats="", reply=""):
    """Build a supervisor response carrying a JSON plan."""
    return _reply(json.dumps({
        "interactive": interactive, "static": static, "stats": stats, "reply": reply,
    }))


def _fake_worker(report_by_role, stats_by_role=None):
    async def fake(agent_role, task_instruction, global_stats, **_):
        for item in (stats_by_role or {}).get(agent_role, []):
            global_stats.append(item)
        report = viz_agent._new_report(agent_role, task_instruction)
        report.update(report_by_role[agent_role])
        return report
    return fake


def test_plot_only_run_uses_template_summary(monkeypatch):
    chat = ScriptedChat([
        _plan(interactive="Histogram of age", stats="None"),
    ])
    monkeypatch.setattr(viz_agent, "chat_no_think", chat)
    monkeypatch.setattr(viz_agent, "get_all_columns_summary_impl", lambda path: SCHEMA)
    monkeypatch.setattr(viz_agent, "_run_worker_agent", _fake_worker({
        "interactive": {"plots": ["Interactive Histogram: Age"], "completed": True},
    }))

    result = asyncio.run(viz_agent.run_analysis(
        [{"role": "user", "content": "Plot age"}], DATA_PATH, "fake", "server.py"
    ))

    assert len(chat.calls) == 1  # planning only, no summary call
    assert "- Interactive Histogram: Age" in result["summary"]


def test_stats_run_writes_summary_without_tools(monkeypatch):
    chat = ScriptedChat([
        _plan(stats="Correlate"),
        _reply("Age and income correlate (r = 0.9)."),
    ])
    monkeypatch.setattr(viz_agent, "chat_no_think", chat)
    monkeypatch.setattr(viz_agent, "get_all_columns_summary_impl", lambda path: SCHEMA)
    stats_item = {"title": "Correlation Analysis", "result": "| r | 0.9 |", "code": ""}
    monkeypatch.setattr(viz_agent, "_run_worker_agent", _fake_worker(
        {"stats": {"stats": ["| r | 0.9 |"], "completed": True}},
        {"stats": [stats_item]},
    ))

    result = asyncio.run(viz_agent.run_analysis(
        [{"role": "user", "content": "Is age related to income?"}], DATA_PATH, "fake", "server.py"
    ))

    assert len(chat.calls) == 2
    assert chat.calls[1]["tools"] is None
    assert "| r | 0.9 |" in chat.calls[1]["messages"][1]["content"]
    assert result["summary"] == "Age and income correlate (r = 0.9)."
    assert result["stats"] == [stats_item]


def test_one_plan_call_can_fill_every_specialist(monkeypatch):
    chat = ScriptedChat([
        _plan(
            interactive="Histogram of age",
            static="Histogram of age for print",
            stats="Correlate age and income",
        ),
    ])
    monkeypatch.setattr(viz_agent, "chat_no_think", chat)

    tasks, _ = asyncio.run(viz_agent._plan_tasks(
        [{"role": "user", "content": "Plots and stats"}], "fake", lambda *_: None
    ))

    assert tasks == [
        ("interactive", "Histogram of age"),
        ("static", "Histogram of age for print"),
        ("stats", "Correlate age and income"),
    ]
    assert chat.calls[0]["tools"] is None
    assert chat.calls[0]["json_schema"] == viz_agent.PLAN_SCHEMA


def test_supervisor_can_answer_directly(monkeypatch):
    chat = ScriptedChat([_plan(reply="The dataset has columns age and income.")])
    monkeypatch.setattr(viz_agent, "chat_no_think", chat)

    result = asyncio.run(viz_agent.run_analysis(
        [{"role": "user", "content": "Which columns are there?"}], DATA_PATH, "fake", "server.py"
    ))

    assert result["summary"] == "The dataset has columns age and income."
    assert result["plots"] == [] and result["stats"] == []


def test_invalid_plan_json_falls_back_to_the_reply_text(monkeypatch):
    chat = ScriptedChat([_reply("Sorry, I cannot plan this.")])
    monkeypatch.setattr(viz_agent, "chat_no_think", chat)
    logs = []

    tasks, reply = asyncio.run(viz_agent._plan_tasks(
        [{"role": "user", "content": "Plot age"}], "fake",
        lambda level, msg: logs.append((level, msg)),
    ))

    assert tasks == [] and reply == "Sorry, I cannot plan this."
    assert any("did not return a valid plan" in msg for _, msg in logs)


@pytest.mark.parametrize("wrap", [
    "{plan}<|eot|>",                      # leaked end-of-turn token
    "```json\n{plan}\n```",               # Markdown code fence
    "Here is the plan:\n{plan}\nDone.",   # surrounding prose
])
def test_plan_json_is_found_despite_surrounding_noise(monkeypatch, wrap):
    plan = json.dumps({"interactive": "Histogram of age", "static": "",
                       "stats": "", "reply": ""})
    chat = ScriptedChat([_reply(wrap.format(plan=plan))])
    monkeypatch.setattr(viz_agent, "chat_no_think", chat)

    tasks, _ = asyncio.run(viz_agent._plan_tasks(
        [{"role": "user", "content": "Plot age"}], "fake", lambda *_: None
    ))

    assert tasks == [("interactive", "Histogram of age")]


def test_leaked_control_tokens_are_removed_from_model_text():
    message = {"role": "assistant", "content": "Summary text.<|im_end|>\n<|endoftext|>"}
    assert viz_agent._model_text(message) == "Summary text."


def test_all_workers_share_one_mcp_server(monkeypatch, fake_server):
    chat = ScriptedChat([_plan(interactive="Histogram", static="Histogram for print")])
    monkeypatch.setattr(viz_agent, "chat_no_think", chat)
    monkeypatch.setattr(viz_agent, "get_all_columns_summary_impl", lambda path: SCHEMA)
    sessions = []

    async def fake_worker(session, agent_role, task_instruction, **_):
        sessions.append(session)
        report = viz_agent._new_report(agent_role, task_instruction)
        report.update({"plots": [f"{agent_role} plot"], "completed": True})
        return report

    monkeypatch.setattr(viz_agent, "_run_worker_agent", fake_worker)

    result = asyncio.run(viz_agent.run_analysis(
        [{"role": "user", "content": "Plot"}], DATA_PATH, "fake", "server.py"
    ))

    assert fake_server == ["server.py"]
    assert len(sessions) == 2 and sessions[0] is sessions[1]
    assert "- interactive plot" in result["summary"]


def test_server_start_failure_gives_failed_reports_not_a_crash(monkeypatch):
    @contextlib.asynccontextmanager
    async def broken_session(mcp_server_script):
        raise FileNotFoundError("mcp_server.py not found")
        yield  # pragma: no cover

    chat = ScriptedChat([_plan(interactive="Histogram")])
    monkeypatch.setattr(viz_agent, "chat_no_think", chat)
    monkeypatch.setattr(viz_agent, "get_all_columns_summary_impl", lambda path: SCHEMA)
    monkeypatch.setattr(viz_agent, "_mcp_session", broken_session)

    result = asyncio.run(viz_agent.run_analysis(
        [{"role": "user", "content": "Plot"}], DATA_PATH, "fake", "server.py"
    ))

    assert "The analysis tools could not be started." in result["summary"]
    assert any(level == "error" and "mcp_server.py not found" in msg
               for level, msg in result["logs"])


def test_activity_log_shows_tools_per_round(monkeypatch, logs):
    bad = ("plot_interactive_boxplot", {"column": "nope", "title": "Box"})
    good_hist = ("plot_interactive_histogram", {"column": "age", "title": "Age"})
    good_box = ("plot_interactive_boxplot", {"column": "age", "title": "Box"})
    chat = ScriptedChat([_reply(calls=[good_hist, bad]), _reply(calls=[good_box])])
    monkeypatch.setattr(viz_agent, "chat_no_think", chat)
    session = FakeSession({
        "plot_interactive_histogram": ["/tmp/run/plots/h.json|||code"],
        "plot_interactive_boxplot": [
            "Error: Column 'nope' not found.", "/tmp/run/plots/b.json|||c",
        ],
    })

    _run_loop("interactive", session, logs)

    assert ("info", "Worker 'interactive' is running 2 tools: "
            "Interactive Histogram, Interactive Box Plot.") in logs
    assert ("info", "Worker 'interactive' is retrying 1 tool: Interactive Box Plot.") in logs


def test_distinct_plots_are_kept_and_exact_repeats_merged():
    plots = []
    viz_agent._record_plot(plots, "plot_interactive_histogram", "/p/hist_age.json|||code A")
    viz_agent._record_plot(plots, "plot_interactive_histogram", "/p/hist_age_2.json|||code B")
    viz_agent._record_plot(plots, "plot_interactive_histogram", "/p/hist_age_3.json|||code A")

    assert [p["code"] for p in plots] == ["code A", "code B"]
    assert plots[0]["path"] == "/p/hist_age_3.json"  # latest file of the repeat
