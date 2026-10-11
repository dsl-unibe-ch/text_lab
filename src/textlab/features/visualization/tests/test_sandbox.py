"""The sandbox for model-written chart code.

Most tests need no container: the code check, the runner (run in this
process) and the plumbing around ``apptainer`` (replaced by a fake). The
tests marked ``container`` start the real sandbox, which needs the Text Lab
image and nested Apptainer (run them with ``scripts/test_on_node.sbatch``).
"""

import json
import os
import pathlib
import subprocess

import pandas as pd
import pytest

from textlab.features.visualization import sandbox, sandbox_runner

TABLE = pd.DataFrame({"group": ["a", "b", "a"], "value": [1.0, 2.0, 3.0]})


# --- Layer 1: the code check ---------------------------------------------


@pytest.mark.parametrize(
    "code",
    [
        "fig = px.histogram(df, x='value')",
        "import plotly.express as px\nfig = px.bar(df, x='group', y='value')",
        "from plotly import graph_objects as go\nfig = go.Figure()",
        "means = {g: d['value'].mean() for g, d in df.groupby('group')}\n"
        "fig = px.bar(x=list(means), y=list(means.values()))",
        "df['value'] = pd.to_numeric(df['value'])\n"
        "for _, row in df.iterrows():\n    print(row['group'])",
        "import seaborn as sns\nfig, ax = plt.subplots()\n"
        "sns.boxplot(data=df, x='group', y='value', ax=ax)",
        "import math\nfrom scipy import stats\n"
        "r = stats.pearsonr(df['value'], df['value'])[0] * math.pi",
        "words = WordCloud().generate(' '.join(df['group'].to_list()))",
    ],
)
def test_ordinary_chart_code_is_allowed(code):
    sandbox.check_code(code)


@pytest.mark.parametrize(
    "code,reason",
    [
        ("import os\nos.system('id')", "importing 'os'"),
        ("import subprocess", "importing 'subprocess'"),
        ("from pathlib import Path", "importing from 'pathlib'"),
        ("from . import x", "importing from '.'"),
        ("__import__('os')", "using '__import__'"),
        ("open('/etc/passwd').read()", "using 'open'"),
        ("eval('1')", "using 'eval'"),
        ("getattr(pd, 'read_csv')", "using 'getattr'"),
        ("x = ().__class__.__bases__", "using the private attribute"),
        ("pd.io.common.os.system('id')", "using '."),
        ("df = pd.read_csv('/etc/passwd')", "using '.read_csv'"),
        ("df.to_csv('/tmp/out.csv')", "using '.to_csv' (writing files)"),
        ("np.save('x.npy', df.values)", "using '.save'"),
    ],
)
def test_file_process_and_internals_access_is_rejected(code, reason):
    with pytest.raises(sandbox.UnsafeCodeError) as error:
        sandbox.check_code(code)
    assert reason in str(error.value)
    assert "(line " in str(error.value)


@pytest.mark.parametrize(
    "code", ["exit()", "quit()", "import sys\nsys.exit(1)"]
)
def test_exit_calls_get_their_own_message(code):
    with pytest.raises(sandbox.ExitCallError) as error:
        sandbox.check_code(code)
    assert str(error.value) == sandbox.EXIT_MESSAGE


def test_syntax_errors_keep_their_line():
    with pytest.raises(SyntaxError) as error:
        sandbox.check_code("x = 1\nif x\n    y = 2")
    assert error.value.lineno == 2


# --- The runner, in this process ---------------------------------------


def run(kind, code, tmp_path):
    request = {"kind": kind, "code": code, "timeout": 10}
    return sandbox_runner.run(request, TABLE.copy(), tmp_path)


def test_the_runner_saves_a_plotly_chart(tmp_path):
    result = run("plotly", "fig = px.bar(df, x='group', y='value')", tmp_path)
    assert result == {"figure": "figure.json"}
    assert json.loads((tmp_path / "figure.json").read_text())["data"]


def test_the_runner_saves_the_current_static_figure(tmp_path):
    code = "sns.histplot(df['value'])"
    assert run("static", code, tmp_path) == {"figure": "figure.png"}
    assert (tmp_path / "figure.png").read_bytes()[:4] == b"\x89PNG"


def test_the_runner_reports_errors_for_the_model(tmp_path):
    result = run("plotly", "x = 1\nfig = px.bar(df, x='Group')", tmp_path)
    assert "At line 2" in result["error"]
    assert "Available columns: group, value" in result["error"]
    missing = run("plotly", "x = 1", tmp_path)
    assert missing["error"].startswith("Error: Your code must assign")


def test_the_runner_checks_the_code_again(tmp_path):
    assert run("plotly", "import os", tmp_path)["error"].startswith(
        "Not allowed: importing 'os'"
    )
    assert run("plotly", "exit()", tmp_path) == {"error": sandbox.EXIT_MESSAGE}


def test_the_builtins_have_no_files_and_limited_imports():
    scope = {"__builtins__": sandbox_runner.safe_builtins()}
    with pytest.raises(NameError):
        exec("open('/etc/passwd')", scope)
    with pytest.raises(ImportError, match="'os' is not allowed"):
        exec("import os", scope)
    exec("import math\nx = math.sqrt(4)", scope)
    assert scope["x"] == 2


# --- The plumbing, with a fake apptainer -------------------------------


@pytest.fixture
def fake_apptainer(monkeypatch):
    """Run the runner in this process instead of in a container."""
    calls = []
    monkeypatch.setattr(sandbox, "sandbox_image", lambda: "/images/tl.sif")

    def fake_run(command, **kwargs):
        calls.append((command, kwargs["env"]))
        work_dir = command[command.index("--bind") + 1].split(":")[0]
        sandbox_runner.serve(pathlib.Path(work_dir))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)
    return calls


def test_a_chart_is_copied_out_and_the_folder_removed(
    tmp_path, fake_apptainer
):
    figure = tmp_path / "chart.json"
    error = sandbox.run_in_sandbox(
        "plotly", "fig = px.bar(df, x='group')", TABLE, str(figure), 10
    )
    assert error is None
    assert json.loads(figure.read_text())["data"]
    assert sorted(os.listdir(tmp_path)) == ["chart.json"]
    ((command, env),) = fake_apptainer
    # The outer container's bind mounts must not reach the sandbox.
    assert env == {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    assert command[-3:] == [
        "-m",
        "textlab.features.visualization.sandbox_runner",
        "/work",
    ]


def test_the_sandbox_is_isolated(tmp_path):
    command = sandbox.sandbox_command("/images/tl.sif", tmp_path)
    assert command[:3] == ["apptainer", "exec", "--containall"]
    assert "--no-home" in command
    assert command[command.index("--network") + 1] == "none"
    binds = [command[i + 1] for i, a in enumerate(command) if a == "--bind"]
    assert binds[0] == f"{tmp_path}:/work"
    assert binds[1].endswith(":/textlab_src:ro")
    assert len(binds) == 2


def test_links_planted_by_the_code_are_not_followed(tmp_path, monkeypatch):
    secret = tmp_path / "secret.txt"
    secret.write_text("private")
    monkeypatch.setattr(sandbox, "sandbox_image", lambda: "/images/tl.sif")

    def planting_run(command, **kwargs):
        work_dir = command[command.index("--bind") + 1].split(":")[0]
        with open(os.path.join(work_dir, "result.json"), "w") as file:
            json.dump({"figure": "figure.json"}, file)
        os.symlink(secret, os.path.join(work_dir, "figure.json"))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(sandbox.subprocess, "run", planting_run)
    figure = tmp_path / "plots"
    figure.mkdir()
    with pytest.raises(sandbox.SandboxUnavailableError):
        sandbox.run_in_sandbox(
            "plotly", "fig = 1", TABLE, str(figure / "c.json"), 10
        )
    assert not (figure / "c.json").exists()
    assert os.listdir(figure) == []


def test_no_image_means_no_custom_code(tmp_path, monkeypatch):
    for name in sandbox.IMAGE_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "data.csv"
    TABLE.to_csv(path, index=False)
    output = sandbox.run_custom_code(
        "plotly",
        str(path),
        "fig = px.bar(df, x='group')",
        "custom",
        ".json",
        "Error executing custom plotly code",
        str,
        10,
    )
    assert output.startswith("Error: custom code cannot run here")


# --- The real sandbox ----------------------------------------------------


@pytest.mark.container
def test_the_real_sandbox_sees_no_home_network_or_storage(tmp_path):
    """Probe the container the sandbox command starts (not the runner)."""
    probe = (
        "import os, socket, pathlib\n"
        "home = os.path.expanduser('~')\n"
        "print('home', os.path.isdir(home) and bool(os.listdir(home)))\n"
        "print('storage', os.path.exists('/storage'))\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 80), 3)\n"
        "    print('network', True)\n"
        "except OSError:\n"
        "    print('network', False)\n"
        "pathlib.Path('/work/ok').write_text('x')\n"
        "try:\n"
        "    pathlib.Path('/textlab_src/x').write_text('x')\n"
        "    print('src_writable', True)\n"
        "except OSError:\n"
        "    print('src_writable', False)\n"
    )
    command = sandbox.sandbox_command(sandbox.sandbox_image(), tmp_path)
    command = command[: command.index("-m")] + ["-c", probe]
    output = subprocess.run(
        command,
        env=sandbox.sandbox_environment(),
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    ).stdout
    assert "home False" in output
    assert "storage False" in output
    assert "network False" in output
    assert "src_writable False" in output
    assert (tmp_path / "ok").read_text() == "x"
