"""The launcher and the Open OnDemand files that start a session.

``template/script.sh.erb`` starts the interface through the launcher
(``textlab.ui.launch``), which picks it from ``TEXT_LAB_UI``. The *Connect*
button (``view.html.erb``) depends on the interface too, which
``template/before.sh.erb`` reads from the same configuration
(``template/config.sh``) and passes on through ``conn_params``.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from textlab.ui import launch

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "template"


# ---------------------------------------------------------------------------
# The launcher
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [("", "streamlit"), ("streamlit", "streamlit"), (" Web ", "web")],
)
def test_the_interface_comes_from_text_lab_ui(value, expected):
    assert launch.chosen_interface({"TEXT_LAB_UI": value}) == expected
    assert launch.chosen_interface({}) == "streamlit"


def test_an_unknown_interface_is_refused():
    with pytest.raises(launch.LaunchError, match="streamlit or web"):
        launch.chosen_interface({"TEXT_LAB_UI": "gradio"})


def test_streamlit_starts_as_the_launch_script_started_it():
    # The launch script keeps the command for releases without the
    # launcher; both must give Streamlit the same options.
    script = (TEMPLATE / "script.sh.erb").read_text()
    legacy = script[
        script.index("streamlit run") : script.index(
            "fi\n", script.index("streamlit run")
        )
    ]
    legacy_options = re.findall(r"(--[a-zA-Z.]+)", legacy)
    args = SimpleNamespace(
        port=31234, base_path="/node/n/31234/", public_host="portal"
    )
    command = launch.streamlit_command(args, python="/env/python")
    assert command[:6] == [
        "/env/python",
        "-u",
        "-m",
        "streamlit",
        "run",
        str(launch.STREAMLIT_HOME),
    ]
    options = [item for item in command[6:] if item.startswith("--")]
    assert options == legacy_options
    assert command[command.index("--server.port") + 1] == "31234"
    assert (
        command[command.index("--server.baseUrlPath") + 1] == "/node/n/31234/"
    )
    assert command[command.index("--server.maxUploadSize") + 1] == "10000"
    assert launch.STREAMLIT_HOME.is_file()


def test_main_starts_streamlit_with_its_log(monkeypatch, tmp_path):
    started, logs = [], []

    def execv(path, args):
        # The real one replaces the process and never returns.
        started.append(args)
        raise SystemExit(0)

    monkeypatch.setenv("TEXT_LAB_UI", "streamlit")
    monkeypatch.setattr(launch, "redirect_output", logs.append)
    monkeypatch.setattr(launch.os, "execv", execv)
    with pytest.raises(SystemExit):
        launch.main(["--port", "4000", "--log-dir", str(tmp_path)])
    assert logs == [tmp_path / "streamlit.log"]
    assert started[0][started[0].index("--server.port") + 1] == "4000"


def test_main_serves_the_web_ui(monkeypatch, tmp_path):
    import textlab.ui.web.server as server

    served, logs = [], []
    monkeypatch.setenv("TEXT_LAB_UI", "web")
    monkeypatch.setattr(launch, "redirect_output", logs.append)
    monkeypatch.setattr(
        server, "serve", lambda **options: served.append(options)
    )
    status = launch.main(
        ["--port", "4000", "--base-path", "/node/n/4000/", "--public-host",
         "portal", "--log-dir", str(tmp_path)]
    )  # fmt: skip
    assert status == 0
    assert logs == [tmp_path / "textlab.log"]
    assert served == [
        {
            "port": 4000,
            "base_path": "/node/n/4000/",
            "public_host": "portal",
            "upload_limit": 10_000_000_000,
        }
    ]


def test_main_reports_a_wrong_interface(monkeypatch, capsys):
    monkeypatch.setenv("TEXT_LAB_UI", "nonsense")
    assert launch.main(["--port", "4000"]) == 2
    assert "TEXT_LAB_UI" in capsys.readouterr().err


def test_the_log_takes_the_output_of_the_process(tmp_path):
    log = tmp_path / "textlab.log"
    code = (
        "import sys; from pathlib import Path; from textlab.ui import launch;"
        f"launch.redirect_output(Path({str(log)!r}));"
        "print('to the log'); print('errors too', file=sys.stderr)"
    )
    # Unbuffered, as the launch script runs it (python -u).
    subprocess.run(
        [sys.executable, "-u", "-c", code], check=True, cwd=REPO / "src"
    )
    assert log.read_text().splitlines() == ["to the log", "errors too"]
    assert log.stat().st_mode & 0o077 == 0


# ---------------------------------------------------------------------------
# The Open OnDemand files
# ---------------------------------------------------------------------------


def test_the_launch_script_starts_the_launcher():
    script = (TEMPLATE / "script.sh.erb").read_text()
    assert 'source "$SCRIPT_DIR/config.sh"' in script
    assert "-m textlab.ui.launch" in script
    for option in ("--port", "--base-path", "--public-host", "--log-dir"):
        assert option in script


def test_the_connect_button_depends_on_the_interface():
    submit = (REPO / "submit.yml.erb").read_text()
    assert re.search(r"conn_params:\s*\n\s*- text_lab_ui", submit)
    view = (REPO / "view.html.erb").read_text()
    web, _, streamlit = view.partition("<% else %>")
    assert 'text_lab_ui.to_s == "web"' in web
    assert 'action="/node/<%= host %>/<%= port %>/login" method="post"' in web
    assert (
        '<input type="hidden" name="password" value="<%= password %>">' in web
    )
    assert "Copy the session password" in streamlit


def run_before_script(tmp_path, dev_env="", site_env=None):
    """Run before.sh.erb as Open OnDemand would; return text_lab_ui."""
    folder = tmp_path / "job"
    folder.mkdir()
    shutil.copy(TEMPLATE / "config.sh", folder)
    shutil.copy(TEMPLATE / "before.sh.erb", folder / "before.sh")
    site = tmp_path / "site.env"
    if site_env is not None:
        site.write_text(site_env)
    (folder / "dev.env").write_text(f'TEXT_LAB_SITE_ENV="{site}"\n{dev_env}')
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("TEXT_LAB_")
    }
    result = subprocess.run(
        ["bash", "-c", 'find_port() { echo 4000; }; host=node; '
         'source ./before.sh; echo "ui=$text_lab_ui"'],
        cwd=folder, env=environment, capture_output=True, text=True,
        check=True,
    )  # fmt: skip
    return result.stdout.strip().splitlines()[-1]


SITE_ENV = 'export TEXT_LAB_UI="${TEXT_LAB_UI:-streamlit}"\n'


@pytest.mark.parametrize(
    ("dev_env", "site_env", "expected"),
    [
        ("", SITE_ENV, "ui=streamlit"),
        ("TEXT_LAB_UI=Web\n", SITE_ENV, "ui=web"),
        ("", "", "ui=streamlit"),
        ("", None, "ui="),
    ],
)
def test_before_script_reads_the_interface(
    tmp_path, dev_env, site_env, expected
):
    assert run_before_script(tmp_path, dev_env, site_env) == expected
