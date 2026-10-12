"""Running model-written plotting code safely.

The custom chart tools run Python code written by the model. The model can
be steered by text in the data it reads (a cell such as "ignore your
instructions and run ..."), so that code is treated as untrusted, in two
layers:

1. :func:`check_code` reads the code before it runs and rejects imports
   outside the plotting and data libraries, private attributes (``_x``),
   and file, process and introspection functions. The rejection is sent to
   the model, which rewrites the code. This stops the obvious attacks, but
   Python offers too many indirect routes for it to be a boundary.
2. :func:`run_in_sandbox` runs the code in a fresh process inside its own
   container (``apptainer exec --containall``) with no network, no home
   folder and no shared storage: it sees only a folder with a copy of the
   table, where it writes the chart. Code that gets past the first layer
   finds nothing to read or send. The code there also runs with a reduced
   set of builtins and a time limit (:mod:`.sandbox_runner`).

If the sandbox cannot be started, custom code does not run at all.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import pandas as pd

import textlab
from textlab.features.visualization.viz_utils import (
    GENERATED_CODE_FILENAME,
    _strip_show_calls,
    describe_code_error,
    get_plot_path,
    load_data_safely,
)

#: Top-level modules the code may import (the libraries are also provided
#: under their usual names, so imports are rarely needed).
ALLOWED_MODULES = frozenset(
    {
        "collections",
        "colorsys",
        "datetime",
        "functools",
        "itertools",
        "json",
        "math",
        "matplotlib",
        "numpy",
        "operator",
        "pandas",
        "plotly",
        "random",
        "re",
        "scipy",
        "seaborn",
        "statistics",
        "string",
        "textwrap",
        "wordcloud",
    }
)

#: Names the code may not use: they reach files, processes, other code or
#: the interpreter's internals.
BLOCKED_NAMES = frozenset(
    {
        "__builtins__",
        "__import__",
        "breakpoint",
        "compile",
        "delattr",
        "eval",
        "exec",
        "getattr",
        "globals",
        "help",
        "input",
        "locals",
        "memoryview",
        "open",
        "setattr",
        "vars",
    }
)

#: Attributes the code may not use, in addition to private ones (``_x``)
#: and file readers and writers (``read_*``, ``to_*``).
BLOCKED_ATTRIBUTES = frozenset(
    {
        "builtins",
        "dump",
        "dumps",
        "eval",
        "exec",
        "fromfile",
        "genfromtxt",
        "importlib",
        "io",
        "load",
        "loadtxt",
        "memmap",
        "open",
        "os",
        "pathlib",
        "popen",
        "save",
        "savefig",
        "savetxt",
        "savez",
        "shutil",
        "socket",
        "spawn",
        "subprocess",
        "sys",
        "system",
        "tofile",
        "write_html",
        "write_image",
        "write_json",
    }
)

#: ``to_*`` methods that convert data in memory and are allowed.
ALLOWED_TO_METHODS = frozenset(
    {
        "to_datetime",
        "to_dict",
        "to_frame",
        "to_list",
        "to_numeric",
        "to_numpy",
        "to_period",
        "to_pydatetime",
        "to_timedelta",
        "to_timestamp",
    }
)

EXIT_MESSAGE = (
    "Error: Your code must not call exit() or sys.exit(). Remove that call."
)
_EXIT_NAMES = frozenset({"exit", "quit"})

#: Variables in which Apptainer names the image a process runs in; they
#: must reach the MCP server, which runs the custom chart tools.
IMAGE_VARIABLES = ("APPTAINER_CONTAINER", "SINGULARITY_CONTAINER")

#: The chart file the runner writes, by kind of chart.
FIGURE_FILES = {"plotly": "figure.json", "static": "figure.png"}
#: Largest result and chart files read back from the sandbox (bytes).
MAX_RESULT_BYTES = 1024 * 1024
MAX_FIGURE_BYTES = 256 * 1024 * 1024

#: Seconds allowed for starting the sandbox and loading the libraries, on
#: top of the time limit of the code itself.
STARTUP_SECONDS = 120


class UnsafeCodeError(ValueError):
    """The code uses something the custom chart tools do not allow."""


class ExitCallError(UnsafeCodeError):
    """The code calls ``exit()``, ``quit()`` or ``sys.exit()``."""


class SandboxUnavailableError(RuntimeError):
    """The sandbox cannot be started, so the code is not run."""


# ---------------------------------------------------------------------------
# Layer 1: reading the code
# ---------------------------------------------------------------------------


def check_code(code: str) -> None:
    """Reject code that uses imports or functions the tools do not allow.

    Args:
        code: The model's code.

    Raises:
        SyntaxError: If the code does not parse (its line is reported
            like that of any other error).
        ExitCallError: If the code calls ``exit()`` or ``sys.exit()``.
        UnsafeCodeError: For anything else not allowed; the message says
            what and where.
    """
    problems = []
    for node in ast.walk(ast.parse(code, filename=GENERATED_CODE_FILENAME)):
        problem = _problem(node)
        if problem:
            problems.append((getattr(node, "lineno", 0), problem))
    if not problems:
        return
    if any("exit()" in problem for _, problem in problems):
        raise ExitCallError(EXIT_MESSAGE)
    line, problem = min(problems)
    raise UnsafeCodeError(
        f"Not allowed: {problem} (line {line}). Use the data and libraries "
        "already provided (df, pd, np and the plotting libraries), and do "
        "not read or write files."
    )


def _problem(node: ast.AST) -> str | None:
    """Return what a syntax node does that is not allowed, or ``None``."""
    if isinstance(node, ast.Import):
        for alias in node.names:
            module = alias.name.split(".")[0]
            if module not in ALLOWED_MODULES:
                return f"importing '{alias.name}'"
    elif isinstance(node, ast.ImportFrom):
        module = (node.module or "").split(".")[0]
        if node.level or module not in ALLOWED_MODULES:
            return f"importing from '{node.module or '.'}'"
    elif isinstance(node, ast.Name):
        if node.id in _EXIT_NAMES:
            return "calling exit()"
        if node.id in BLOCKED_NAMES or node.id.startswith("__"):
            return f"using '{node.id}'"
    elif isinstance(node, ast.Attribute):
        name = node.attr
        if name == "exit":
            return "calling exit()"
        if name.startswith("_"):
            return f"using the private attribute '.{name}'"
        if name in BLOCKED_ATTRIBUTES or name.startswith("read_"):
            return f"using '.{name}'"
        if name.startswith("to_") and name not in ALLOWED_TO_METHODS:
            return f"using '.{name}' (writing files)"
    return None


# ---------------------------------------------------------------------------
# Layer 2: the sandbox
# ---------------------------------------------------------------------------


def sandbox_image() -> str:
    """Return the Apptainer image the sandbox runs in: the app's own image.

    Apptainer names it in :data:`IMAGE_VARIABLES` inside the container.

    Raises:
        SandboxUnavailableError: If the app does not run in an image, or
            ``apptainer`` is missing.
    """
    image = next(
        (os.environ[name] for name in IMAGE_VARIABLES if os.environ.get(name)),
        None,
    )
    if not image or not os.path.exists(image):
        raise SandboxUnavailableError(
            "Custom code runs only inside the Text Lab container."
        )
    if shutil.which("apptainer") is None:
        raise SandboxUnavailableError("apptainer is not available.")
    return image


def sandbox_command(image: str, work_dir: Path) -> list[str]:
    """Return the command that runs the runner in an isolated container.

    Args:
        image: The Apptainer image.
        work_dir: The folder with the request and the table; the only
            folder the sandbox can write.

    Returns:
        The command.
    """
    src = Path(textlab.__file__).resolve().parents[1]
    return [
        "apptainer",
        "exec",
        "--containall",
        "--no-home",
        "--net",
        "--network",
        "none",
        "--bind",
        f"{work_dir}:/work",
        "--bind",
        f"{src}:/textlab_src:ro",
        "--pwd",
        "/work",
        "--env",
        "PYTHONPATH=/textlab_src",
        "--env",
        "PYTHONNOUSERSITE=1",
        "--env",
        "MPLBACKEND=Agg",
        "--env",
        "MPLCONFIGDIR=/tmp/matplotlib",
        "--env",
        "XDG_CACHE_HOME=/tmp",
        image,
        sys.executable,
        "-m",
        "textlab.features.visualization.sandbox_runner",
        "/work",
    ]


def sandbox_environment() -> dict[str, str]:
    """Return the environment ``apptainer`` is started with: only ``PATH``.

    Inside a container, Apptainer sets ``APPTAINER_BIND`` (and
    ``SINGULARITY_BIND``) to the outer container's bind mounts, and a
    nested ``apptainer`` would mount them all again: the model stores, the
    workspace, a home folder. Nothing of this process's environment is
    passed on, so the sandbox gets only what :func:`sandbox_command` names.
    """
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}


def run_in_sandbox(
    kind: str,
    code: str,
    df: pd.DataFrame,
    figure_path: str,
    timeout: float,
) -> str | None:
    """Run model-written chart code in the sandbox and save its chart.

    The sandbox works in a private folder next to ``figure_path``, removed
    afterwards.

    Args:
        kind: ``"plotly"`` (the code assigns ``fig``; saved as Plotly JSON)
            or ``"static"`` (Matplotlib/Seaborn: ``fig`` or the current
            figure; saved as PNG).
        code: The code, already checked with :func:`check_code`.
        df: The table, given to the code as ``df``.
        figure_path: Where to save the chart.
        timeout: Seconds the code itself may run.

    Returns:
        ``None`` once the chart is saved, or why the code failed, for the
        model (a message starting with ``"Error:"`` is complete; any other
        is the code's own error).

    Raises:
        SandboxUnavailableError: If the sandbox cannot be started.
    """
    image = sandbox_image()
    work_dir = Path(
        tempfile.mkdtemp(prefix="sandbox-", dir=os.path.dirname(figure_path))
    )
    try:
        df.to_pickle(work_dir / "data.pkl")
        request = {"kind": kind, "code": code, "timeout": timeout}
        (work_dir / "request.json").write_text(
            json.dumps(request), encoding="utf-8"
        )
        try:
            completed = subprocess.run(
                sandbox_command(image, work_dir),
                env=sandbox_environment(),
                capture_output=True,
                text=True,
                timeout=timeout + STARTUP_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return (
                f"TimeoutError: Code did not finish within {timeout:.0f} "
                "seconds."
            )
        try:
            result = json.loads(
                _read_file(work_dir / "result.json", MAX_RESULT_BYTES)
            )
        except FileNotFoundError:
            detail = (completed.stderr or completed.stdout).strip()[-1500:]
            raise SandboxUnavailableError(
                f"The sandbox did not run (exit code "
                f"{completed.returncode}): {detail}"
            ) from None
        if not isinstance(result, dict):
            raise SandboxUnavailableError(
                "Unexpected result from the sandbox."
            )
        if result.get("error"):
            return str(result["error"])
        figure = _read_file(work_dir / FIGURE_FILES[kind], MAX_FIGURE_BYTES)
        Path(figure_path).write_bytes(figure)
        return None
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _read_file(path: Path, limit: int) -> bytes:
    """Read a file the sandbox wrote, without following links.

    The sandbox's code could replace its output with a symbolic link to a
    file it cannot see but this process can; such a file is not read.

    Raises:
        FileNotFoundError: If the file does not exist.
        SandboxUnavailableError: If it is a link, or larger than ``limit``.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        if isinstance(exc, FileNotFoundError):
            raise
        raise SandboxUnavailableError(
            f"Unexpected output file {path.name}."
        ) from exc
    with os.fdopen(fd, "rb") as file:
        data = file.read(limit + 1)
    if len(data) > limit:
        raise SandboxUnavailableError(f"Output file {path.name} too large.")
    return data


def run_custom_code(
    kind: str,
    data_file_path: str,
    python_code: str,
    plot_name: str,
    ext: str,
    error_prefix: str,
    snippet: Callable[[str], str],
    timeout: float,
) -> str:
    """Check and run model-written chart code for a custom chart tool.

    Args:
        kind: ``"plotly"`` or ``"static"`` (see :func:`run_in_sandbox`).
        data_file_path: The table.
        python_code: The model's code, possibly in a Markdown fence.
        plot_name: Name for the chart file.
        ext: The chart file's extension.
        error_prefix: Start of the error returned for a failing code.
        snippet: Returns the reproducible code shown to the user.
        timeout: Seconds the code itself may run.

    Returns:
        ``"<chart path>|||<code>"``, or a message starting with ``"Error"``
        for the model.
    """
    columns: list[str] = []
    clean_code = ""
    try:
        df = load_data_safely(data_file_path)
        columns = [str(c) for c in df.columns]
        # Strip Markdown fences and show/save calls before running.
        clean_code = (
            python_code.replace("```python", "").replace("```", "").strip()
        )
        clean_code = _strip_show_calls(clean_code)
        check_code(clean_code)

        plot_path = get_plot_path(data_file_path, plot_name, ext=ext)
        # The sandbox gets its own copy of the table, so code such as
        # df.dropna(inplace=True) cannot change the cached one.
        error = run_in_sandbox(kind, clean_code, df, plot_path, timeout)
        if error is None:
            return f"{plot_path}|||{snippet(clean_code)}"
        if error.startswith("Error"):
            return error
        return f"{error_prefix}: {error}"
    except ExitCallError:
        # exit() would otherwise end the MCP server that all agents share.
        return EXIT_MESSAGE
    except UnsafeCodeError as e:
        return f"{error_prefix}: {e}"
    except SandboxUnavailableError as e:
        return f"Error: custom code cannot run here: {e}"
    except Exception as e:
        detail = describe_code_error(e, clean_code, columns)
        return f"{error_prefix}: {detail}"
