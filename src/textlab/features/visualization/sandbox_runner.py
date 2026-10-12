"""The process that runs model-written chart code inside the sandbox.

Started by :func:`.sandbox.run_in_sandbox` as
``python -m textlab.features.visualization.sandbox_runner /work`` in an
isolated container. It reads ``request.json`` and ``data.pkl`` from the
folder, runs the code with a reduced set of builtins and a time limit, and
writes the chart (``figure.json`` or ``figure.png``) and ``result.json``
(``{"figure": name}`` or ``{"error": message}``) there.
"""

from __future__ import annotations

import builtins
import json
import resource
import sys
from pathlib import Path
from typing import Any

#: Builtins the code may use: data handling and arithmetic, no files,
#: imports outside :data:`.sandbox.ALLOWED_MODULES` or introspection.
SAFE_BUILTIN_NAMES = (
    "abs",
    "all",
    "any",
    "bool",
    "callable",
    "chr",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "format",
    "frozenset",
    "hash",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "ord",
    "pow",
    "print",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "type",
    "zip",
    "ArithmeticError",
    "AttributeError",
    "Exception",
    "IndexError",
    "KeyError",
    "NameError",
    "RuntimeError",
    "StopIteration",
    "TypeError",
    "ValueError",
    "ZeroDivisionError",
)

#: Largest file the code's process may write (bytes).
MAX_FILE_BYTES = 512 * 1024 * 1024


def safe_builtins() -> dict[str, Any]:
    """Return the builtins for the code, with imports limited."""
    from textlab.features.visualization.sandbox import ALLOWED_MODULES

    allowed = {name: getattr(builtins, name) for name in SAFE_BUILTIN_NAMES}

    def limited_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name.split(".")[0] not in ALLOWED_MODULES:
            raise ImportError(f"Importing '{name}' is not allowed.")
        return builtins.__import__(name, globals, locals, fromlist, level)

    allowed["__import__"] = limited_import
    return allowed


def run(request: dict[str, Any], df: Any, work_dir: Path) -> dict[str, Any]:
    """Run the code of a request and save its chart.

    Args:
        request: ``kind`` (``"plotly"`` or ``"static"``), ``code`` and
            ``timeout`` (seconds).
        df: The table, given to the code as ``df``.
        work_dir: Where the chart is saved.

    Returns:
        ``{"figure": file name}`` or ``{"error": message for the model}``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from textlab.features.visualization.sandbox import (
        EXIT_MESSAGE,
        ExitCallError,
        UnsafeCodeError,
        check_code,
    )
    from textlab.features.visualization.viz_utils import (
        describe_code_error,
        run_generated_code,
    )

    code = request["code"]
    columns = [str(c) for c in df.columns]
    scope: dict[str, Any] = {
        "__builtins__": safe_builtins(),
        "pd": pd,
        "np": np,
        "df": df,
    }
    if request["kind"] == "plotly":
        import plotly.express as px
        import plotly.graph_objects as go

        scope.update(px=px, go=go)
    else:
        import seaborn as sns
        from wordcloud import WordCloud

        scope.update(sns=sns, plt=plt, WordCloud=WordCloud)
        plt.close("all")

    try:
        check_code(code)
        run_generated_code(code, scope, request["timeout"])
    except (SystemExit, ExitCallError):
        return {"error": EXIT_MESSAGE}
    except UnsafeCodeError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        return {"error": describe_code_error(exc, code, columns)}

    if request["kind"] == "plotly":
        fig = scope.get("fig")
        if fig is None:
            return {
                "error": "Error: Your code must assign the Plotly object to "
                "a variable named 'fig'."
            }
        fig.write_json(str(work_dir / "figure.json"))
        return {"figure": "figure.json"}

    fig = scope.get("fig")
    if fig is None:
        if not plt.get_fignums():
            return {
                "error": "Error: Your code did not produce any matplotlib "
                "figure."
            }
        fig = plt.gcf()
    fig.savefig(work_dir / "figure.png", bbox_inches="tight", dpi=300)
    plt.close("all")
    return {"figure": "figure.png"}


def serve(work_dir: Path) -> None:
    """Run the request in a folder and write ``result.json`` there."""
    import pandas as pd

    request = json.loads((work_dir / "request.json").read_text("utf-8"))
    df = pd.read_pickle(work_dir / "data.pkl")
    try:
        result = run(request, df, work_dir)
    except Exception as exc:
        # Saving the chart failed; the code's own errors are caught in run().
        result = {"error": f"{type(exc).__name__}: {exc}"}
    (work_dir / "result.json").write_text(json.dumps(result), "utf-8")


def main(argv: list[str]) -> int:
    """Run the request in the folder given on the command line."""
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE_BYTES, MAX_FILE_BYTES))
    serve(Path(argv[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
