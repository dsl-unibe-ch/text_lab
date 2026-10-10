"""
Utility functions for the AI Visualization Engine.
Handles file I/O, memory-safe data loading, and path generation.
"""

import contextlib
import hashlib
import os
import re
import signal
import sys
import threading
from functools import lru_cache
from typing import Iterator

import pandas as pd

from textlab.features.visualization.viz_config import MAX_ROWS

# Tracks whether the most recent load_data_safely call truncated the file at MAX_ROWS.
# Read by tools so they can surface a warning to the agent / UI.
LAST_LOAD_TRUNCATED: dict[str, bool] = {}

# Plot names are built from column lists and can get very long (e.g. a heatmap
# over 30 columns). Most filesystems cap a file name at 255 bytes.
MAX_PLOT_NAME_CHARS: int = 100


def shorten_text(value: object, limit: int) -> str:
    """Return ``value`` as a single-line string of at most ``limit`` characters.

    Whitespace (including newlines) is collapsed so long free text never ends
    up verbatim in a model prompt; truncated values end with ``"..."``.
    """
    flat = " ".join(str(value).split())
    return flat if len(flat) <= limit else flat[: max(limit - 3, 0)] + "..."


def _read_csv_with_fallback(file_path: str, sep: str = ",", nrows: int | None = None) -> pd.DataFrame:
    """Try UTF-8 first (most common), fall back to latin1 to avoid silent mangling."""
    try:
        return pd.read_csv(file_path, sep=sep, nrows=nrows, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(file_path, sep=sep, nrows=nrows, encoding="latin1")


def _read_excel_safely(file_path: str, max_rows: int) -> pd.DataFrame:
    """
    Read an Excel file using openpyxl in read-only streaming mode for .xlsx files,
    which avoids loading the entire workbook into memory at once.
    Falls back to standard pd.read_excel for .xls files (xlrd doesn't support streaming).
    """
    if file_path.lower().endswith(".xls"):
        return pd.read_excel(file_path, nrows=max_rows)

    try:
        import openpyxl
        wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
        ws = wb.active
        row_iter = ws.iter_rows(values_only=True)
        header = next(row_iter, None)
        if header is None:
            wb.close()
            return pd.DataFrame()
        data = []
        for row in row_iter:
            if len(data) >= max_rows:
                break
            data.append(row)
        wb.close()
        return pd.DataFrame(data, columns=header)
    except ImportError:
        return pd.read_excel(file_path, nrows=max_rows)


def save_data_file(file_bytes: bytes, file_name: str, run_dir: str) -> str:
    """
    Save the uploaded file bytes to a temporary workspace directory.

    Args:
        file_bytes: The raw bytes of the uploaded file.
        file_name: The original name of the uploaded file.
        run_dir: The directory where the file should be saved.

    Returns:
        The absolute path to the saved file.
    """
    file_extension = os.path.splitext(file_name)[1].lower()
    data_file_path = os.path.join(run_dir, f"uploaded_data{file_extension}")
    
    with open(data_file_path, "wb") as f:
        f.write(file_bytes)
        
    return data_file_path


def get_fast_data_preview(file_path: str, file_name: str, nrows: int = 5) -> pd.DataFrame | None:
    """
    Read only the first few rows of a dataset directly from disk to save memory.
    This is used by the UI to quickly preview the data before passing it to the AI.

    Args:
        file_path: The path to the saved data file.
        file_name: The original name of the file (used for extension checking).
        nrows: The number of rows to read.

    Returns:
        A pandas DataFrame containing the top rows, or None if reading fails.
    """
    lower_name = file_name.lower()
    try:
        if lower_name.endswith(".csv"):
            return _read_csv_with_fallback(file_path, nrows=nrows)
        if lower_name.endswith(".tsv"):
            return _read_csv_with_fallback(file_path, sep="\t", nrows=nrows)
        if lower_name.endswith((".xls", ".xlsx")):
            return _read_excel_safely(file_path, nrows)
        if lower_name.endswith(".json"):
            try:
                return pd.read_json(file_path, lines=True, nrows=nrows)
            except (ValueError, TypeError):
                return pd.read_json(file_path).head(nrows)
        return None
    except Exception as e:
        print(f"Error reading data preview for {file_name}: {e}", file=sys.stderr)
        return None


@lru_cache(maxsize=8)
def _load_data_cached(file_path: str, mtime: float, size: int) -> pd.DataFrame:
    """Cache-keyed loader. mtime+size invalidate the cache when the file changes."""
    lower_name = file_path.lower()
    if lower_name.endswith(".csv"):
        df = _read_csv_with_fallback(file_path, nrows=MAX_ROWS + 1)
    elif lower_name.endswith(".tsv"):
        df = _read_csv_with_fallback(file_path, sep="\t", nrows=MAX_ROWS + 1)
    elif lower_name.endswith((".xls", ".xlsx")):
        df = _read_excel_safely(file_path, MAX_ROWS + 1)
    elif lower_name.endswith(".json"):
        try:
            df = pd.read_json(file_path, lines=True, nrows=MAX_ROWS + 1)
        except (ValueError, TypeError):
            df = pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file type: {os.path.basename(file_path)}")

    truncated = len(df) > MAX_ROWS
    LAST_LOAD_TRUNCATED[file_path] = truncated
    if truncated:
        df = df.iloc[:MAX_ROWS].copy()
    return df


def load_data_safely(file_path: str) -> pd.DataFrame:
    """
    Load data from disk with safety limits to prevent Out-Of-Memory (OOM) crashes.
    Cached by (path, mtime, size) so repeated tool calls in the same MCP session
    don't re-parse the file.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        ValueError: If the file type is unsupported.
        RuntimeError: If pandas fails to read the file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")

    try:
        stat = os.stat(file_path)
        return _load_data_cached(file_path, stat.st_mtime, stat.st_size)
    except (FileNotFoundError, ValueError):
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to load data safely: {str(e)}")


def was_last_load_truncated(file_path: str) -> bool:
    """Returns True if the last load_data_safely call for this path hit MAX_ROWS."""
    return LAST_LOAD_TRUNCATED.get(file_path, False)


@contextlib.contextmanager
def time_limit(seconds: float) -> Iterator[None]:
    """Raise ``TimeoutError`` inside the block if it runs longer than ``seconds``.

    Used around model-written code: an endless loop would otherwise block the
    MCP server, which all workers of an analysis share. Implemented with
    ``SIGALRM``, so it is only active in the main thread on POSIX systems
    (where the MCP server runs its tools); elsewhere the block runs unlimited.

    Note that a long-running C call (e.g. a single huge NumPy operation) is
    only interrupted once control returns to Python.
    """
    usable = (
        seconds > 0
        and hasattr(signal, "SIGALRM")
        and threading.current_thread() is threading.main_thread()
    )
    if not usable:
        yield
        return

    def _on_timeout(signum, frame):
        raise TimeoutError(f"Code did not finish within {seconds:.0f} seconds.")

    previous_handler = signal.signal(signal.SIGALRM, _on_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


# File name given to model-written code when compiling it, so tracebacks can
# be matched back to the generated code's own lines.
GENERATED_CODE_FILENAME: str = "<generated code>"
# Upper bound for the column list appended to code errors sent to the model.
ERROR_COLUMNS_MAX_CHARS: int = 600


def run_generated_code(code: str, scope: dict, timeout: float) -> None:
    """Compile and execute model-written code in ``scope`` under a time limit.

    Compiling under ``GENERATED_CODE_FILENAME`` lets ``describe_code_error``
    report the exact failing line of the generated code.

    Raises:
        SyntaxError: If the code does not compile.
        TimeoutError: If it runs longer than ``timeout`` seconds.
        Exception: Whatever the generated code itself raises.
    """
    compiled = compile(code, GENERATED_CODE_FILENAME, "exec")
    with time_limit(timeout):
        exec(compiled, scope)


def _generated_code_line(exc: BaseException) -> int | None:
    """Return the line of the generated code where ``exc`` was raised, if any.

    For errors raised inside library calls the innermost generated-code frame
    is used, i.e. the line of the model's code that made the failing call.
    """
    if isinstance(exc, SyntaxError) and exc.filename == GENERATED_CODE_FILENAME:
        return exc.lineno
    line_no = None
    tb = exc.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == GENERATED_CODE_FILENAME:
            line_no = tb.tb_lineno
        tb = tb.tb_next
    return line_no


def describe_code_error(exc: BaseException, code: str, columns: list[str]) -> str:
    """Explain a failure of model-written code so the model can fix it in one retry.

    The bare exception text is often too terse to act on (a pandas
    ``KeyError`` is just the quoted key). This adds the exception type, the
    failing line of the generated code, and the dataset's column names.

    Args:
        exc: The exception raised while compiling or running the code.
        code: The code that was executed.
        columns: The dataset's column names (empty if the data did not load).

    Returns:
        A multi-line description, e.g.::

            KeyError: 'Radius_mean'
            At line 3: fig = px.scatter(df, x='Radius_mean', y='area_mean')
            Available columns: radius_mean, texture_mean, ...
    """
    lines = [f"{type(exc).__name__}: {exc}"]
    line_no = _generated_code_line(exc)
    code_lines = code.splitlines()
    if line_no and 1 <= line_no <= len(code_lines):
        lines.append(f"At line {line_no}: {code_lines[line_no - 1].strip()}")
    if columns:
        listed = shorten_text(", ".join(columns), ERROR_COLUMNS_MAX_CHARS)
        lines.append(f"Available columns: {listed}")
    return "\n".join(lines)


def get_plot_path(data_file_path: str, plot_name: str, ext: str = ".json") -> str:
    """
    Generate a safe, unique file path for saving a generated plot.

    The name is sanitised to word characters and hyphens; names longer than
    ``MAX_PLOT_NAME_CHARS`` are shortened and suffixed with a hash. If a file
    with that name already exists (e.g. two histograms of the same column, or
    a plot from an earlier chat turn), a numeric suffix is added so earlier
    plots are never overwritten.

    Args:
        data_file_path: The path to the source data file (used to locate the run directory).
        plot_name: The descriptive name of the plot.
        ext: The file extension for the plot (e.g., '.json', '.png').

    Returns:
        The absolute path where the plot should be saved.
    """
    run_dir = os.path.dirname(data_file_path)
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Sanitize the plot name: Keep only alphanumeric characters, underscores, and hyphens
    safe_plot_name = re.sub(r"[^\w\-]", "", plot_name.replace(" ", "_")).rstrip("_")
    if not safe_plot_name:
        safe_plot_name = "plot"
    if len(safe_plot_name) > MAX_PLOT_NAME_CHARS:
        # Keep a readable prefix and add a short hash of the full name so
        # different long names still map to different files.
        digest = hashlib.sha1(safe_plot_name.encode("utf-8")).hexdigest()[:10]
        safe_plot_name = f"{safe_plot_name[:MAX_PLOT_NAME_CHARS]}_{digest}"

    base = os.path.join(plot_dir, safe_plot_name)
    plot_path = f"{base}{ext}"
    counter = 2
    # The MCP server runs tools one at a time, so check-then-write is safe here.
    while os.path.exists(plot_path):
        plot_path = f"{base}_{counter}{ext}"
        counter += 1
    return plot_path


def split_comma_list(value: str | None) -> list[str]:
    """Split a comma-separated tool argument into its non-empty, stripped parts."""
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def format_plot_output(plot_path: str, code: str, r_snippet: str = "") -> str:
    """Build a plot tool's result: ``"path|||python code"``, plus ``"|||R code"``.

    The R part is only present when the run asked for R code (see r_code.py).
    """
    output = f"{plot_path}|||{code}"
    return f"{output}|||{r_snippet}" if r_snippet else output


def _strip_show_calls(code: str) -> str:
    """
    Remove standalone display/save calls from model-generated code before exec.

    Strips:
      - fig.show() / plt.show()   — clear the active figure on non-interactive backends
      - plt.savefig(...)           — model may save to an arbitrary/wrong path;
                                     the tool always saves explicitly afterwards
    """
    # Remove show() calls
    code = re.sub(r"^\s*(fig|plt)\.show\(\)\s*$", "", code, flags=re.MULTILINE)
    # Remove plt.savefig(...) — matches single-line calls (balanced or not)
    code = re.sub(r"^\s*plt\.savefig\([^\n]*\)\s*$", "", code, flags=re.MULTILINE)
    return code.rstrip()


def generate_code_snippet(plot_code: str, data_file_path: str | None = None) -> str:
    """
    Format the raw plot generation code into a complete, runnable script string.

    Args:
        plot_code: The core logic used to generate the plot.
        data_file_path: Optional source file path; its extension is used to choose
            the appropriate pandas reader in the generated snippet.

    Returns:
        A formatted Python script string including imports and data loading.
    """
    loader = "df = pd.read_csv('your_data.csv')"
    if data_file_path:
        ext = os.path.splitext(data_file_path)[1].lower()
        if ext == ".tsv":
            loader = "df = pd.read_csv('your_data.tsv', sep='\\t')"
        elif ext in (".xls", ".xlsx"):
            loader = f"df = pd.read_excel('your_data{ext}')"
        elif ext == ".json":
            loader = "df = pd.read_json('your_data.json')"

    clean = _strip_show_calls(plot_code)
    return (
        "import pandas as pd\n"
        "import plotly.express as px\n"
        "import plotly.graph_objects as go\n\n"
        "# Load Data\n"
        f"{loader}\n\n"
        "# Generate Plot\n"
        f"{clean}\n"
        "fig.show()"
    )