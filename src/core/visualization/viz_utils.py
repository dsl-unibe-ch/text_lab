"""
Utility functions for the AI Visualization Engine.
Handles file I/O, memory-safe data loading, and path generation.
"""

import os
import re
import sys

import pandas as pd

from core.visualization.viz_config import MAX_ROWS


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
            return pd.read_csv(file_path, nrows=nrows, encoding="latin1")
        if lower_name.endswith(".tsv"):
            return pd.read_csv(file_path, sep="\t", nrows=nrows, encoding="latin1")
        if lower_name.endswith((".xls", ".xlsx")):
            return pd.read_excel(file_path, nrows=nrows)
        if lower_name.endswith(".json"):
            return pd.read_json(file_path).head(nrows)
        return None
    except Exception as e:
        print(f"Error reading data preview for {file_name}: {e}", file=sys.stderr)
        return None


def load_data_safely(file_path: str) -> pd.DataFrame:
    """
    Load data from disk with safety limits to prevent Out-Of-Memory (OOM) crashes.
    This is used by the MCP Server when executing plot commands.

    Args:
        file_path: The absolute path to the data file.

    Returns:
        A pandas DataFrame containing the loaded data (capped at MAX_ROWS).

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        ValueError: If the file type is unsupported.
        RuntimeError: If pandas fails to read the file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")

    lower_name = file_path.lower()
    try:
        if lower_name.endswith(".csv"):
            return pd.read_csv(file_path, nrows=MAX_ROWS, encoding="latin1")
        if lower_name.endswith(".tsv"):
            return pd.read_csv(file_path, sep="\t", nrows=MAX_ROWS, encoding="latin1")
        if lower_name.endswith((".xls", ".xlsx")):
            return pd.read_excel(file_path, nrows=MAX_ROWS)
        if lower_name.endswith(".json"):
            # JSON is difficult to chunk efficiently in pandas, so we load entirely.
            return pd.read_json(file_path)
        
        raise ValueError(f"Unsupported file type: {os.path.basename(file_path)}")
    except Exception as e:
        raise RuntimeError(f"Failed to load data safely: {str(e)}")


def get_plot_path(data_file_path: str, plot_name: str, ext: str = ".json") -> str:
    """
    Generate a safe, unique file path for saving a generated plot.

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

    return os.path.join(plot_dir, f"{safe_plot_name}{ext}")


def generate_code_snippet(plot_code: str) -> str:
    """
    Format the raw plot generation code into a complete, runnable script string.

    Args:
        plot_code: The core logic used to generate the plot.

    Returns:
        A formatted Python script string including imports and data loading.
    """
    return (
        "import pandas as pd\n"
        "import plotly.express as px\n\n"
        "# Load Data\n"
        "df = pd.read_csv('your_data.csv')\n\n"
        "# Generate Plot\n"
        f"{plot_code}\n"
        "fig.show()"
    )