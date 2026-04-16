"""
Model Context Protocol (MCP) Server for the AI Visualization Engine.
Registers both interactive (Plotly) and static (Matplotlib) tools.
"""

import logging
import os
import sys

# Ensure the 'src' directory is in the Python path ---
# Because this script is launched as an isolated subprocess by the MCP stdio client,
# we must explicitly tell Python where the root 'src' directory is located.
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from mcp.server.fastmcp import FastMCP

from core.visualization.plot_data import get_column_summary_impl
from core.visualization.plot_interactive import (
    generate_custom_plotly_impl,
    plot_boxplot_impl as interactive_boxplot,
    plot_histogram_impl as interactive_histogram,
    plot_lineplot_impl as interactive_lineplot,
    plot_scatterplot_impl as interactive_scatterplot,
)
from core.visualization.plot_static import (
    generate_custom_static_plot_impl,
    plot_static_boxplot_impl,
    plot_static_histogram_impl,
    plot_static_lineplot_impl,
    plot_static_scatterplot_impl,
)

# Configure strict logging to prevent interference with stdout/stderr JSON-RPC
logging.basicConfig(level=logging.ERROR)
logging.getLogger("mcp").setLevel(logging.ERROR)

mcp = FastMCP("Data Visualization MCP Server")


# --- INTERACTIVE TOOLS (Plotly) ---

@mcp.tool()
def plot_interactive_histogram(
    data_file_path: str, column: str, title: str, color_column: str | None = None
) -> str:
    """Generates a web-ready interactive Plotly histogram."""
    return interactive_histogram(data_file_path, column, title, color_column)


@mcp.tool()
def plot_interactive_scatterplot(
    data_file_path: str, x_column: str, y_column: str, title: str, color_column: str | None = None
) -> str:
    """Generates a web-ready interactive Plotly scatter plot."""
    return interactive_scatterplot(data_file_path, x_column, y_column, title, color_column)


@mcp.tool()
def plot_interactive_boxplot(
    data_file_path: str, x_column: str, y_column: str, title: str, color_column: str | None = None
) -> str:
    """Generates a web-ready interactive Plotly box plot."""
    return interactive_boxplot(data_file_path, x_column, y_column, title, color_column)


@mcp.tool()
def plot_interactive_lineplot(
    data_file_path: str, x_column: str, y_column: str, title: str, color_column: str | None = None
) -> str:
    """Generates a web-ready interactive Plotly line plot."""
    return interactive_lineplot(data_file_path, x_column, y_column, title, color_column)


@mcp.tool()
def generate_custom_plotly(
    data_file_path: str, python_code: str, plot_filename_keyword: str
) -> str:
    """Executes custom Python code (px, pd) to generate complex Plotly charts."""
    return generate_custom_plotly_impl(data_file_path, python_code, plot_filename_keyword)


# --- STATIC TOOLS (Matplotlib/Seaborn) ---

@mcp.tool()
def plot_static_histogram(
    data_file_path: str, column: str, title: str, x_label: str
) -> str:
    """Generates a static Matplotlib/Seaborn histogram (for papers/publications)."""
    return plot_static_histogram_impl(data_file_path, column, title, x_label)


@mcp.tool()
def plot_static_scatterplot(
    data_file_path: str, x_column: str, y_column: str, title: str, x_label: str, y_label: str, hue_column: str | None = None
) -> str:
    """Generates a static Matplotlib/Seaborn scatter plot (for papers/publications)."""
    return plot_static_scatterplot_impl(data_file_path, x_column, y_column, title, x_label, y_label, hue_column)


@mcp.tool()
def plot_static_boxplot(
    data_file_path: str, x_column: str, y_column: str, title: str, x_label: str, y_label: str
) -> str:
    """Generates a static Matplotlib/Seaborn box plot (for papers/publications)."""
    return plot_static_boxplot_impl(data_file_path, x_column, y_column, title, x_label, y_label)


@mcp.tool()
def plot_static_lineplot(
    data_file_path: str, x_column: str, y_column: str, title: str, x_label: str, y_label: str, hue_column: str | None = None
) -> str:
    """Generates a static Matplotlib/Seaborn line plot (for papers/publications)."""
    return plot_static_lineplot_impl(data_file_path, x_column, y_column, title, x_label, y_label, hue_column)


@mcp.tool()
def generate_custom_static_plot(
    data_file_path: str, python_code: str, plot_filename_keyword: str
) -> str:
    """Executes custom Python code (plt, sns, pd) to generate complex static charts."""
    return generate_custom_static_plot_impl(data_file_path, python_code, plot_filename_keyword)


if __name__ == "__main__":
    mcp.run(transport="stdio")