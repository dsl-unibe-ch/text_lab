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
    plot_correlation_heatmap_impl as interactive_correlation_heatmap,
    plot_histogram_impl as interactive_histogram,
    plot_lineplot_impl as interactive_lineplot,
    plot_scatterplot_impl as interactive_scatterplot,
)
from core.visualization.plot_static import (
    generate_custom_static_plot_impl,
    plot_static_boxplot_impl,
    plot_static_correlation_heatmap_impl,
    plot_static_histogram_impl,
    plot_static_lineplot_impl,
    plot_static_scatterplot_impl,
    plot_static_wordcloud_impl,
)

from core.visualization.stats_analysis import (
    run_correlation_impl,
    run_group_comparison_impl,
    run_linear_regression_impl,
    rank_target_correlations_impl,
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
def plot_interactive_correlation_heatmap(
    data_file_path: str, title: str, method: str = "pearson"
) -> str:
    """
    Generates an interactive Plotly correlation heatmap for all numeric columns.
    Use this to visualize relationships between all numeric features at once.
    method must be 'pearson' or 'spearman'.
    """
    return interactive_correlation_heatmap(data_file_path, title, method)


@mcp.tool()
def generate_custom_plotly(
    data_file_path: str, python_code: str, plot_filename_keyword: str
) -> str:
    """Executes custom Python code (px, pd) to generate complex Plotly charts."""
    return generate_custom_plotly_impl(data_file_path, python_code, plot_filename_keyword)


@mcp.tool()
def get_column_summary(data_file_path: str, column: str) -> str:
    """
    Analyzes a specific column in the dataset and returns a statistical summary.
    Use this to check values, ranges, or unique items before selecting plotting parameters.
    """
    return get_column_summary_impl(data_file_path, column)


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


@mcp.tool()
def plot_static_wordcloud(
    data_file_path: str,
    text_column: str,
    title: str = "Word Cloud",
    extra_stopwords: str | None = None,
) -> str:
    """
    Generates a static Word Cloud image from a column containing text data.
    Use this when the user wants to visualize the most frequent terms in a dataset.
    extra_stopwords: optional comma-separated words to exclude (e.g. "said,also,one").
    """
    return plot_static_wordcloud_impl(data_file_path, text_column, title, extra_stopwords)


@mcp.tool()
def plot_static_correlation_heatmap(
    data_file_path: str, title: str, method: str = "pearson"
) -> str:
    """
    Generates a publication-ready Seaborn correlation heatmap for all numeric columns.
    Use this when the user explicitly asks for static or publication figures.
    method must be 'pearson' or 'spearman'.
    """
    return plot_static_correlation_heatmap_impl(data_file_path, title, method)


# --- STATISTICAL TOOLS ---

@mcp.tool()
def run_correlation(
    data_file_path: str, x_column: str, y_column: str, method: str = "pearson"
) -> str:
    """
    Computes statistical correlation (pearson, spearman) between two numeric columns.
    Use this to mathematically verify relationships before plotting scatterplots.
    """
    return run_correlation_impl(data_file_path, x_column, y_column, method)


@mcp.tool()
def run_group_comparison(
    data_file_path: str, target_col: str, group_col: str
) -> str:
    """
    Performs T-tests (2 groups) or ANOVA (>2 groups) to see if a numeric variable 
    (target_col) differs significantly across categories (group_col).
    Use this before generating boxplots.
    """
    return run_group_comparison_impl(data_file_path, target_col, group_col)


@mcp.tool()
def run_linear_regression(
    data_file_path: str, target_col: str, predictor_cols: list[str]
) -> str:
    """
    Runs an OLS Linear Regression. 
    target_col is the dependent variable (Y).
    predictor_cols is a list of independent variables (X). 
    CRITICAL: predictor_cols MUST be a valid JSON array of strings, e.g., ["col1", "col2"].
    """
    return run_linear_regression_impl(data_file_path, target_col, predictor_cols)

@mcp.tool()
def rank_target_correlations(
    data_file_path: str, target_col: str, method: str = "pearson"
) -> str:
    """
    Calculates and ranks the correlation between a single target column and all other 
    numeric columns in the dataset at once. Use this tool when the user wants to rank, 
    sort, or find top features related to a specific outcome column like diagnosis.
    """
    return rank_target_correlations_impl(data_file_path, target_col, method)


if __name__ == "__main__":
    mcp.run(transport="stdio")