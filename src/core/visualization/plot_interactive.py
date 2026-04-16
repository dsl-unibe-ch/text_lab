"""
Interactive Plotting Module for the AI Visualization Engine.
Generates web-ready interactive Plotly charts (.json).
"""

import pandas as pd
import plotly.express as px

from core.visualization.viz_utils import (
    generate_code_snippet,
    get_plot_path,
    load_data_safely,
)


def plot_histogram_impl(
    data_file_path: str, column: str, title: str, color_column: str | None = None
) -> str:
    """
    Generates and saves an interactive Plotly histogram.
    """
    try:
        df = load_data_safely(data_file_path)
        if column not in df.columns:
            return f"Error: Column '{column}' not found in data."

        if color_column and color_column not in df.columns:
            color_column = None

        fig = px.histogram(
            df, 
            x=column, 
            color=color_column, 
            title=title, 
            template="plotly_white"
        )

        plot_path = get_plot_path(data_file_path, f"hist_{column}", ext=".json")
        fig.write_json(plot_path)

        color_arg = f", color='{color_column}'" if color_column else ""
        code_logic = (
            f"fig = px.histogram(df, x='{column}'{color_arg}, "
            f"title='{title}', template='plotly_white')"
        )
        code = generate_code_snippet(code_logic)

        return f"{plot_path}|||{code}"
    except Exception as e:
        return f"Error plotting interactive histogram: {str(e)}"


def plot_scatterplot_impl(
    data_file_path: str,
    x_column: str,
    y_column: str,
    title: str,
    color_column: str | None = None,
) -> str:
    """
    Generates and saves an interactive Plotly scatter plot.
    """
    try:
        df = load_data_safely(data_file_path)
        if x_column not in df.columns or y_column not in df.columns:
            return f"Error: Columns '{x_column}' or '{y_column}' not found."

        if color_column and color_column not in df.columns:
            color_column = None

        fig = px.scatter(
            df,
            x=x_column,
            y=y_column,
            color=color_column,
            title=title,
            template="plotly_white",
        )

        plot_path = get_plot_path(
            data_file_path, f"scatter_{x_column}_vs_{y_column}", ext=".json"
        )
        fig.write_json(plot_path)

        color_arg = f", color='{color_column}'" if color_column else ""
        code_logic = (
            f"fig = px.scatter(df, x='{x_column}', y='{y_column}'{color_arg}, "
            f"title='{title}', template='plotly_white')"
        )
        code = generate_code_snippet(code_logic)

        return f"{plot_path}|||{code}"
    except Exception as e:
        return f"Error plotting interactive scatterplot: {str(e)}"


def plot_boxplot_impl(
    data_file_path: str,
    x_column: str,
    y_column: str,
    title: str,
    color_column: str | None = None,
) -> str:
    """
    Generates and saves an interactive Plotly box plot.
    """
    try:
        df = load_data_safely(data_file_path)
        if x_column not in df.columns or y_column not in df.columns:
            return f"Error: Columns '{x_column}' or '{y_column}' not found."

        if color_column and color_column not in df.columns:
            color_column = None

        fig = px.box(
            df,
            x=x_column,
            y=y_column,
            color=color_column,
            title=title,
            template="plotly_white",
        )

        plot_path = get_plot_path(
            data_file_path, f"boxplot_{y_column}_by_{x_column}", ext=".json"
        )
        fig.write_json(plot_path)

        color_arg = f", color='{color_column}'" if color_column else ""
        code_logic = (
            f"fig = px.box(df, x='{x_column}', y='{y_column}'{color_arg}, "
            f"title='{title}', template='plotly_white')"
        )
        code = generate_code_snippet(code_logic)

        return f"{plot_path}|||{code}"
    except Exception as e:
        return f"Error plotting interactive boxplot: {str(e)}"


def plot_lineplot_impl(
    data_file_path: str,
    x_column: str,
    y_column: str,
    title: str,
    color_column: str | None = None,
) -> str:
    """
    Generates and saves an interactive Plotly line plot.
    """
    try:
        df = load_data_safely(data_file_path)
        if x_column not in df.columns or y_column not in df.columns:
            return f"Error: Columns '{x_column}' or '{y_column}' not found."

        if color_column and color_column not in df.columns:
            color_column = None

        fig = px.line(
            df,
            x=x_column,
            y=y_column,
            color=color_column,
            title=title,
            template="plotly_white",
        )

        plot_path = get_plot_path(
            data_file_path, f"lineplot_{y_column}_over_{x_column}", ext=".json"
        )
        fig.write_json(plot_path)

        color_arg = f", color='{color_column}'" if color_column else ""
        code_logic = (
            f"fig = px.line(df, x='{x_column}', y='{y_column}'{color_arg}, "
            f"title='{title}', template='plotly_white')"
        )
        code = generate_code_snippet(code_logic)

        return f"{plot_path}|||{code}"
    except Exception as e:
        return f"Error plotting interactive lineplot: {str(e)}"


def generate_custom_plotly_impl(
    data_file_path: str, python_code: str, plot_filename_keyword: str
) -> str:
    """
    Executes custom Python code to generate complex interactive Plotly charts.
    """
    try:
        df = load_data_safely(data_file_path)

        # Prepare a secure local scope for `exec`
        local_scope = {
            "pd": pd,
            "px": px,
            "df": df,
            "data_file_path": data_file_path,
        }

        # Strip markdown formatting if the LLM provided it
        clean_code = python_code.replace("```python", "").replace("```", "").strip()

        # Execute the custom code
        exec(clean_code, {}, local_scope)

        # The system prompt enforces that the LLM must assign the output to 'fig'
        if "fig" not in local_scope:
            return (
                "Error: Your code must assign the Plotly object to a variable "
                "named 'fig'."
            )

        fig = local_scope["fig"]
        plot_path = get_plot_path(
            data_file_path, f"custom_{plot_filename_keyword}", ext=".json"
        )
        fig.write_json(plot_path)

        full_user_code = generate_code_snippet(clean_code)

        return f"{plot_path}|||{full_user_code}"

    except Exception as e:
        return f"Error executing custom plotly code: {str(e)}"