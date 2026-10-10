"""Interactive Plotting Module for the AI Visualization Engine.

Generates web-ready interactive Plotly charts (.json).
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from textlab.features.visualization import r_code
from textlab.features.visualization.viz_config import CUSTOM_CODE_TIMEOUT
from textlab.features.visualization.viz_utils import (
    _strip_show_calls,
    describe_code_error,
    format_plot_output,
    generate_code_snippet,
    get_plot_path,
    load_data_safely,
    run_generated_code,
    split_comma_list,
)


def plot_histogram_impl(
    data_file_path: str,
    column: str,
    title: str,
    color_column: str | None = None,
) -> str:
    """Generates and saves an interactive Plotly histogram."""
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
            template="plotly_white",
        )

        plot_path = get_plot_path(
            data_file_path, f"hist_{column}", ext=".json"
        )
        fig.write_json(plot_path)

        color_arg = f", color='{color_column}'" if color_column else ""
        code_logic = (
            f"fig = px.histogram(df, x='{column}'{color_arg}, "
            f"title='{title}', template='plotly_white')"
        )
        code = generate_code_snippet(code_logic, data_file_path)

        r_snippet = r_code.build(
            r_code.histogram,
            data_file_path,
            column,
            title,
            color_column,
            pd.api.types.is_numeric_dtype(df[column]),
            interactive=True,
        )
        return format_plot_output(plot_path, code, r_snippet)
    except Exception as e:
        return f"Error plotting interactive histogram: {str(e)}"


def plot_scatterplot_impl(
    data_file_path: str,
    x_column: str,
    y_column: str,
    title: str,
    color_column: str | None = None,
) -> str:
    """Generates and saves an interactive Plotly scatter plot."""
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
        code = generate_code_snippet(code_logic, data_file_path)

        r_snippet = r_code.build(
            r_code.scatter,
            data_file_path,
            x_column,
            y_column,
            title,
            color_column,
            bool(color_column)
            and pd.api.types.is_numeric_dtype(df[color_column]),
            interactive=True,
        )
        return format_plot_output(plot_path, code, r_snippet)
    except Exception as e:
        return f"Error plotting interactive scatterplot: {str(e)}"


def plot_boxplot_impl(
    data_file_path: str,
    x_column: str,
    y_column: str,
    title: str,
    color_column: str | None = None,
) -> str:
    """Generates and saves an interactive Plotly box plot."""
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
        code = generate_code_snippet(code_logic, data_file_path)

        r_snippet = r_code.build(
            r_code.boxplot,
            data_file_path,
            x_column,
            y_column,
            title,
            color_column,
            interactive=True,
        )
        return format_plot_output(plot_path, code, r_snippet)
    except Exception as e:
        return f"Error plotting interactive boxplot: {str(e)}"


def plot_lineplot_impl(
    data_file_path: str,
    x_column: str,
    y_column: str,
    title: str,
    color_column: str | None = None,
) -> str:
    """Generates and saves an interactive Plotly line plot."""
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
        code = generate_code_snippet(code_logic, data_file_path)

        r_snippet = r_code.build(
            r_code.lineplot,
            data_file_path,
            x_column,
            y_column,
            title,
            color_column,
            interactive=True,
        )
        return format_plot_output(plot_path, code, r_snippet)
    except Exception as e:
        return f"Error plotting interactive lineplot: {str(e)}"


def plot_barchart_impl(
    data_file_path: str,
    x_column: str,
    y_column: str,
    title: str,
    color_column: str | None = None,
    aggregation: str = "mean",
) -> str:
    """Generates and saves an interactive Plotly bar chart.

    Args:
        data_file_path: The dataset file.
        x_column: The column on the x axis.
        y_column: The column on the y axis.
        title: The chart title.
        color_column: Optional column to colour the bars by.
        aggregation: How to aggregate y values per x category ('mean', 'sum',
            'count', 'median').
    """
    try:
        df = load_data_safely(data_file_path)
        if x_column not in df.columns or y_column not in df.columns:
            return f"Error: Columns '{x_column}' or '{y_column}' not found."

        # Colouring by the x or y column adds nothing and would duplicate a
        # grouping key, so it is dropped.
        if color_column and (
            color_column not in df.columns
            or color_column in (x_column, y_column)
        ):
            color_column = None

        valid_aggs = {"mean", "sum", "count", "median"}
        if aggregation not in valid_aggs:
            aggregation = "mean"

        group_cols = [x_column] + ([color_column] if color_column else [])
        if y_column in group_cols:
            # e.g. x=y='diagnosis' to count rows per category. Aggregating a
            # grouping key fails on reset_index ("cannot insert ..., already
            # exists"), so count the rows per group instead.
            agg_df = df.groupby(group_cols).size().reset_index(name="count")
            plot_y = "count"
            agg_code = (
                "agg_df = "
                f"df.groupby({group_cols!r}).size().reset_index(name='count')"
            )
        else:
            agg_df = getattr(
                df.groupby(group_cols)[y_column], aggregation
            )().reset_index()
            plot_y = y_column
            agg_code = (
                "agg_df = "
                f"df.groupby({group_cols!r})['{y_column}'].{aggregation}()"
                ".reset_index()"
            )

        color_arg = f", color='{color_column}'" if color_column else ""
        barmode = "group" if color_column else "relative"
        fig = px.bar(
            agg_df,
            x=x_column,
            y=plot_y,
            color=color_column,
            barmode=barmode,
            title=title,
            template="plotly_white",
        )

        plot_path = get_plot_path(
            data_file_path, f"bar_{y_column}_by_{x_column}", ext=".json"
        )
        fig.write_json(plot_path)

        code_logic = (
            f"{agg_code}\n"
            f"fig = px.bar(agg_df, x='{x_column}', y='{plot_y}'{color_arg},\n"
            f"    barmode='{barmode}', title='{title}', "
            "template='plotly_white')"
        )
        code = generate_code_snippet(code_logic, data_file_path)
        r_snippet = r_code.build(
            r_code.barchart_interactive,
            data_file_path,
            x_column,
            y_column,
            title,
            color_column,
            aggregation,
            count_rows=y_column in group_cols,
        )
        return format_plot_output(plot_path, code, r_snippet)
    except Exception as e:
        return f"Error plotting interactive bar chart: {str(e)}"


def plot_scatter_matrix_impl(
    data_file_path: str,
    columns: str,
    title: str,
    color_column: str | None = None,
) -> str:
    """Generate and save an interactive Plotly scatter matrix (pair plot).

    Args:
        data_file_path: The dataset file.
        columns: Comma-separated list of numeric column names to include.
        title: The chart title.
        color_column: Optional categorical column to colour points by (e.g.
            'diagnosis').
    """
    try:
        df = load_data_safely(data_file_path)

        col_list = [c.strip() for c in columns.split(",") if c.strip()]
        missing = [c for c in col_list if c not in df.columns]
        if missing:
            return f"Error: columns not found in data: {', '.join(missing)}"

        if color_column and color_column not in df.columns:
            color_column = None

        fig = px.scatter_matrix(
            df,
            dimensions=col_list,
            color=color_column,
            title=title,
            template="plotly_white",
        )
        fig.update_traces(diagonal_visible=False, showupperhalf=False)

        plot_path = get_plot_path(
            data_file_path,
            f"scatter_matrix_{'_'.join(col_list[:3])}",
            ext=".json",
        )
        fig.write_json(plot_path)

        color_arg = f", color='{color_column}'" if color_column else ""
        code_logic = (
            f"cols = {col_list!r}\n"
            f"fig = px.scatter_matrix(df, dimensions=cols{color_arg},\n"
            f"    title='{title}', template='plotly_white')\n"
            "fig.update_traces(diagonal_visible=False, showupperhalf=False)"
        )
        code = generate_code_snippet(code_logic, data_file_path)
        r_snippet = r_code.build(
            r_code.pair_plot,
            data_file_path,
            col_list,
            title,
            color_column,
            interactive=True,
        )
        return format_plot_output(plot_path, code, r_snippet)
    except Exception as e:
        return f"Error plotting scatter matrix: {str(e)}"


def plot_correlation_heatmap_impl(
    data_file_path: str,
    title: str,
    method: str = "pearson",
    column_filter: str | None = None,
) -> str:
    """Generates an interactive Plotly correlation heatmap.

    Args:
        data_file_path: The dataset file.
        title: The chart title.
        method: The correlation method, such as 'pearson' or 'spearman'.
        column_filter: Optional comma-separated column names or suffix patterns
            (e.g. "_mean" to select only columns ending in _mean). When
            omitted, all numeric columns are used.
    """
    try:
        df = load_data_safely(data_file_path)
        numeric_df = df.select_dtypes(include="number")

        if column_filter:
            filters = [
                f.strip() for f in column_filter.split(",") if f.strip()
            ]
            selected = [
                c
                for c in numeric_df.columns
                if c in filters or any(c.endswith(f) for f in filters)
            ]
            if len(selected) < 2:
                return (
                    f"Error: column_filter '{column_filter}' matched fewer "
                    "than 2 columns. "
                    "Available numeric columns: "
                    f"{', '.join(numeric_df.columns)}"
                )
            numeric_df = numeric_df[selected]

        if numeric_df.shape[1] < 2:
            return (
                "Error: Need at least 2 numeric columns to generate a "
                "correlation heatmap."
            )

        corr_matrix = numeric_df.corr(method=method).round(2)

        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title=title,
        )
        fig.update_layout(template="plotly_white")

        safe_filter = (
            column_filter.replace(",", "_").replace(" ", "")
            if column_filter
            else "all"
        )
        plot_path = get_plot_path(
            data_file_path, f"corr_heatmap_{safe_filter}_{method}", ext=".json"
        )
        fig.write_json(plot_path)

        filter_code = ""
        if column_filter:
            filter_code = (
                "filters = "
                f"{repr([f.strip() for f in column_filter.split(',') if f.strip()])}\n"  # noqa: E501
                f"numeric_df = df.select_dtypes(include='number')\n"
                "numeric_df = numeric_df[[c for c in numeric_df.columns if c "
                "in filters or any(c.endswith(f) for f in filters)]]\n"
            )
        else:
            filter_code = "numeric_df = df.select_dtypes(include='number')\n"

        code_logic = (
            f"{filter_code}"
            f"corr_matrix = numeric_df.corr(method='{method}').round(2)\n"
            f"fig = px.imshow(corr_matrix, text_auto=True, aspect='auto',\n"
            "    color_continuous_scale='RdBu_r', zmin=-1, zmax=1, "
            f"title='{title}')\n"
            "fig.update_layout(template='plotly_white')"
        )
        code = generate_code_snippet(code_logic, data_file_path)
        r_snippet = r_code.build(
            r_code.correlation_heatmap,
            data_file_path,
            title,
            method,
            split_comma_list(column_filter),
            interactive=True,
        )
        return format_plot_output(plot_path, code, r_snippet)
    except Exception as e:
        return f"Error plotting correlation heatmap: {str(e)}"


def generate_custom_plotly_impl(
    data_file_path: str, python_code: str, plot_filename_keyword: str
) -> str:
    """Run model-written Python code that draws an interactive Plotly chart."""
    columns: list[str] = []
    clean_code = ""
    try:
        df = load_data_safely(data_file_path)
        columns = [str(c) for c in df.columns]

        # Local scope for `exec`. Pass as a single dict (used as both globals
        # and locals) so that nested scopes like list comprehensions can also
        # resolve `df`, `pd`, etc. Using exec(code, {}, locals) would make
        # injected names invisible inside comprehensions and function defs due
        # to Python 3's exec scoping rules.
        local_scope = {
            "pd": pd,
            "px": px,
            "go": go,
            "np": np,
            # A copy, because load_data_safely returns a cached DataFrame
            # shared with every later tool call in this server process.
            # Generated code such as df.dropna(inplace=True) must not change
            # the data for them.
            "df": df.copy(),
            # data_file_path is intentionally NOT exposed: the model should use
            # df directly and must not call pd.read_csv() or reference file
            # paths.
        }

        # Strip markdown formatting and any fig.show() / plt.show() calls
        # before exec.
        clean_code = (
            python_code.replace("```python", "").replace("```", "").strip()
        )
        clean_code = _strip_show_calls(clean_code)

        run_generated_code(clean_code, local_scope, CUSTOM_CODE_TIMEOUT)

        # The system prompt enforces that the LLM must assign the output to
        # 'fig'
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

        full_user_code = generate_code_snippet(clean_code, data_file_path)

        return f"{plot_path}|||{full_user_code}"

    except SystemExit:
        # exit()/sys.exit() in generated code would otherwise shut down the MCP
        # server that every worker of the analysis shares.
        return (
            "Error: Your code must not call exit() or sys.exit(). Remove that "
            "call."
        )
    except Exception as e:
        detail = describe_code_error(e, clean_code, columns)
        return f"Error executing custom plotly code: {detail}"
