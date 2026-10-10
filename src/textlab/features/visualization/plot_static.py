"""Static Plotting Module for the AI Visualization Engine.

Generates publication-ready Matplotlib and Seaborn charts (.png).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import seaborn as sns
from wordcloud import STOPWORDS, WordCloud

from textlab.features.visualization import r_code
from textlab.features.visualization.viz_config import CUSTOM_CODE_TIMEOUT
from textlab.features.visualization.viz_utils import (
    _strip_show_calls,
    describe_code_error,
    format_plot_output,
    get_plot_path,
    load_data_safely,
    run_generated_code,
    split_comma_list,
)


def _generate_static_code_snippet(
    plot_code: str,
    grid_figure: bool = False,
    data_file_path: str | None = None,
) -> str:
    """Format plot code into a complete, runnable Matplotlib/Seaborn script.

    Args:
        plot_code: The core logic used to generate the plot.
        grid_figure: When True, omit the ``plt.figure()`` call. Use this for
            seaborn grid plots (pairplot, FacetGrid) and heatmaps that create
            their own figure internally.
        data_file_path: Optional source file path; its extension is used to
            choose the appropriate pandas reader in the generated snippet.

    Returns:
        A formatted Python script string including imports and data loading.
    """
    clean = _strip_show_calls(plot_code)
    figure_line = "" if grid_figure else "plt.figure(figsize=(10, 6))\n"

    loader = "df = pd.read_csv('your_data.csv')"
    if data_file_path:
        ext = os.path.splitext(data_file_path)[1].lower()
        if ext == ".tsv":
            loader = "df = pd.read_csv('your_data.tsv', sep='\\t')"
        elif ext in (".xls", ".xlsx"):
            loader = f"df = pd.read_excel('your_data{ext}')"
        elif ext == ".json":
            loader = "df = pd.read_json('your_data.json')"

    return (
        "import matplotlib.pyplot as plt\n"
        "import pandas as pd\n"
        "import seaborn as sns\n\n"
        "# Load Data\n"
        f"{loader}\n\n"
        "# Generate Plot\n"
        f"{figure_line}"
        f"{clean}\n"
        "plt.savefig('output.png', bbox_inches='tight', dpi=300)\n"
        "plt.show()"
    )


def plot_static_histogram_impl(
    data_file_path: str, column: str, title: str, x_label: str
) -> str:
    """Generates and saves a static Seaborn histogram."""
    try:
        df = load_data_safely(data_file_path)
        if column not in df.columns:
            return f"Error: Column '{column}' not found in data."

        plt.figure(figsize=(10, 6))
        sns.histplot(df[column], kde=True)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel("Frequency")

        plot_path = get_plot_path(data_file_path, f"hist_{column}", ext=".png")
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close()

        code = _generate_static_code_snippet(
            f"sns.histplot(df['{column}'], kde=True)\n"
            f"plt.title('{title}')\n"
            f"plt.xlabel('{x_label}')\n"
            "plt.ylabel('Frequency')",
            data_file_path=data_file_path,
        )
        r_snippet = r_code.build(
            r_code.histogram,
            data_file_path,
            column,
            title,
            None,
            pd.api.types.is_numeric_dtype(df[column]),
            interactive=False,
            x_label=x_label,
        )
        return format_plot_output(plot_path, code, r_snippet)
    except Exception as e:
        return f"Error plotting static histogram: {str(e)}"


def plot_static_scatterplot_impl(
    data_file_path: str,
    x_column: str,
    y_column: str,
    title: str,
    x_label: str,
    y_label: str,
    hue_column: str | None = None,
) -> str:
    """Generates and saves a static Seaborn scatter plot."""
    try:
        df = load_data_safely(data_file_path)
        if x_column not in df.columns or y_column not in df.columns:
            return f"Error: Columns '{x_column}' or '{y_column}' not found."

        if hue_column and hue_column not in df.columns:
            hue_column = None

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=x_column, y=y_column, hue=hue_column)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plot_path = get_plot_path(
            data_file_path, f"scatter_{x_column}_vs_{y_column}", ext=".png"
        )
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close()

        hue_arg = f", hue='{hue_column}'" if hue_column else ""
        code = _generate_static_code_snippet(
            f"sns.scatterplot(data=df, x='{x_column}', "
            f"y='{y_column}'{hue_arg})\n"
            f"plt.title('{title}')\n"
            f"plt.xlabel('{x_label}')\n"
            f"plt.ylabel('{y_label}')",
            data_file_path=data_file_path,
        )
        r_snippet = r_code.build(
            r_code.scatter,
            data_file_path,
            x_column,
            y_column,
            title,
            hue_column,
            bool(hue_column) and pd.api.types.is_numeric_dtype(df[hue_column]),
            interactive=False,
            x_label=x_label,
            y_label=y_label,
        )
        return format_plot_output(plot_path, code, r_snippet)
    except Exception as e:
        return f"Error plotting static scatterplot: {str(e)}"


def plot_static_boxplot_impl(
    data_file_path: str,
    x_column: str,
    y_column: str,
    title: str,
    x_label: str,
    y_label: str,
) -> str:
    """Generates and saves a static Seaborn box plot."""
    try:
        df = load_data_safely(data_file_path)
        if x_column not in df.columns or y_column not in df.columns:
            return f"Error: Columns '{x_column}' or '{y_column}' not found."

        plt.figure(figsize=(12, 7))
        sns.boxplot(data=df, x=x_column, y=y_column)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if not ptypes.is_numeric_dtype(df[x_column]):
            plt.xticks(rotation=45)

        plot_path = get_plot_path(
            data_file_path, f"boxplot_{y_column}_by_{x_column}", ext=".png"
        )
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close()

        code = _generate_static_code_snippet(
            f"sns.boxplot(data=df, x='{x_column}', y='{y_column}')\n"
            f"plt.title('{title}')\n"
            f"plt.xlabel('{x_label}')\n"
            f"plt.ylabel('{y_label}')\n"
            "plt.xticks(rotation=45)",
            data_file_path=data_file_path,
        )
        r_snippet = r_code.build(
            r_code.boxplot,
            data_file_path,
            x_column,
            y_column,
            title,
            None,
            interactive=False,
            x_label=x_label,
            y_label=y_label,
        )
        return format_plot_output(plot_path, code, r_snippet)
    except Exception as e:
        return f"Error plotting static boxplot: {str(e)}"


def generate_custom_static_plot_impl(
    data_file_path: str, python_code: str, plot_filename_keyword: str
) -> str:
    """Run model-written Python code that draws a Matplotlib/Seaborn chart.

    The LLM may either assign its figure to `fig`, or just call
    seaborn/matplotlib directly; we save the current active figure in either
    case.
    """
    columns: list[str] = []
    clean_code = ""
    try:
        df = load_data_safely(data_file_path)
        columns = [str(c) for c in df.columns]

        # Pass as a single dict (used as both globals and locals) so that
        # nested scopes like list comprehensions can also resolve `df`, `sns`,
        # etc. Using exec(code, {}, locals) would hide injected names inside
        # comprehensions and function defs due to Python 3's exec scoping
        # rules.
        local_scope = {
            "pd": pd,
            "np": np,
            "sns": sns,
            "plt": plt,
            "WordCloud": WordCloud,
            # A copy, because load_data_safely returns a cached DataFrame
            # shared with every later tool call in this server process.
            # Generated code such as df.dropna(inplace=True) must not change
            # the data for them.
            "df": df.copy(),
            # data_file_path is intentionally NOT exposed: the model should use
            # df directly and must not call pd.read_csv() or reference file
            # paths.
        }

        clean_code = (
            python_code.replace("```python", "").replace("```", "").strip()
        )
        # Strip plt.show() / fig.show() calls before exec — on non-interactive
        # backends (Agg on HPC), plt.show() clears the active figure, making
        # it impossible to retrieve and save it afterwards.
        clean_code = _strip_show_calls(clean_code)

        # Close any stale figures from previous runs in this process.
        plt.close("all")

        run_generated_code(clean_code, local_scope, CUSTOM_CODE_TIMEOUT)

        # Prefer an explicit `fig` if the LLM created one; otherwise use the
        # currently active matplotlib figure.
        fig = local_scope.get("fig")
        if fig is None:
            if not plt.get_fignums():
                return (
                    "Error: Your code did not produce any matplotlib figure."
                )
            fig = plt.gcf()

        plot_path = get_plot_path(
            data_file_path,
            f"custom_static_{plot_filename_keyword}",
            ext=".png",
        )
        fig.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close("all")

        full_user_code = _generate_static_code_snippet(
            clean_code, data_file_path=data_file_path
        )

        return f"{plot_path}|||{full_user_code}"

    except SystemExit:
        # exit()/sys.exit() in generated code would otherwise shut down the MCP
        # server that every worker of the analysis shares.
        plt.close("all")
        return (
            "Error: Your code must not call exit() or sys.exit(). Remove that "
            "call."
        )
    except Exception as e:
        plt.close("all")
        detail = describe_code_error(e, clean_code, columns)
        return f"Error executing custom static plot code: {detail}"


def plot_static_lineplot_impl(
    data_file_path: str,
    x_column: str,
    y_column: str,
    title: str,
    x_label: str,
    y_label: str,
    hue_column: str | None = None,
) -> str:
    """Generates and saves a static Seaborn line plot."""
    try:
        df = load_data_safely(data_file_path)
        if x_column not in df.columns or y_column not in df.columns:
            return f"Error: Columns '{x_column}' or '{y_column}' not found."

        if hue_column and hue_column not in df.columns:
            hue_column = None

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x=x_column, y=y_column, hue=hue_column)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if not ptypes.is_numeric_dtype(df[x_column]):
            plt.xticks(rotation=45)

        plot_path = get_plot_path(
            data_file_path, f"lineplot_{y_column}_over_{x_column}", ext=".png"
        )
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close()

        hue_arg = f", hue='{hue_column}'" if hue_column else ""
        code = _generate_static_code_snippet(
            f"sns.lineplot(data=df, x='{x_column}', y='{y_column}'{hue_arg})\n"
            f"plt.title('{title}')\n"
            f"plt.xlabel('{x_label}')\n"
            f"plt.ylabel('{y_label}')\n"
            "plt.xticks(rotation=45)",
            data_file_path=data_file_path,
        )
        r_snippet = r_code.build(
            r_code.lineplot,
            data_file_path,
            x_column,
            y_column,
            title,
            hue_column,
            interactive=False,
            x_label=x_label,
            y_label=y_label,
        )
        return format_plot_output(plot_path, code, r_snippet)
    except Exception as e:
        return f"Error plotting static lineplot: {str(e)}"


def plot_static_barchart_impl(
    data_file_path: str,
    x_column: str,
    y_column: str,
    title: str,
    x_label: str,
    y_label: str,
    hue_column: str | None = None,
    aggregation: str = "mean",
) -> str:
    """Generates and saves a static Seaborn bar chart.

    Args:
        data_file_path: The dataset file.
        x_column: The column on the x axis.
        y_column: The column on the y axis.
        title: The chart title.
        x_label: The x-axis label.
        y_label: The y-axis label.
        hue_column: Optional column to split the bars by colour.
        aggregation: How to aggregate y values per x category ('mean', 'sum',
            'count', 'median').
    """
    try:
        df = load_data_safely(data_file_path)
        if x_column not in df.columns or y_column not in df.columns:
            return f"Error: Columns '{x_column}' or '{y_column}' not found."

        if hue_column and hue_column not in df.columns:
            hue_column = None

        valid_aggs = {"mean", "sum", "count", "median"}
        if aggregation not in valid_aggs:
            aggregation = "mean"

        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=df,
            x=x_column,
            y=y_column,
            hue=hue_column,
            estimator=aggregation,
            errorbar="sd" if aggregation in {"mean", "median"} else None,
        )
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if not ptypes.is_numeric_dtype(df[x_column]):
            plt.xticks(rotation=45, ha="right")

        plot_path = get_plot_path(
            data_file_path, f"bar_{y_column}_by_{x_column}", ext=".png"
        )
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close()

        hue_arg = f", hue='{hue_column}'" if hue_column else ""
        code = _generate_static_code_snippet(
            f"sns.barplot(data=df, x='{x_column}', y='{y_column}'{hue_arg},\n"
            f"    estimator='{aggregation}', errorbar='sd')\n"
            f"plt.title('{title}')\n"
            f"plt.xlabel('{x_label}')\n"
            f"plt.ylabel('{y_label}')\n"
            "plt.xticks(rotation=45, ha='right')",
            data_file_path=data_file_path,
        )
        r_snippet = r_code.build(
            r_code.barchart_static,
            data_file_path,
            x_column,
            y_column,
            title,
            x_label,
            y_label,
            hue_column,
            aggregation,
        )
        return format_plot_output(plot_path, code, r_snippet)
    except Exception as e:
        return f"Error plotting static bar chart: {str(e)}"


def plot_static_wordcloud_impl(
    data_file_path: str,
    text_column: str,
    title: str = "Word Cloud",
    extra_stopwords: str | None = None,
) -> str:
    """Generates and saves a static Word Cloud from a text column.

    Args:
        data_file_path: The dataset file.
        text_column: The column holding the text.
        title: The chart title.
        extra_stopwords: Optional comma-separated list of additional words to
            exclude (e.g., "the,and,said"). Combined with the built-in English
            stopword list.
    """
    try:
        df = load_data_safely(data_file_path)
        if text_column not in df.columns:
            return f"Error: Column '{text_column}' not found in data."

        text_data = " ".join(df[text_column].dropna().astype(str))

        if not text_data.strip():
            return (
                "Error: The specified text column is empty or contains only "
                "null values."
            )

        stopword_set = set(STOPWORDS)
        if extra_stopwords:
            stopword_set.update(
                w.strip().lower()
                for w in extra_stopwords.split(",")
                if w.strip()
            )

        wordcloud = WordCloud(
            width=1200,
            height=600,
            background_color="white",
            colormap="viridis",
            max_words=200,
            stopwords=stopword_set,
        ).generate(text_data)

        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.title(title, fontsize=16, pad=20)
        plt.axis("off")

        plot_path = get_plot_path(
            data_file_path, f"wordcloud_{text_column}", ext=".png"
        )
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close()

        extra_code = ""
        if extra_stopwords:
            extra_code = (
                f"from wordcloud import STOPWORDS\n"
                "stopwords = set(STOPWORDS) | "
                f"{{{', '.join(repr(w.strip()) for w in extra_stopwords.split(',') if w.strip())}}}\n"  # noqa: E501
            )
        code = _generate_static_code_snippet(
            "from wordcloud import "
            f"WordCloud{''.join(['', chr(10) + extra_code] if extra_code else [''])}\n"  # noqa: E501
            f"text_data = ' '.join(df['{text_column}'].dropna().astype(str))\n"
            "wordcloud = WordCloud(width=1200, height=600, "
            "background_color='white',\n"
            "    colormap='viridis', "
            f"stopwords={'stopwords' if extra_stopwords else 'None'})"
            ".generate(text_data)\n"
            "plt.imshow(wordcloud, interpolation='bilinear')\n"
            f"plt.title('{title}', fontsize=16, pad=20)\n"
            "plt.axis('off')",
            data_file_path=data_file_path,
        )
        r_snippet = r_code.build(
            r_code.wordcloud,
            data_file_path,
            text_column,
            title,
            split_comma_list(extra_stopwords),
        )
        return format_plot_output(plot_path, code, r_snippet)
    except Exception as e:
        return f"Error plotting word cloud: {str(e)}"


_PAIRPLOT_MAX_ROWS = (
    5_000  # sns.pairplot is O(n²) — cap to avoid hangs on large datasets
)


def plot_static_pairplot_impl(
    data_file_path: str,
    columns: str,
    title: str,
    hue_column: str | None = None,
) -> str:
    """Generate a Seaborn pair plot (scatter matrix) of numeric columns.

    Args:
        data_file_path: The dataset file.
        columns: Comma-separated list of numeric column names to include.
        title: The chart title.
        hue_column: Optional categorical column to colour the points by (e.g.
            'diagnosis').
    """
    try:
        df = load_data_safely(data_file_path)

        col_list = [c.strip() for c in columns.split(",") if c.strip()]
        missing = [c for c in col_list if c not in df.columns]
        if missing:
            return f"Error: columns not found in data: {', '.join(missing)}"

        if hue_column and hue_column not in df.columns:
            return f"Error: hue_column '{hue_column}' not found in data."

        plot_cols = col_list + ([hue_column] if hue_column else [])
        plot_df = df[plot_cols].dropna()

        # Cap rows to prevent multi-minute hangs on large datasets.
        sampled = len(plot_df) > _PAIRPLOT_MAX_ROWS
        if sampled:
            plot_df = plot_df.sample(_PAIRPLOT_MAX_ROWS, random_state=42)

        display_title = title
        if sampled:
            display_title = f"{title} (sampled {_PAIRPLOT_MAX_ROWS:,} rows)"

        sns.set_theme(style="ticks")
        g = sns.pairplot(
            plot_df,
            hue=hue_column,
            vars=col_list,
            palette="husl",
            diag_kind="kde",
            plot_kws={"alpha": 0.6},
        )
        if display_title:
            g.fig.suptitle(display_title, y=1.02, fontsize=14)

        safe_name = "_".join(col_list[:3])
        plot_path = get_plot_path(
            data_file_path, f"pairplot_{safe_name}", ext=".png"
        )
        g.fig.savefig(plot_path, bbox_inches="tight", dpi=150)
        plt.close("all")

        hue_code = f", hue='{hue_column}'" if hue_column else ""
        sample_code = (
            (
                f"if len(plot_df) > {_PAIRPLOT_MAX_ROWS}:\n"
                f"    plot_df = plot_df.sample({_PAIRPLOT_MAX_ROWS}, "
                "random_state=42)\n"
            )
            if sampled
            else ""
        )
        code_body = (
            f"import seaborn as sns\n"
            f"cols = {col_list!r}\n"
            "plot_df = "
            f"df[cols{' + [' + repr(hue_column) + ']' if hue_column else ''}].dropna()\n"  # noqa: E501
            + sample_code
            + f"sns.set_theme(style='ticks')\n"
            f"g = sns.pairplot(plot_df, vars=cols{hue_code}, palette='husl', "
            f"diag_kind='kde', plot_kws={{'alpha': 0.6}})\n"
            f"g.fig.suptitle('{display_title}', y=1.02, fontsize=14)\n"
        )
        code = _generate_static_code_snippet(
            code_body, grid_figure=True, data_file_path=data_file_path
        )
        r_snippet = r_code.build(
            r_code.pair_plot,
            data_file_path,
            col_list,
            title,
            hue_column,
            interactive=False,
            sample_rows=_PAIRPLOT_MAX_ROWS if sampled else None,
        )
        return format_plot_output(plot_path, code, r_snippet)
    except Exception as e:
        return f"Error generating pair plot: {str(e)}"


def plot_static_correlation_heatmap_impl(
    data_file_path: str,
    title: str,
    method: str = "pearson",
    column_filter: str | None = None,
) -> str:
    """Generates a publication-ready Seaborn correlation heatmap.

    Args:
        data_file_path: The dataset file.
        title: The chart title.
        method: The correlation method, such as 'pearson' or 'spearman'.
        column_filter: Optional comma-separated column names or glob-style
            suffix (e.g. "_mean" to select only columns ending in _mean). When
            omitted, all numeric columns are used.
    """
    try:
        df = load_data_safely(data_file_path)
        numeric_df = df.select_dtypes(include="number")

        if column_filter:
            filters = [
                f.strip() for f in column_filter.split(",") if f.strip()
            ]
            # Support either exact names or suffix patterns like "_mean"
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

        corr_matrix = numeric_df.corr(method=method)
        n_cols = corr_matrix.shape[1]
        fig_size = max(8, n_cols)

        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title(title, fontsize=14, pad=15)

        safe_filter = (
            column_filter.replace(",", "_").replace(" ", "")
            if column_filter
            else "all"
        )
        plot_path = get_plot_path(
            data_file_path,
            f"static_corr_heatmap_{safe_filter}_{method}",
            ext=".png",
        )
        fig.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close(fig)

        # Build snippet that reflects the actual column selection used.
        if column_filter:
            filters_repr = repr(
                [f.strip() for f in column_filter.split(",") if f.strip()]
            )
            filter_snippet = (
                f"filters = {filters_repr}\n"
                f"numeric_df = df.select_dtypes(include='number')\n"
                f"numeric_df = numeric_df[[c for c in numeric_df.columns "
                f"if c in filters or any(c.endswith(f) for f in filters)]]\n"
                f"corr_matrix = numeric_df.corr(method='{method}')\n"
            )
        else:
            filter_snippet = (
                "corr_matrix = "
                f"df.select_dtypes(include='number').corr(method='{method}')\n"
            )

        code = _generate_static_code_snippet(
            filter_snippet
            + f"fig, ax = plt.subplots(figsize=({fig_size}, {fig_size}))\n"
            + "sns.heatmap(corr_matrix, annot=True, fmt='.2f', "
            "cmap='coolwarm',\n"
            + "    center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, "
            "ax=ax)\n" + f"ax.set_title('{title}', fontsize=14, pad=15)",
            grid_figure=True,
            data_file_path=data_file_path,
        )
        r_snippet = r_code.build(
            r_code.correlation_heatmap,
            data_file_path,
            title,
            method,
            split_comma_list(column_filter),
            interactive=False,
        )
        return format_plot_output(plot_path, code, r_snippet)
    except Exception as e:
        return f"Error plotting static correlation heatmap: {str(e)}"
