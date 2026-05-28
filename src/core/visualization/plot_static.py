"""
Static Plotting Module for the AI Visualization Engine.
Generates publication-ready Matplotlib and Seaborn charts (.png).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import seaborn as sns
from wordcloud import STOPWORDS, WordCloud

from core.visualization.viz_utils import get_plot_path, load_data_safely


def _generate_static_code_snippet(plot_code: str) -> str:
    """
    Format the raw plot generation code into a complete, runnable script string
    for Matplotlib and Seaborn.

    Args:
        plot_code: The core logic used to generate the plot.

    Returns:
        A formatted Python script string including imports and data loading.
    """
    from core.visualization.viz_utils import _strip_show_calls
    clean = _strip_show_calls(plot_code)
    return (
        "import matplotlib.pyplot as plt\n"
        "import pandas as pd\n"
        "import seaborn as sns\n\n"
        "# Load Data\n"
        "df = pd.read_csv('your_data.csv')\n\n"
        "# Generate Plot\n"
        "plt.figure(figsize=(10, 6))\n"
        f"{clean}\n"
        "plt.show()"
    )


def plot_static_histogram_impl(
    data_file_path: str, column: str, title: str, x_label: str
) -> str:
    """
    Generates and saves a static Seaborn histogram.
    """
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
            "plt.ylabel('Frequency')"
        )
        return f"{plot_path}|||{code}"
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
    """
    Generates and saves a static Seaborn scatter plot.
    """
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
            f"sns.scatterplot(data=df, x='{x_column}', y='{y_column}'{hue_arg})\n"
            f"plt.title('{title}')\n"
            f"plt.xlabel('{x_label}')\n"
            f"plt.ylabel('{y_label}')"
        )
        return f"{plot_path}|||{code}"
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
    """
    Generates and saves a static Seaborn box plot.
    """
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
            "plt.xticks(rotation=45)"
        )
        return f"{plot_path}|||{code}"
    except Exception as e:
        return f"Error plotting static boxplot: {str(e)}"


def generate_custom_static_plot_impl(
    data_file_path: str, python_code: str, plot_filename_keyword: str
) -> str:
    """
    Executes custom Python code to generate complex Matplotlib/Seaborn charts.
    The LLM may either assign its figure to `fig`, or just call seaborn/matplotlib
    directly; we save the current active figure in either case.
    """
    try:
        df = load_data_safely(data_file_path)

        # Pass as a single dict (used as both globals and locals) so that nested
        # scopes like list comprehensions can also resolve `df`, `sns`, etc.
        # Using exec(code, {}, locals) would hide injected names inside comprehensions
        # and function defs due to Python 3's exec scoping rules.
        local_scope = {
            "pd": pd,
            "np": np,
            "sns": sns,
            "plt": plt,
            "WordCloud": WordCloud,
            "df": df,
            # data_file_path is intentionally NOT exposed: the model should use df
            # directly and must not call pd.read_csv() or reference file paths.
        }

        clean_code = python_code.replace("```python", "").replace("```", "").strip()

        # Close any stale figures from previous runs in this process.
        plt.close("all")

        exec(clean_code, local_scope)

        # Prefer an explicit `fig` if the LLM created one; otherwise use the
        # currently active matplotlib figure.
        fig = local_scope.get("fig")
        if fig is None:
            if not plt.get_fignums():
                return "Error: Your code did not produce any matplotlib figure."
            fig = plt.gcf()

        plot_path = get_plot_path(
            data_file_path, f"custom_static_{plot_filename_keyword}", ext=".png"
        )
        fig.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close("all")

        full_user_code = _generate_static_code_snippet(clean_code)

        return f"{plot_path}|||{full_user_code}"

    except Exception as e:
        plt.close("all")
        return f"Error executing custom static plot code: {str(e)}"
    

def plot_static_lineplot_impl(
    data_file_path: str,
    x_column: str,
    y_column: str,
    title: str,
    x_label: str,
    y_label: str,
    hue_column: str | None = None,
) -> str:
    """
    Generates and saves a static Seaborn line plot.
    """
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
            "plt.xticks(rotation=45)"
        )
        return f"{plot_path}|||{code}"
    except Exception as e:
        return f"Error plotting static lineplot: {str(e)}"
    

def plot_static_wordcloud_impl(
    data_file_path: str,
    text_column: str,
    title: str = "Word Cloud",
    extra_stopwords: str | None = None,
) -> str:
    """
    Generates and saves a static Word Cloud from a text column.

    Args:
        extra_stopwords: Optional comma-separated list of additional words to exclude
            (e.g., "the,and,said"). Combined with the built-in English stopword list.
    """
    try:
        df = load_data_safely(data_file_path)
        if text_column not in df.columns:
            return f"Error: Column '{text_column}' not found in data."

        text_data = " ".join(df[text_column].dropna().astype(str))
        
        if not text_data.strip():
            return "Error: The specified text column is empty or contains only null values."

        stopword_set = set(STOPWORDS)
        if extra_stopwords:
            stopword_set.update(w.strip().lower() for w in extra_stopwords.split(",") if w.strip())

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

        stopwords_arg = f", extra_stopwords='{extra_stopwords}'" if extra_stopwords else ""
        extra_code = ""
        if extra_stopwords:
            extra_code = (
                f"from wordcloud import STOPWORDS\n"
                f"stopwords = set(STOPWORDS) | {{{', '.join(repr(w.strip()) for w in extra_stopwords.split(',') if w.strip())}}}\n"
            )
        code = _generate_static_code_snippet(
            f"from wordcloud import WordCloud{''.join(['', chr(10) + extra_code] if extra_code else [''])}\n"
            f"text_data = ' '.join(df['{text_column}'].dropna().astype(str))\n"
            f"wordcloud = WordCloud(width=1200, height=600, background_color='white',\n"
            f"    colormap='viridis', stopwords={'stopwords' if extra_stopwords else 'None'}).generate(text_data)\n"
            "plt.imshow(wordcloud, interpolation='bilinear')\n"
            f"plt.title('{title}', fontsize=16, pad=20)\n"
            "plt.axis('off')"
        )
        return f"{plot_path}|||{code}"
    except Exception as e:
        return f"Error plotting word cloud: {str(e)}"


def plot_static_correlation_heatmap_impl(
    data_file_path: str, title: str, method: str = "pearson"
) -> str:
    """
    Generates a publication-ready Seaborn correlation heatmap for all numeric columns.
    """
    try:
        df = load_data_safely(data_file_path)
        numeric_df = df.select_dtypes(include="number")

        if numeric_df.shape[1] < 2:
            return "Error: Need at least 2 numeric columns to generate a correlation heatmap."

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

        plot_path = get_plot_path(
            data_file_path, f"static_corr_heatmap_{method}", ext=".png"
        )
        fig.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close(fig)

        code = _generate_static_code_snippet(
            f"corr_matrix = df.select_dtypes(include='number').corr(method='{method}')\n"
            f"sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',\n"
            f"    center=0, vmin=-1, vmax=1, square=True, linewidths=0.5)\n"
            f"plt.title('{title}', fontsize=14, pad=15)"
        )
        return f"{plot_path}|||{code}"
    except Exception as e:
        return f"Error plotting static correlation heatmap: {str(e)}"