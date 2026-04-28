"""
Static Plotting Module for the AI Visualization Engine.
Generates publication-ready Matplotlib and Seaborn charts (.png).
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

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
    return (
        "import matplotlib.pyplot as plt\n"
        "import pandas as pd\n"
        "import seaborn as sns\n\n"
        "# Load Data\n"
        "df = pd.read_csv('your_data.csv')\n\n"
        "# Generate Plot\n"
        "plt.figure(figsize=(10, 6))\n"
        f"{plot_code}\n"
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
    """
    try:
        df = load_data_safely(data_file_path)

        local_scope = {
            "pd": pd,
            "sns": sns,
            "plt": plt,
            "df": df,
            "data_file_path": data_file_path,
        }

        clean_code = python_code.replace("```python", "").replace("```", "").strip()

        plt.clf()
        plt.figure(figsize=(10, 6))

        exec(clean_code, {}, local_scope)

        plot_path = get_plot_path(
            data_file_path, f"custom_static_{plot_filename_keyword}", ext=".png"
        )
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close()

        full_user_code = _generate_static_code_snippet(clean_code)

        return f"{plot_path}|||{full_user_code}"

    except Exception as e:
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
    data_file_path: str, text_column: str, title: str = "Word Cloud"
) -> str:
    """
    Generates and saves a static Word Cloud from a text column.
    """
    try:
        df = load_data_safely(data_file_path)
        if text_column not in df.columns:
            return f"Error: Column '{text_column}' not found in data."

        # Combine all text in the column into a single string, dropping nulls
        text_data = " ".join(df[text_column].dropna().astype(str))
        
        if not text_data.strip():
            return "Error: The specified text column is empty or contains only null values."

        # Generate the word cloud
        wordcloud = WordCloud(
            width=1200, 
            height=600, 
            background_color="white", 
            colormap="viridis",
            max_words=200
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

        code = _generate_static_code_snippet(
            "from wordcloud import WordCloud\n\n"
            f"text_data = ' '.join(df['{text_column}'].dropna().astype(str))\n"
            "wordcloud = WordCloud(width=1200, height=600, background_color='white', colormap='viridis').generate(text_data)\n"
            "plt.imshow(wordcloud, interpolation='bilinear')\n"
            f"plt.title('{title}', fontsize=16, pad=20)\n"
            "plt.axis('off')"
        )
        return f"{plot_path}|||{code}"
    except Exception as e:
        return f"Error plotting word cloud: {str(e)}"