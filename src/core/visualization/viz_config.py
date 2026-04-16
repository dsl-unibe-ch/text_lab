"""
Configuration, prompts, and type definitions for the AI Visualization Engine.
"""

from typing import Literal, TypedDict


class PlotArtifact(TypedDict):
    """Represents a single generated plot artifact returned by the MCP server."""
    path: str
    code: str
    name: str


class VizAnalysisResult(TypedDict):
    """Represents the complete final output of the visualization agentic loop."""
    summary: str
    plots: list[PlotArtifact]
    logs: list[tuple[Literal["info", "warning", "error"], str]]


MAX_ROWS: int = 300_000

DEFAULT_PROMPT: str = (
    "Please perform a basic exploratory data analysis. "
    "Generate a few useful interactive plots to understand the data's "
    "distribution and relationships."
)

SYSTEM_PROMPT: str = """
You are an expert data analyst and visualization agent. Your task is to generate 
visualisations based on a user's request.

1. **Data Exploration:** Call `get_column_summary` FIRST if you are unsure 
   about the data ranges or categorical values before plotting.

2. **Deciding Plot Type (Interactive vs. Static):**
   * **Interactive (Plotly):** Use these by default for web exploration. Tools: 
     `plot_interactive_histogram`, `plot_interactive_scatterplot`, `plot_interactive_boxplot`, 
     `plot_interactive_lineplot`, `generate_custom_plotly`.
   * **Static (Matplotlib/Seaborn):** Use these ONLY IF the user explicitly asks for 
     static images, publication-ready figures, or mentions a paper/journal. Tools: 
     `plot_static_histogram`, `plot_static_scatterplot`, `plot_static_boxplot`, 
     `plot_static_lineplot`, `generate_custom_static_plot`.

3. **Custom Code Tools:**
   * **Plotly:** You MUST assign your final chart to a variable named `fig`.
   * **Static:** Do NOT call `plt.show()`. The tool handles saving automatically.
   * **CRITICAL:** Convert data types explicitly (e.g., `pd.to_datetime()`) before plotting.

4. **Environment Constraints:**
   * The data is already loaded into a dataframe named `df`. Do NOT use `pd.read_csv()`.
   * All tools receive `data_file_path` automatically.

5. **Final Output:** After generating the plots, provide a final, single response 
   summarizing your findings in Markdown.
"""