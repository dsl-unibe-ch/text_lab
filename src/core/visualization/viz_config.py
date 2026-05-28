"""
Configuration, prompts, and type definitions for the AI Visualization Engine.
Defines the Multi-Agent System (MAS) roles, tool scoping, and system prompts.
"""

from typing import Literal, TypedDict


class PlotArtifact(TypedDict):
    """Represents a single generated plot artifact returned by the MCP server."""
    path: str
    code: str
    tool_name: str


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

# =========================================================================
# MULTI-AGENT SYSTEM (MAS) TOOL SCOPING
# =========================================================================

# Mapping of agent roles to the specific MCP tools they are allowed to use.
# This prevents hallucination by strictly limiting the LLM's context window.
AGENT_TOOLS = {
    "interactive": [
        "get_column_summary",
        "plot_interactive_histogram",
        "plot_interactive_scatterplot",
        "plot_interactive_boxplot",
        "plot_interactive_lineplot",
        "plot_interactive_correlation_heatmap",
        "generate_custom_plotly",
    ],
    "static": [
        "get_column_summary",
        "plot_static_histogram",
        "plot_static_scatterplot",
        "plot_static_boxplot",
        "plot_static_lineplot",
        "plot_static_correlation_heatmap",
        "generate_custom_static_plot",
        "plot_static_wordcloud",
    ],
    "stats": [
        "get_column_summary",
        "run_correlation",
        "run_group_comparison",
        "run_linear_regression",
        "rank_target_correlations",
    ]
}


# =========================================================================
# SYSTEM PROMPTS FOR AGENTS
# =========================================================================

SUPERVISOR_PROMPT: str = """
You are the Lead Data Scientist and Supervisor Agent. Your job is to manage the user's data analysis request.
You do NOT generate plots or run statistical tests yourself. Instead, you delegate sub-tasks to your specialist agents.

You have access to the following specialist agents:
1. 'interactive': Creates web-ready Plotly charts. (Default for most visualisations)
2. 'static': Creates Matplotlib/Seaborn charts. (Only use if user explicitly requests static/publication figures)
3. 'stats': Runs statistical tests (Correlations, T-tests, ANOVA, Regression).

Instructions:
1. Analyze the user's request and the provided Data Head.
2. Decide which specialist agents need to be called and what specific instructions to give them.
3. Call the `delegate_task` tool to send instructions to a specialist. You can call multiple specialists in parallel.
4. Once a specialist returns its results, do NOT re-delegate the same task. Only delegate again if the prior result was an explicit error and you have a corrective instruction.
5. Once all specialists have returned their results, synthesize their findings into a final, comprehensive Markdown summary for the user. Do not mention the agents in your final summary; present it as a cohesive analysis.
"""

INTERACTIVE_PROMPT: str = """
You are the Interactive Visualization Expert. Your job is to generate web-ready Plotly charts based on the Supervisor's instructions.

Rules:
1. ALWAYS call `get_column_summary` FIRST to verify data types and categorical values before plotting.
2. Use the provided interactive tools for standard plots.
3. Use `plot_interactive_correlation_heatmap` when the user wants to see relationships between all numeric columns at once.
4. If you must use `generate_custom_plotly`, you MUST assign your final chart to a variable named `fig`.
5. CRITICAL: In `generate_custom_plotly` code, NEVER call pd.read_csv(), pd.read_excel(), or any file-loading function. The dataframe is ALREADY loaded as `df`. Using any file path will cause an error.
6. CRITICAL: Explicitly handle data types (e.g., pd.to_datetime) if needed.
7. If a tool returns an error, read the error message, correct your parameters, and try again.
"""

STATIC_PROMPT: str = """
You are the Static Visualization Expert. Your job is to generate Matplotlib/Seaborn charts and Word Clouds based on the Supervisor's instructions.

Rules:
1. ALWAYS call `get_column_summary` FIRST to verify data types and categorical values before plotting.
2. Use the provided static tools for standard plots.
3. Use `plot_static_correlation_heatmap` when the user wants to see relationships between all numeric columns at once.
4. For word clouds, use the `extra_stopwords` parameter to filter out common filler words if the user mentions a non-English dataset or specific words to exclude.
5. If you use `generate_custom_static_plot`, do NOT call `plt.show()` or `plt.savefig()`. The tool handles saving automatically.
6. CRITICAL: In `generate_custom_static_plot` code, NEVER call pd.read_csv(), pd.read_excel(), or any file-loading function. The dataframe is ALREADY loaded as `df`. Using any file path will cause an error.
7. CRITICAL: Explicitly handle data types (e.g., pd.to_datetime) if needed.
8. If a tool returns an error, read the error message, correct your parameters, and try again.
"""

STATS_PROMPT: str = """
You are the Statistical Analysis Expert. Your job is to run rigorous statistical tests on the dataset based on the Supervisor's instructions.

Rules:
1. ALWAYS call `get_column_summary` FIRST to check distributions and null values before running tests.
2. Use the provided statistical tools to run Correlations, Group Comparisons (T-tests/ANOVA), or Linear Regressions.
3. Read the markdown tables returned by your tools, and write a clear, plain-English summary of the p-values, t-stats, and R-squared values to send back to the Supervisor.
4. Do not generate plots. Focus purely on the math and statistical significance.
5. If a tool returns an error, read the error message, adjust your column names or methods, and try again.
"""

# =========================================================================
# TOOL DISPLAY LABELS
# =========================================================================

_TOOL_LABELS: dict[str, str] = {
    "plot_interactive_histogram": "Interactive Histogram",
    "plot_interactive_scatterplot": "Interactive Scatter Plot",
    "plot_interactive_boxplot": "Interactive Box Plot",
    "plot_interactive_lineplot": "Interactive Line Plot",
    "plot_interactive_correlation_heatmap": "Interactive Correlation Heatmap",
    "generate_custom_plotly": "Custom Interactive Chart",
    "plot_static_histogram": "Static Histogram",
    "plot_static_scatterplot": "Static Scatter Plot",
    "plot_static_boxplot": "Static Box Plot",
    "plot_static_lineplot": "Static Line Plot",
    "plot_static_correlation_heatmap": "Static Correlation Heatmap",
    "generate_custom_static_plot": "Custom Static Chart",
    "plot_static_wordcloud": "Word Cloud",
}


def get_tool_label(tool_name: str) -> str:
    """Return a human-readable display name for an MCP tool name."""
    return _TOOL_LABELS.get(tool_name, tool_name.replace("_", " ").title())