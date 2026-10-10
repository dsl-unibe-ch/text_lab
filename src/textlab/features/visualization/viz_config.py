"""
Configuration, prompts, and type definitions for the AI Visualization Engine.
Defines the Multi-Agent System (MAS) roles, tool scoping, and system prompts.
"""

import os
from typing import Literal, TypedDict


class PlotArtifact(TypedDict):
    """Represents a single generated plot artifact returned by the MCP server."""
    path: str
    code: str
    r_code: str     # R equivalent of ``code``; empty unless R code was requested
    tool_name: str


class StatsArtifact(TypedDict):
    """Represents a single statistical analysis result returned by the stats agent."""
    title: str      # short human-readable label, e.g. "T-test: radius_mean by diagnosis"
    result: str     # the markdown table / summary text returned by the stats tool
    code: str       # the reproducible Python code snippet embedded in the result
    r_code: str     # R equivalent of ``code``; empty unless R code was requested


class VizAnalysisResult(TypedDict):
    """Represents the complete final output of the visualization agentic loop."""
    summary: str
    plots: list[PlotArtifact]
    stats: list[StatsArtifact]
    logs: list[tuple[Literal["info", "warning", "error"], str]]


MAX_ROWS: int = 300_000


def _env_int(name: str) -> int | None:
    """Return a positive integer from the environment, or None if unset/invalid."""
    try:
        value = int(os.environ.get(name, "0"))
    except ValueError:
        return None
    return value if value > 0 else None


# =========================================================================
# MODEL RUNTIME OPTIONS
# =========================================================================

# Low temperature makes tool selection and argument filling more deterministic,
# which matters most for the small models this app runs.
AGENT_TEMPERATURE: float = 0.1

# Context window for agent calls. Left unset by default because a num_ctx that
# differs from the one used elsewhere (e.g. the Chat page) makes Ollama reload
# the model on every switch. Set TEXTLAB_AGENT_NUM_CTX (e.g. 16384) when the
# Ollama server log reports "truncating input prompt" during an analysis.
AGENT_NUM_CTX: int | None = _env_int("TEXTLAB_AGENT_NUM_CTX")

# HTTP timeout (seconds) for each agent model call, so a request Ollama never
# answers fails the step instead of freezing the analysis. Generous enough to
# include loading a large model from shared storage on the first call.
AGENT_REQUEST_TIMEOUT: float = float(_env_int("TEXTLAB_AGENT_TIMEOUT") or 300)

# Limits for tool execution. Model-written plotting code is stopped inside the
# MCP server after CUSTOM_CODE_TIMEOUT seconds, so an endless loop cannot block
# the server all workers share. TOOL_CALL_TIMEOUT is the client-side safety net
# for any tool call and must stay above CUSTOM_CODE_TIMEOUT to leave time for
# saving the figure.
CUSTOM_CODE_TIMEOUT: float = 60
TOOL_CALL_TIMEOUT: float = 180

AGENT_OPTIONS: dict[str, float | int] = {"temperature": AGENT_TEMPERATURE}
if AGENT_NUM_CTX:
    AGENT_OPTIONS["num_ctx"] = AGENT_NUM_CTX

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
        "plot_interactive_histogram",
        "plot_interactive_scatterplot",
        "plot_interactive_boxplot",
        "plot_interactive_lineplot",
        "plot_interactive_barchart",
        "plot_interactive_scatter_matrix",
        "plot_interactive_correlation_heatmap",
        "generate_custom_plotly",
    ],
    "static": [
        "plot_static_histogram",
        "plot_static_scatterplot",
        "plot_static_boxplot",
        "plot_static_lineplot",
        "plot_static_barchart",
        "plot_static_pairplot",
        "plot_static_correlation_heatmap",
        "generate_custom_static_plot",
        "plot_static_wordcloud",
    ],
    "stats": [
        "run_correlation",
        "run_group_comparison",
        "run_association_test",
        "run_linear_regression",
        "run_logistic_regression",
        "rank_target_correlations",
    ]
}


# =========================================================================
# SYSTEM PROMPTS FOR AGENTS
# =========================================================================

SUPERVISOR_PROMPT: str = """
You are the Lead Data Scientist and Supervisor Agent. You plan the user's data analysis request and hand the work to specialist agents.
You do NOT generate plots or run statistical tests yourself.

You have access to the following specialist agents:
1. 'interactive': Creates web-ready Plotly charts. (Default for most visualisations)
2. 'static': Creates Matplotlib/Seaborn/WordCloud charts. (Only use if user explicitly requests static/publication figures or a word cloud)
3. 'stats': Runs pure statistical tests — correlations, group comparisons (t-test, ANOVA, Mann-Whitney, Kruskal-Wallis), associations between two categorical columns (chi-square, Fisher), linear and logistic regression. Returns numbers and tables ONLY. It cannot produce any visual output.

CRITICAL ROUTING RULES:
- Any task that must produce a visual output (chart, plot, image, word cloud, heatmap) MUST go to 'interactive' or 'static' — even if computing those visuals requires first running statistics internally.
- NEVER delegate a visualization request to 'stats'. The stats agent cannot create images.
- Word clouds, heatmaps, and pair plots are visualizations — always route them to 'static' (if static is requested) or 'interactive'.

PLANNING RULES:
1. Analyze the user's request and the dataset summary given below.
2. Respond with ONE JSON object with exactly these string fields: "interactive", "static", "stats" and "reply". You will not get another chance to delegate.
3. For EACH of the three specialists decide whether the request needs it. Write a short, self-contained instruction for every specialist that is needed, and leave the field empty ("") for the others.
   Name only the columns the task actually needs. Never list every column of the dataset, and skip identifier columns (e.g. 'id') and empty columns.
4. Requests often need several specialists at once. Examples:
   - "plots and statistical analysis" -> fill "interactive" AND "stats".
   - "an interactive plot and a static version for publication" -> fill "interactive" AND "static" with the same plots.
   - "a correlation heatmap and the strongest correlations" -> fill "interactive" AND "stats".
5. "reply" is only used when the request needs no plots and no statistics (e.g. a question about which columns exist): then leave the three specialist fields empty and answer the user in "reply" (Markdown). Otherwise leave "reply" empty.
6. NEVER include file paths, directory names, or storage locations in any text.
"""

SUMMARY_PROMPT: str = """
You are the Lead Data Scientist. Specialist agents have finished the user's data analysis request; their results are given below.

Write a concise, cohesive Markdown summary for the user:
1. Report the key statistical findings with their numbers (p-values, coefficients, R², correlations) and explain what they mean in plain English.
2. Briefly list the visualisations that were generated. They are displayed automatically below your summary.
3. If some steps could not be completed, say so briefly.
4. Only use numbers that appear in the results. NEVER invent statistics.
5. Do not mention the agents or tools. NEVER include file paths, directory names, or storage locations — all files are temporary.
"""

INTERACTIVE_PROMPT: str = """
You are the Interactive Visualization Expert. Your job is to generate web-ready Plotly charts based on the Supervisor's instructions.

The dataset schema is provided above. The data is already loaded — use this schema to select the correct column names.

Rules:
1. Use the provided interactive tools for standard plots. Do NOT call `get_all_columns_summary` — the schema is already given.
2. Use `plot_interactive_barchart` for bar/column charts. Choose the appropriate aggregation ('mean', 'sum', 'count', 'median').
3. Use `plot_interactive_scatter_matrix` for pair plots or multi-feature distribution charts.
4. Use `plot_interactive_correlation_heatmap` when the user wants to see relationships between numeric columns. Use `column_filter` (e.g. '_mean') to restrict to a column subset.
5. If you must use `generate_custom_plotly`, you MUST assign your final chart to a variable named `fig`.
6. CRITICAL: In `generate_custom_plotly` code, NEVER call pd.read_csv(), pd.read_excel(), or any file-loading function. The dataframe is ALREADY loaded as `df`. Using any file path will cause an error.
7. CRITICAL: Explicitly handle data types (e.g., pd.to_datetime) if needed.
8. Make ALL the plot tool calls the task needs in a single response. Once they succeed your work is finished.
9. If a tool returns an error, read the error message, correct your parameters, and try again. Do not regenerate plots that already succeeded.
"""

STATIC_PROMPT: str = """
You are the Static Visualization Expert. Your job is to generate Matplotlib/Seaborn charts and Word Clouds based on the Supervisor's instructions.

The dataset schema is provided above. The data is already loaded — use this schema to select the correct column names.

Rules:
1. Do NOT call `get_all_columns_summary` — the schema is already given above.
2. Use the provided static tools for standard plots.
3. Use `plot_static_barchart` for bar/column charts. Choose the appropriate aggregation ('mean', 'sum', 'count', 'median').
4. Use `plot_static_pairplot` for pair plots / scatter matrices. Pass only numeric column names in `columns` (comma-separated) and the optional categorical column in `hue_column` (e.g. 'diagnosis'). NEVER include string/categorical columns in the `columns` parameter.
5. Use `plot_static_correlation_heatmap` when the user wants to see relationships between numeric columns.
   - Use the `column_filter` parameter to restrict to a subset: pass a suffix like '_mean' to select all columns ending in _mean, or pass exact comma-separated column names.
   - Example: column_filter='_mean' selects all columns ending in _mean.
6. For word clouds weighted by correlation strength, use `generate_custom_static_plot` with code that:
   a. Computes correlations between columns and the target column.
   b. Uses the absolute correlation values as word frequencies for WordCloud.
   c. Do NOT call plt.show() or plt.savefig() — the tool handles saving automatically.
7. For other word clouds, use the `extra_stopwords` parameter to filter common filler words.
8. If you use `generate_custom_static_plot`, NEVER call `plt.show()` or `plt.savefig()` in the code. The tool handles saving automatically.
9. CRITICAL: In `generate_custom_static_plot` code, NEVER call pd.read_csv(), pd.read_excel(), or any file-loading function. The dataframe is ALREADY loaded as `df`. Using any file path will cause an error.
10. CRITICAL: Explicitly handle data types (e.g., pd.to_datetime) if needed.
11. Make ALL the plot tool calls the task needs in a single response. Once they succeed your work is finished.
12. If a tool returns an error, read the error message, correct your parameters, and try again. Do not regenerate plots that already succeeded.
"""

STATS_PROMPT: str = """
You are the Statistical Analysis Expert. Your job is to run rigorous statistical tests on the dataset using your tools.

The dataset schema is provided above. The data is already loaded — you do NOT need to load a file.

CRITICAL RULES — follow these exactly:
1. You MUST call the appropriate stats tool immediately. NEVER answer with numbers, p-values, or statistics from your own knowledge — always call the tool and return its output.
2. Choose the tool by the types of the columns involved:
   - numeric vs numeric: `run_correlation` (one pair) or `rank_target_correlations` (one target against all numeric columns).
   - numeric vs groups of a categorical column: `run_group_comparison` (t-test / ANOVA).
     Use method='nonparametric' for ordinal data (Likert scales, ratings, ranks), for small groups, or when a parametric run reports doubtful normality.
   - categorical vs categorical: `run_association_test` (chi-square / Fisher).
   - numeric outcome explained by predictors: `run_linear_regression`.
   - binary outcome (two values such as yes/no) explained by predictors: `run_logistic_regression`.
3. For both regression tools, the `predictor_cols` argument MUST be a JSON array, e.g. ["col1", "col2"].
4. Make ALL the tool calls the task needs in a single response.
5. After the tools return, you will be asked for a short plain-English interpretation of the key numbers (p-values, effect sizes, odds ratios, R²).
6. Do not generate plots. Focus purely on numbers and statistical significance.
7. If a tool returns an error, correct the column names or parameters, or switch to the tool the error suggests, and try again.
"""

# =========================================================================
# TOOL DISPLAY LABELS
# =========================================================================

_TOOL_LABELS: dict[str, str] = {
    "plot_interactive_histogram": "Interactive Histogram",
    "plot_interactive_scatterplot": "Interactive Scatter Plot",
    "plot_interactive_boxplot": "Interactive Box Plot",
    "plot_interactive_lineplot": "Interactive Line Plot",
    "plot_interactive_barchart": "Interactive Bar Chart",
    "plot_interactive_scatter_matrix": "Interactive Scatter Matrix",
    "plot_interactive_correlation_heatmap": "Interactive Correlation Heatmap",
    "generate_custom_plotly": "Custom Interactive Chart",
    "plot_static_histogram": "Static Histogram",
    "plot_static_scatterplot": "Static Scatter Plot",
    "plot_static_boxplot": "Static Box Plot",
    "plot_static_lineplot": "Static Line Plot",
    "plot_static_barchart": "Static Bar Chart",
    "plot_static_pairplot": "Static Pair Plot",
    "plot_static_correlation_heatmap": "Static Correlation Heatmap",
    "generate_custom_static_plot": "Custom Static Chart",
    "plot_static_wordcloud": "Word Cloud",
    "run_correlation": "Correlation Analysis",
    "run_group_comparison": "Group Comparison",
    "run_association_test": "Association Test (Chi-square / Fisher)",
    "run_logistic_regression": "Logistic Regression",
    "run_linear_regression": "Linear Regression (OLS)",
    "rank_target_correlations": "Feature Correlation Ranking",
}


def get_tool_label(tool_name: str) -> str:
    """Return a human-readable display name for an MCP tool name."""
    return _TOOL_LABELS.get(tool_name, tool_name.replace("_", " ").title())