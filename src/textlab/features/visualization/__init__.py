"""Backend for the Visualization feature.

Data visualization and statistics driven by an LLM agent through an internal
MCP server.

Moved from ``src/core`` with only import and path updates. The modules are
reorganized when the feature is refactored (see ``docs/dev/architecture.md``):

- ``viz_agent``: the agent that plans and runs an analysis.
- ``mcp_server``: MCP server exposing the plotting and statistics tools; runs
  as its own process.
- ``plot_data``: data exploration tools.
- ``plot_interactive``: interactive Plotly charts.
- ``plot_static``: static Matplotlib and Seaborn charts.
- ``stats_analysis``: statistical tests.
- ``r_code``: R equivalents of the tools.
- ``viz_config``: configuration, prompts and types.
- ``viz_utils``: data loading and file helpers.
"""
