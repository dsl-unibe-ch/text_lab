# Visualization

Data visualization and statistics driven by an LLM agent through an internal
MCP server.

**Status:** moved, not refactored yet. The modules were moved here from
`src/core` with only import and path updates; the package docstring in
`__init__.py` says what each one does.

- Modules: `mcp_server.py`, `plot_data.py`, `plot_interactive.py`, `plot_static.py`, `r_code.py`, `stats_analysis.py`, `viz_agent.py`, `viz_config.py`, `viz_utils.py`
- UI: `Visualize_Data.py` and the data analysis in `Chat.py`, both in
  `src/textlab/ui/streamlit/pages/`
- Tests: `tests/test_viz_agent.py`, `tests/test_viz_plots.py`

Once the feature is refactored, this file documents its pipeline, public API,
files written to disk, configuration and tests, as described in the developer
guide ([architecture](../../../../docs/dev/architecture.md#feature-packages)).
