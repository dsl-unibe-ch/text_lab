"""Backend for the Visualization feature: charts and statistics by agents.

Interfaces use :mod:`.service`; runs go through :mod:`.runs`. The modules
behind it:

- ``viz_agent``: the supervisor and specialist agents.
- ``viz_config``: prompts, the tools of each agent and limits.
- ``mcp_server``: the tools, served over MCP by a process of their own
  (``python -m textlab.features.visualization.mcp_server``).
- ``plot_data``, ``plot_interactive``, ``plot_static``, ``stats_analysis``:
  the tools' implementations; ``r_code``: their R equivalents.
- ``models``, ``preview``, ``reports``, ``viz_utils``: requests and
  results, previews, downloads, data loading.

See ``README.md`` in this folder for the pipeline and the files written.
"""
