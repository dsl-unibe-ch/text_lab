# Knowledge Graph

Knowledge graphs built from collections of scientific papers, using Grobid for
parsing and an LLM for topic extraction.

**Status:** moved, not refactored yet. The modules were moved here from
`src/core` with only import and path updates; the package docstring in
`__init__.py` says what each one does.

- Modules: `kg_engine.py`
- UI: `Knowledge_Graph.py` in `src/textlab/ui/streamlit/pages/`
- Tests: none yet

Once the feature is refactored, this file documents its pipeline, public API,
files written to disk, configuration and tests, as described in the developer
guide ([architecture](../../../../docs/dev/architecture.md#feature-packages)).
