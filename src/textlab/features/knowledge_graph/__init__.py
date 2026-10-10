"""Backend for the Knowledge Graph feature.

Knowledge graphs built from collections of scientific papers, using Grobid for
parsing and an LLM for topic extraction.

Moved from ``src/core`` with only import and path updates. The modules are
reorganized when the feature is refactored (see ``docs/dev/architecture.md``):

- ``kg_engine``: Grobid server and parsing, corpus table, LLM topic extraction
  and graph building.
"""
