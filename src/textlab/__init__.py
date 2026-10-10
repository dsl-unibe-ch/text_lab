"""Text Lab: AI and NLP tools for research data on an HPC cluster.

The package is split into a backend and user interfaces:

- ``textlab.features``: one package per feature (transcription, OCR,
  translation, ...). Each holds the feature's processing logic and never
  imports a user interface.
- ``textlab.common``: code shared by several features.
- ``textlab.ui``: user interfaces. Today that is the Streamlit app; a web
  frontend or other interfaces can be added next to it later.

See ``docs/dev/architecture.md`` for the rules that keep these layers apart.
"""
