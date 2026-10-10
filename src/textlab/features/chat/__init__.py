"""Backend for the Chat feature.

Private chat with local LLMs over the user's own documents and data files.

Moved from ``src/core`` with only import and path updates. The modules are
reorganized when the feature is refactored (see ``docs/dev/architecture.md``):

- ``chat_engine``: Ollama client helpers, document reading and chat generation.
  Meeting Notes, Translate and Visualization also use its Ollama helpers.
"""
