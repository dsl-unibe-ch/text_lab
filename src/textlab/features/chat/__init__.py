"""Backend for the Chat feature.

Private chat with local LLMs over the user's own documents and data files.

Moved from ``src/core`` with only import and path updates. The modules are
reorganized when the feature is refactored (see ``docs/dev/architecture.md``):

- ``chat_engine``: document reading, chat generation, the data-analysis
  router and chat history formatting. Its Ollama helpers moved to
  ``textlab.common.ollama``.
"""
