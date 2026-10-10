"""Backend for the Meeting Notes feature.

Meeting notes generated from a recording: transcription followed by LLM
summarization.

Moved from ``src/core`` with only import and path updates. The modules are
reorganized when the feature is refactored (see ``docs/dev/architecture.md``):

- ``summarize_engine``: chunked LLM summarization of a transcript into
  structured notes.
"""
