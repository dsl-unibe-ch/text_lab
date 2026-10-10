# Chat

Private chat with local LLMs over the user's own documents and data files.

**Status:** moved, not refactored yet. The modules were moved here from
`src/core` with only import and path updates; the package docstring in
`__init__.py` says what each one does.

- Modules: `chat_engine.py`
- UI: `Chat.py` in `src/textlab/ui/streamlit/pages/`
- Tests: none yet

Once the feature is refactored, this file documents its pipeline, public API,
files written to disk, configuration and tests, as described in the developer
guide ([architecture](../../../../docs/dev/architecture.md#feature-packages)).
