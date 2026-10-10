# Transcription

Audio transcription with WhisperX, including speaker diarization, VAD
pre-filtering and Swiss German models.

**Status:** moved, not refactored yet. The modules were moved here from
`src/core` with only import and path updates; the package docstring in
`__init__.py` says what each one does.

- Modules: `transcribe_engine.py`, `transcribe_worker.py`
- UI: `Transcribe.py` in `src/textlab/ui/streamlit/pages/`
- Tests: none yet

Once the feature is refactored, this file documents its pipeline, public API,
files written to disk, configuration and tests, as described in the developer
guide ([architecture](../../../../docs/dev/architecture.md#feature-packages)).
