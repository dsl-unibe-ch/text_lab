# Transcription

Audio transcription with WhisperX, including speaker diarization, VAD
pre-filtering and Swiss German models.

**Status:** not migrated yet. The code currently lives in:

- `src/core/transcribe_engine.py`
- `src/core/transcribe_worker.py`
- UI: `src/pages/Transcribe.py`

Once the feature is refactored, this file documents its pipeline, public API,
files written to disk, configuration and tests, as described in the developer
guide ([architecture](../../../../docs/dev/architecture.md#feature-packages)).
