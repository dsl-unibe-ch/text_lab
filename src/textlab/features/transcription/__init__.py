"""Backend for the Transcription feature.

Audio transcription with WhisperX, including speaker diarization, VAD
pre-filtering and Swiss German models.

The code still lives in the files below and moves here during the refactor (see
``docs/dev/architecture.md``):

- ``src/core/transcribe_engine.py``
- ``src/core/transcribe_worker.py``
"""
