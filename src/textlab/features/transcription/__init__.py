"""Backend for the Transcription feature.

Audio transcription with WhisperX, including speaker diarization, VAD
pre-filtering and Swiss German models.

Moved from ``src/core`` with only import and path updates. The modules are
reorganized when the feature is refactored (see ``docs/dev/architecture.md``):

- ``transcribe_engine``: audio decoding, language detection, VAD and the
  transcript exports (CSV, SRT, VTT, ELAN).
- ``transcribe_worker``: subprocess that runs the WhisperX pipeline for the
  Meeting Notes Generator.
"""
