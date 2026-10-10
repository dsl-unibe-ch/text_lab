"""Backend for the Transcription feature.

Audio transcription with WhisperX, including speaker diarization, VAD
pre-filtering and Swiss German models. Interfaces call
:func:`service.run_transcription`, which runs the pipeline in a worker
process; see ``README.md`` in this folder.

- ``service``: the pipeline and its entry points, and staging of uploads.
- ``models``: options and results.
- ``audio``: decoding, language detection and voice activity detection.
- ``formats``: the transcript exports and their readers.
- ``whisper_models``: which model to use for a language.
- ``worker``: the worker process's entry point.
- ``cli``: batch command (placeholder).
"""
