"""Backend for the Meeting Notes feature.

Structured notes from a recording or transcript, written by a local LLM,
with each point citing the transcript lines it is based on. Transcription
comes from the transcription feature; ``service`` is the public API, built on
``transcript`` (numbered segments) and ``citations`` (checking and showing
citations). See ``README.md`` in this folder.
"""
