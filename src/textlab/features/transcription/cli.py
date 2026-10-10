"""Command-line interface for batch transcription (planned, not implemented).

Intended for transcribing many recordings in a Slurm batch job, where the
interactive app is not practical. It will call the same backend service as the
Transcribe page, so results match the app.

Planned usage (a sketch, settled when implemented)::

    textlab transcribe INPUT_DIR --out OUTPUT_DIR [--model NAME]
        [--language CODE] [--diarize]

This module only reserves the location. See ``deploy/sbatch/README.md``.
"""
