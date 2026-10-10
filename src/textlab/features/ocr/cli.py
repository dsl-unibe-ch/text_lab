"""Command-line interface for batch OCR (planned, not implemented).

Intended for running OCR over large document collections in a Slurm batch job,
where the interactive app is not practical. It will call the same backend
service as the OCR page, so results match the app.

Planned usage (a sketch, settled when implemented)::

    textlab ocr INPUT_DIR --out OUTPUT_DIR [--engine NAME]
        [--searchable-pdf]

This module only reserves the location. See ``deploy/sbatch/README.md``.
"""
