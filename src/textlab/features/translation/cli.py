"""Command-line interface for batch translation (planned, not implemented).

Intended for translating many documents in a Slurm batch job, where the
interactive app is not practical. It will call the same backend service as the
Translate page, so results match the app.

Planned usage (a sketch, settled when implemented)::

    textlab translate INPUT_DIR --out OUTPUT_DIR --target-lang CODE

This module only reserves the location. See ``deploy/sbatch/README.md``.
"""
