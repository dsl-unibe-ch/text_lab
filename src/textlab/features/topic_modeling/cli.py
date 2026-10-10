"""Command-line interface for batch topic modeling (planned, not implemented).

Intended for fitting topic models on large corpora in a Slurm batch job, where
the interactive app is not practical. It will call the same backend service as
the Topic Modeling page, so results match the app.

Planned usage (a sketch, settled when implemented)::

    textlab topics INPUT_FILE --text-column NAME --out OUTPUT_DIR
        [--method NAME]

This module only reserves the location. See ``deploy/sbatch/README.md``.
"""
