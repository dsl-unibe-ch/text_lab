"""Command-line interface for building corpora (planned, not implemented).

Intended for parsing large paper collections in a Slurm batch job, where
waiting in the interactive app is not practical. It will call
``service.build_corpus`` and, optionally, ``service.extract_topics``, so the
corpus folder is the same as the app's and the page can draw its graphs.

Planned usage (a sketch, settled when implemented)::

    textlab knowledge-graph PAPERS_DIR [--out LOCATION] [--topics MODEL]

This module only reserves the location. See ``deploy/sbatch/README.md``.
"""
