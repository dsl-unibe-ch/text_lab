"""Entry point for the ``textlab`` command (planned, not implemented).

It will dispatch ``textlab <feature> ...`` to the ``cli`` module of each
feature package, for example ``textlab transcribe`` to
``textlab.features.transcription.cli``. The command is not registered in
``pyproject.toml`` until the first feature command exists.

This module only reserves the location. See ``deploy/sbatch/README.md``.
"""
