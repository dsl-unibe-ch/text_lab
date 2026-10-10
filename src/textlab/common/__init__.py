"""Backend code shared by several features.

Examples are the LLM client, GPU management, upload and HTML safety checks
and the model configuration. Like ``textlab.features``, this package never
imports a user interface.

The code still lives in the files below and moves here during the refactor
(see ``docs/dev/architecture.md``):

- ``src/core/gpu_manager.py``
- ``src/core/upload_safety.py``
- ``src/core/html_safety.py``
- ``src/core/model_config.py``
- ``src/language_mappings.py``
- ``src/config/models.json``
"""
