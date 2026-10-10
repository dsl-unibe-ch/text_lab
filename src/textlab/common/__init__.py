"""Backend code shared by several features.

Like ``textlab.features``, this package never imports a user interface.

Foundation for all features:

- ``config``: site settings from the environment (``deploy/site.env``).
- ``storage``: the private per-job workspace for temporary user files.

Moved from ``src/core`` and ``src`` with only import and path updates:

- ``gpu_manager``: frees the GPU before a feature runs.
- ``html_safety``: sanitizer for OCR-produced table HTML.
- ``language_mappings``: language names and codes for the OCR, transcription
  and translation pages.
- ``model_config``: the LLM lists offered in the app, read from
  ``models.json`` and an optional ``models.local.json`` next to it.
- ``upload_safety``: safeguards for uploaded files and ZIP archives.
"""
