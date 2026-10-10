"""Backend code shared by several features.

Like ``textlab.features``, this package never imports a user interface. The
exception still to be removed is ``gpu_manager``, which shows Streamlit
messages.

Moved from ``src/core`` and ``src`` with only import and path updates:

- ``artifacts``: per-user directory for files that Chat and Visualization
  generate.
- ``gpu_manager``: frees the GPU before a feature runs.
- ``html_safety``: sanitizer for OCR-produced table HTML.
- ``language_mappings``: language names and codes for the OCR, transcription
  and translation pages.
- ``model_config``: the LLM lists offered in the app, read from ``models.json``
  and an optional ``models.local.json`` next to it.
- ``upload_safety``: safeguards for uploaded files and ZIP archives.
"""
