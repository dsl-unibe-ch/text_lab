# Survey

Extraction of filled-in survey and form responses from scanned documents, with
templates and batch scoring.

**Status:** not migrated yet. The code currently lives in:

- `src/core/form_extract.py`
- `src/core/survey_batch.py`
- `src/core/survey_label.py`
- `src/core/survey_template.py`
- `src/tools/survey_cli.py`
- UI: `src/pages/OCR.py (survey review, currently hidden)`

Once the feature is refactored, this file documents its pipeline, public API,
files written to disk, configuration and tests, as described in the developer
guide ([architecture](../../../../docs/dev/architecture.md#feature-packages)).
