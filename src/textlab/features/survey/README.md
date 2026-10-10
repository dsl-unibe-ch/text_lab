# Survey

Extraction of filled-in survey and form responses from scanned documents, with
templates and batch scoring.

**Status:** moved, not refactored yet. The modules were moved here from
`src/core` with only import and path updates; the package docstring in
`__init__.py` says what each one does.

- Modules: `cli.py`, `form_extract.py`, `survey_batch.py`, `survey_label.py`, `survey_template.py`
- UI: the survey review in `src/textlab/ui/streamlit/pages/OCR.py` (currently
  hidden)
- Tests: `tests/test_survey_template.py`

Once the feature is refactored, this file documents its pipeline, public API,
files written to disk, configuration and tests, as described in the developer
guide ([architecture](../../../../docs/dev/architecture.md#feature-packages)).
