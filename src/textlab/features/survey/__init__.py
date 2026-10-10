"""Backend for the Survey feature.

Extraction of filled-in survey and form responses from scanned documents, with
templates and batch scoring.

Moved from ``src/core`` with only import and path updates. The modules are
reorganized when the feature is refactored (see ``docs/dev/architecture.md``):

- ``cli``: command-line batch extraction (``python -m
  textlab.features.survey.cli``).
- ``form_extract``: question-level response extraction.
- ``survey_batch``: reading a batch of questionnaires against a template.
- ``survey_label``: naming the controls on a synthesized blank form.
- ``survey_template``: the printed form, learned once per batch.
"""
