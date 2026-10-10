"""Backend for the Survey feature: answers of filled-in paper questionnaires.

Interfaces and the OCR feature use :mod:`.service`. The modules behind it:

- ``survey_template``: the blank form, learned once per batch, and its
  response controls.
- ``survey_label``: names of questions, answers and options, read from the
  form's printed text.
- ``survey_batch``: reading a batch of questionnaires against the form,
  tables, overlays and rebuilt exports.
- ``form_extract``: question-level extraction with a vision model
  (experimental, hidden in the app).
- ``cli``: command line for developers (``python -m
  textlab.features.survey.cli``).

See ``README.md`` in this folder for the pipeline and the files written.
"""
