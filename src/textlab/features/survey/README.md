# Survey

Reads the answers of filled-in paper questionnaires from scans. In a batch of
the same questionnaire, the blank form is learned from the batch itself and
every respondent is read against it, giving one row per respondent with a
certainty beside every answer. Used by the OCR feature's batch in survey batch
mode ("Extract questionnaire responses" on the OCR page).

## Pipeline

Questionnaire batches (`service.QuestionnaireBatch`):

1. **Learn the blank form** (`survey_template.build_template`): each page of
   every questionnaire is registered onto a reference copy (ORB features and
   a homography), and the median of up to 15 aligned copies cancels the
   respondents' ink, leaving the blank form. Response controls (checkboxes,
   circles) are found on the blank.
2. **Name the answers** (`survey_label`): the PaddleOCR-VL worker reads the
   blank's printed text, which names questions and options; controls are
   grouped into answers (a row, a rating scale, a vertical option list) and
   Tesseract reads row stems the layout model missed. This step is an
   enrichment: when it fails, the batch still works with generated names.
3. **Read each questionnaire** (`survey_batch.read_document`): each page is
   registered onto the blank, the blank is subtracted, and the remaining ink
   in each control decides checked, unchecked or uncertain, with a certainty
   from the margin to the thresholds.
4. **Export** (`survey_batch.write_batch_outputs`): a `survey/` folder with
   the tables (one row per respondent, one line per answer, a review queue),
   the template and overlay images, and a README for users; plus each
   questionnaire's own `survey_answers.csv` and its answers in the OCR
   document's exports.
5. **Review** (optional): the user checks the overlays and drops controls
   that are not really on the form; the exports are rebuilt from the stored
   readings without reading the questionnaires again.

With fewer than 6 questionnaires, an option most respondents marked can
survive on the blank; the batch then carries a warning.

Question-level extraction (`form_extract.extract_page_forms`) is a second,
experimental method for single documents: a vision model reads each question
section of a recognized page. It is hidden in the app; the OCR pipeline runs
it with `extract_survey=True`, and only with models listed in
`TEXTLAB_APPROVED_SURVEY_MODELS`.

## Public API

Everything other features and interfaces need is in `service.py`:

| Name | Purpose |
|---|---|
| `QuestionnaireBatch.learn(files, on_progress=, vl_session=)` | Learn the blank form from a batch; returns the batch |
| `QuestionnaireBatch.read(path, document, output_dir, folder)` | Read one questionnaire, write its answers, add them to its OCR document |
| `QuestionnaireBatch.keep_document(folder, document)` | Keep the document, without images, for rebuilding exports |
| `QuestionnaireBatch.write_outputs(folder)` | Write the `survey/` folder |
| `QuestionnaireBatch.overlays()`, `.overview()` | Review: the form with its controls outlined, one row per answer |
| `QuestionnaireBatch.drop_controls(ids, zip_bytes)` | Remove controls and rebuild the result ZIP |
| `warning`, `control_count`, `answer_count` | For the user: a small-batch warning and the size of the form |
| `extract_page_forms`, `SameLayoutTemplate` | Question-level extraction (experimental) |

The OCR batch uses it like this (simplified from `ocr.service`):

```python
from textlab.features.survey.service import QuestionnaireBatch

survey = QuestionnaireBatch.learn(files, vl_session=vl_session)
for path in files:
    document = process_document(path, ...)
    skip = survey.read(path, document, output_dir, folder)
    write_document_outputs(document, output_dir, skip_tables=skip, ...)
    survey.keep_document(folder, document)
survey.write_outputs(results_dir / "survey")
```

## Layout

```
survey/
├── service.py          # the API above
├── survey_template.py  # registration, the blank form, control detection
├── survey_label.py     # names of questions, answers and options
├── survey_batch.py     # reading, tables, overlays, exports, rebuilds
├── form_extract.py     # question-level extraction (experimental)
├── cli.py              # command line for developers (below)
└── tests/
```

## Files written

| What | Where | Removed |
|---|---|---|
| Blank form images for the layout model | `survey/blank-*` in the job workspace | When the layout is read |
| Exports while they are rebuilt | `survey/exports-*`, `survey/rebuild-*` in the job workspace | When the rebuilt ZIP is returned |

The questionnaires themselves, their page images and the batch's outputs are
in the OCR batch's job folder (`ocr/batch-*`); see the OCR README.

## Configuration

| Setting | Used for |
|---|---|
| `TEXTLAB_APPROVED_SURVEY_MODELS` | Vision models allowed for question-level extraction |
| `TEXTLAB_SURVEY_CONTRACT` | Output format asked of the model in question-level extraction |

The questionnaire batch uses no settings; the PaddleOCR-VL worker is
configured by the OCR feature.

## Tests

In `tests/`, without GPU or models:

- `test_survey_template.py`: registration, the consensus blank, control
  detection, mark reads, structure, labels, tables, scoring and rebuilds, on
  drawn forms.
- `test_service.py`: the service with the batch reader replaced by fakes:
  progress, answer files, document hand-over, rebuilds.

Question-level extraction is tested in `features/ocr/tests/test_enrichments.py`
with a fake vision client.

## Command line (developers)

`cli.py` runs the questionnaire batch without the app, for testing on a
folder of scans:

```bash
python -m textlab.features.survey.cli run --input scans/ --out results/
```

`build-template` and `read` split the same work when the template needs a
human pass in between; `prune-template` drops controls from a template;
`sheet` and `score` compare reads against hand-made answers. It writes to
the folders given and is not part of the app.
