# Historic Handwriting

Transcribes handwritten historical German documents (Kurrent, medieval
hands) line by line. kraken's segmenter finds the lines on each page, and a
handwriting recognition model of the Digital Humanities group of the
University of Bern ([dh-unibe](https://huggingface.co/dh-unibe)) reads
each line. It is separate from the OCR feature, whose pipeline reads
printed documents; the two share no code.

Users reach it on the OCR page, in the **Historic German handwriting**
expander below the legacy engines. The UI is in `ui/streamlit/handwriting/`
(`panel.handwriting_expander`), which shows the models a run will use
(`HandwritingModel.models_used`) before the user uploads anything.

## Pipeline

`service.transcribe_document` (one PDF or image) and
`service.transcribe_batch` (a ZIP archive) run the same four steps. A
batch sends all pages of all files through each worker at once, so every
model is loaded once per batch.

1. **Pages** (`documents.page_images`): a PDF is rendered at 300 DPI with
   `pdftoppm`; every frame of an image file (a TIFF can have several)
   becomes an RGB PNG.
2. **Line detection** (`detect_worker.py`): kraken's baseline segmenter
   (`blla`, with the default model that ships with kraken, so nothing is
   downloaded) finds the text lines on each page, in a worker in the
   image's `kraken_backend` environment. It returns one boundary polygon
   per whole line, in kraken's reading order: lines are ordered within the
   text regions it finds (columns, marginal notes), and the regions among
   each other. `lines.line_boxes` turns each polygon into its box, keeping
   that order and dropping boxes too small to hold writing. Each line is
   cropped with a margin of 10% of its height and scaled down to the
   model's `max_pixels`, as in training.
3. **Recognition** (`worker.py`, `recognizer.py`): the chosen model reads
   every crop, in a worker in the app's environment. Lines are read in
   batches of 8, halved when the GPU runs out of memory.
4. **Exports** (`documents.py`): plain text, JSON (lines with boxes) and
   one PAGE XML file per page, which eScriptorium and Transkribus import
   for correction. The XML coordinates are on the page image the lines
   were found on: for a single image that is the uploaded file itself, so
   the XML names it; rendered PDF pages and TIFF frames are included next
   to the XML as `page_0001.png`, ...

Both workers exit when they are done, so the GPU is free afterwards and a
crash cannot take down the app. Before a run, the panel calls
`gpu_manager.prepare_gpu(HANDWRITING)`, which releases the other features'
models; `gpu_manager` also knows both workers, to stop leftovers.

## Models

The models offered are configured in `common/models.json` under
`handwriting_models`, or in `models.local.json` next to it, so a model can
be added or swapped without a code change. Each entry names:

| Field | Meaning |
|---|---|
| `key` | Stable identifier, used in options |
| `label`, `description` | Shown on the page |
| `base_model` | Hugging Face vision-language model, loaded with `AutoModelForImageTextToText` |
| `adapter` | Hugging Face repository of a LoRA adapter for it, or `null` |
| `prompt` | The instruction sent with each line; use the one the model was trained with |
| `max_pixels` | Crops larger than this are scaled down; the training value |

The code treats every entry the same way. Today both entries are LoRA
adapters on `Qwen/Qwen3-VL-4B-Instruct`:

| Model | Material | Reported CER |
|---|---|---|
| `dh-unibe/qwen3vl-german-xix-v2` | Kurrent, 19th century, mainly Swiss records | 7.7% on unseen hands (Federal Council minutes 1848-1903) |
| `dh-unibe/qwen3vl-medieval-german-v3` | Medieval and early modern German, 14th-17th century | about 14% |

They were chosen because they run with the packages already in the image.
dh-unibe's TrOCR models would also run without a rebuild, but they need a
second recognition path; the kraken models need a rebuild (kraken is not
in the image).

### Adapters without `peft`

`peft` is not in the image. `recognizer.merge_lora` adds each adapted
layer's update, `W + (lora_alpha / r) * B @ A` (`lora_alpha / sqrt(r)` for
rsLoRA), to the base weights after loading, which is what `peft`'s
`merge_and_unload` computes, so generation then runs at the base model's
speed. Adapters that change more than the linear layers (DoRA, per-module
ranks, `modules_to_save`, other tensors) or do not match the model are
refused with an error rather than merged wrongly. The processor is loaded
from the adapter repository, which ships the training tokenizer and chat
template, or from the base model when the adapter has none.

## Public API

Interfaces use `service.py`:

| Name | Purpose |
|---|---|
| `available_models()` | The configured models (`HandwritingModel`), in order |
| `HandwritingOptions(model, previews=True)` | Model key, and whether to draw page previews |
| `transcribe_document(name, data, options, on_progress=)` | One PDF or image; returns a `TranscriptionRun` (document, text, JSON, previews, ZIP) |
| `transcribe_batch(archive, options, on_progress=)` | A ZIP archive; returns a `BatchRun` (result ZIP, counts, models used) |
| `INPUT_EXTENSIONS` | The file types read |
| `LINE_DETECTOR` | Name of the line detector, for display |
| `HandwritingError` | A failure with a message for users |

A worker failure raises `common.jobs.WorkerError`, with the worker's
traceback in `details`.

```python
from textlab.features.handwriting import service

options = service.HandwritingOptions(model="german-kurrent-xix")
run = service.transcribe_document("letter.jpg", image_bytes, options)
print(run.text)
```

## Layout

```
handwriting/
├── service.py        # the API above: steps, progress, outputs
├── models.py         # model configuration, options, results
├── lines.py          # line polygons to boxes, crop boxes
├── documents.py      # page images, crops, previews, text/JSON/PAGE XML, ZIPs
├── detect_worker.py  # line detection with kraken (kraken_backend environment)
├── worker.py         # recognition worker (app environment)
├── recognizer.py     # model loading, LoRA merge, batched generation
└── tests/
```

## Files written

| What | Where | Removed |
|---|---|---|
| The upload, page images, line crops, result files | `handwriting/run-*` in the job workspace | When the run ends |
| A batch's files, page images, crops and result folders | `handwriting/batch-*` | When the batch ends |
| The workers' request, progress and result | `handwriting/job-*` | When the worker ends |

Results are returned in memory. Model weights are downloaded into the
Hugging Face store (`HF_HOME`) on first use; kraken's segmentation model
is part of the kraken package in the image.

## Configuration

| Setting | Used for |
|---|---|
| `handwriting_models` in `common/models.json` | The models offered |
| `KRAKEN_BACKEND_PYTHON` | Another interpreter for the line-detection worker; optional |
| `HF_HOME` | Hugging Face model store, set by the image |

## Tests

In `tests/`, without GPU or models:

- `test_lines.py`: polygons to boxes in the detector's order, crops.
- `test_models.py`: the configured models load; results and citation.
- `test_recognizer.py`: the LoRA merge on a tiny model, refused adapters,
  batching and the out-of-memory fallback.
- `test_documents.py`: TIFF frames, crops, previews, PAGE XML, ZIPs.
- `test_service.py`: the service with fake workers: steps, outputs, batch
  layout, cleanup.
- `test_workers.py`: what the detection worker may import; the detection
  worker run through `common.jobs` with fake `kraken` and `torch`
  modules; reading kraken's line boundaries; the recognition worker with
  a fake model.
- `test_handwriting_page.py`: the panel imports only the service and uses
  only the names it exports; there is no separate page; the OCR page draws
  it last, below the legacy engines.
