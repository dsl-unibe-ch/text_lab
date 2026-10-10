# OCR

Text, tables, figures, formulas and layout from PDFs and images, with an
automatic pipeline that picks the route per page, plus searchable-PDF export
and optional figure descriptions. Used by the OCR page and, for scanned PDF
pages, by Translation.

**Status:** the automatic pipeline is refactored (phase 6a). Manual engine
selection (EasyOCR, PaddleOCR, OlmOCR, GLM-OCR) still runs its engines in
`ui/streamlit/ocr/legacy.py` with `ocr_engine.py` and `paddle_ocr_worker.py`;
phase 6b moves it behind engine adapters here.

## Pipeline

`pipeline.process_document` turns one PDF or image into a
`doc_ir.Document`, routing each page:

1. **Born-digital lane** (`native.py`): a PDF page with a real, usable text
   layer is read with PyMuPDF, text and embedded images, without a model.
   Pages with equations, a mis-encoded text layer (broken `ToUnicode`
   maps) or too little text are not "usable" and take the next lane, and
   so does every page when the user asks for the highest quality.
2. **PaddleOCR-VL lane** (`vl_session.py`): the page is rasterized (200
   DPI, 300 for survey extraction) and recognized by the PaddleOCR-VL
   worker. Its layout blocks become regions; checkbox regions are
   classified and mark glyphs checked against the ink (`marks.py`,
   `markup_detect.py`).
3. **Optional steps**: figure descriptions by a local vision model through
   Ollama (`vision_enrich.py`), survey response extraction
   (`features/survey/form_extract.py`, hidden in the app) and a searchable
   PDF whose invisible text layer is placed with Tesseract
   (`searchable_pdf.py`).
4. **Exports** (`doc_ir.py`): Markdown with figure assets, plain text, Word,
   JSON, table CSVs, form responses and a bundle of everything, each with
   the models used, for citation.

The PaddleOCR-VL worker runs in the image's `paddle_vl_backend` environment
(`common/container.py`), started as
`python -m textlab.features.ocr.paddle_vl_worker`. It loads its weights in
about 25 s, so a batch keeps one worker running for all files
(`VLWorkerSession`). Before it starts, Ollama models are unloaded
(`common.ollama.release_models`), since both do not fit on one GPU; when
another feature needs the GPU, `gpu_manager` stops the worker.

## Public API

Interfaces use `service.py` and the result model `doc_ir.py`:

| Name | Purpose |
|---|---|
| `OcrOptions` | Highest quality, figure descriptions, searchable PDF and its language, survey options |
| `recognize_document(name, data, options, on_progress=)` | One PDF or image; returns a `DocumentResult` (document, summary, downloads) |
| `recognize_batch(archive, options, on_progress=)` | A ZIP of files; returns a `BatchResult` with the result ZIP and, in survey batch mode, the questionnaire results |
| `DocumentDownloads` | Every download of a document; `refresh_responses()` after a review |
| `document_summary(document)` | Counts for an overview |
| `process_document(path, workspace_dir, ...)` | The pipeline itself, on a file; used by Translation |
| `TESSERACT_LANGUAGES`, `render_layout_preview`, ... | Re-exported for interfaces |

A minimal use:

```python
from textlab.features.ocr import doc_ir, service

result = service.recognize_document(
    "scan.pdf", pdf_bytes, service.OcrOptions(searchable_pdf=True)
)
markdown = doc_ir.to_markdown(result.document)
pdf = result.downloads.searchable_pdf
```

## Layout

```
ocr/
├── service.py           # the API above
├── pipeline.py          # process_document(): routing, assembly
├── native.py            # born-digital lane
├── vl_session.py        # PaddleOCR-VL worker: sessions and one-shot runs
├── paddle_vl_worker.py  # the worker (paddle_vl_backend environment)
├── marks.py             # checkbox and mark handling on a page
├── markup_detect.py     # mark detection in images
├── rasters.py           # page images: decode, encode, crop, downscale
├── layout_preview.py    # page image with regions outlined
├── searchable_pdf.py    # invisible text layer
├── vision_enrich.py     # vision-model client, figure descriptions
├── doc_ir.py            # result model and exports
├── ocr_engine.py        # manual engines (until phase 6b)
├── paddle_ocr_worker.py # PaddleOCR 2 worker (until phase 6b)
├── cli.py               # batch command (placeholder)
└── tests/
```

## Files written

| What | Where | Removed |
|---|---|---|
| The upload, page images, crops | `ocr/job-*` in the job workspace | When the document is done |
| A batch's files, page images and result folders | `ocr/batch-*` in the job workspace | Page images after each file; the rest when the batch is done |

Results are returned in memory; nothing is written outside the workspace.

## Configuration

| Setting | Used for |
|---|---|
| `PADDLE_VL_BACKEND_PYTHON` | Another interpreter for the PaddleOCR-VL worker; optional |
| `TEXTLAB_VL_STALL_TIMEOUT` | Seconds without worker output before it is stopped (900) |
| `TEXTLAB_VISION_MODEL` | Ollama model for figure descriptions |
| `PADDLEX_HOME`, `PADDLE_PDX_CACHE_HOME` | PaddleX model store, mounted by the launch script |

## Tests

In `tests/`, without GPU or models unless marked:

- `test_regression.py`: the result model, routing (text layer, math,
  mis-encoded text, decorative images), the worker protocol with stub
  workers, stall detection.
- `test_vl_session.py`: the resident worker loads once per batch, recovers
  from crashes and falls back to a one-shot worker.
- `test_markup.py`, `test_noisy_scan.py`: mark detection on drawn forms.
- `test_searchable_pdf.py`: word placement and the PDF text layer.
- `test_enrichments.py`: figure descriptions and survey extraction with a
  fake vision client.

Stub workers are scripts that speak the worker's protocol; pass them as
`worker_path` with `backend_python=sys.executable`.

## Batch use (planned)

`cli.py` is a placeholder for a `textlab ocr` command for Slurm batch jobs.
It will call `service.recognize_batch` with a progress callback that writes
log lines.
