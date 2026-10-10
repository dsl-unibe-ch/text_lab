# OCR

Text, tables, figures, formulas and layout from PDFs and images, with an
automatic pipeline that picks the route per page, plus searchable-PDF export
and optional figure descriptions. Manual engine selection runs one chosen
engine instead (EasyOCR, PaddleOCR, OlmOCR or GLM-OCR) for its plain text.
Used by the OCR page and, for scanned PDF pages, by Translation.

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

In survey batch mode, the batch also reads every file as a filled-in
questionnaire, through the Survey feature (`survey.service`).

## Manual engines

`manual.py` runs one engine chosen by the user on a document or a ZIP batch.
Each engine is an adapter in `engines/` with the interface of
`engines.base.Engine`: `prepare()` gets its model ready once per run, and
`recognize()` reads one file and returns an `EngineOutput` (text per page,
the engine's raw result, optional preview images).

| Engine | Adapter | Runs | Reads |
|---|---|---|---|
| EasyOCR | `easy_ocr.py` | In the app process; one reader per language stays loaded, released by `gpu_manager` | Page images |
| PaddleOCR | `paddle_ocr.py` | `python -m textlab.features.ocr.engines.paddle_ocr_worker` in the `paddle_backend` environment | Page images |
| OlmOCR | `olm_ocr.py` | `python -m olmocr.pipeline` in the `olmocr_backend` environment, with vLLM | Whole PDFs (an image becomes a one-page PDF) |
| GLM-OCR | `glm_ocr.py` | A vision model on the session's Ollama server, pulled on first use | Page images, scaled to 2048 px |

Engines that read page images subclass `engines.base.PageEngine`, which
renders PDFs with `pdftoppm` first. The PaddleOCR worker may import only
`engines/payloads.py` from `textlab`, since its environment does not have
the app's packages; a test checks this.

To add an engine, write an adapter in `engines/` and add it to
`manual.ENGINES`. The page lists the engines from there, with a language
choice for engines that set `languages` and a mode choice for those that
set `modes`.

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
| `ENGINES` | The manual engines by name, with their language and mode choices |
| `recognize_with_engine(name, data, engine, options, on_progress=)` | One file with a manual engine; returns an `EngineRun` (text, JSON, previews, ZIP) |
| `recognize_batch_with_engine(archive, engine, options, on_progress=)` | A ZIP of files with a manual engine; returns the result ZIP |
| `EngineOptions`, `EngineError` | Settings of a manual engine run; its error, with `details` for developers |
| `html_table`, `contains_html_table` | A table in a manual engine's text, as a data frame |

A minimal use:

```python
from textlab.features.ocr import doc_ir, service

result = service.recognize_document(
    "scan.pdf", pdf_bytes, service.OcrOptions(searchable_pdf=True)
)
markdown = doc_ir.to_markdown(result.document)
pdf = result.downloads.searchable_pdf

run = service.recognize_with_engine(
    "scan.png", png_bytes, "EasyOCR", service.EngineOptions(language="de")
)
text = run.text
```

## Layout

```
ocr/
├── service.py           # the API above
├── archives.py          # ZIP input and output of batches
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
├── manual.py            # manual engine selection: runs and outputs
├── engines/
│   ├── base.py              # engine interface, page images, results
│   ├── easy_ocr.py          # EasyOCR
│   ├── paddle_ocr.py        # PaddleOCR 2, through its worker
│   ├── paddle_ocr_worker.py # the worker (paddle_backend environment)
│   ├── olm_ocr.py           # OlmOCR
│   ├── glm_ocr.py           # GLM-OCR on Ollama
│   ├── payloads.py          # engine results to JSON (used by the worker)
│   └── previews.py          # preview images of the results
├── cli.py               # batch command (placeholder)
└── tests/
```

## Files written

| What | Where | Removed |
|---|---|---|
| The upload, page images, crops | `ocr/job-*` in the job workspace | When the document is done |
| A batch's files, page images and result folders | `ocr/batch-*` in the job workspace | Page images after each file; the rest when the batch is done |
| A manual engine run: the upload or batch, page images, engine files, results | `ocr/manual-*` in the job workspace | Page images after each file; the rest when the run is done |

Results are returned in memory; nothing is written outside the workspace.

## Configuration

| Setting | Used for |
|---|---|
| `PADDLE_VL_BACKEND_PYTHON` | Another interpreter for the PaddleOCR-VL worker; optional |
| `PADDLE_BACKEND_PYTHON` | Another interpreter for the PaddleOCR worker; optional |
| `OLMOCR_BACKEND_PYTHON` | Another interpreter for OlmOCR; optional |
| `OLMOCR_GPU_MEMORY_UTILIZATION` | Share of GPU memory OlmOCR's vLLM may take (0.6) |
| `TEXTLAB_GLM_OCR_MODEL` | Ollama model of the GLM-OCR engine (`glm-ocr:latest`) |
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
- `test_service.py`: the service with a fake pipeline: job folders, batch
  layout, the shared worker session.
- `test_engines.py`: the engine adapters with fake models and a stub
  PaddleOCR worker; what the worker may import.
- `test_manual.py`: manual engine runs and batches with a fake engine:
  outputs, ZIP layout, cleanup.
- `test_ocr_page.py`: the page and `ui/streamlit/ocr/` import only the
  service and `doc_ir`, and use only names the service exports.

Stub workers are scripts that speak the worker's protocol; pass them as
`worker_path` with `backend_python=sys.executable`.

## Batch use (planned)

`cli.py` is a placeholder for a `textlab ocr` command for Slurm batch jobs.
It will call `service.recognize_batch` with a progress callback that writes
log lines.
