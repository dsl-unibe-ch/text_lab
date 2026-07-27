# TextLab: Automatic Structured OCR Page — Implementation Plan

> Companion to `report.md` (open-source OCR landscape, 2026-07-15). Intended to be executed directly on UBELIX against a clone of `dsl-unibe-ch/text_lab`.

## Context

TextLab (`dsl-unibe-ch/text_lab`, Streamlit on UBELIX/Apptainer, offline-only) currently makes the user pick one of four OCR engines (EasyOCR, PaddleOCR, olmOCR, GLM-OCR) plus a language or a GLM mode, and every engine flattens its output to a plain text string — layout boxes, tables, figures, and confidence are discarded. Table "detection" is a regex for `<table>` in the output text.

The driving use case is a support request: batches of scanned **surveys full of markup sections** (questions answered by coloring/crossing circles or boxes). The overnight research report recommends a confidence-aware pipeline over a typed intermediate representation (IR) rather than another string-returning engine.

**Decisions made:**
- **Engine:** PaddleOCR-VL-1.6 as the single automatic engine (PP-DocLayoutV3 layout + 0.9B recognition VLM, Apache-2.0, ~2 GB weights, 109 languages so no language selector). Its layout taxonomy natively detects `text, table, image, chart, formula, footnote, header, footer, seal, checkbox`, and the recognition model handles tables/formulas/charts. A cheap native-text fast lane handles born-digital PDFs.
- **UI:** redesign the existing OCR page — the automatic pipeline is the default flow; the four legacy engines move into an "Advanced / legacy engines" expander unchanged.
- **Survey batch aggregation** (one row per respondent) is deferred to a later iteration, but per-page checkbox/mark **state detection must be solid now** since it is the foundation.

**Key open risk:** PaddleOCR-VL's checkbox *recognition* is a claimed capability (since 1.5) whose output format is undocumented. The plan validates it empirically on real survey pages and includes a geometric fallback classifier.

## Repo setup

Clone `https://github.com/dsl-unibe-ch/text_lab` on UBELIX, create branch `feature/auto-ocr`. All paths below are relative to that repo.

## Implementation

### 1. Backend worker — `src/core/paddle_vl_worker.py` (new)
Mirror the existing subprocess pattern of `src/core/paddle_ocr_worker.py` (isolated conda env, `TEXTLAB_..._RESULT_JSON=` marker-line protocol), but run the `PaddleOCRVL` pipeline from `paddleocr>=3.6` (`[doc-parser]` extra):
- Input: list of page image paths (+ optional per-page DPI metadata).
- Per page, emit: `parsing_res_list` (each block: `block_bbox`, `block_label`, `block_content`, `block_order`), `layout_det_res` (label + score + coordinate per region), the pipeline's markdown, and base64 PNG crops for non-text regions (`image`, `chart`, `seal`, `checkbox`, figure-like labels) cut from the source page with a small margin.
- Offline hygiene as today: `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`, pinned model cache dir, no network at runtime. Native Paddle inference first (simplest in Apptainer); vLLM serving is a later optimization, not in scope.

### 2. Typed IR — `src/core/doc_ir.py` (new)
Light version of the report's §4.1 IR (plain dataclasses/dicts, JSON-serializable):
- `document → pages → regions`; region: `id, type, bbox, reading_order, content {text|html|latex}, confidence {layout, ocr}, asset (crop path/b64), source ("native"|"paddleocr-vl-1.6"), warnings`.
- Adapter `from_paddle_vl(page_json) -> Page` mapping `block_label` to region types (map `vision_footnote`/`footnote` → footnote, `image`/`chart` → figure, etc.).
- Deterministic exporters: `to_markdown(with assets)`, `to_json`, `tables_to_dataframes` (parse each table region's HTML via the existing `extract_html_table` logic in `src/core/ocr_engine.py`, generalized to per-region instead of whole-text regex), ZIP bundling.

### 3. Markup/checkbox handling — `src/core/markup_detect.py` (new)
For every region labeled `checkbox` (and configurable extra labels):
- Keep the crop + layout confidence in the IR.
- Parse VL `block_content` for state glyphs/markup (☑/☐/✗ or similar) — **verify empirically what 1.6 actually emits** on test survey pages; this is the first task of this module.
- Geometric fallback (OpenCV on the crop): detect the circle/box outline, compute interior ink/fill ratio vs. outline baseline → `checked/unchecked/uncertain` with a score. This covers survey marks (colored circles, crossed boxes) that the VL may miss or mislabel.
- Region output: `{state: checked|unchecked|uncertain, method: vl|geometric, score}`; `uncertain` regions are surfaced in the UI for review, never silently dropped.

### 4. Born-digital fast lane — `src/core/auto_ocr.py` (new orchestrator)
- Per PDF page, use `pypdfium2` (already an olmOCR dependency; verify availability, else PyMuPDF) to measure text coverage.
- Pages with a real text layer → extract text (+ embedded images) directly into the IR with `source: "native"`; pages without → rasterize at 200 DPI (`pdftoppm -r 200`, replacing the current default-DPI call) → VL worker. Images always go to the VL worker.
- The orchestrator returns one IR document regardless of route; the UI never asks.

### 5. UI rework — `src/pages/OCR.py` (modify)
- Default flow: upload → single **"Parse document"** button. No engine, language, or mode selectors.
- Results (from IR), in tabs:
  - **Document** — rendered markdown in reading order (assets inlined).
  - **Tables** — one `st.dataframe` per table region + per-table CSV download.
  - **Figures** — crops with linked captions/footnotes.
  - **Markup** — checkbox/mark regions: crop thumbnail, detected state, confidence, method; uncertain ones highlighted.
  - **Layout preview** — page image with colored boxes per region type (reuse drawing helpers in `ocr_engine.py`).
- Downloads: Markdown+assets ZIP, canonical JSON, all-tables CSV ZIP, full bundle — keep the existing session-state + self-cleaning job-dir pattern (`OCR_JOBS_BASE_DIR`, aggressive `rmtree`).
- Batch ZIP mode: same auto pipeline per file; per-file `document.md`, `document.json`, `tables/`, `assets/` in the result ZIP (structure mirrors input as today).
- Legacy: current engine-selection UI and code paths move under `st.expander("Advanced: legacy engines")`, functionally untouched.

### 6. Packaging & docs
- Extend the Paddle backend conda env (or add `paddle_vl_backend`) with `paddleocr[doc-parser]>=3.6` + paddlepaddle-gpu; locate the Apptainer/environment definition in the repo and add model pre-staging (PP-DocLayoutV3 + PaddleOCR-VL-1.6 weights, ~2–3 GB) into the image/model cache.
- Update `docs/ocr.md`: automatic mode is the default, legacy engines documented as advanced; keep the privacy section accurate (same local-only, self-cleaning flow).

## Verification
1. **Worker smoke test** (standalone, before UI): run on (a) a scanned survey page with marked circles/boxes, (b) a born-digital paper PDF, (c) a page with a table — confirm typed regions, table dataframe, figure crops, and route selection.
2. **Checkbox validation micro-benchmark**: 10–20 real survey pages (request samples from the support requester, or synthesize filled forms), hand-labeled states; report accuracy of VL-only vs. geometric fallback vs. combined. Gate: uncertain-flagging must catch what it gets wrong.
3. **Streamlit end-to-end**: run the app, upload each test document through the new page, exercise all tabs and downloads, then confirm legacy expander engines still run.
4. **Offline check**: run with network blocked (models pre-staged) — no runtime downloads.
5. Record peak VRAM and seconds/page for the UBELIX sizing note.

## Out of scope (explicitly deferred)
- Survey batch → one-row-per-respondent spreadsheet aggregation (next iteration, builds on the Markup regions in the IR).
- Docling born-digital lane, GROBID, vLLM serving, cross-page table merging, review-queue tooling — all per the report's later phases.
