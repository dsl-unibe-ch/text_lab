# OCR

Text, table and layout extraction from images and PDFs, with an automatic
pipeline and manual engine selection (olmOCR, GLM-OCR, PaddleOCR, EasyOCR),
plus searchable-PDF export.

**Status:** moved, not refactored yet. The modules were moved here from
`src/core` with only import and path updates; the package docstring in
`__init__.py` says what each one does.

- Modules: `auto_ocr.py`, `doc_ir.py`, `markup_detect.py`, `ocr_engine.py`, `paddle_ocr_worker.py`, `paddle_vl_worker.py`, `searchable_pdf.py`, `vision_enrich.py`
- UI: `OCR.py` in `src/textlab/ui/streamlit/pages/`
- Tests: `tests/test_enrichments.py`, `tests/test_markup.py`, `tests/test_noisy_scan.py`, `tests/test_regression.py`, `tests/test_searchable_pdf.py`, `tests/test_vl_session.py`

Once the feature is refactored, this file documents its pipeline, public API,
files written to disk, configuration and tests, as described in the developer
guide ([architecture](../../../../docs/dev/architecture.md#feature-packages)).
