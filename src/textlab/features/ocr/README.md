# OCR

Text, table and layout extraction from images and PDFs, with an automatic
pipeline and manual engine selection (olmOCR, GLM-OCR, PaddleOCR, EasyOCR),
plus searchable-PDF export.

**Status:** not migrated yet. The code currently lives in:

- `src/core/ocr_engine.py`
- `src/core/auto_ocr.py`
- `src/core/doc_ir.py`
- `src/core/searchable_pdf.py`
- `src/core/markup_detect.py`
- `src/core/vision_enrich.py`
- `src/core/paddle_ocr_worker.py`
- `src/core/paddle_vl_worker.py`
- UI: `src/pages/OCR.py`

Once the feature is refactored, this file documents its pipeline, public API,
files written to disk, configuration and tests, as described in the developer
guide ([architecture](../../../../docs/dev/architecture.md#feature-packages)).
