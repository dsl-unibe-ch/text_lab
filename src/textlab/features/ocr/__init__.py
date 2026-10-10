"""Backend for the OCR feature.

Text, table and layout extraction from images and PDFs, with an automatic
pipeline and manual engine selection (olmOCR, GLM-OCR, PaddleOCR, EasyOCR),
plus searchable-PDF export.

The code still lives in the files below and moves here during the refactor (see
``docs/dev/architecture.md``):

- ``src/core/ocr_engine.py``
- ``src/core/auto_ocr.py``
- ``src/core/doc_ir.py``
- ``src/core/searchable_pdf.py``
- ``src/core/markup_detect.py``
- ``src/core/vision_enrich.py``
- ``src/core/paddle_ocr_worker.py``
- ``src/core/paddle_vl_worker.py``
"""
