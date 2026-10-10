"""Backend for the OCR feature.

Text, table and layout extraction from images and PDFs, with an automatic
pipeline and manual engine selection (olmOCR, GLM-OCR, PaddleOCR, EasyOCR),
plus searchable-PDF export.

Moved from ``src/core`` with only import and path updates. The modules are
reorganized when the feature is refactored (see ``docs/dev/architecture.md``):

- ``auto_ocr``: the automatic OCR pipeline.
- ``doc_ir``: typed intermediate representation of an OCR result.
- ``markup_detect``: checkbox and survey-mark detection.
- ``ocr_engine``: result handling and previews for the manually selected
  engines.
- ``paddle_ocr_worker``: PaddleOCR subprocess, run in the ``paddle_backend``
  environment.
- ``paddle_vl_worker``: PaddleOCR-VL subprocess, run in the
  ``paddle_vl_backend`` environment.
- ``searchable_pdf``: PDF export with an invisible text layer.
- ``vision_enrich``: local vision-model client and optional enrichments.
"""
