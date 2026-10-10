"""Text, tables and layout from PDFs and images.

Interfaces use :mod:`.service` and the result model :mod:`.doc_ir`. The
automatic pipeline (:mod:`.pipeline`) reads born-digital pages directly
(:mod:`.native`) and recognizes the others with the PaddleOCR-VL worker
(:mod:`.vl_session`, :mod:`.paddle_vl_worker`); :mod:`.marks` and
:mod:`.markup_detect` handle checkboxes and marks, :mod:`.searchable_pdf`
the invisible text layer and :mod:`.vision_enrich` figure descriptions.

Manual engine selection still uses :mod:`.ocr_engine` and
:mod:`.paddle_ocr_worker` (until refactor phase 6b).

See ``README.md`` in this folder for the pipeline and the files written.
"""
