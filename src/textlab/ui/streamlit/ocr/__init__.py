"""Parts of the OCR page (``pages/OCR.py``), which is too large for one file.

* :mod:`.state`: session-state keys shared by the page's workflows.
* :mod:`.results`: the result of one document: downloads and tabs.
* :mod:`.review`: reviewing extracted survey responses.
* :mod:`.legacy`: manual engine selection (EasyOCR, PaddleOCR, OlmOCR,
  GLM-OCR); still holds its engine code and is replaced by engine adapters
  in the OCR feature (refactor phase 6b).
"""
