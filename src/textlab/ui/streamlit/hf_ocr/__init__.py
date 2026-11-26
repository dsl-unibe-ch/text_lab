"""Hugging Face OCR models on the OCR page (``pages/OCR.py``).

* :mod:`.panel`: the model name, prompt, the user's downloaded models,
  upload, progress and results, drawn in an expander below historic
  handwriting.

It is a package of its own, not part of :mod:`textlab.ui.streamlit.ocr`,
because it uses :mod:`textlab.features.hf_ocr` rather than the OCR
feature.
"""
