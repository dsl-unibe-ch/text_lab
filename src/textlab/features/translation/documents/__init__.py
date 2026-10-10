"""Translating documents in their own format.

Each translator takes the file's bytes (or text) and a ``translate_fn``
from :func:`textlab.features.translation.engine.make_translate_fn`, and
returns the same kind of file with the prose translated:

* :mod:`.markdown`: Markdown and plain text, keeping the structure.
* :mod:`.office`: Word, Excel and PowerPoint files.
* :mod:`.pdf`: a native PDF rebuilt with the translation in place.
* :mod:`.pdf_markdown`: a PDF converted to Markdown (with OCR for scanned
  pages), then translated.
* :mod:`.pdf_workflow`: both PDF outputs, built and checked separately, with
  a validation report.

:mod:`.pdf_checks` decides per page whether text can be read natively or
needs OCR, :mod:`.pdf_extract` extracts it, and :mod:`.pdf_blocks` holds the
heuristics for equations and text over figures.
"""
