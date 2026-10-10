# Translation

Document and text translation with local models, preserving formatting for
Office files and PDFs.

**Status:** moved, not refactored yet. The modules were moved here from
`src/core` with only import and path updates; the package docstring in
`__init__.py` says what each one does.

- Modules: `chunking.py`, `engine.py`, `format.py`, `gpu_memory.py`, `gpu_profile.py`, `hf_backend.py`, `lang_detect.py`, `messages.py`, `ollama_backend.py`, `pdf_checks.py`, `pdf_extract.py`, `pdf_workflow.py`, `quality.py`, `review.py`, `shield.py`
- UI: `Translate.py` in `src/textlab/ui/streamlit/pages/`
- Tests: `tests/test_translation.py`, `tests/test_translation_documents.py`, `tests/test_translation_gpu.py`, `tests/test_translation_limits.py`, `tests/test_translation_ollama.py`, `tests/test_translation_page.py`, `tests/test_translation_pdf.py`, `tests/test_translation_review.py`, `tests/test_translation_shield.py`, `tests/test_translation_syntax.py`

Once the feature is refactored, this file documents its pipeline, public API,
files written to disk, configuration and tests, as described in the developer
guide ([architecture](../../../../docs/dev/architecture.md#feature-packages)).
