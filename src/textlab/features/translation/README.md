# Translation

Translates pasted text and whole documents with local models: Hugging Face
translation models (NLLB-200, MADLAD-400, OPUS-MT) on the session's GPU, or
an LLM on the session's Ollama server. Documents come back in their own
format (Markdown, plain text, subtitles, Word, Excel, PowerPoint, PDF). Used
by the Translate page.

## Pipeline

Text (`service.translate_text`):

1. **Reflow**: sentences that the source hard-wrapped over several lines are
   rejoined (`documents.markdown.reflow_soft_wraps`).
2. **Shield**: links, inline code, math, HTML, placeholders and glossary
   terms are replaced by markers the model cannot alter (`shield`), and
   restored afterwards; damaged markers fail the translation
   (`ProtectedContentError`) instead of returning broken text.
3. **Translate** the remaining prose with the selected backend (`engine`).
   Hugging Face backends split it with the model's own tokenizer and
   batch it (`hf_backend`); Ollama gets chunks sized to its context
   (`ollama_backend`). An output cut off by a model limit is retried in
   smaller pieces, never returned incomplete (`chunking`).

Documents (`service.translate_documents`), for each file:

1. **Detect the language** from paragraphs spread across the file
   (`lang_detect`), unless the user turned detection off. A file already in
   the target language is reported and skipped.
2. **Translate in the file's format** (`documents`): only the prose is sent
   through the text pipeline above; structure, styles and layout are kept.
   One translator per source language holds a sentence cache, so text
   repeated across files and outputs is translated once.
3. **PDFs** give two outputs, built and checked separately
   (`documents.pdf_workflow`): Markdown (native text, plus OCR for scanned
   pages) and a reconstructed PDF. A failed output is reported and never
   discards the other; a JSON validation report lists both.
4. **Review file**: the source and translation side by side, as HTML and
   Word (`review`).

Translation runs in the Streamlit process, not in a worker: the text editor
relies on a model that stays loaded between clicks. Another feature that
needs the GPU releases it through `common.gpu_manager`, which calls
`engine.free_translation_vram`.

## Public API

Everything interfaces need is in `service.py`:

| Name | Purpose |
|---|---|
| `TranslationOptions` | Backend, languages, Ollama model, formality, glossary |
| `DocumentOptions` | PDF outputs, LaTeX for equations, language detection, review file |
| `translate_text(text, options, on_progress=, on_notice=)` | Translate pasted text |
| `unpack_uploads(uploads)` | Replace ZIP archives by their supported members |
| `translate_documents(files, options, document_options, on_progress=)` | Translate a batch; returns `DocumentResults` |
| `translate_document(name, data, translate_fn, target_code, ...)` | One file, in its format |
| `load_backend`, `backend_ready`, `load_signature` | Load a model ahead of use and check that it is still loaded |
| `supports_pair(backend, source, target)` | Whether OPUS-MT has a model for a pair |
| `review_files`, `outputs_zip`, `mime_type` | Review files and downloads |

Other modules the page uses directly: `engine` (the backend table
`BACKENDS` and formality choices), `lang_detect.detect_language`,
`gpu_profile.detect_gpu_profile` and `messages.describe_error`.

A minimal use:

```python
from textlab.features.translation import service

options = service.TranslationOptions(
    backend="nllb",
    source_code="deu_Latn",
    source_name="German",
    target_code="eng_Latn",
    target_name="English",
)
service.load_backend(options)
english = service.translate_text("Guten Tag.", options)

results = service.translate_documents([("report.docx", data)], options)
for name, content in results.outputs:
    ...
```

## Layout

```
translation/
├── service.py         # the API above
├── engine.py          # backend table, model loading, dispatch
├── hf_backend.py      # tokenizer-aware batching and recovery
├── ollama_backend.py  # prompted translation within a context budget
├── chunking.py        # lossless splitting, limit errors
├── shield.py          # markers for protected content and glossary terms
├── gpu_memory.py      # the lock serializing model use, CUDA cleanup
├── gpu_profile.py     # batch sizes for the allocated GPU
├── lang_detect.py     # source-language detection
├── messages.py        # error messages for users
├── review.py          # side-by-side review files
├── documents/
│   ├── markdown.py      # Markdown and plain text
│   ├── office.py        # Word, Excel, PowerPoint
│   ├── pdf.py           # PDF rebuilt with the translation in place
│   ├── pdf_markdown.py  # PDF to Markdown, then translation
│   ├── pdf_workflow.py  # both PDF outputs and the validation report
│   ├── pdf_checks.py    # per-page routing: native, OCR or blank
│   ├── pdf_extract.py   # native extraction; OCR for the pages that need it
│   └── pdf_blocks.py    # equation and figure-label heuristics
├── cli.py             # batch command (placeholder)
└── tests/
```

`documents/pdf_extract.py` is the only place that uses another feature: it
calls the OCR service (`features.ocr.service.process_document`) for scanned
pages and reads the result model (`features.ocr.doc_ir`).

## Files written

| What | Where | Removed |
|---|---|---|
| PDF pages sent to OCR, and OCR's own temporary files | `translation/ocr-*` in the job workspace | When the document's OCR finishes |

Everything else stays in memory: uploads, translations and the results the
page offers for download. Nothing is written outside the workspace.

## Configuration

No site settings. Models are read from `HF_HOME`, the model store the
launch script mounts, and are downloaded there on first use if missing. The
Ollama models offered are listed in `common/models.json`; the server
address comes from `OLLAMA_HOST`, set by the launch script.

## Tests

In `tests/`, all without models, GPU or Ollama:

- `test_service.py`: dispatch per format, PDF outputs and reports, ZIP
  uploads, batches with language detection and failures, review files, the
  download ZIP, progress and time estimates.
- `test_translation.py`: reflowing hard-wrapped text.
- `test_translation_shield.py`: protected content and glossary terms.
- `test_translation_limits.py`: splitting, limits and retries.
- `test_translation_ollama.py`: context budgets and completion checks.
- `test_translation_gpu.py`: model loading and release, batch sizes, the
  lock, Ollama model ownership.
- `test_translation_documents.py`, `test_translation_pdf.py`: Office and
  PDF formats, page routing, OCR subsets, layout checks.
- `test_translation_review.py`: review files, language detection, messages.
- `test_translation_page.py`: the page's imports resolve.
- `test_translation_syntax.py`: the sources parse as Python 3.11.

## Batch use (planned)

`cli.py` is a placeholder for a `textlab translate` command for Slurm batch
jobs. It will call `service.translate_documents` with a progress callback
that writes log lines.
