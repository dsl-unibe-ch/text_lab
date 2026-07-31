# Optical Character Recognition (OCR)

The OCR tool extracts editable text, tables, figures, and formulas from scanned PDFs or images, and preserves the document's layout in a structured result. Optional local analysis can describe figures.

## Supported Input

* **Documents:** PDF (`.pdf`). *Note: For multi-page PDFs, processing may take a while per page depending on layout density.*
* **Images:** PNG, JPG, JPEG, BMP, TIFF
* **Archives:** ZIP (`.zip`) *(for batch processing multiple documents or images at once).*

## Automatic mode (default)

The recommended workflow is fully automatic — **there is no engine, language, or mode to choose.** Just upload a file and press **Parse document**.

Under the hood, Text Lab:

* **Detects layout** with a single multilingual model (PaddleOCR-VL, 109 languages), classifying every region as text, title, table, image/chart, formula, footnote, header/footer, seal, or **checkbox**.
* **Reads structure**, not just text: tables become spreadsheets, formulas become LaTeX, and figures are cropped out.
* **Takes a fast lane for born-digital PDFs:** pages that already contain a real text layer are read directly (no AI vision needed); only scanned/image pages go through the vision model.
* Optionally **describes detected figures/images** with Text Lab's local vision-language model. Generated descriptions and visible-text transcriptions remain separate from printed OCR content.

Results are shown in tabs:

* **Document** — the full text rendered in reading order, with tables, figures, and formulas inline.
* **Tables** — each detected table as an interactive spreadsheet, with a per-table CSV download.
* **Figures** — cropped images, charts, and seals, plus generated descriptions when requested.
* **Layout preview** — the page image with colour-coded boxes for each region type.

### Downloads

* **Searchable PDF:** the original pages, unchanged to the eye, with an invisible text layer so the scan becomes selectable and searchable. Produced automatically for a single document; for a batch it is a tick-box, since the cost multiplies by the number of files. The indexed text is Text Lab's own OCR result — Tesseract runs a second pass purely to locate each word, and never overrides what was transcribed. Table cells are included, since they really are printed; formulas are left out, because the exported LaTeX is not what the reader sees. Tables align less precisely than prose — the reading order of a grid and of flowing text do not always agree — so a mis-set cell stays searchable but highlights a wider span.

    The document language is detected automatically, per page, from the text Text Lab already extracted, so mixed-language documents are handled page by page. It only affects how precisely each word is positioned: on a scanned German questionnaire, using `deu` placed 87.6% of words on their exact box against 82.1% for `eng`. Pages with too little text keep the English default, and you can always set the language explicitly.
* **Plain text (.txt):** the text in reading order, with tables rendered as aligned columns and no markup. Best for text analysis or pasting elsewhere.
* **Word (.docx):** an editable document with real headings, Word tables, and embedded figures — for revising or citing the result.
* **Markdown + assets (.zip):** clean Markdown with the cropped figures alongside. Best for LLM/RAG ingestion.
* **JSON:** the canonical structured representation (regions, bounding boxes, confidence, markup states) for developers and further analysis.
* **Tables (.csv .zip):** one CSV per detected table.
* **Full bundle (.zip):** Markdown, plain text, `.docx`, JSON, assets, and table CSVs together.

Formats are generated from the same structured result, so they always agree; nothing is re-OCR'd per download.

### Which models produced the result (for citation)

Expand **Models used (for citation)** under the downloads to see exactly which models ran on your document. The same summary is written as `models_used.txt` in the bundle, once at the root of a batch result, and as a `models` block in each document's JSON. It is read off what actually ran rather than restated, so it stays correct if the configured models change:

* **Text recognition** — `PaddleOCR-VL 1.6 (PaddleOCR 3.7 / PaddleX 3.7)` for scanned pages and images. Born-digital pages taken by the fast lane report `PyMuPDF text extraction (no recognition model)`, because no recognition model is involved in reading an existing text layer.
* **Figure descriptions** — the local vision-language model, by name and tag (by default `qwen3-vl:30b-a3b-instruct`, served through Ollama), listed only when descriptions were actually generated.
* **Searchable-PDF word geometry** — `Tesseract`, with its version and the language pack used. It supplies word positions only and never contributes text.

When quoting results in a publication, keep the distinction: printed text is *transcribed* by the recognition model, while figure descriptions are *generated* by a vision-language model and are not part of the source document.

### Batch processing (ZIP)

Choose **Batch OCR (ZIP)**, upload a `.zip` of PDFs/images, and press **Parse batch**. The result ZIP mirrors your original folder structure and each file gets every format the single-document page offers, written by the same code: `document.md`, `document.txt`, `document.docx`, `document.json`, `document_searchable.pdf` (when the tick-box is set), plus `tables/` and `assets/` folders. A single `models_used.txt` sits at the root of the result rather than in every folder, listing every model that ran anywhere in the batch — a batch can mix scans and born-digital PDFs, which take different lanes.

### Survey/form response extraction (not enabled)

A question-level survey/form response extractor is present in the codebase but **switched off in the interface** while it is validated against a representative multi-document benchmark. It renders complete 300-DPI question sections to the local vision-language model, assigns its own IDs, verifies each marked position against an echoed choice label, derives selection constraints independently, and flags empty or inconsistent responses; OCR text and table HTML are never modified by it. Developers can reach it through `auto_ocr.process_document(..., extract_survey=True)`, or re-expose the UI controls, the **Responses** tab, and the form-responses CSV by setting `SURVEY_EXTRACTION_UI_ENABLED = True` in `src/pages/OCR.py`.

## Advanced: legacy engines

The previous engine-picker workflow is still available under the **“Advanced: legacy engines”** expander for cases where you want a specific engine's plain-text output:

1. **EasyOCR:** general-purpose text across many languages.
2. **PaddleOCR:** strong on complex or multi-column layouts.
3. **OlmOCR:** tuned for converting scientific PDFs into clean Markdown.
4. **GLM-OCR:** a large vision model with selectable Text / Table / Figure extraction modes.

Each legacy engine returns a single plain-text (or Markdown) string per page, exactly as before. For most documents — and for anything with tables or checkboxes — the automatic mode above is recommended.

> *Warning:* Do not close the tab while a "Running" indicator is active.

---

##  Data Privacy & Security

Text Lab is designed to handle highly sensitive, confidential, and proprietary documents (including unredacted PDFs, medical records, or unpublished manuscripts). We utilize a strict **"Local-Only, Self-Cleaning"** architecture to ensure your documents remain secure.

Here is exactly what happens to your data when you use the OCR tool:

* **100% Local Processing:** Your documents are **never** sent to external cloud services or APIs (like Adobe, Google Cloud Vision, or AWS). All text extraction is performed entirely on the University of Bern's secure UBELIX high-performance computing nodes.
* **Isolated User Workspaces:** The automatic pipeline, advanced AI vision models (like OlmOCR), and Batch ZIP processing all need to read files from disk (uploads, rasterised pages, cropped regions). To accommodate this, Text Lab generates a unique, temporary workspace located strictly within your private University home directory (`$HOME/ondemand_text_lab_ocr_jobs`). Other users on the cluster cannot access this space.
* **Instant Auto-Deletion (Self-Cleaning):** The exact moment the AI finishes extracting the text (or if the process encounters an error), the application runs an aggressive `shutil.rmtree()` command. **This guarantees that the entire temporary workspace—including your original documents, intermediate images, and raw data files—is instantly and permanently deleted from the hard drive.**
* **Ephemeral Results:** The final extracted text and tables presented on your screen are stored strictly in your browser's volatile memory (`st.session_state`). When you close the tab, refresh the page, or your HPC job ends, all traces of the document and its extracted text are destroyed by Python's garbage collector.
* **No AI Training:** The vision models only perform *inference* (looking at the image to extract text). They do not learn from your documents, and your data is never used to train or improve the AI.
