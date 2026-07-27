# Optical Character Recognition (OCR)

The OCR tool extracts editable text, tables, figures, and formulas from scanned PDFs or images, and preserves the document's layout in a structured result. Optional local analysis can describe figures or extract question-level survey responses.

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
* Optionally runs the **experimental survey/form response extractor** on complete 300-DPI question sections with the local vision-language model. Paddle-recognised controls and conservative option-pattern evidence determine which sections are submitted; ordinary tables and isolated circle-like shapes are excluded. The response model receives the section image, not Paddle's inferred answer schema, and returns one fixed structure for simple questions, conditional subquestions, ratings, and matrices. TextLab assigns IDs, verifies each marked position against an echoed choice label, derives selection constraints independently, and flags empty or inconsistent responses. OCR text and table HTML are immutable. Until a representative multi-document benchmark approves a model, all extracted groups are flagged for review.

Results are shown in tabs:

* **Document** — the full text rendered in reading order, with tables, figures, and formulas inline.
* **Tables** — each detected table as an interactive spreadsheet, with a per-table CSV download.
* **Figures** — cropped images, charts, and seals, plus generated descriptions when requested.
* **Responses** — question/row-level selected, cancelled, or ambiguous responses and their evidence crops when survey extraction was requested. Review flags are shown instead of silently changing the OCR.
* **Layout preview** — the page image with colour-coded boxes for each region type.

### Downloads

* **Markdown + assets (.zip):** clean Markdown with the cropped figures alongside.
* **JSON:** the canonical structured representation (regions, bounding boxes, confidence, markup states) for developers and further analysis.
* **Tables (.csv .zip):** one CSV per detected table.
* **Form responses (.csv):** one row per question or matrix row when survey responses were extracted.
* **Full bundle (.zip):** Markdown, JSON, assets, table CSVs, and semantic form-response CSV together.

### Batch processing (ZIP)

Choose **Batch OCR (ZIP)**, upload a `.zip` of PDFs/images, and press **Parse batch**. The result ZIP mirrors your original folder structure, with a `document.md`, `document.json`, a `tables/` folder, an `assets/` folder, and (when requested) `form_responses.csv` for each file.

When **Extract survey/form responses** is enabled, **All files use the same questionnaire layout** appears. This learns normalized question crop locations from the first document and reuses them for later files. Each response image is still reconstructed and analysed independently; answer states are never copied between respondents. A later optimization will freeze a validated first-document structure and use an ID-only mark pass for subsequent files, but that switch remains disabled until mandatory layout-drift checks are benchmarked.

For reproducible A/B audits, `schema-free-v2` is the default survey contract and `paddle-id-v1` retains the repaired Paddle-derived baseline. Version 2 clamps crops to question boundaries, removes explicitly numbered adjacent-question spill, enforces one-row versus matrix structure, scopes printed multiple-answer cues to the current subquestion, and flags uncertain or missing matrix marks. The audit helper accepts `--contract` and saves exact crops, prompts, schemas, raw replies, generation settings, and a manual boundary/choice-clipping review template. The old `schema-free-v1` command spelling is accepted as an alias for v2 so it cannot silently run the weaker validator. Ranking questions and continuous mark-anywhere-on-a-line scales are currently out of scope.

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
