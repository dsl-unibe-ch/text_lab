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

A question-level survey/form response extractor is present in the codebase but **switched off in the interface** while it is validated against a representative multi-document benchmark. It renders complete 300-DPI question sections to the local vision-language model, assigns its own IDs, verifies each marked position against an echoed choice label, derives selection constraints independently, and flags empty or inconsistent responses; OCR text and table HTML are never modified by it. Developers can reach it through `OcrOptions(extract_survey=True)` in `textlab.features.ocr.service`, or re-expose the UI controls, the **Responses** tab, and the form-responses CSV by setting `SURVEY_EXTRACTION_UI_ENABLED = True` in `src/textlab/ui/streamlit/ocr/state.py`.

## Advanced: legacy engines

The previous engine-picker workflow is still available under the **“Advanced: legacy engines”** expander for cases where you want a specific engine's plain-text output:

1. **EasyOCR:** general-purpose text across many languages.
2. **PaddleOCR:** strong on complex or multi-column layouts.
3. **OlmOCR:** tuned for converting scientific PDFs into clean Markdown.
4. **GLM-OCR:** a large vision model with selectable Text / Table / Figure extraction modes.

Each legacy engine returns plain text (or Markdown) per page. For most documents — and for anything with tables or checkboxes — the automatic mode above is recommended.

The downloads of a single document are the whole text (`.txt`), the engine's result with positions and scores (`.json`; OlmOCR gives its own `.jsonl` record), and a ZIP with both plus one text and JSON file per page. In a batch, each file gets a folder with the same files, mirroring your ZIP's folders. If an engine fails on one file, the batch stops with an error naming it.

> *Warning:* Do not close the tab while a "Running" indicator is active.

## Historic German handwriting

The **"Historic German handwriting (Kurrent / medieval manuscripts)"**
expander, below the legacy engines, transcribes **handwritten historical
German documents** such as letters, minutes and charters. It uses
handwriting recognition models of the Digital Humanities group of the
University of Bern ([dh-unibe](https://huggingface.co/dh-unibe)), trained
specifically on such material. The rest of the OCR page is unaffected, and
the two keep their results apart.

### Models

Choose the model trained on material closest to your documents. The
**Models used** box under the model choice shows, before you upload
anything, which models will process your documents, with links to their
pages on Hugging Face.

| Choice in Text Lab | Model | Trained on | Character error rate |
|---|---|---|---|
| **German Kurrent, 19th century** | [`dh-unibe/qwen3vl-german-xix-v2`](https://huggingface.co/dh-unibe/qwen3vl-german-xix-v2) | Handwritten German in Kurrent script, 19th century, mainly Swiss records such as Federal Council minutes and parliamentary protocols | 7.7% on hands it has not seen (Federal Council minutes 1848-1903) |
| **Medieval German, 14th-17th century** | [`dh-unibe/qwen3vl-medieval-german-v3`](https://huggingface.co/dh-unibe/qwen3vl-medieval-german-v3) | Medieval and early modern German hands: the Rats- und Richtebücher, the Bullinger correspondence, the Königsfelden charters and the AAEB collections | about 14% |

Both are fine-tuned versions (LoRA adapters) of the vision-language model
[`Qwen/Qwen3-VL-4B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct).
The text lines are found by PaddleOCR's text detector
(`PP-OCRv5_server_det`).

A character error rate of 7.7% means that about one character in thirteen
is wrong; 14% is about one in seven.

### How it works

1. **Find the lines.** The text detector finds the text on each page.
   Pieces of text side by side at the same height are joined into lines,
   and the lines are put in reading order: top to bottom, and left to right
   for lines next to each other.
2. **Read each line.** The chosen model reads every line separately. The
   models were trained on single lines and cannot read a whole page at
   once, which is why the page is split first.

### Use

* **Input:** PDF (rendered at 300 DPI), PNG, JPG, JPEG, TIFF (every page of
  a multi-page TIFF) and BMP, or a ZIP of these. Scans at 300 DPI or more
  give the best results.
* **Single document:** upload a file and press **Transcribe**. The result
  shows, page by page, the image with the detected lines outlined and
  numbered, next to the numbered transcription.
* **Batch (ZIP):** upload a `.zip` and press **Transcribe batch**. The
  result ZIP mirrors your folders, with one folder per file.

### Downloads

* **Text (.txt):** one line per detected line, pages separated by a blank
  line.
* **JSON:** every line with its position on the page (in pixels of the page
  image).
* **All files with PAGE XML (.zip):** the text and JSON, plus one PAGE XML
  file per page in `page_xml/`, which can be imported into
  [eScriptorium](https://escriptorium.inria.fr/) or Transkribus to correct
  the transcription against the image. For a PDF or a multi-page TIFF the
  page images the positions refer to are included next to the XML; for a
  single image the XML refers to your uploaded file.
* **Citation:** `models_used.txt` in the ZIP, and **Models used (for
  citation)** under the downloads, name the line detector and the
  recognition model with the base model it adapts, for example
  `dh-unibe/qwen3vl-german-xix-v2 (LoRA adapter on Qwen/Qwen3-VL-4B-Instruct)`.

### Limitations

Treat the result as a draft to correct, not as an edition.

* **Errors are expected.** The error rates above were measured on the
  models' test material; on hands, scribes or periods unlike the training
  data they are higher. The models are language models, so a wrong reading
  can still look like fluent, plausible German. Check names, numbers and
  dates against the image in particular.
* **German historical handwriting only.** The models were trained on
  19th-century Kurrent and on medieval and early modern German hands. They
  are not meant for print, modern handwriting or other languages; use the
  automatic mode above for those.
* **Line detection is the weak point.** The text detector is a general one,
  not trained on manuscripts. It can miss faint or very short lines, split
  a line in two, or join lines that touch, and it can struggle with dense,
  strongly slanted or curved writing. A missed line is missing from the
  text, and two lines detected as one are read as one, so compare the
  outlined lines with the page.
* **Simple reading order, no layout.** Lines are ordered top to bottom and
  left to right. Two-column pages can come out interleaved, and marginal
  notes and interlinear additions are placed by their position, not by
  where they belong in the text. Paragraphs, tables, headings and figures
  are not recognised.
* **No context between lines.** Each line is read on its own, so a word
  hyphenated across two lines is not joined, and the model cannot use the
  previous line to resolve an unclear word.
* **Transcription conventions.** Abbreviations, special characters and
  spelling follow the conventions of the training transcriptions, which may
  differ from your editorial guidelines.
* **No confidence scores and no searchable PDF.** Unlike the automatic
  mode, there is no per-word confidence and no searchable PDF export.
* **Speed.** Every run first loads the model onto the GPU, which takes
  about a minute; after that, a page takes a few seconds. A large batch can
  take a long time.

---

##  Data Privacy & Security

Text Lab is designed to handle highly sensitive, confidential, and proprietary documents (including unredacted PDFs, medical records, or unpublished manuscripts). We utilize a strict **"Local-Only, Self-Cleaning"** architecture to ensure your documents remain secure.

Here is exactly what happens to your data when you use the OCR tool:

* **100% Local Processing:** Your documents are **never** sent to external cloud services or APIs (like Adobe, Google Cloud Vision, or AWS). All text extraction is performed entirely on the University of Bern's secure UBELIX high-performance computing nodes.
* **Private session workspace:** The automatic pipeline, advanced AI vision models (like OlmOCR), historic German handwriting, and Batch ZIP processing all need to read files from disk (uploads, rasterised pages, cropped regions and line crops). Text Lab keeps these in a private workspace that is created for your session on the compute node and that only your user account can open. Nothing is written to your home directory.
* **Automatic deletion:** As soon as the extraction finishes, or fails, the working folder of that run is deleted, including your original documents, intermediate images and raw data files. If a run is interrupted, for example because the session ends, the whole session workspace is deleted when the session ends.
* **Results in your browser session:** The extracted text and tables shown on the page are kept only in your open session (Streamlit's `st.session_state`). When you close the tab, reload the page, or your Text Lab session ends, they are gone. Download anything you want to keep.
* **No AI Training:** The vision models only perform *inference* (looking at the image to extract text). They do not learn from your documents, and your data is never used to train or improve the AI.
