# OCR

!!! note "Draft for the new interface"
    This guide describes the OCR page of the new Text Lab interface. It
    replaces [the current guide](../ocr.md) when the new interface goes
    live. So far it covers **Standard**, the default; the other tools (choosing an
    engine, historic handwriting, Hugging Face models) follow as they are
    built.

OCR reads the text in scans, photos and PDFs: the text in reading order,
with headings, tables, formulas and figures, and keeps where each part
sits on the page. You get it back as plain text, Word, Markdown, a
searchable PDF and more.

Everything runs on the University's computing cluster. Your documents are
never sent to outside services such as Adobe, Google or Amazon.

## Quick start

1. Keep **Standard** and drop a PDF or an image on the form, or a ZIP of
   many.
2. Press **Read the document**.

The job runs in the background; you can leave the page and come back. When
it is done, the record shows each page beside the text read from it, and
the downloads.

## What it reads

PDF, PNG, JPG, BMP and TIFF files, or a ZIP of them. A ZIP is read as a
batch (see [A batch](#a-batch)).

## How it reads

There is no engine or language to choose. Each page is looked at first:

- **A page with its own text**, as in a PDF made on a computer, is read
  from that text, at once and exactly.
- **A scan or a photo** is read by a recognition model (PaddleOCR-VL, 109
  languages), which also finds the layout: titles, text, tables, figures,
  formulas, headers and footers, seals and check boxes. Tables become real
  tables, formulas LaTeX, and figures are cut out. It takes from a few
  seconds to about a minute a page.

Pages whose own text is unusable (equations that come out as junk, a
mis-encoded font) go to the model as well.

## Options

- **Searchable PDF** (on by default): the original pages with an invisible
  layer of the text, so you can search, select and copy in any PDF reader.
  A second pass places each word; the text itself is always the one read
  above. **Language, for placing the words** helps that pass place the
  words more precisely. It is not the language of the recognition, which
  reads 109 languages by itself: the list holds the languages installed
  for placing words (English, German, French, Italian, Spanish). *Detect
  page by page* works for most documents, also ones that mix languages;
  in other languages the words are placed a little less precisely, so a
  search may highlight a wider span.
- **Highest quality**: every page goes through the recognition model, also
  pages with their own text. Best for forms and complex layouts; slower.
- **Describe figures**: a vision model describes each figure and reads the
  text in it. The descriptions are *generated*: they are not part of the
  document and are always shown apart from it. Loading the model takes
  about a minute.

## The result

- **Pages**: each page beside its text. The regions the model found are
  outlined on the page in the colour of their type, with a legend; point
  at an outline, or at a block of text, and both are marked. Click an
  outline to bring its text into view. Choose the page above it, or step
  through with the arrows.
- **Tables**: each table, with its CSV.
- **Figures**: the figures, charts and seals, with their captions and,
  when asked for, their generated descriptions.
- **Models**: which models made the result, for citation. Printed text is
  *transcribed*; figure descriptions are *generated*. The same list is in
  `models_used.txt` in the **Everything** download, and under `models` in
  the JSON.

Check boxes are shown as read: filled, empty, or with a dashed outline when
the reading is not certain. The record says how many were uncertain.

### Downloads

| Download | What it is |
|---|---|
| **Everything** (`_bundle.zip`) | All of the below in one ZIP |
| **Plain text** (`.txt`) | The text in reading order, tables as aligned columns. For text analysis or pasting |
| **Word** (`.docx`) | Headings, tables and figures, to edit or cite |
| **Markdown with figures** (`_markdown.zip`) | Markdown and an `assets/` folder of the figures. For language models and search tools |
| **Searchable PDF** (`_searchable.pdf`) | The original pages with an invisible text layer |
| **JSON with boxes** (`.json`) | Every region with its box, confidence and type, for further analysis |
| **Tables, CSV** (`_tables.zip`) | One CSV per table |

All of them come from the same result, so they agree with each other and
with the page.

## A batch

Drop a ZIP of PDFs and images. The recognition model is loaded once for all
of them, so a batch is much faster than the files one by one. The record
lists each file with its pages, how many were read from their own text,
what was found and how long it took. The download is one ZIP with a folder
per file, as in your archive, each with `document.md`, `document.txt`,
`document.docx`, `document.json`, the tables and figures, and
`document_searchable.pdf` when you asked for it. `models_used.txt` at the
top lists every model that ran in the batch.

**Cancel** stops a job at once; nothing is kept then.

## Your data

- Everything runs on the University's computing cluster, UBELIX. Your
  documents are never sent to outside services, and the models do not
  learn from them.
- Uploads, the page images made while reading, and the results are kept in
  your session's private folder on the compute node, which only your
  account can open. Uploads and page images are deleted when their job
  ends; results when you remove them or the session ends.
