# Translate

Translate turns text and documents from one language into another. You can paste a short text and see the translation next to it, or upload whole files (Word, PowerPoint, Excel, PDF, Markdown, plain text and subtitles) and download them translated, in the same format and layout as the original.

Typical uses are reading foreign-language articles and reports, preparing a translated version of a presentation or handout, translating survey answers or interview transcripts, and checking how a technical term is rendered in another language.

All processing runs on the university cluster. Your texts and documents are never sent to an external translation service such as DeepL or Google Translate.

## Quick start

**Translating a short text**

1. Choose a **Translation backend**. If you are unsure, keep the default, **NLLB-200 Distilled**.
2. Select the **Source language** and **Target language**.
3. Open the **Text** tab and paste your text into the left panel.
4. Click **Load model** and wait until the green "ready" message appears.
5. Click **Translate**. The translation appears in the right panel. Use **Copy translation** to copy it.

**Translating documents**

1. Choose a **Translation backend** and the **Target language**.
2. Open the **Document** tab and drop one or more files (or a ZIP archive) into the upload area.
3. Keep **Detect each document's language** and **Add side-by-side review file** ticked.
4. Click **Translate document(s)**.
5. Click the **Download** button to save the result.

The model is loaded automatically for documents, so there is no *Load model* step in the Document tab.

## Choosing a translation backend

The backend is the translation model that does the work. The choice applies to both tabs.

| | **NLLB-200 Distilled (600M)** | **NLLB-200 (3.3B)** | **MADLAD-400 (3B)** | **OPUS-MT** | **LLM (Ollama)** |
|---|---|---|---|---|---|
| **Best for** | Everyday use; the default | Higher quality, long technical text | Less widely spoken languages | Fast translation between common European language pairs | Dialects, informal text, control over tone |
| **Speed** | Fast | Slower | Slower | Very fast | Depends on the model |
| **Languages** | All languages in the list | All languages in the list | All languages in the list | Only some pairs | All languages in the list |
| **GPU memory** | Small | Needs a larger GPU | Needs a larger GPU | Small | Depends on the model |
| **Formality control** | No | No | No | No | Yes |

* **NLLB-200 Distilled** is a good starting point for almost everything.
* **NLLB-200 (3.3B)** gives noticeably better results on long or specialised sentences. If it fails to load, your session's GPU is too small; relaunch Text Lab with a larger GPU or use the distilled version.
* **MADLAD-400** is worth trying when the result from NLLB is poor for a less common language.
* **OPUS-MT** uses a separate small model for each language pair. If no model exists for your pair, Text Lab tells you so and you should switch to NLLB.
* **LLM (Ollama)** uses a general-purpose language model (the same kind used by Chat) instead of a dedicated translation model. When you select it, an extra **LLM model** menu appears; the models offered depend on the GPU of your session. LLMs cope better with dialects, colloquial language and unusual phrasing, and accept much longer passages in one piece, but they are slower and occasionally paraphrase more freely.

### Formality

When you use the **LLM (Ollama)** backend, a **Formality** option appears:

* **default**: the model follows the tone of the source text.
* **formal**: a polite, professional register (for example *Sie* in German, *vous* in French).
* **informal**: a relaxed, conversational register with informal forms of address.

The other backends do not support this option, so it is hidden for them.

## Languages

The language menus offer 59 languages, including all major European languages, Arabic, Hebrew, Persian, Hindi, Urdu, Bengali, Chinese (Simplified and Traditional), Japanese, Korean, and many others.

* Source and target must be different.
* Click the **⇄** button between the two menus to swap source and target. In the Text tab this also swaps the text in the two panels, so you can translate your result back.

## Glossary (term lock)

The glossary makes sure that specific words or phrases are always translated the way you want: names of institutes, product names, technical terms, or words that should not be translated at all.

1. Open **Glossary / term lock** above the tabs.
2. In each row, enter the term as it appears in your source text in the **Source** column, and the exact wording you want in the translation in the **Target** column. To keep a term untranslated, enter it identically in both columns.
3. Add more rows as needed (up to 50 terms).

The heading of the section shows how many terms are active. Rows with an empty source or target are ignored.

* The glossary applies to both the Text and the Document tab.
* Longer terms take priority, so an entry for `University of Bern` is used instead of an entry for `Bern` where both match.
* Matching ignores upper and lower case by default, so `Bern` also matches `BERN`. Tick **Case-sensitive matching** to match only the exact spelling.
* In languages written with Latin, Cyrillic or Greek letters, terms are matched as whole words, so `art` does not match inside `article`. In other scripts (for example Chinese or Arabic) a term also matches inside longer words.
* The target term is inserted exactly as you typed it. It is not adapted to grammar (for example plural or case endings), so prefer the form that fits most sentences.

## Translating text

The Text tab is a split-screen editor: your text on the left, the translation on the right.

### Entering text

* Paste or type your text into the left panel. Below it you see the number of characters and words.
* Texts up to **5,000 characters** are recommended. Longer texts still work, but the counter turns red and translation takes longer; for long material, the Document tab is more convenient.
* Text copied from a PDF or an email often has line breaks in the middle of sentences. Text Lab joins these lines before translating, so the model sees complete sentences. Breaks after a finished sentence and blank lines between paragraphs are kept.

### Detecting the source language

If you do not know the language of a text, click **Detect**. A badge shows the detected language and how certain the detection is:

* **green** (85% or more): reliable.
* **yellow** (60–84%): probably right, but check.
* **red** (below 60%): uncertain; choose the language yourself.

If the detected language differs from the selected source language, click **Use ... as source** to switch. Detection recognises 20 common languages; for other languages, select the source language manually.

### Loading the model

Before the first translation, click **Load model**. This loads the translation model onto the GPU of your session.

* The first time a model is used it may need to be downloaded, which can take one to two minutes. Afterwards loading takes only seconds.
* You need to load again whenever you change the backend, the LLM model or the language pair. The *Translate* button stays disabled until the selected model is ready.

### Translating

Click **Translate**. A progress bar is shown for longer texts. When the translation is ready:

* Select text in the right panel to copy part of it, or click **Copy translation** to copy everything.
* Below the panel you see the character and word count and the backend used.

### What stays unchanged

Some parts of a text should never be translated. Text Lab protects them automatically, so the model cannot alter them:

* web addresses (URLs), email addresses and file paths
* Markdown links and images (the visible link text is translated, the address is not)
* inline code and code blocks
* LaTeX formulas such as `$E = mc^2$`
* HTML tags
* placeholders such as `{name}` or `%s`

If the model damages one of these protected parts, Text Lab retries the affected passage in smaller pieces. If that also fails, it shows an error instead of giving you a silently broken translation.

## Translating documents

### Supported files

| Format | What is translated | What is kept |
|---|---|---|
| **Word** (`.docx`) | Body text, tables, headers and footers, footnotes and endnotes, comments | Paragraph styles, headings, lists, tables, images, hyperlinks |
| **PowerPoint** (`.pptx`) | Text in all shapes, text boxes, tables and grouped shapes, speaker notes | Slide layout, positions, images, colours, tables |
| **Excel** (`.xlsx`) | Text cells and cell comments | Numbers, dates, formulas, sheet names, column widths, merged cells, styles |
| **PDF** (`.pdf`) | All text, including scanned pages (see [PDF files](#pdf-files)) | Layout (PDF output), or structure and figures (Markdown output) |
| **Markdown** (`.md`) | Headings, paragraphs, lists, tables, block quotes | Code blocks, links, formulas, HTML |
| **Plain text** (`.txt`) | All text | Paragraph structure |
| **Subtitles** (`.srt`, `.vtt`) | The subtitle text | One line per line, so the subtitle structure is not merged |
| **ZIP** (`.zip`) | Any mix of the formats above | Files are translated one by one |

Older formats (`.doc`, `.ppt`, `.xls`) are not supported. Open them in Office and save them in the newer format first. Files of other types inside a ZIP archive are skipped with a warning.

**Known limitations**

* **Word and PowerPoint**: each paragraph is translated as a whole, because word order changes between languages. As a consequence, formatting of single words inside a paragraph (for example one bold word in a sentence) is lost; the paragraph takes the formatting of its first word. Paragraph styles, headings and lists are kept.
* **Excel**: sheet names are not translated, because formulas refer to them. Text inside charts and images is not translated.
* **Text in images** is not translated in any format.

### Uploading

Drop your files into the upload area or click it to browse. You can select several files at once, upload a ZIP archive, or combine both. All files are translated into the same target language with the same backend and glossary.

If your session has no GPU, Text Lab shows a note: translation then runs on the CPU, which is much slower, and scanned PDFs cannot be processed.

### Options

* **Detect each document's language** (on by default): Text Lab reads several paragraphs spread across each document and determines its language. This lets you upload documents in different source languages in one batch. If the language cannot be identified with confidence, the **Source language** selected above is used. Turn this off to use the selected source language for every file.
* **Add side-by-side review file** (on by default): adds a file showing the original and the translation paragraph by paragraph, as a web page (`.html`) and a Word document. This is the easiest way to check a translation. See [Reviewing a translation](#reviewing-a-translation).

### PDF files

PDF is a format for printing, not for editing, so Text Lab offers two kinds of output. When you upload a PDF, a **PDF outputs** menu appears:

* **Markdown** (text, tables and figures; best for reading): the content is extracted as a structured text document with headings, tables and the figures saved as images, then translated. This is usually the most readable and reliable result.
* **PDF** (same layout as the original): the translated text is placed back into the original pages, in the same position. Images, lines and the page design are kept.

Both are selected by default. Select only what you need: each output takes extra time. Text that appears in both is translated only once.

**Things to know about the PDF output**

* Translations are often longer than the original. When a translation does not fit into its original space, the font is made smaller so that nothing is cut off. In tight layouts text can become very small; the validation report tells you where.
* Equations are kept as they are, untranslated.
* Labels inside figures and text drawn on top of images are not translated.
* The PDF output is created only if every page has real, selectable text. Scanned pages cannot be rebuilt as a PDF; use the Markdown output instead (see below).

**Scanned PDFs**

Scanned pages (photographs or scans of paper documents) contain no text, only images. For the Markdown output, Text Lab first reads these pages with text recognition (OCR) and then translates the result. This needs a suitable GPU. If your session's GPU is not eligible, the Markdown output is blocked with a message rather than silently skipping pages; relaunch Text Lab with a larger GPU.

**Equations in Markdown**

By default, equations in the Markdown output are copied from the PDF as they are. Tick **Convert equations to LaTeX in the Markdown (slower)** to have pages with equations read by OCR so that formulas appear as proper LaTeX. Use this for scientific documents where you need the formulas to be editable.

**Validation report**

Every translated PDF comes with a `.translation-report.json` file. It lists, page by page, how each page was processed, which outputs were created, which were blocked and why, and any warnings (for example shrunk text or untranslated figure labels). The same information appears on the page under **PDF validation reports** after the run. If one output fails its checks, the other is still delivered.

### Running the translation

Click **Translate document(s)**. A progress panel shows:

* which file is being translated (for example `[2/5] report.docx`),
* the current stage (detecting language, translating sentences, reconstructing the file),
* the elapsed time and, during translation, an estimate of the time left.

Translating large documents can take several minutes. Text Lab warns you when the upload is larger than 5 MB. Click **Cancel** to stop; no files are produced in that case.

If a document is already in the target language, it is skipped with a message.

## Downloading the results

When the run is finished, a summary shows the elapsed time and the detected language of each file.

* **One output file**: a **Download** button for that file.
* **Several output files**: a **Download translated ZIP** button. If you uploaded several documents, each gets its own folder in the ZIP.

The results stay on the page until you start a new translation, so you can download them more than once.

### File names

Translated files keep the original name with the target language code added, for example for a German translation:

| File | Content |
|---|---|
| `report.deu_Latn.docx` | The translated document, same format as the original |
| `report.deu_Latn.side-by-side.html` | Review file, opens in any web browser |
| `report.deu_Latn.side-by-side.docx` | Review file as a Word table |
| `paper.deu_Latn.md` or `paper.deu_Latn.md.zip` | Markdown output of a PDF; a ZIP with an `assets/` folder when the PDF contains figures |
| `paper.deu_Latn.pdf` | PDF output with the original layout |
| `paper.deu_Latn.translation-report.json` | Validation report for a PDF |
| `notes.txt.ERROR.txt` | Explanation for a file that could not be translated |

In the ZIP download, a Markdown bundle from a PDF is unpacked, so the `.md` file and its `assets/` folder sit next to each other and the figures display correctly in a Markdown viewer.

## Reviewing a translation

Machine translation is very good but not perfect. Always review translations before you publish or share them, especially for legal, medical or official texts.

The **side-by-side review file** shows a numbered table with the original on the left and the translation on the right, paragraph by paragraph. It works for every input format and does not depend on the layout, so it is the quickest way to:

* spot sentences that were translated incorrectly or left out,
* check that names and technical terms are consistent (add them to the [glossary](#glossary-term-lock) and translate again if not),
* hand the translation to a native speaker for proofreading. The Word version can be edited and commented on directly.

## Tips for good results

* **Choose the right source language**, or keep language detection on. A wrong source language is the most common cause of poor translations.
* **Use the glossary** for names, institutions, abbreviations and field-specific terms. It is the most effective way to get consistent terminology.
* **Try another backend** if the result is unsatisfying: NLLB-200 (3.3B) for long technical sentences, MADLAD-400 for less common languages, LLM (Ollama) for dialects and informal language.
* **Prefer Word over PDF** when you have both. Word files contain the real document structure, so the translated file is cleaner.
* **For PDFs, read the Markdown output first.** It is usually more reliable than the rebuilt PDF; use the PDF output when you need the original look.

## Troubleshooting

Click a problem to see what to do.

??? question "The Translate button is greyed out"
    Click **Load model** first, and wait for the green "ready" message. You need to load again after changing the backend, the LLM model or the languages. The button is also disabled when the source panel is empty or the source and target languages are the same.

??? question "'OPUS-MT does not support direct translation between ...'"
    OPUS-MT only has models for some language pairs. Switch to NLLB-200 or MADLAD-400, which support all languages in the list.

??? question "The model fails to load or 'CUDA out of memory'"
    The selected model is too large for your session's GPU. Choose a smaller backend (NLLB-200 Distilled or OPUS-MT) or a smaller LLM, or relaunch Text Lab with a larger GPU.

??? question "'Part of the text is too long for the selected model to translate in one piece'"
    A single sentence or passage exceeds what the model can process at once. Split very long sentences, or switch to the LLM (Ollama) backend, which accepts much longer input.

??? question "'The model could not finish translating part of the text'"
    The translation of a passage was longer than the model can produce. Try NLLB-200 (3.3B), or the LLM (Ollama) backend if you already use it.

??? question "'The model did not keep the document's formatting (links, code, tables) intact'"
    The model damaged protected content such as links or code even after retrying. Try the LLM (Ollama) backend, or a larger LLM model if you already use it.

??? question "Detection shows a language code 'outside the dropdown'"
    The detected language is not available for translation, or detection was not confident. Select the source language yourself.

??? question "A document was skipped as 'already in' the target language"
    Language detection found that the file is already in the target language. If this is wrong, untick **Detect each document's language** and select the source language manually.

??? question "Only the Markdown version of my PDF was created"
    The PDF contains scanned or image-only pages, or uses a font that cannot display the translated characters. These cannot be rebuilt as a PDF. Use the Markdown output; the validation report names the affected pages.

??? question "'Scanned pages could not be read' or the Markdown output is blocked"
    The scanned pages need text recognition (OCR), which requires a suitable GPU. Relaunch Text Lab with a larger GPU and try again.

??? question "'This PDF is password-protected'"
    Remove the password (for example by saving an unprotected copy in your PDF reader) and upload the file again.

??? question "Text in the translated PDF is very small"
    The translation was longer than the original and had to be shrunk to fit. Use the Markdown output or the side-by-side review file for reading, or translate the original Word file if you have it.

??? question "Bold or italic words inside a sentence are lost in Word or PowerPoint"
    Each paragraph is translated as a whole, so formatting of single words cannot be carried over reliably. Reapply it in the translated file where needed.

??? question "Translation is very slow"
    Check whether the Document tab says *No GPU detected*; translation on the CPU is much slower. Large batches and the 3B models also take longer. Select only the PDF outputs you need, and leave *Convert equations to LaTeX* off unless you need it.

For help with translating material for your project, contact the Data Science Lab (DSL).
