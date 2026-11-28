# Translation

!!! note "Draft for the new interface"
    This guide describes the Translation page of the new Text Lab
    interface. It replaces [the current guide](../translate.md) when the
    new interface goes live.

Translation turns text and documents from one language into another. Type
or paste a text and read the translation next to it, or upload whole files
(Word, PowerPoint, Excel, PDF, Markdown, plain text and subtitles) and get
them back translated, in the same format and layout.

Everything runs on the University's computing cluster. Your texts and
documents are never sent to an outside translation service such as DeepL
or Google Translate.

## Quick start

**A text**

1. Keep **Text**, and choose the language under **To**. Under **From**,
   keep **Detect language** or choose the language yourself.
2. Type or paste the text on the left.
3. Press **Translate**, or <kbd>Ctrl</kbd> + <kbd>Enter</kbd>. The
   translation appears on the right; **Copy** copies it.

The first translation with an engine loads it, which can take a minute or
two; later ones take a second or two. Tick **Translate as I type** to
translate whenever you stop typing for a moment.

**Documents**

1. Choose **Documents**, and the language under **To**.
2. Drop one or more files, or a ZIP archive, on the form.
3. Press **Translate documents**.

The job runs in the background: you can leave the page and come back. When
it is done, download the files, or compare source and translation on the
page.

## Languages

The menus offer 62 languages, the most used ones first: all major European
languages, Arabic, Hebrew, Persian, Hindi, Urdu, Bengali, Chinese
(Simplified and Traditional), Japanese, Korean and many more.

- **Detect language** finds the language of a text when you translate it,
  and the menu then names it, for example *Detect language (German)*. It
  recognises 20 common languages; for others, or when it is not sure,
  Text Lab asks you to choose the language yourself. For documents, it
  finds each document's language, so one batch can mix languages.
- The button between the menus swaps the languages. With a translation on
  the right, it also moves the translation to the left and translates it
  back, which is a quick check of the result.

## Engine

The engine is the model that translates. Open **Engine** to change it; the
choice applies to texts and documents.

| Engine | Best for |
|---|---|
| **NLLB-200, 600M** | Everyday use, 200 languages. The default |
| **NLLB-200, 3.3B** | Long or specialised sentences. Slower; needs a GPU with much memory |
| **MADLAD-400, 3B** | Languages with little written text online, when NLLB's result is poor |
| **OPUS-MT** | Fast translation between common European languages; only some pairs have a model |
| **Language model** | Dialects, informal text, and a formal or informal tone. Slower |

With **Language model**, choose the model (the ones offered depend on your
session's GPU) and the **Formality**: *Default* follows the text,
*Formal* writes politely (*Sie*, *vous*), *Informal* casually (*du*, *tu*).

The engine stays loaded while you translate, until another tool needs the
GPU: then the next translation loads it again.

## Glossary

The glossary makes names and terms come out the way you want: names of
institutes, products, technical terms, or words that must stay
untranslated. Open **Glossary**, write the term as it appears in the source
on the left and its translation on the right; **Add a term** adds a row,
up to 50. To keep a term untranslated, write it the same on both sides.

- The glossary applies to texts and documents.
- Longer terms are matched first, so *University of Bern* wins over *Bern*.
- Upper and lower case are ignored unless you tick **Match upper and lower
  case exactly**.
- In languages written with Latin, Cyrillic or Greek letters, terms match
  whole words only, so *art* does not match inside *article*.
- The translation is put in as you wrote it, not adapted to grammar: write
  the form that fits most sentences.

Links, e-mail addresses, file paths, code, LaTeX formulas, HTML tags and
placeholders such as `{name}` are always kept as they are. If a model
damages one of them, Text Lab tries again in smaller pieces, and reports
an error rather than giving you a broken translation.

## Text

- The editor takes up to 20,000 characters; the counter below the text
  shows how many you have. For longer texts, save them as a `.txt` file
  and translate them as a document.
- Text copied from a PDF or an e-mail often breaks lines in the middle of
  sentences. Text Lab joins those lines first, so the model sees whole
  sentences; breaks after a finished sentence and blank lines stay.
- Below the translation you see the languages, the engine, the number of
  words and how long it took.
- The page keeps your text, the translation and your settings for the
  session: reload the page and they are still there.

Translations use the GPU one at a time, like all jobs. If a long job is
running, for example a transcription, the translation says that it waits
for it, and offers to cancel that job. **Stop** stops a translation.

## Documents

| Format | What is translated | What is kept |
|---|---|---|
| **Word** (`.docx`) | Body text, tables, headers and footers, footnotes, comments | Styles, headings, lists, tables, images, links |
| **PowerPoint** (`.pptx`) | Text in shapes, text boxes, tables, grouped shapes, speaker notes | Layout, positions, images, colours |
| **Excel** (`.xlsx`) | Text cells and comments | Numbers, dates, formulas, sheet names, widths, merged cells, styles |
| **PDF** (`.pdf`) | All text, scanned pages included | The layout (PDF output), or structure and figures (Markdown output) |
| **Markdown** (`.md`) | Headings, paragraphs, lists, tables, quotes | Code, links, formulas, HTML |
| **Text** (`.txt`) | All text | Paragraphs |
| **Subtitles** (`.srt`, `.vtt`) | The subtitles | Every line and its timing |
| **ZIP** (`.zip`) | Any of the above | Each file is translated on its own; others are left out and listed |

Older formats (`.doc`, `.ppt`, `.xls`) are not read: save them in the newer
format first. Text in images is not translated. In Word and PowerPoint each
paragraph is translated as a whole, so the formatting of single words
inside it (one bold word) is lost.

### Outputs

- **PDF files become** *Markdown* (text, tables and figures: the best to
  read and reuse) and *PDF* in the layout of the original. Choose what you
  need: each takes time, though text shared by both is translated once.
  Scanned pages are read with text recognition (OCR) first, which needs a
  suitable GPU. **Equations as LaTeX** reads the pages with equations with
  OCR too, so formulas become editable LaTeX in the Markdown; otherwise
  they are kept as they are.
- **Side-by-side file**: source and translation paragraph by paragraph, as
  a web page and a Word table, the easiest way to check a translation or
  hand it to a proofreader.

Every output is checked on its own. If one fails, for example a PDF whose
translation cannot fit its pages, the result says why, and the other
outputs are kept. Each PDF also gets a report of the checks
(`.translation-report.json`).

### The result

The job's record lists each document with its language and its files. With
several files, **All files** downloads them in one ZIP, a folder per
document. **Compare source and translation** shows the document's passages
side by side on the page, the first 300 of them; the side-by-side file has
all. A document that could not be translated says why: for example, it is
in the target language already, or its language could not be detected.

File names get the target language's code, for example for English:
`report.eng_Latn.docx`, `report.eng_Latn.side-by-side.html`,
`paper.eng_Latn.pdf`, `paper.eng_Latn.md` (a ZIP with an `assets/` folder
when the PDF has figures) and `paper.eng_Latn.translation-report.json`.

**Cancel** stops a job; nothing is kept then.

## Your data

- Everything runs on the University's computing cluster, UBELIX. Your
  texts and documents are never sent to outside services, and the models
  do not learn from them.
- Uploads and results are kept in your session's private folder on the
  compute node, which only your account can open. Uploads are deleted when
  their job ends; results when you remove them or the session ends. The
  editor's text and translation are kept in the session's memory until it
  ends.

## Tips

- **Check the source language.** A wrong one is the most common cause of
  a poor translation.
- **Use the glossary** for names, institutions and terms of your field.
- **Try another engine** when a result is poor: NLLB-200 3.3B for long
  technical sentences, MADLAD-400 for less common languages, a language
  model for dialects and informal language.
- **Prefer Word to PDF** when you have both: the translated file is
  cleaner.
- **Review before you publish.** Machine translation is good, not perfect:
  read the side-by-side file, especially for legal, medical or official
  texts.

## Troubleshooting

??? question "The translation waits"
    Another job is using the GPU, and translations take their turn. Wait,
    or cancel that job from the message.

??? question "'OPUS-MT has no model from ... into ...'"
    OPUS-MT has models for some language pairs only. Choose NLLB-200 or
    MADLAD-400, which translate all languages in the list.

??? question "An engine fails to load, or 'out of memory'"
    The model is too large for your session's GPU. Choose NLLB-200, 600M
    or OPUS-MT, a smaller language model, or start Text Lab with a larger
    GPU.

??? question "'The language of the text could not be told' or 'not certain'"
    Choose the language under **From**. Detection knows 20 common
    languages and needs a sentence or two.

??? question "'Part of the text is too long ...' or 'could not finish translating ...'"
    A passage is longer than the model can take or write at once. Split
    very long sentences, or use a language model, which takes much longer
    passages.

??? question "Only the Markdown of my PDF was made"
    The PDF has scanned pages, or a font that cannot show the translation.
    Such pages cannot be rebuilt as a PDF; the result and the report name
    them.

??? question "Text in the translated PDF is very small"
    The translation was longer than the original and was shrunk to fit.
    Read the Markdown or the side-by-side file, or translate the Word file
    if you have it.

For help with translating material for your project, contact the Data
Science Lab (DSL).
