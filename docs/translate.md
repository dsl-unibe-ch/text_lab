# Translate

The Translate feature provides advanced, secure translation for both raw text and complex documents. It uses Large Language Models (LLMs) to deliver high-quality translations while preserving the original structure and formatting of your files.

## Workflows

The tool is divided into two distinct workflows:

### Text Translation

A split-screen editor designed for translating text snippets.

*   **Auto-detection:** Automatically detects the source language.
*   **Counters:** Provides character and word counts.
*   **Glossary / Term-lock:** Ensure specific terms remain untranslated or are translated exactly as you specify.
*   **Formality Control:** Adjust the tone of the translation (e.g., formal or informal).
*   **Language-swap:** Quickly switch between source and target languages.

### Document Translation

Upload entire files or batches of files for translation. The tool parses the document, translates the content, and reconstructs the file in its exact original format.

*   **Supported Formats:** `.md`, `.txt`, `.srt`, `.vtt`, `.pdf`, `.docx`, `.xlsx`, `.pptx`, or a `.zip` archive containing any mix of these.
*   **Markup Preservation:** Structural elements like Markdown links, LaTeX equations, inline code, HTML tags, and formatting are carefully preserved intact.
*   **Batch Upload:** Process multiple files at once by uploading them in a ZIP archive.

## Data Privacy & Security

All translations are performed locally on the university cluster. Your sensitive text and documents are never sent to external translation services (like DeepL or Google Translate), ensuring complete data privacy and security.
