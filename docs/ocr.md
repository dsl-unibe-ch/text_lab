# Optical Character Recognition (OCR)

The OCR tool allows you to extract editable text from scanned PDF documents or images.

## Supported Input
* **Files:** PDF documents (`.pdf`).
* **Note:** If your PDF has multiple pages, the process may take several minutes per page depending on the density of the text.

## Choosing an Engine

Text Lab provides three different AI engines for text extraction:

1.  **EasyOCR (Default):** Good for general purpose text and supports many languages.
2.  **PaddleOCR:** Often performs better on documents with complex layouts or tables.
3.  **OlmOCR:** A specialized pipeline for converting PDFs into clean Markdown.

## How to Use

1.  **Upload PDF:** Select your file.
2.  **Select Engine:** Choose one of the engines listed above.
3.  **Run OCR:** Click the button to start processing.
    * *Warning:* Do not close the tab while the "Running" indicator is active.

## Outputs

Once finished, you will see a side-by-side preview of the detected text boxes and the extracted content. You can download:
* **Extracted Text (.txt)**
* **Structured Data (.jsonl):** Useful for developers or data analysis.
* **Full Package (.zip):** Includes text, JSON, and layout images.
