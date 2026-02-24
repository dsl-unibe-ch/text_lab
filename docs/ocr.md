# Optical Character Recognition (OCR)

The OCR tool allows you to extract editable text, tables, and structured data from scanned PDF documents or raw images.

## Supported Input
* **Documents:** PDF (`.pdf`). *Note: If your PDF has multiple pages, the process may take several minutes per page depending on the density of the text.*
* **Images:** PNG, JPG, JPEG, BMP, TIFF

## Choosing an Engine

Text Lab provides four different AI engines for text extraction, each with its own strengths:

1.  **EasyOCR:** Good for general-purpose text and supports a wide variety of languages.
2.  **PaddleOCR:** Often performs better on documents with complex layouts or multi-column text.
3.  **OlmOCR:** A specialized pipeline optimized for converting scientific PDFs into clean Markdown.
4.  **GLM-OCR:** A state-of-the-art large vision model. It is exceptional for highly complex layouts and allows you to specifically target what to extract (e.g., General Text, specific Tables, or Figures).

## How to Use

1.  **Upload File:** Select your PDF or image.
2.  **Select Engine:** Choose one of the engines listed above.
    * *If using EasyOCR/PaddleOCR:* Select the language of the document.
    * *If using GLM-OCR:* Select the extraction mode (Text, Table, or Figure).
3.  **Run OCR:** Click the button to start processing.
    * *Warning:* Do not close the tab while the "Running" indicator is active.

## Outputs

Once finished, you will see a side-by-side preview of the document and the extracted content. You can download:
* **Extracted Text (.txt):** The raw text or Markdown.
* **Table Data (.csv):** If the AI detects a structured table, you can download it directly as a CSV for Excel/Python.
* **Structured Data (.jsonl):** Useful for developers or bulk data analysis.
* **Full Package (.zip):** Includes all text, JSON, and layout images in one convenient package.

---

## 🔒 Data Privacy & Security

Text Lab is designed to handle highly sensitive, confidential, and proprietary documents (including unredacted PDFs, medical records, or unpublished manuscripts). We utilize a strict **"Local-Only, Self-Cleaning"** architecture to ensure your documents remain secure.

Here is exactly what happens to your data when you use the OCR tool:

* **100% Local Processing:** Your documents are **never** sent to external cloud services or APIs (like Adobe, Google Cloud Vision, or AWS). All text extraction is performed entirely on the University of Bern's secure UBELIX high-performance computing nodes.
* **Isolated User Workspaces:** While simple text is processed in memory, advanced AI vision models (like OlmOCR) require reading files from a hard drive. To accommodate this, Text Lab generates a unique, temporary workspace located strictly within your private University home directory (`$HOME/ondemand_text_lab_ocr_jobs`). Other users on the cluster cannot access this space.
* **Instant Auto-Deletion (Self-Cleaning):** The exact moment the AI finishes extracting the text (or if the process encounters an error), the application runs an aggressive `shutil.rmtree()` command. **This guarantees that the entire temporary workspace—including your original document, intermediate images, and raw data files—is instantly and permanently deleted from the hard drive.**
* **Ephemeral Results:** The final extracted text and tables presented on your screen are stored strictly in your browser's volatile memory (`st.session_state`). When you close the tab, refresh the page, or your HPC job ends, all traces of the document and its extracted text are destroyed by Python's garbage collector.
* **No AI Training:** The vision models only perform *inference* (looking at the image to extract text). They do not learn from your documents, and your data is never used to train or improve the AI.