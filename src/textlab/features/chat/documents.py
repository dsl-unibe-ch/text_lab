"""Reading the files a user attaches to a chat: PDFs, text and tables.

Their text is put in front of the user's question, so the model answers
from it.
"""

from __future__ import annotations

from collections.abc import Sequence
from io import BytesIO

import fitz
import pandas as pd

#: Largest attachment read, in megabytes; larger files are skipped.
MAX_FILE_MB = 10


def read_pdf(file_bytes: bytes) -> str:
    """Extract text content from a PDF file.

    Args:
        file_bytes (bytes): The raw bytes of the PDF file.

    Returns:
        str: The extracted text or an error message.
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text_blocks = [page.get_text() for page in doc]
        return "\n".join(text_blocks)
    except Exception as e:
        return f"[Error reading PDF: {str(e)}]"


def read_txt(file_bytes: bytes) -> str:
    """Decode a plain text file.

    Args:
        file_bytes (bytes): The raw bytes of the text file.

    Returns:
        str: The decoded string or an error message.
    """
    try:
        return file_bytes.decode("utf-8")
    except Exception as e:
        return f"[Error reading TXT: {str(e)}]"


def read_tabular(file_bytes: bytes, file_name: str) -> str:
    """Parse CSV or Excel files into a Markdown table representation.

    Truncates to the first 100 rows to prevent context overflow.

    Args:
        file_bytes (bytes): The raw bytes of the tabular file.
        file_name (str): The name of the file.

    Returns:
        str: A Markdown-formatted table or an error message.
    """
    try:
        if file_name.endswith(".csv"):
            df = pd.read_csv(BytesIO(file_bytes))
        else:
            df = pd.read_excel(BytesIO(file_bytes))

        if len(df) > 100:
            return (
                f"[Dataset truncated. First 100 of {len(df)} "
                f"rows]:\n{df.head(100).to_markdown()}"
            )
        return df.to_markdown()
    except Exception as e:
        return f"[Error reading Table: {str(e)}]"


def read_documents(
    files: Sequence[tuple[str, bytes]],
) -> tuple[str, list[str]]:
    """Read attached files into one context block for the model.

    PDFs, text files and tables (CSV, Excel) are read; files larger than
    :data:`MAX_FILE_MB` are skipped with a warning.

    Args:
        files: The attachments, as ``(name, content)`` pairs.

    Returns:
        The context, with each file's text between start and end markers
        (empty if no file had text), and warnings for users.
    """
    if not files:
        return "", []

    file_context = "### User Uploaded File Content:\n"
    warnings = []
    has_valid_text = False

    for name, file_bytes in files:
        file_size_mb = len(file_bytes) / (1024 * 1024)
        if file_size_mb > MAX_FILE_MB:
            warnings.append(
                f"File '{name}' exceeds the {MAX_FILE_MB}MB limit "
                f"({file_size_mb:.1f}MB). Skipped."
            )
            continue

        file_name = name.lower()
        extracted_text = ""

        if file_name.endswith(".pdf"):
            extracted_text = read_pdf(file_bytes)
        elif file_name.endswith(".txt"):
            extracted_text = read_txt(file_bytes)
        elif file_name.endswith((".csv", ".xlsx", ".xls")):
            extracted_text = read_tabular(file_bytes, file_name)

        if extracted_text:
            has_valid_text = True
            file_context += f"\n--- Start of file: {name} ---\n"
            file_context += extracted_text
            file_context += f"\n--- End of file: {name} ---\n"

    if not has_valid_text:
        return "", warnings

    return file_context, warnings
