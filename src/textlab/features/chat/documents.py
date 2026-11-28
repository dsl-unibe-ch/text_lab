"""Reading the files a user attaches to a chat: PDFs, text, JSON and tables.

Their text is put in front of the user's question, so the model answers
from it. Files that give no text (scanned PDFs, unreadable or unsupported
files) are left out with a warning for the user, so that the model is
never asked about a document it cannot see.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from io import BytesIO

import fitz
import pandas as pd

#: Largest attachment read, in megabytes; larger files are skipped.
MAX_FILE_MB = 10

#: Rows of a table included as text; the data tools use the whole table.
MAX_TABLE_ROWS = 100

#: File extensions read as context, without the dot.
DOCUMENT_TYPES = ("pdf", "txt", "json", "csv", "tsv", "xls", "xlsx")

#: Most files a conversation can have attached at a time.
MAX_FILES = 4

#: Extensions of the tables among :data:`DOCUMENT_TYPES`.
TABLE_TYPES = ("csv", "tsv", "xls", "xlsx")


@dataclasses.dataclass(frozen=True)
class Document:
    """An attached file, read for the model.

    Attributes:
        name: The file's name.
        size: Its size in bytes.
        text: Its text for the model; empty if it could not be read.
        problem: Why it could not be read, for users; empty if it was.
    """

    name: str
    size: int
    text: str = ""
    problem: str = ""

    @property
    def kind(self) -> str:
        """The file's type for people: ``"PDF"``, ``"Table"``, ``"Text"``."""
        suffix = (
            self.name.rsplit(".", 1)[-1].lower() if "." in self.name else ""
        )
        if suffix in TABLE_TYPES:
            return "Table"
        if suffix == "pdf":
            return "PDF"
        if suffix == "json":
            return "JSON"
        return "Text"


def read_pdf(file_bytes: bytes) -> str:
    """Extract the text layer of a PDF file.

    Args:
        file_bytes (bytes): The raw bytes of the PDF file.

    Returns:
        str: The text of all pages; empty (or only whitespace) for a
        scanned PDF without a text layer.

    Raises:
        ValueError: If the PDF is password-protected.
        Exception: Any error of PyMuPDF for a damaged file.
    """
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        if doc.needs_pass:
            raise ValueError("the PDF is password-protected")
        return "\n".join(page.get_text() for page in doc)


def read_txt(file_bytes: bytes) -> str:
    """Decode a UTF-8 text file (a leading byte order mark is dropped).

    Args:
        file_bytes (bytes): The raw bytes of the text file.

    Returns:
        str: The decoded text.

    Raises:
        UnicodeDecodeError: If the file is not UTF-8.
    """
    return file_bytes.decode("utf-8-sig")


def read_json(file_bytes: bytes) -> str:
    """Decode a JSON file and check that it is valid JSON.

    The text is kept as written rather than reformatted, which would add
    tokens.

    Args:
        file_bytes (bytes): The raw bytes of the JSON file.

    Returns:
        str: The JSON text.

    Raises:
        ValueError: If the file is not UTF-8 or not valid JSON.
    """
    text = read_txt(file_bytes)
    json.loads(text)
    return text


def read_tabular(file_bytes: bytes, file_name: str) -> str:
    """Parse CSV, TSV or Excel files into a Markdown table representation.

    Truncates to the first :data:`MAX_TABLE_ROWS` rows to keep the context
    small.

    Args:
        file_bytes (bytes): The raw bytes of the tabular file.
        file_name (str): The name of the file, in lower case.

    Returns:
        str: A Markdown-formatted table.

    Raises:
        Exception: Any error of pandas for a file it cannot parse.
    """
    if file_name.endswith(".csv"):
        df = pd.read_csv(BytesIO(file_bytes))
    elif file_name.endswith(".tsv"):
        df = pd.read_csv(BytesIO(file_bytes), sep="\t")
    else:
        df = pd.read_excel(BytesIO(file_bytes))

    if len(df) > MAX_TABLE_ROWS:
        return (
            f"[Dataset truncated. First {MAX_TABLE_ROWS} of {len(df)} "
            f"rows]:\n{df.head(MAX_TABLE_ROWS).to_markdown()}"
        )
    return df.to_markdown()


def _read_file(name: str, file_bytes: bytes) -> str:
    """Return the text of one attachment by its file type.

    Raises:
        ValueError: For a file type that is not read as context.
        Exception: Any error of the reader for that file type.
    """
    file_name = name.lower()
    if file_name.endswith(".pdf"):
        return read_pdf(file_bytes)
    if file_name.endswith(".txt"):
        return read_txt(file_bytes)
    if file_name.endswith(".json"):
        return read_json(file_bytes)
    if file_name.endswith((".csv", ".tsv", ".xlsx", ".xls")):
        return read_tabular(file_bytes, file_name)
    raise ValueError("this file type is not supported")


def read_document(name: str, data: bytes) -> Document:
    """Read one attached file for the model.

    Args:
        name: The file's name; its extension gives its type.
        data: Its content.

    Returns:
        The document, with its text, or with the problem that kept it from
        being read: a file larger than :data:`MAX_FILE_MB`, a file that
        cannot be read, or a PDF without a text layer.
    """
    size = len(data)
    file_size_mb = size / (1024 * 1024)
    if file_size_mb > MAX_FILE_MB:
        return Document(
            name,
            size,
            problem=(
                f"File '{name}' exceeds the {MAX_FILE_MB}MB limit "
                f"({file_size_mb:.1f}MB). Skipped."
            ),
        )

    try:
        text = _read_file(name, data)
    except Exception as e:
        return Document(
            name,
            size,
            problem=f"File '{name}' could not be read ({e}). Skipped.",
        )

    if not text.strip():
        if name.lower().endswith(".pdf"):
            problem = (
                f"File '{name}' contains no text, probably because it "
                "is scanned. Convert it with OCR first. Skipped."
            )
        else:
            problem = f"File '{name}' is empty. Skipped."
        return Document(name, size, problem=problem)
    return Document(name, size, text=text)


def document_context(documents: Sequence[Document]) -> str:
    """Return the text of read documents as one context block.

    Args:
        documents: The documents, as :func:`read_document` returns them;
            those without text are left out.

    Returns:
        Each document's text between start and end markers, under a
        heading; empty if no document has text.
    """
    readable = [document for document in documents if document.text]
    if not readable:
        return ""
    file_context = "### User Uploaded File Content:\n"
    for document in readable:
        file_context += f"\n--- Start of file: {document.name} ---\n"
        file_context += document.text
        file_context += f"\n--- End of file: {document.name} ---\n"
    return file_context


def read_documents(
    files: Sequence[tuple[str, bytes]],
) -> tuple[str, list[str]]:
    """Read attached files into one context block for the model.

    PDFs, text and JSON files and tables (CSV, TSV, Excel) are read. Files
    larger than :data:`MAX_FILE_MB`, files that cannot be read and PDFs
    without a text layer are skipped with a warning.

    Args:
        files: The attachments, as ``(name, content)`` pairs.

    Returns:
        The context, with each file's text between start and end markers
        (empty if no file had text), and warnings for users.
    """
    documents = [read_document(name, data) for name, data in files]
    warnings = [document.problem for document in documents if document.problem]
    return document_context(documents), warnings
