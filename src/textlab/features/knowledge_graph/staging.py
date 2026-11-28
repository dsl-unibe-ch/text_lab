"""Uploaded papers: a PDF, or a ZIP of PDFs, put into a folder to read.

ZIP subfolders are flattened, keeping the PDFs' names, by which a corpus
recognises papers it has read. Two PDFs of one name are refused, so none
overwrites or hides another. Archive paths and expanded sizes go through
the shared upload safeguards, and every file must hold a PDF header.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

from textlab.common.storage import get_workspace
from textlab.common.upload_safety import safe_upload_name, safe_zip_members
from textlab.features.knowledge_graph.corpus import list_pdfs

#: Bytes at the start of a file searched for the PDF header.
HEADER_BYTES = 1024


def stage_papers(upload: Path, folder: Path, name: str = "") -> list[Path]:
    """Put the PDFs of an uploaded PDF or ZIP file into a folder.

    Args:
        upload: The uploaded file, on disk.
        folder: Where to put the PDFs; created if needed.
        name: The file's name as uploaded, which says whether it is a PDF
            or a ZIP; the upload's own name if empty.

    Returns:
        The PDFs, sorted by name.

    Raises:
        ValueError: If the file is neither a PDF nor a ZIP, is not valid,
            or holds no PDF; the message is for users.
    """
    name = safe_upload_name(name or upload.name)
    extension = _extension(name)
    folder.mkdir(parents=True, exist_ok=True)
    if extension == ".pdf":
        with upload.open("rb") as file:
            _check_header(name, file.read(HEADER_BYTES))
        shutil.copyfile(upload, folder / name)
    else:
        try:
            with ZipFile(upload) as archive:
                _extract_pdfs(archive, folder)
        except BadZipFile as exc:
            raise ValueError("The uploaded ZIP archive is not valid.") from exc
    return list_pdfs(folder)


@contextmanager
def staged_papers(name: str, data: bytes) -> Iterator[Path]:
    """Stage a PDF or ZIP in a private folder removed after processing.

    Args:
        name: The browser-supplied filename.
        data: The uploaded PDF or ZIP bytes.

    Yields:
        A temporary folder containing the uploaded PDF papers.

    Raises:
        ValueError: If the input is invalid, unsafe, or contains no PDFs.
    """
    name = safe_upload_name(name)
    extension = _extension(name)
    with get_workspace().temp_dir("knowledge_graph", prefix="upload-") as root:
        if extension == ".pdf":
            _check_header(name, data)
            (root / name).write_bytes(data)
        else:
            try:
                with ZipFile(BytesIO(data)) as archive:
                    _extract_pdfs(archive, root)
            except BadZipFile as exc:
                raise ValueError(
                    "The uploaded ZIP archive is not valid."
                ) from exc
        yield root


def _extension(name: str) -> str:
    """Return a PDF's or ZIP's extension, or refuse another file."""
    extension = Path(name).suffix.lower()
    if extension not in {".pdf", ".zip"}:
        raise ValueError("Upload a PDF document or ZIP archive of PDFs.")
    return extension


def _check_header(name: str, start: bytes) -> None:
    """Refuse a file whose first bytes hold no PDF header."""
    if b"%PDF-" not in start[:HEADER_BYTES]:
        raise ValueError(f"{name} does not contain a PDF header.")


def _extract_pdfs(archive: ZipFile, folder: Path) -> None:
    """Write the PDFs of an archive into a folder, without its subfolders.

    Raises:
        ValueError: If two PDFs share a name, one is not a PDF, or there
            is none.
    """
    seen: set[str] = set()
    for member in safe_zip_members(archive, allowed_extensions={".pdf"}):
        relative = PurePosixPath(member.filename.replace("\\", "/"))
        if "__MACOSX" in relative.parts or relative.name.startswith("._"):
            continue
        filename = safe_upload_name(member.filename)
        if filename.casefold() in seen:
            raise ValueError(
                f"More than one PDF is named {filename}. "
                "Give each paper a unique filename."
            )
        seen.add(filename.casefold())
        data = archive.read(member)
        _check_header(filename, data)
        (folder / filename).write_bytes(data)
    if not seen:
        raise ValueError("The ZIP archive contains no PDF papers.")
