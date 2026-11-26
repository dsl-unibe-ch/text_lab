"""Receiving uploads: the size limit, form limits and saving files.

Starlette parses a multipart form into temporary files under ``TMPDIR``,
which the launch script places in the session's private workspace, so an
upload is never held in memory whole. A page reads the form with
:func:`read_form`, checks the files and copies them into its job's
``inputs`` folder with :func:`save` (in a worker thread, as it can take
seconds for large files)::

    form = await uploads.read_form(request, max_files=1)
    upload = uploads.single_file(form, "file", suffixes=AUDIO_EXTENSIONS)
    job = runner.create(title=upload.filename, page="transcription")
    path = await run_in_threadpool(uploads.save, upload, job.inputs)

:class:`BodyLimit` refuses requests larger than the upload limit, also when
they do not announce their size.
"""

from __future__ import annotations

import shutil
from collections.abc import Collection
from pathlib import Path, PurePosixPath

from fastapi import Request
from starlette.datastructures import FormData, Headers, UploadFile
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from textlab.common.upload_safety import safe_upload_name
from textlab.ui.web.render import file_size

#: Bytes copied at a time when saving an upload.
CHUNK_SIZE = 1024 * 1024


class UploadError(ValueError):
    """An upload that cannot be used; the message is for the user."""


class BodyTooLargeError(UploadError):
    """A request is larger than the upload limit."""

    def __init__(self, limit: int):
        """Name the limit in the message.

        Args:
            limit: The upload limit, in bytes.
        """
        super().__init__(
            f"The upload is larger than Text Lab accepts ({file_size(limit)})."
        )
        self.limit = limit


class BodyLimit:
    """ASGI middleware that refuses request bodies over a size limit.

    A body that announces a larger size (``Content-Length``) is refused
    when the app starts reading it; one that grows beyond the limit is
    refused as soon as it does. Either way, reading raises
    :class:`BodyTooLargeError`, which the app turns into an error message.
    """

    def __init__(self, app: ASGIApp, limit: int):
        """Wrap ``app``.

        Args:
            app: The ASGI app.
            limit: The largest body accepted, in bytes.
        """
        self.app = app
        self.limit = limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        """Pass the request on, counting the bytes of its body."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        announced = Headers(scope=scope).get("content-length", "")
        too_large = announced.isdigit() and int(announced) > self.limit
        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            if too_large:
                raise BodyTooLargeError(self.limit)
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.limit:
                    raise BodyTooLargeError(self.limit)
            return message

        await self.app(scope, limited_receive, send)


async def read_form(
    request: Request, *, max_files: int = 1, max_fields: int = 30
) -> FormData:
    """Read a form with uploads.

    Args:
        request: The request.
        max_files: Most files the form may hold.
        max_fields: Most other fields it may hold.

    Returns:
        The form. Its files stay open until the response is sent.

    Raises:
        BodyTooLargeError: If the request is larger than the upload limit.
        fastapi.HTTPException: If the form is malformed or holds too many
            files or fields (400).
    """
    return await request.form(max_files=max_files, max_fields=max_fields)


def files(form: FormData, field: str) -> list[UploadFile]:
    """Return the files sent in a form field, without empty ones.

    Args:
        form: The form.
        field: The field's name.

    Returns:
        The files that have a name; a file input left empty sends none.
    """
    return [
        value
        for value in form.getlist(field)
        if isinstance(value, UploadFile) and value.filename
    ]


def display_name(upload: UploadFile) -> str:
    """Return an upload's file name without any folders, for display.

    Args:
        upload: The upload.

    Returns:
        The name the browser sent, reduced to its last part.
    """
    return safe_upload_name(upload.filename or "")


def check_suffix(upload: UploadFile, suffixes: Collection[str]) -> None:
    """Check that an upload's file name ends in one of ``suffixes``.

    Args:
        upload: The upload.
        suffixes: Accepted endings with their dot, such as ``".mp3"``.

    Raises:
        UploadError: If the name has another ending.
    """
    suffix = PurePosixPath(upload.filename or "").suffix.lower()
    if suffix not in {item.lower() for item in suffixes}:
        accepted = ", ".join(sorted(item.lstrip(".") for item in suffixes))
        raise UploadError(
            f"{display_name(upload)} is not a file this page reads. "
            f"Accepted: {accepted}."
        )


def single_file(
    form: FormData, field: str, *, suffixes: Collection[str] | None = None
) -> UploadFile:
    """Return the one file of a form field, checked.

    Args:
        form: The form.
        field: The field's name.
        suffixes: Accepted file name endings, or ``None`` for any.

    Returns:
        The upload.

    Raises:
        UploadError: If the field holds no file or one with another ending.
    """
    found = files(form, field)
    if not found:
        raise UploadError("Choose a file first.")
    if suffixes is not None:
        check_suffix(found[0], suffixes)
    return found[0]


def save(
    upload: UploadFile, folder: Path, used: set[str] | None = None
) -> Path:
    """Copy an upload into ``folder`` under a safe, unique name.

    Blocks while copying; call it from a worker thread
    (``run_in_threadpool``) in async routes.

    Args:
        upload: The upload.
        folder: The folder, usually a job's ``inputs``.
        used: Names given so far in this folder, compared without case;
            the new name is added. Defaults to the names in the folder.

    Returns:
        The saved file. Its name is the upload's own, without any folders,
        with ``_2``, ``_3`` ... before the extension if it is taken.
    """
    if used is None:
        used = {path.name.casefold() for path in folder.iterdir()}
    name = display_name(upload)
    stem, suffix = PurePosixPath(name).stem, PurePosixPath(name).suffix
    candidate, counter = name, 1
    while candidate.casefold() in used:
        counter += 1
        candidate = f"{stem}_{counter}{suffix}"
    used.add(candidate.casefold())

    path = folder / candidate
    upload.file.seek(0)
    with path.open("xb") as target:
        shutil.copyfileobj(upload.file, target, CHUNK_SIZE)
    return path
