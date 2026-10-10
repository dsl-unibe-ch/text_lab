"""Filesystem safeguards for user-supplied files and ZIP archives."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import shutil
import stat
from typing import Collection, Optional
import zipfile


MAX_ARCHIVE_MEMBERS = int(os.environ.get("TEXTLAB_MAX_ARCHIVE_MEMBERS", "1000"))
MAX_ARCHIVE_MEMBER_BYTES = int(
    os.environ.get("TEXTLAB_MAX_ARCHIVE_MEMBER_BYTES", str(500 * 1024**2))
)
MAX_ARCHIVE_TOTAL_BYTES = int(
    os.environ.get("TEXTLAB_MAX_ARCHIVE_TOTAL_BYTES", str(1024**3))
)
MAX_ARCHIVE_COMPRESSION_RATIO = float(
    os.environ.get("TEXTLAB_MAX_ARCHIVE_COMPRESSION_RATIO", "200")
)


def safe_upload_name(name: str, fallback: str = "upload") -> str:
    """Return only the filename portion of a browser-supplied name."""
    normalized = str(name or "").replace("\\", "/").rstrip("/")
    filename = PurePosixPath(normalized).name
    if filename in {"", ".", ".."}:
        return fallback
    return filename


def unique_output_directory(relative_path, used: set[str]) -> Path:
    """Return a stable result directory without colliding with earlier inputs."""
    source = PurePosixPath(str(relative_path).replace("\\", "/"))
    base = source.parent / (source.stem or "document")
    candidate = base
    extension = source.suffix.lower().lstrip(".") or "file"
    counter = 1
    while candidate.as_posix().casefold() in used:
        counter += 1
        suffix = f"_{extension}" if counter == 2 else f"_{extension}_{counter}"
        candidate = base.parent / f"{base.name}{suffix}"
    used.add(candidate.as_posix().casefold())
    return Path(*candidate.parts)


def safe_zip_members(
    archive: zipfile.ZipFile,
    *,
    allowed_extensions: Optional[Collection[str]] = None,
) -> list[zipfile.ZipInfo]:
    """Validate and return regular ZIP members within configured budgets."""
    allowed = (
        {str(extension).lower() for extension in allowed_extensions}
        if allowed_extensions is not None
        else None
    )
    members = [info for info in archive.infolist() if not info.is_dir()]
    if len(members) > MAX_ARCHIVE_MEMBERS:
        raise ValueError(
            f"ZIP contains {len(members)} files; the limit is {MAX_ARCHIVE_MEMBERS}."
        )

    selected = []
    total_size = 0
    seen = set()
    for info in members:
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise ValueError(f"ZIP contains a symbolic link: {info.filename}")
        relative = PurePosixPath(info.filename.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"ZIP member escapes the destination: {info.filename}")
        if not relative.parts or relative.name in {"", ".", ".."}:
            continue
        if allowed is not None and Path(relative.name).suffix.lower() not in allowed:
            continue
        key = relative.as_posix().casefold()
        if key in seen:
            raise ValueError(f"ZIP contains duplicate paths: {relative.as_posix()}")
        seen.add(key)
        if info.flag_bits & 0x1:
            raise ValueError(f"Encrypted ZIP members are not supported: {info.filename}")
        if info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
            raise ValueError(
                f"ZIP member {info.filename} expands beyond the per-file limit."
            )
        ratio = info.file_size / max(1, info.compress_size)
        if ratio > MAX_ARCHIVE_COMPRESSION_RATIO:
            raise ValueError(
                f"ZIP member {info.filename} has a suspicious compression ratio."
            )
        total_size += info.file_size
        if total_size > MAX_ARCHIVE_TOTAL_BYTES:
            raise ValueError("ZIP expands beyond the total size limit.")
        selected.append(info)

    return selected


def extract_zip_safely(
    archive: zipfile.ZipFile,
    destination,
    *,
    allowed_extensions: Optional[Collection[str]] = None,
) -> list[Path]:
    """Extract regular files after bounding paths and expanded archive size."""
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    selected = safe_zip_members(archive, allowed_extensions=allowed_extensions)

    extracted = []
    for info in selected:
        relative = PurePosixPath(info.filename.replace("\\", "/"))
        target = (root / Path(*relative.parts)).resolve()
        if root not in target.parents:
            raise ValueError(f"ZIP member escapes the destination: {info.filename}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source, target.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        extracted.append(target)
    return extracted