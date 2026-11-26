"""The user's own folder of downloaded Hugging Face OCR models.

Models chosen in this tool are not put into the shared model store, which
only the maintainers fill: each user's models go into a folder in their
home directory, :data:`DEFAULT_DIR`, laid out as a Hugging Face cache
(``models--owner--name``). ``TEXT_LAB_HF_OCR_DIR`` names another folder,
for example on a larger project storage.

The folder counts towards the user's storage quota, so the tool lists the
downloaded models with their size and lets users delete them.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
from pathlib import Path

#: Default folder, in the user's home directory.
DEFAULT_DIR = Path("~/.cache/textlab/hf_ocr_models")
#: Environment variable naming another folder.
DIR_VARIABLE = "TEXT_LAB_HF_OCR_DIR"

_PREFIX = "models--"


@dataclasses.dataclass(frozen=True)
class StoredModel:
    """A model in the user's folder.

    Attributes:
        repo_id: The model, as ``owner/name``.
        size_bytes: Disk space it takes.
    """

    repo_id: str
    size_bytes: int


def models_dir() -> Path:
    """Return the user's model folder; it may not exist yet."""
    configured = os.environ.get(DIR_VARIABLE, "").strip()
    return Path(configured or DEFAULT_DIR).expanduser()


def _cache_name(repo_id: str) -> str:
    return _PREFIX + repo_id.replace("/", "--")


def _folder_size(folder: Path) -> int:
    """Return the size of a folder's files; links are not followed.

    A Hugging Face cache keeps every file once under ``blobs/`` and links
    to it from ``snapshots/``, so links are not counted twice.
    """
    total = 0
    for root, _dirs, names in os.walk(folder):
        for name in names:
            path = Path(root) / name
            if not path.is_symlink():
                try:
                    total += path.stat().st_size
                except OSError:
                    pass
    return total


def stored_models(folder: Path | None = None) -> list[StoredModel]:
    """Return the models in the user's folder, by name.

    Args:
        folder: The folder; by default :func:`models_dir`.
    """
    folder = models_dir() if folder is None else folder
    if not folder.is_dir():
        return []
    models = []
    for entry in sorted(folder.iterdir()):
        if entry.is_dir() and entry.name.startswith(_PREFIX):
            owner_name = entry.name[len(_PREFIX) :].split("--", 1)
            if len(owner_name) == 2:
                models.append(
                    StoredModel("/".join(owner_name), _folder_size(entry))
                )
    return models


def delete_model(repo_id: str, folder: Path | None = None) -> bool:
    """Delete a model from the user's folder.

    Args:
        repo_id: The model, as ``owner/name``.
        folder: The folder; by default :func:`models_dir`.

    Returns:
        Whether the model was there.
    """
    folder = models_dir() if folder is None else folder
    target = folder / _cache_name(repo_id)
    if target.parent != folder or not target.is_dir():
        return False
    shutil.rmtree(target)
    return True


def format_size(size_bytes: int) -> str:
    """Return a size in GB or MB, for display."""
    if size_bytes >= 1024**3:
        return f"{size_bytes / 1024**3:.1f} GB"
    return f"{size_bytes / 1024**2:.0f} MB"
