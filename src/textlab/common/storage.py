"""Where Text Lab writes temporary user data: one private workspace per job.

Uploaded files, intermediate results and generated charts are the user's
data. They go into a single workspace directory per job, never into the
user's home directory or the deployed source tree, so that nothing is left
behind when the session ends.

The launch script creates the workspace, exports its path as
``TEXT_LAB_WORKDIR`` and deletes it when the job ends. By default it lives in
the job's ``$TMPDIR``, which Slurm clusters usually provide per job and clean
up themselves (on UBELIX, ``/scratch/local/<job id>``), so the data is also
removed if the job is killed. A site can choose another location with
``TEXT_LAB_WORKDIR_BASE`` in its site configuration.

Without ``TEXT_LAB_WORKDIR`` (tests, scripts run by hand), a private folder
under the system temporary directory is used instead.

Files a user deliberately saves somewhere they choose, such as the
Knowledge Graph output folder, are not temporary and do not go here.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import shutil
import stat
import tempfile
from collections.abc import Iterator
from pathlib import Path

from textlab.common.config import get_settings

#: Permissions of every directory the workspace creates: owner only.
PRIVATE_MODE = 0o700

LOGGER = logging.getLogger(__name__)


class WorkspaceError(RuntimeError):
    """The workspace cannot be used safely."""


class Workspace:
    """A private directory tree for one job's temporary files.

    Features write into named areas (``"chat"``, ``"ocr"``, ...) so the
    files of different features stay apart. Every directory is created with
    owner-only permissions.

    Attributes:
        root: The workspace directory.
    """

    def __init__(self, root: Path | str):
        """Use ``root`` as the workspace directory.

        Args:
            root: The workspace directory. It is created on first use.
        """
        self.root = Path(root)

    def dir(self, *parts: str) -> Path:
        """Return a directory inside the workspace, creating it if needed.

        Args:
            *parts: Path components below the root, usually the feature
                area, e.g. ``dir("ocr")``.

        Returns:
            The directory, with owner-only permissions on every level from
            the root down.

        Raises:
            WorkspaceError: If a directory exists but belongs to another
                user.
        """
        path = self.root
        _ensure_private_dir(path, parents=True)
        for part in parts:
            path = path / part
            _ensure_private_dir(path)
        return path

    def make_temp_dir(self, area: str, prefix: str = "") -> Path:
        """Create a new uniquely named directory in an area.

        Use this for directories that must outlive a single function call,
        such as the folder holding a chat session's uploads. The caller may
        remove it; otherwise it is removed with the workspace when the job
        ends.

        Args:
            area: The feature area, e.g. ``"chat"``.
            prefix: Prefix of the directory name.

        Returns:
            The new directory, with owner-only permissions.
        """
        return Path(tempfile.mkdtemp(prefix=prefix, dir=self.dir(area)))

    @contextlib.contextmanager
    def temp_dir(self, area: str, prefix: str = "") -> Iterator[Path]:
        """Provide a directory in an area that is removed afterwards.

        Args:
            area: The feature area, e.g. ``"visualization"``.
            prefix: Prefix of the directory name.

        Yields:
            The directory. It and its contents are deleted when the
            ``with`` block ends, also on errors, and also when a tool left
            read-only files or folders in it.
        """
        path = self.make_temp_dir(area, prefix=prefix)
        try:
            yield path
        finally:
            remove_tree(path)


def remove_tree(path: Path) -> None:
    """Delete a directory tree, including read-only parts.

    Some tools (model caches, PDF libraries) create read-only files or
    folders; those are made writable and removed. Whatever still cannot be
    removed is logged rather than raised, so cleaning up never hides the
    result or the error of the work before it.

    Args:
        path: The directory to delete; nothing happens if it is missing.
    """

    def make_writable_and_retry(function, failed_path, _exc_info):
        try:
            os.chmod(failed_path, stat.S_IRWXU)
            if function is not os.rmdir:
                os.chmod(os.path.dirname(failed_path), stat.S_IRWXU)
            function(failed_path)
        except OSError:
            LOGGER.warning("Could not remove %s", failed_path)

    if path.exists():
        # onerror, not onexc: the image runs Python 3.11.
        shutil.rmtree(path, onerror=make_writable_and_retry)


def default_root() -> Path:
    """Return the workspace root used when ``TEXT_LAB_WORKDIR`` is unset.

    Returns:
        A per-user folder under the system temporary directory, which
        follows ``$TMPDIR``.
    """
    return Path(tempfile.gettempdir()) / f"textlab-{os.getuid()}"


@functools.cache
def get_workspace() -> Workspace:
    """Return the workspace of this process.

    Returns:
        The workspace at ``TEXT_LAB_WORKDIR``, or at :func:`default_root`
        when that is not set. Tests call ``get_workspace.cache_clear()``
        after changing the environment.
    """
    root = get_settings().workdir or default_root()
    return Workspace(root)


def _ensure_private_dir(path: Path, parents: bool = False) -> None:
    """Create ``path`` with owner-only permissions, or tighten it.

    An existing directory is accepted only if the current user owns it:
    under a shared temporary directory another user could otherwise create
    it first and read everything written into it.

    Args:
        path: The directory.
        parents: Also create missing parent directories. Only the root
            needs this; the parents are outside the workspace and keep the
            default permissions.

    Raises:
        WorkspaceError: If the directory belongs to another user.
    """
    path.mkdir(mode=PRIVATE_MODE, parents=parents, exist_ok=True)
    if path.stat().st_uid != os.getuid():
        raise WorkspaceError(
            f"Refusing to use {path}: it belongs to another user."
        )
    os.chmod(path, PRIVATE_MODE)
