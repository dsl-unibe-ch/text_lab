"""Showing backend progress on a Streamlit page."""

from __future__ import annotations

from types import TracebackType

import streamlit as st

from textlab.common.progress import Progress


class StatusBox:
    """A ``st.status`` box that shows a backend's progress updates.

    Use it as a context manager and pass it as the ``on_progress`` callback;
    each new message replaces the previous one. When the block ends, the box
    shows ``complete_label`` or, if it raised, ``error_label``. A page
    refresh stops the script, which also stops a worker process (see
    :func:`textlab.common.jobs.run_worker`).

    Example::

        with StatusBox("Running transcription pipeline...") as status:
            result = run_transcription(files, options, on_progress=status)
    """

    def __init__(
        self,
        label: str,
        *,
        complete_label: str = "Done.",
        error_label: str = "Failed.",
        expanded: bool = True,
        caption: str | None = "To cancel, refresh the page.",
    ):
        """Configure the box; it appears when the block is entered.

        Args:
            label: Title shown until the first update arrives.
            complete_label: Title shown when the block ends normally.
            error_label: Title shown when the block raises.
            expanded: Whether the box starts expanded.
            caption: Hint shown under the title, or ``None``.
        """
        self._label = label
        self._complete_label = complete_label
        self._error_label = error_label
        self._expanded = expanded
        self._caption = caption
        self._status = None
        self._message = None
        self._last = None

    def __enter__(self) -> StatusBox:
        """Draw the box and return the callback."""
        self._status = st.status(self._label, expanded=self._expanded)
        self._status.__enter__()
        if self._caption:
            st.caption(self._caption)
        self._message = st.empty()
        return self

    def __call__(self, progress: Progress) -> None:
        """Show a progress update if its message is new.

        Args:
            progress: The update.
        """
        if progress.message and progress.message != self._last:
            self._last = progress.message
            self._message.info(progress.message)
            self._status.update(label=progress.message)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Mark the box complete or failed and close it.

        Streamlit stops a script for a rerun or refresh by raising a
        ``BaseException``; the box is left as it is then, since the page is
        about to be redrawn.
        """
        if exc_type is None:
            self._status.update(
                label=self._complete_label, state="complete", expanded=False
            )
        elif issubclass(exc_type, Exception):
            self._status.update(
                label=self._error_label, state="error", expanded=True
            )
        self._status.__exit__(exc_type, exc, traceback)
