"""Progress reporting and cancellation for long-running backend work.

Backend functions never draw progress bars. They accept an ``on_progress``
callback and call it with :class:`Progress` updates; each interface decides
how to show them (a Streamlit status box, a log line in a batch job, a
server-sent event in a web frontend). Work that can be stopped accepts a
``cancel`` event and calls :func:`check_cancelled` between steps.
"""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Callable, Mapping
from typing import Any


@dataclasses.dataclass(frozen=True)
class Progress:
    """One progress update.

    Attributes:
        message: Short description of the current step, shown to the user.
        fraction: Share of the work done, from 0.0 to 1.0, or ``None`` when
            it is not known.
    """

    message: str
    fraction: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the update as a JSON-serializable dictionary."""
        return {"message": self.message, "fraction": self.fraction}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Progress:
        """Rebuild an update from :meth:`to_dict` output.

        Args:
            data: The dictionary.

        Returns:
            The progress update.
        """
        return cls(str(data.get("message", "")), data.get("fraction"))


#: Signature of the callbacks that receive progress updates.
ProgressCallback = Callable[[Progress], None]


def no_progress(progress: Progress) -> None:
    """Ignore a progress update; the default ``on_progress`` callback.

    Args:
        progress: The update, discarded.
    """


class CancelledError(Exception):
    """The work was stopped through its cancel event."""


def check_cancelled(cancel: threading.Event | None) -> None:
    """Stop the current work if cancellation was requested.

    Args:
        cancel: The cancel event passed by the caller, or ``None``.

    Raises:
        CancelledError: If the event is set.
    """
    if cancel is not None and cancel.is_set():
        raise CancelledError("The work was cancelled.")
