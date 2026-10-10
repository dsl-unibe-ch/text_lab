import threading

import pytest

from textlab.common.progress import (
    CancelledError,
    Progress,
    check_cancelled,
)


def test_progress_round_trips_through_a_dict():
    update = Progress("Aligning...", 0.25)
    assert Progress.from_dict(update.to_dict()) == update


def test_check_cancelled_only_raises_when_set():
    check_cancelled(None)
    event = threading.Event()
    check_cancelled(event)
    event.set()
    with pytest.raises(CancelledError):
        check_cancelled(event)
