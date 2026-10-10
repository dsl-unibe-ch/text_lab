"""A worker for testing textlab.common.jobs; its request picks a behavior."""

import os
import time

import numpy as np

from textlab.common.jobs import worker_main
from textlab.common.progress import Progress


def handle(request, report):
    """Report progress, then succeed, fail, crash or hang."""
    report(Progress("working", 0.5))
    mode = request["mode"]
    if mode == "fail":
        raise ValueError("bad input")
    if mode == "crash":
        os._exit(3)
    if mode == "hang":
        time.sleep(60)
    return {"echo": request.get("value"), "number": np.float32(1.5)}


if __name__ == "__main__":
    worker_main(handle)
