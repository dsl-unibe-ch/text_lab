"""Shared sys.path setup for the plain-python test scripts.

The suites run in two places:
* inside the container (``text_lab_main`` env has cv2/pandas/lxml), or
* on the dev host, where cv2 comes from a pip ``--target`` directory pointed
  to by the ``TEXTLAB_TESTDEPS`` environment variable.
"""

import os
import pathlib
import sys

REPO_SRC = str(pathlib.Path(__file__).resolve().parents[1] / "src")

deps = os.environ.get("TEXTLAB_TESTDEPS")
if deps and deps not in sys.path:
    sys.path.insert(0, deps)
if REPO_SRC not in sys.path:
    sys.path.insert(0, REPO_SRC)
