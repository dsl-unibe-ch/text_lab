"""Where the Text Lab image keeps its separate Python environments.

The image (``deploy/container/text_lab.def``) has one conda environment for
the app and separate ones for engines whose dependencies conflict with it.
Workers for those engines run with that environment's interpreter. This is
the image's layout, the same on every cluster, so it is not a site setting;
an environment variable can still point at another interpreter, for
example when testing a new engine version.
"""

from __future__ import annotations

import os

#: Root of the image's conda environments.
CONDA_ENVS = "/opt/conda/envs"

#: PaddleOCR-VL document parser (the automatic OCR pipeline).
PADDLE_VL_ENV = "paddle_vl_backend"
#: PaddleOCR 2 (the PaddleOCR engine of manual engine selection).
PADDLE_ENV = "paddle_backend"
#: olmOCR (the OlmOCR engine of manual engine selection).
OLMOCR_ENV = "olmocr_backend"


def env_python(env: str, *overrides: str) -> str:
    """Return the Python interpreter of one of the image's environments.

    Args:
        env: The environment name, such as :data:`PADDLE_VL_ENV`.
        *overrides: Environment variables that, when set, name the
            interpreter to use instead, checked in order.

    Returns:
        The interpreter's path.
    """
    for variable in overrides:
        value = os.environ.get(variable)
        if value:
            return value
    return f"{CONDA_ENVS}/{env}/bin/python"
