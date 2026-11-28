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
#: WhisperX 3.8 and pyannote 4 (the transcription worker).
TRANSCRIPTION_ENV = "transcription_backend"
#: kraken (line detection of the Historic German handwriting tool).
KRAKEN_ENV = "kraken_backend"
#: Transformers 5 (the Hugging Face OCR models tool).
HF_OCR_ENV = "hf_ocr_backend"


def paddle_environment(env: dict[str, str]) -> dict[str, str]:
    """Add the settings Paddle's workers need to their environment.

    Paddle stalls or crashes in its OpenMP and BLAS threads with more than
    one thread (a worker spun one core for 15 minutes with the GPU idle
    when the session's ``OMP_NUM_THREADS=16`` reached it), so the thread
    counts are set to 1 whatever the caller's environment says. They have
    to be set before the worker starts: numpy's BLAS reads them when it is
    imported. Model source checks over the internet are turned off.

    Args:
        env: A worker's environment, changed in place.

    Returns:
        The same environment.
    """
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")
    env.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    return env


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
