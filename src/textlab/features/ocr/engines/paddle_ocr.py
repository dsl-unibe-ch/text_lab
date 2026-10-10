"""PaddleOCR 2: text lines, strong on multi-column layouts.

PaddleOCR needs its own environment (``common.container.PADDLE_ENV``), so
it runs in a worker process (:mod:`.paddle_ocr_worker`) that reads all pages
of a document and prints the result as one marked JSON line.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from textlab.common import container, jobs
from textlab.common.language_mappings import PADDLEOCR_LANGUAGE_MAPPING
from textlab.common.progress import ProgressCallback
from textlab.features.ocr.engines.base import (
    EngineError,
    EngineOptions,
    EngineOutput,
    PageEngine,
    PageText,
    Preview,
    page_done,
    process_output,
    run_process,
)

WORKER_MODULE = "textlab.features.ocr.engines.paddle_ocr_worker"
#: Prefix of the worker's result line on stdout.
RESULT_MARKER = "TEXTLAB_PADDLEOCR_RESULT_JSON="


def run_worker(
    images: list[Path],
    language: str,
    *,
    backend_python: str | None = None,
    worker_path: str | None = None,
) -> list[dict[str, Any]]:
    """Recognize page images in the PaddleOCR worker.

    Args:
        images: The page images, in page order.
        language: A PaddleOCR language code, such as ``"en"``.
        backend_python: The worker's interpreter; by default the one of the
            image's PaddleOCR environment, or ``PADDLE_BACKEND_PYTHON``.
        worker_path: Run this script instead of :data:`WORKER_MODULE`; for
            tests with a stub worker.

    Returns:
        One dictionary per page with ``page``, ``image``, ``text``, ``raw``
        and ``rendered_png_b64``.

    Raises:
        EngineError: If the worker fails or prints no result.
    """
    python = backend_python or container.env_python(
        container.PADDLE_ENV, "PADDLE_BACKEND_PYTHON"
    )
    target = [worker_path] if worker_path else ["-m", WORKER_MODULE]
    command = [python, *target, "--lang", language, *map(str, images)]
    env = jobs.worker_environment(python)
    env.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")
    env.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    result = run_process(command, env)
    if result.returncode != 0:
        raise EngineError(
            "PaddleOCR backend failed.", details=process_output(result)
        )
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(RESULT_MARKER):
            return json.loads(line[len(RESULT_MARKER) :]).get("pages", [])
    raise EngineError(
        "PaddleOCR backend did not return JSON.",
        details=process_output(result),
    )


def _decode_png(value: str | None) -> bytes | None:
    """Decode base64 PNG data from the worker, or return ``None``."""
    if not value:
        return None
    try:
        return base64.b64decode(value)
    except Exception:
        return None


class PaddleOcr(PageEngine):
    """PaddleOCR 2 with text-line orientation, in its worker."""

    name = "PaddleOCR"
    languages = PADDLEOCR_LANGUAGE_MAPPING

    def recognize_pages(
        self,
        images: list[Path],
        options: EngineOptions,
        on_progress: ProgressCallback,
    ) -> EngineOutput:
        """Read page images (see :meth:`PageEngine.recognize_pages`)."""
        pages = run_worker(images, options.language)
        output = EngineOutput(pages=[])
        for index, page in enumerate(pages, start=1):
            raw = page.get("raw", [])
            output.pages.append(PageText(index, page.get("text", ""), raw))
            if options.previews:
                preview = _decode_png(page.get("rendered_png_b64"))
                if not preview:
                    from textlab.features.ocr.engines.previews import (
                        render_paddle_preview,
                    )

                    image = Path(page.get("image") or images[index - 1])
                    preview, _layout = render_paddle_preview(image, raw)
                if preview:
                    output.previews.append(Preview(preview))
            page_done(on_progress, self.name, index, len(images))
        return output
