"""EasyOCR: text lines in many languages, in the app process.

A reader per language stays loaded between runs. When another feature needs
the GPU, ``common.gpu_manager`` releases them (:func:`release_readers`).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from textlab.common import gpu_manager
from textlab.common.language_mappings import EASYOCR_LANGUAGE_MAPPING
from textlab.common.progress import ProgressCallback, check_cancelled
from textlab.features.ocr.engines.base import (
    EngineOptions,
    EngineOutput,
    PageEngine,
    PageText,
    Preview,
    TextLine,
    page_done,
)
from textlab.features.ocr.engines.payloads import polygon_points

_READERS: dict[str, Any] = {}
_LOCK = threading.Lock()


def get_reader(language: str) -> Any:
    """Return the EasyOCR reader for a language, loading it once.

    Args:
        language: An EasyOCR language code, such as ``"en"``.

    Returns:
        An ``easyocr.Reader`` on the GPU.
    """
    with _LOCK:
        reader = _READERS.get(language)
        if reader is None:
            import easyocr

            reader = easyocr.Reader([language], gpu=True)
            _READERS[language] = reader
        return reader


def release_readers() -> None:
    """Drop the loaded readers, so their GPU memory can be freed."""
    with _LOCK:
        _READERS.clear()


gpu_manager.register(
    gpu_manager.OCR, "Unloaded EasyOCR model", release_readers
)


class EasyOcr(PageEngine):
    """EasyOCR, with a paragraph per detected text block."""

    name = "EasyOCR"
    description = (
        "Printed text in many languages, block by block. Quick, and good "
        "on photos and signs; reads no tables."
    )
    languages = EASYOCR_LANGUAGE_MAPPING
    default_language = "English"

    def prepare(
        self, options: EngineOptions, on_progress: ProgressCallback
    ) -> None:
        """Load the reader for the run's language."""
        get_reader(options.language)

    def recognize_pages(
        self,
        images: list[Path],
        options: EngineOptions,
        on_progress: ProgressCallback,
        cancel: threading.Event | None = None,
    ) -> EngineOutput:
        """Read page images (see :meth:`PageEngine.recognize_pages`)."""
        from textlab.features.ocr.engines.previews import (
            render_easyocr_preview,
        )

        reader = get_reader(options.language)
        output = EngineOutput(pages=[])
        for index, image in enumerate(images, start=1):
            check_cancelled(cancel)
            found = reader.readtext(str(image), detail=1, paragraph=True)
            text = "\n".join(item[1] for item in found)
            output.pages.append(
                PageText(index, text, found, lines=text_lines(found))
            )
            if options.previews:
                boxes, layout = render_easyocr_preview(image, found)
                if boxes and layout:
                    output.previews.append(Preview(boxes, layout))
            page_done(on_progress, self.name, index, len(images))
        return output


def text_lines(found: list[Any]) -> list[TextLine]:
    """Return EasyOCR's text blocks for one page as lines with outlines.

    Args:
        found: EasyOCR's ``readtext`` result: ``(box, text, ...)`` items.

    Returns:
        The blocks with an outline, in reading order.
    """
    lines = []
    for item in found:
        if not isinstance(item, list | tuple) or len(item) < 2:
            continue
        points = polygon_points(item[0])
        if points is not None:
            lines.append(TextLine.of(str(item[1]).strip(), points))
    return lines
