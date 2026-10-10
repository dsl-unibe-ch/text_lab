"""GLM-OCR: a vision model on the Ollama server, with extraction modes.

Each page image is sent to the model with the chosen mode as the prompt:
text, tables (returned as HTML) or figures. The model is pulled on first use.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import ollama

from textlab.common.ollama import (
    canonical_model_name,
    installed_model_names,
    message_text,
)
from textlab.common.progress import Progress, ProgressCallback
from textlab.features.ocr.engines.base import (
    EngineError,
    EngineOptions,
    EngineOutput,
    PageEngine,
    PageText,
    Preview,
    page_done,
)

#: The Ollama model.
MODEL = os.environ.get("TEXTLAB_GLM_OCR_MODEL", "glm-ocr:latest")
#: The prompts GLM-OCR was trained with, one per kind of content.
MODES = ("Text Recognition", "Table Recognition", "Figure Recognition")
#: Longest image side sent to the model, in pixels.
MAX_IMAGE_SIDE = 2048
#: Context window for one page.
CONTEXT_TOKENS = 8192


def ensure_model(
    on_progress: ProgressCallback, model: str = MODEL, client: Any = ollama
) -> None:
    """Pull the model unless the Ollama server already has it.

    Args:
        on_progress: Told when a download starts.
        model: The model name.
        client: The ``ollama`` module or an ``ollama.Client``.

    Raises:
        EngineError: If the download fails.
    """
    installed = installed_model_names(client)
    if installed is not None and canonical_model_name(model) in installed:
        return
    on_progress(Progress(f"Pulling model '{model}'..."))
    try:
        client.pull(model)
    except Exception as error:
        raise EngineError(f"Failed to pull GLM-OCR model: {error}") from error


def page_png(image_path: Path, max_side: int = MAX_IMAGE_SIDE) -> bytes | None:
    """Return a page image as PNG data, scaled down to ``max_side``.

    Args:
        image_path: The page image.
        max_side: Longest side of the result, in pixels.

    Returns:
        The PNG data, or ``None`` if the image cannot be read or encoded.
    """
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        return None
    h, w = image.shape[:2]
    if h > max_side or w > max_side:
        scale = max_side / max(h, w)
        image = cv2.resize(
            image,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(".png", image)
    return encoded.tobytes() if ok else None


class GlmOcr(PageEngine):
    """GLM-OCR on the Ollama server, one request per page."""

    name = "GLM-OCR"
    modes = MODES

    def prepare(
        self, options: EngineOptions, on_progress: ProgressCallback
    ) -> None:
        """Pull the model if the server does not have it."""
        ensure_model(on_progress)

    def recognize_pages(
        self,
        images: list[Path],
        options: EngineOptions,
        on_progress: ProgressCallback,
    ) -> EngineOutput:
        """Read page images (see :meth:`PageEngine.recognize_pages`).

        A page that fails gets an error note as its text, so one bad page
        does not lose the others.
        """
        prompt = options.mode or MODES[0]
        output = EngineOutput(pages=[])
        for index, image_path in enumerate(images, start=1):
            png = page_png(image_path)
            if png is None:
                text = "[Error encoding image]"
            else:
                text = _recognize(png, prompt, index)
                if options.previews:
                    output.previews.append(Preview(png))
            output.pages.append(PageText(index, text, {"content": text}))
            page_done(on_progress, self.name, index, len(images))
        return output


def _recognize(png: bytes, prompt: str, page: int) -> str:
    """Return the model's answer for one page, or an error note."""
    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt, "images": [png]}],
            options={"temperature": 0, "num_ctx": CONTEXT_TOKENS},
        )
    except Exception as error:
        return f"[Error processing page {page}: {error}]"
    return message_text(_message(response))


def _message(response: Any) -> Any:
    """Return the message of a chat response object or dictionary."""
    if isinstance(response, dict):
        return response.get("message")
    return getattr(response, "message", None)
