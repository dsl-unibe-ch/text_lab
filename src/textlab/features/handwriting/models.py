"""Data classes of the handwriting feature: models, options and results.

The recognition models offered are configured in ``common/models.json``
under ``handwriting_models`` (or ``models.local.json`` next to it), so a
model can be added or replaced without a code change. Every entry is a
Hugging Face vision-language model, optionally with a LoRA adapter, that
reads one cropped text line per request.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from textlab.common.model_config import load_model_config

#: Key of the model list in the model configuration.
CONFIG_KEY = "handwriting_models"

#: The text detector that finds the lines (PaddleOCR, in the image's
#: ``paddle_backend`` environment).
LINE_DETECTOR = "PP-OCRv5_server_det"


class HandwritingError(RuntimeError):
    """A transcription could not be done, with a message for users."""


@dataclasses.dataclass(frozen=True)
class HandwritingModel:
    """A recognition model users can choose.

    Attributes:
        key: Stable identifier, used in options and results.
        label: The name shown to users.
        description: What the model was trained on and how well it reads.
        base_model: Hugging Face repository of the vision-language model.
        adapter: Hugging Face repository of a LoRA adapter for the base
            model, or ``None`` to use the base model as it is. The adapter
            repository's processor is used when it has one.
        prompt: The instruction sent with every line image; it should be
            the one the model was trained with.
        max_pixels: Line images larger than this are scaled down to it,
            as in training; ``None`` keeps the processor's own limit.
    """

    key: str
    label: str
    description: str
    base_model: str
    adapter: str | None
    prompt: str
    max_pixels: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the model as JSON-serializable data."""
        return dataclasses.asdict(self)

    def models_used(self) -> dict[str, str]:
        """Return the models a transcription with this model runs.

        Known before a run, so interfaces can show it up front; results
        report the same (:meth:`DocumentTranscription.models_used`).
        """
        recognition = self.base_model
        if self.adapter:
            recognition = f"{self.adapter} (LoRA adapter on {recognition})"
        return {
            "line_detection": f"PaddleOCR {LINE_DETECTOR}",
            "recognition": recognition,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HandwritingModel:
        """Create a model from :meth:`to_dict` output or a config entry.

        Raises:
            ValueError: If a required field is missing.
        """
        try:
            return cls(
                key=str(data["key"]),
                label=str(data["label"]),
                description=str(data.get("description", "")),
                base_model=str(data["base_model"]),
                adapter=data.get("adapter") or None,
                prompt=str(data["prompt"]),
                max_pixels=data.get("max_pixels"),
            )
        except KeyError as error:
            raise ValueError(
                f"Handwriting model entry is missing {error}: {dict(data)}"
            ) from error


def available_models() -> list[HandwritingModel]:
    """Return the configured recognition models, in the configured order.

    Raises:
        ValueError: If an entry is incomplete or two share a key.
    """
    entries = load_model_config().get(CONFIG_KEY, [])
    models = [HandwritingModel.from_dict(entry) for entry in entries]
    keys = [model.key for model in models]
    if len(set(keys)) != len(keys):
        raise ValueError(f"Duplicate handwriting model keys: {keys}")
    return models


def get_model(key: str) -> HandwritingModel:
    """Return the configured model with this key.

    Raises:
        HandwritingError: If no model has this key.
    """
    for model in available_models():
        if model.key == key:
            return model
    raise HandwritingError(f"Unknown handwriting model: {key!r}")


@dataclasses.dataclass(frozen=True)
class HandwritingOptions:
    """Settings for a transcription.

    Attributes:
        model: Key of the recognition model (:func:`available_models`).
        previews: Draw page images with the detected lines outlined.
    """

    model: str
    previews: bool = True


@dataclasses.dataclass
class Line:
    """One text line on a page.

    Attributes:
        box: The line's bounding box on the page image, in pixels, as
            ``(left, top, right, bottom)``.
        text: The transcribed text.
    """

    box: tuple[int, int, int, int]
    text: str = ""


@dataclasses.dataclass
class PageTranscription:
    """The lines of one page, in reading order.

    Attributes:
        page: The page number, from 1.
        width: Width of the page image, in pixels.
        height: Height of the page image, in pixels.
        lines: The lines, in reading order.
        image_name: File name of the page image in the PAGE XML export.
    """

    page: int
    width: int
    height: int
    lines: list[Line]
    image_name: str = ""

    @property
    def text(self) -> str:
        """The page's text, one line per detected line."""
        return "\n".join(line.text for line in self.lines)


@dataclasses.dataclass
class DocumentTranscription:
    """The transcription of one PDF or image.

    Attributes:
        name: The file's name.
        model: The recognition model used.
        pages: The pages, in order.
    """

    name: str
    model: HandwritingModel
    pages: list[PageTranscription]

    @property
    def text(self) -> str:
        """The text of all pages, separated by blank lines."""
        return "\n\n".join(page.text for page in self.pages)

    @property
    def line_count(self) -> int:
        """The number of lines on all pages."""
        return sum(len(page.lines) for page in self.pages)

    def models_used(self) -> dict[str, str]:
        """Return the models that produced the result, for citation."""
        return self.model.models_used()

    def to_dict(self) -> dict[str, Any]:
        """Return the transcription as JSON-serializable data."""
        return {
            "file": self.name,
            "model": self.model.label,
            "models_used": self.models_used(),
            "pages": [
                {
                    "page": page.page,
                    "width": page.width,
                    "height": page.height,
                    "lines": [
                        {"box": list(line.box), "text": line.text}
                        for line in page.lines
                    ],
                }
                for page in self.pages
            ],
        }


@dataclasses.dataclass
class TranscriptionRun:
    """The outcome of transcribing one document.

    Attributes:
        document: The transcription.
        text_name: File name for the text download.
        json: The transcription as JSON.
        json_name: File name for the JSON download.
        previews: PNG images of the pages with the lines outlined and
            numbered, if requested.
        zip_bytes: Text, JSON and one PAGE XML file per page in one ZIP.
    """

    document: DocumentTranscription
    text_name: str
    json: str
    json_name: str
    previews: list[bytes]
    zip_bytes: bytes

    @property
    def text(self) -> str:
        """The text of all pages."""
        return self.document.text


@dataclasses.dataclass
class BatchRun:
    """The outcome of transcribing a ZIP archive.

    Attributes:
        zip_bytes: The results, one folder per file, mirroring the archive.
        files: The number of files transcribed.
        lines: The number of lines transcribed in all files.
        models_used: The models that produced the results, for citation.
    """

    zip_bytes: bytes
    files: int
    lines: int
    models_used: dict[str, str]
