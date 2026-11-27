"""Data classes of the Hugging Face OCR tool: options and results.

Users name any vision-language model on the Hugging Face Hub that
Transformers can load without running the repository's own code. The
model is downloaded into their own folder (:mod:`.store`) and reads one
page image per request, with a prompt the user can change.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Mapping
from typing import Any

#: Instruction sent with every page when the user gives none. Models
#: trained with a specific prompt read best with that one; their model
#: cards name it.
DEFAULT_PROMPT = (
    "Transcribe all text in this image in reading order. Return only the "
    "text, without comments."
)
#: Most tokens generated for one page.
DEFAULT_MAX_NEW_TOKENS = 2048
#: Upper limit users can set, so a run that never stops generating ends.
MAX_NEW_TOKENS_LIMIT = 8192

HUB_URL = "https://huggingface.co"

#: A Hub repository name: ``owner/name``, as the Hub allows them.
_REPO_ID = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$"
)


class HfOcrError(RuntimeError):
    """A run could not be done, with a message for users."""


def parse_repo_id(text: str) -> str:
    """Return the ``owner/name`` of a model, from its name or its URL.

    Args:
        text: What the user typed, for example ``owner/name`` or
            ``https://huggingface.co/owner/name``.

    Returns:
        The repository name.

    Raises:
        HfOcrError: If the text does not name a Hub model.
    """
    value = text.strip()
    for prefix in (f"{HUB_URL}/", "huggingface.co/"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
    value = value.split("?")[0].split("#")[0].strip("/")
    parts = value.split("/")
    if len(parts) > 2 and parts[2] in ("tree", "blob", "resolve"):
        value = "/".join(parts[:2])
    if not _REPO_ID.match(value) or ".." in value or value.endswith(".git"):
        raise HfOcrError(
            f"{text.strip() or 'This'} is not the name of a Hugging Face "
            "model. Enter it as owner/name, for example "
            "Qwen/Qwen2.5-VL-3B-Instruct."
        )
    return value


@dataclasses.dataclass(frozen=True)
class HfOcrOptions:
    """How to read the pages.

    Attributes:
        repo_id: The model, as ``owner/name``.
        prompt: The instruction sent with every page; empty for models
            that take no instruction.
        max_new_tokens: Most tokens generated for one page.
    """

    repo_id: str
    prompt: str = DEFAULT_PROMPT
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS

    def __post_init__(self) -> None:
        """Check the values.

        Raises:
            HfOcrError: If the model name or the token limit is invalid.
        """
        object.__setattr__(self, "repo_id", parse_repo_id(self.repo_id))
        if not 1 <= self.max_new_tokens <= MAX_NEW_TOKENS_LIMIT:
            raise HfOcrError(
                "The token limit must be between 1 and "
                f"{MAX_NEW_TOKENS_LIMIT}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the options as JSON-serializable data."""
        return dataclasses.asdict(self)


@dataclasses.dataclass
class PageText:
    """The text read on one page.

    Attributes:
        page: Page number in the document, from 1.
        text: What the model returned for the page.
    """

    page: int
    text: str


@dataclasses.dataclass
class DocumentText:
    """The text of one document, page by page.

    Attributes:
        name: The document's file name.
        pages: One entry per page, in order.
    """

    name: str
    pages: list[PageText]

    @property
    def text(self) -> str:
        """All pages, separated by a blank line."""
        return "\n\n".join(page.text for page in self.pages)

    def to_dict(self) -> dict[str, Any]:
        """Return the document as JSON-serializable data."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ModelUsed:
    """The model that produced a result, for citation.

    Attributes:
        repo_id: The model, as ``owner/name``.
        revision: The commit of the repository that was loaded, if known.
    """

    repo_id: str
    revision: str | None = None

    @property
    def url(self) -> str:
        """The model's page on the Hub."""
        return f"{HUB_URL}/{self.repo_id}"

    def citation(self) -> str:
        """The model as one line, with its revision when known."""
        if self.revision:
            return f"{self.repo_id} (revision {self.revision})"
        return self.repo_id

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelUsed:
        """Create the record from the worker's result."""
        return cls(data["repo_id"], data.get("revision"))


@dataclasses.dataclass
class OcrRun:
    """A finished single-document run, ready for display and download.

    Attributes:
        document: The text, page by page.
        model: The model that read it.
        prompt: The instruction that was sent with every page.
        stem: Base name for the downloads.
        zip_bytes: Text, JSON and ``model_used.txt`` in one ZIP.
        previews: One PNG of each page, scaled down for display.
    """

    document: DocumentText
    model: ModelUsed
    prompt: str
    stem: str
    zip_bytes: bytes
    previews: list[bytes] = dataclasses.field(default_factory=list)

    def to_json(self) -> str:
        """Return the document, model and prompt as indented JSON."""
        return result_json(self.document, self.model, self.prompt)


@dataclasses.dataclass(frozen=True)
class BatchFile:
    """One document of a batch, as it was read.

    Attributes:
        name: Its path in the archive.
        pages: How many pages it has.
        characters: How many characters the model returned for them.
    """

    name: str
    pages: int
    characters: int


@dataclasses.dataclass
class BatchRun:
    """A finished batch: the result ZIP and a summary.

    Attributes:
        zip_bytes: One folder per document, mirroring the archive.
        files: Number of documents read.
        pages: Number of pages read.
        model: The model that read them.
        items: Each document's overview, in the archive's order.
    """

    zip_bytes: bytes
    files: int
    pages: int
    model: ModelUsed
    items: tuple[BatchFile, ...] = ()


def result_json(document: DocumentText, model: ModelUsed, prompt: str) -> str:
    """Return a document's result as indented JSON."""
    data = {
        "document": document.name,
        "model": {"repo_id": model.repo_id, "revision": model.revision},
        "prompt": prompt,
        "pages": [dataclasses.asdict(page) for page in document.pages],
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


def model_used_text(model: ModelUsed, prompt: str) -> str:
    """Return the model and prompt as text, for ``model_used.txt``."""
    return (
        "Text read with a Hugging Face model in Text Lab.\n\n"
        f"Model: {model.citation()}\n"
        f"Model page: {model.url}\n"
        f"Prompt: {prompt or '(none)'}\n"
    )
