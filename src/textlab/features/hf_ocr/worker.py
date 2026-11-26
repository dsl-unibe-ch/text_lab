"""Worker: read page images with a Hugging Face vision-language model.

Runs in the image's Transformers 5 environment (``common.container.
HF_OCR_ENV``), started by :func:`.service.read_pages` through
:mod:`textlab.common.jobs`::

    python -m textlab.features.hf_ocr.worker JOB_DIR

The request is ``{"repo_id", "prompt", "max_new_tokens", "cache_dir",
"images": [path, ...]}``; the result is ``{"texts": [...], "model":
{"repo_id", "revision"}}``, one text per image.

The same code serves every model; nothing depends on which one it is.
A model is accepted when Transformers can load it as
``AutoModelForImageTextToText`` without running code from its repository
(``trust_remote_code=False``), and it is public and not gated: no token is
ever sent. Before downloading, the model's size is compared with the free
GPU memory, so a model that cannot fit is refused without filling the
user's folder.

The app's packages are not installed in this environment: only the
standard library, its own packages and the standard-library-only modules
``textlab.common.jobs`` and ``textlab.common.progress`` may be imported. A
test checks this.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from textlab.common.jobs import worker_main
from textlab.common.progress import Progress, ProgressCallback

#: Longest side of a page image given to the model, in pixels. Processors
#: resize further to their own limits; this keeps very large scans from
#: costing more GPU memory than any model needs.
MAX_PAGE_SIDE = 2048
#: Bytes per parameter of a model loaded in bfloat16.
BYTES_PER_PARAMETER = 2
#: Share of the free GPU memory the weights may take; the rest is for the
#: page and the generated text.
WEIGHTS_SHARE = 0.85


class OcrModelError(Exception):
    """The model cannot be used, with a message for users.

    The service shows its message as it is (see
    :func:`.service.read_pages`).
    """


def read(request: Mapping[str, Any], report: ProgressCallback) -> dict:
    """Read the requested page images with the requested model.

    Args:
        request: See the module docstring.
        report: Receives progress updates.

    Returns:
        ``{"texts": [...], "model": {"repo_id", "revision"}}``.

    Raises:
        OcrModelError: If the model cannot be used.
    """
    cache_dir = Path(request["cache_dir"])
    # Keep the download's chunk cache in the user's folder, not in the
    # shared Hugging Face store the image points HF_HOME at.
    os.environ["HF_XET_CACHE"] = str(cache_dir / "xet")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    import torch

    repo_id = request["repo_id"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report(Progress(f"Checking {repo_id} on Hugging Face...", None))
    hub_revision = check_model(repo_id, free_gpu_bytes(torch))

    report(
        Progress(
            f"Loading {repo_id} (the first use downloads it to your "
            "folder)...",
            None,
        )
    )
    model, processor = load_model(repo_id, cache_dir, device, torch)
    revision = getattr(model.config, "_commit_hash", None) or hub_revision

    images = list(request["images"])
    texts = []
    for index, image in enumerate(images, start=1):
        try:
            texts.append(
                read_page(
                    model,
                    processor,
                    page_image(image),
                    request["prompt"],
                    int(request["max_new_tokens"]),
                    torch,
                )
            )
        except torch.cuda.OutOfMemoryError as error:
            raise OcrModelError(
                f"The GPU ran out of memory on page {index}. Try a smaller "
                "model or a lower token limit."
            ) from error
        report(
            Progress(
                f"Reading pages... {index}/{len(images)}",
                index / len(images),
            )
        )
    return {
        "texts": texts,
        "model": {"repo_id": repo_id, "revision": revision},
    }


def free_gpu_bytes(torch: Any) -> int | None:
    """Return the free memory of the GPU, or ``None`` without one."""
    if not torch.cuda.is_available():
        return None
    free, _total = torch.cuda.mem_get_info()
    return int(free)


def check_model(repo_id: str, free_bytes: int | None) -> str | None:
    """Check a model on the Hub before anything is downloaded.

    Args:
        repo_id: The model.
        free_bytes: Free GPU memory, or ``None`` to skip the size check.

    Returns:
        The repository's current commit, for citation.

    Raises:
        OcrModelError: If the model does not exist, is private or gated,
            is only an adapter, or is too large for the GPU.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    try:
        info = HfApi().model_info(repo_id, token=False)
    except GatedRepoError as error:
        raise OcrModelError(_GATED.format(repo_id=repo_id)) from error
    except RepositoryNotFoundError as error:
        raise OcrModelError(
            f"There is no public model {repo_id} on Hugging Face. Check the "
            "name; private models cannot be used."
        ) from error
    if info.gated:
        raise OcrModelError(_GATED.format(repo_id=repo_id))
    files = {sibling.rfilename for sibling in info.siblings or ()}
    if "adapter_config.json" in files and "config.json" not in files:
        raise OcrModelError(
            f"{repo_id} is an adapter (LoRA), not a full model: it only "
            "holds changes to another model, and this tool loads full "
            "models only. Look on Hugging Face for a merged version."
        )
    parameters = getattr(info.safetensors, "total", None)
    check_size(repo_id, parameters, free_bytes)
    return info.sha


_GATED = (
    "{repo_id} is gated: Hugging Face requires accepting its conditions "
    "with an account first. This tool only downloads public models "
    "without an account."
)


def check_size(
    repo_id: str, parameters: int | None, free_bytes: int | None
) -> None:
    """Refuse a model whose weights cannot fit into the free GPU memory.

    Args:
        repo_id: The model, for the message.
        parameters: Its number of parameters, if the Hub knows it.
        free_bytes: Free GPU memory, if there is a GPU.

    Raises:
        OcrModelError: If the model is too large.
    """
    if not parameters or not free_bytes:
        return
    needed = parameters * BYTES_PER_PARAMETER
    if needed > free_bytes * WEIGHTS_SHARE:
        raise OcrModelError(
            f"{repo_id} has {parameters / 1e9:.1f} billion parameters and "
            f"needs about {needed / 1024**3:.0f} GB of GPU memory; "
            f"{free_bytes / 1024**3:.0f} GB are free. Choose a smaller model."
        )


def load_model(repo_id: str, cache_dir: Path, device: str, torch: Any):
    """Load a model and its processor into the user's folder.

    Returns:
        ``(model, processor)``.

    Raises:
        OcrModelError: If Transformers cannot load the model as an image
            and text to text model without running the repository's code.
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    common = {"cache_dir": str(cache_dir), "token": False}
    try:
        processor = AutoProcessor.from_pretrained(
            repo_id, trust_remote_code=False, **common
        )
        model = AutoModelForImageTextToText.from_pretrained(
            repo_id, dtype=dtype, trust_remote_code=False, **common
        )
    except (ValueError, KeyError) as error:
        raise OcrModelError(
            f"{repo_id} cannot be used here: Transformers cannot load it as "
            "an image-and-text model without running code from its "
            f"repository, which this tool does not do. ({error})"
        ) from error
    except OSError as error:
        raise OcrModelError(
            f"{repo_id} could not be downloaded or read: {error}"
        ) from error
    try:
        model.to(device)
    except torch.cuda.OutOfMemoryError as error:
        raise OcrModelError(
            f"{repo_id} does not fit into the GPU's memory. Choose a "
            "smaller model."
        ) from error
    model.eval()
    return model, processor


def page_image(path: str) -> Any:
    """Open a page image as RGB, at most :data:`MAX_PAGE_SIDE` pixels."""
    from PIL import Image

    with Image.open(path) as image:
        page = image.convert("RGB")
    page.thumbnail((MAX_PAGE_SIDE, MAX_PAGE_SIDE))
    return page


def read_page(
    model: Any,
    processor: Any,
    image: Any,
    prompt: str,
    max_new_tokens: int,
    torch: Any,
) -> str:
    """Return the text a model generates for one page image.

    A processor with a chat template gets the page and the prompt as one
    user message, as chat models are trained; one without gets them
    directly. Only the generated part of the output is returned.
    """
    if getattr(processor, "chat_template", None):
        content: list[dict[str, str]] = [{"type": "image"}]
        if prompt:
            content.append({"type": "text", "text": prompt})
        text = processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(text=[text], images=[image], return_tensors="pt")
    else:
        arguments: dict[str, Any] = {"images": image, "return_tensors": "pt"}
        if prompt:
            arguments["text"] = prompt
        inputs = processor(**arguments)
    inputs = inputs.to(model.device).to(model.dtype)
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    if "input_ids" in inputs and not model.config.is_encoder_decoder:
        output = output[:, inputs["input_ids"].shape[1] :]
    decode = getattr(processor, "batch_decode", None)
    if decode is None:
        decode = processor.tokenizer.batch_decode
    return decode(output, skip_special_tokens=True)[0].strip()


if __name__ == "__main__":
    worker_main(read)
