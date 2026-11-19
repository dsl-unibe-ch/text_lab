"""Line recognition with a Hugging Face vision-language model.

A :class:`Recognizer` holds a model and its processor and reads line images
in batches, one line per sequence, with the prompt the model was trained
with. A LoRA adapter is merged into the base model's weights when the
model is loaded (:func:`merge_lora`), so the ``peft`` package is not
needed and generation runs at the speed of the base model.

Runs in the recognition worker (:mod:`.worker`). PyTorch and Transformers
are imported inside the functions, so the module imports without them.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from textlab.features.handwriting.models import (
    HandwritingError,
    HandwritingModel,
)

LOGGER = logging.getLogger(__name__)

#: Lines read in one generation call; halved when the GPU runs out of
#: memory.
BATCH_SIZE = 8
#: Most tokens generated for one line; a long handwritten line has well
#: under a hundred.
MAX_NEW_TOKENS = 256

#: Files of a LoRA adapter repository.
ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
#: Prefix of the parameter names in a saved PEFT adapter.
PEFT_PREFIX = "base_model.model."


def merge_lora(
    model: Any, state: Mapping[str, Any], config: Mapping[str, Any]
) -> int:
    """Add a LoRA adapter's weight updates to a model, in place.

    Each adapted linear layer ``W`` becomes ``W + scale * B @ A``, with
    ``scale = lora_alpha / r`` (``lora_alpha / sqrt(r)`` for rsLoRA), which
    is what ``peft``'s ``merge_and_unload`` computes. Adapter variants that
    change more than that (DoRA, per-module ranks, extra trained modules)
    are refused rather than merged wrongly.

    Args:
        model: The base model, a ``torch.nn.Module``.
        state: The adapter's tensors, as saved by ``peft``.
        config: The adapter's ``adapter_config.json``.

    Returns:
        The number of layers updated.

    Raises:
        HandwritingError: If the adapter is of an unsupported kind or does
            not match the model.
    """
    import torch

    _check_adapter_config(config)
    rank = config["r"]
    alpha = config["lora_alpha"]
    scale = (
        alpha / math.sqrt(rank) if config.get("use_rslora") else (alpha / rank)
    )

    prefixes = set()
    for name in state:
        for suffix in (".lora_A.weight", ".lora_B.weight"):
            if name.endswith(suffix):
                prefixes.add(name[: -len(suffix)])
                break
        else:
            raise HandwritingError(
                f"The adapter has a tensor this merge does not handle: {name}"
            )

    modules = dict(model.named_modules())
    merged = 0
    with torch.no_grad():
        for prefix in sorted(prefixes):
            target = prefix.removeprefix(PEFT_PREFIX)
            module = modules.get(target)
            weight = getattr(module, "weight", None)
            if weight is None:
                raise HandwritingError(
                    f"The adapter does not match the model: no layer {target}"
                )
            lora_a = state[f"{prefix}.lora_A.weight"]
            lora_b = state[f"{prefix}.lora_B.weight"]
            delta = lora_b.to(weight.device, torch.float32) @ lora_a.to(
                weight.device, torch.float32
            )
            if config.get("fan_in_fan_out"):
                delta = delta.T
            if delta.shape != weight.shape:
                raise HandwritingError(
                    f"The adapter does not match the model: {target} has "
                    f"shape {tuple(weight.shape)}, the update "
                    f"{tuple(delta.shape)}"
                )
            weight.add_((delta * scale).to(weight.dtype))
            merged += 1
    if not merged:
        raise HandwritingError("The adapter holds no LoRA weights.")
    return merged


def _check_adapter_config(config: Mapping[str, Any]) -> None:
    """Refuse adapters that :func:`merge_lora` would merge wrongly."""
    problems = []
    if config.get("peft_type", "LORA") != "LORA":
        problems.append(f"type {config.get('peft_type')}")
    if config.get("use_dora"):
        problems.append("DoRA")
    if config.get("rank_pattern") or config.get("alpha_pattern"):
        problems.append("per-module ranks")
    if config.get("modules_to_save"):
        problems.append("extra trained modules")
    if problems:
        raise HandwritingError(
            "This adapter cannot be merged: " + ", ".join(problems)
        )
    if not config.get("r") or not config.get("lora_alpha"):
        raise HandwritingError("The adapter configuration has no r or alpha.")


class Recognizer:
    """A loaded recognition model that reads line images.

    Attributes:
        spec: The model's configuration.
        batch_size: Lines per generation call; lowered after running out
            of GPU memory.
    """

    def __init__(
        self, spec: HandwritingModel, model: Any, processor: Any, device: str
    ) -> None:
        """Wrap a loaded model and processor (see :func:`load_recognizer`)."""
        self.spec = spec
        self.batch_size = BATCH_SIZE
        self._model = model
        self._processor = processor
        self._device = device

    def read(
        self,
        images: Sequence[Any],
        on_done: Callable[[int], None] | None = None,
    ) -> list[str]:
        """Return the text of each line image, in order.

        Args:
            images: RGB ``PIL.Image`` line crops.
            on_done: Called with the number of lines read so far.

        Returns:
            One text per image.
        """
        import torch

        texts: list[str] = []
        while len(texts) < len(images):
            batch = images[len(texts) : len(texts) + self.batch_size]
            try:
                texts.extend(self._generate(batch))
            except torch.cuda.OutOfMemoryError:
                if self.batch_size == 1:
                    raise
                self.batch_size //= 2
                torch.cuda.empty_cache()
                LOGGER.warning(
                    "Out of GPU memory; reading %d lines at a time",
                    self.batch_size,
                )
                continue
            if on_done is not None:
                on_done(len(texts))
        return texts

    def _generate(self, images: Sequence[Any]) -> list[str]:
        """Read one batch of line images."""
        import torch

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.spec.prompt},
                ],
            }
        ]
        prompt = self._processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[prompt] * len(images),
            images=list(images),
            padding=True,
            return_tensors="pt",
        ).to(self._device)
        with torch.inference_mode():
            output = self._model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
            )
        generated = output[:, inputs["input_ids"].shape[1] :]
        decoded = self._processor.batch_decode(
            generated, skip_special_tokens=True
        )
        return [text.strip() for text in decoded]


def load_recognizer(
    spec: HandwritingModel, device: str | None = None
) -> Recognizer:
    """Load a recognition model, merging its adapter if it has one.

    Weights are downloaded into the Hugging Face cache on first use.

    Args:
        spec: The model's configuration.
        device: ``"cuda"`` or ``"cpu"``; by default the GPU if there is
            one.

    Returns:
        The recognizer.
    """
    import torch
    from transformers import AutoModelForImageTextToText

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        spec.base_model, dtype=dtype
    )
    model.to(device)
    if spec.adapter:
        config, state = _load_adapter(spec.adapter)
        merged = merge_lora(model, state, config)
        LOGGER.info("Merged %d LoRA layers of %s", merged, spec.adapter)
    model.eval()
    return Recognizer(spec, model, _load_processor(spec), device)


def _load_adapter(repo: str) -> tuple[dict, dict]:
    """Download a LoRA adapter; return its configuration and tensors."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    with open(hf_hub_download(repo, ADAPTER_CONFIG), encoding="utf-8") as fh:
        config = json.load(fh)
    return config, load_file(hf_hub_download(repo, ADAPTER_WEIGHTS))


def _load_processor(spec: HandwritingModel) -> Any:
    """Load the adapter's processor, or the base model's if it has none.

    Batches are padded on the left, so every line's answer starts right
    after its prompt.
    """
    from transformers import AutoProcessor

    processor = None
    if spec.adapter:
        try:
            processor = AutoProcessor.from_pretrained(spec.adapter)
        except (OSError, ValueError, KeyError):
            LOGGER.info(
                "No processor in %s; using the base model's", spec.adapter
            )
    if processor is None:
        processor = AutoProcessor.from_pretrained(spec.base_model)
    processor.tokenizer.padding_side = "left"
    return processor
