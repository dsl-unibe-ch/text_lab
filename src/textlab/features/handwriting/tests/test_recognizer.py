"""Merging LoRA adapters and reading lines in batches, without a model."""

import pytest

from textlab.features.handwriting import recognizer
from textlab.features.handwriting.models import (
    HandwritingError,
    HandwritingModel,
)

torch = pytest.importorskip("torch")

CONFIG = {"peft_type": "LORA", "r": 2, "lora_alpha": 4}


class Tiny(torch.nn.Module):
    """A model with the module layout of a PEFT-adapted language model."""

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.proj = torch.nn.Linear(3, 4, bias=False)
        torch.nn.init.zeros_(self.model.proj.weight)


def adapter_state(prefix="base_model.model.model.proj"):
    lora_a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    lora_b = torch.ones(4, 2)
    return {
        f"{prefix}.lora_A.weight": lora_a,
        f"{prefix}.lora_B.weight": lora_b,
    }, lora_b @ lora_a


def test_lora_updates_are_added_with_alpha_over_r():
    model = Tiny()
    state, product = adapter_state()
    assert recognizer.merge_lora(model, state, CONFIG) == 1
    assert torch.allclose(model.model.proj.weight, 2 * product)


def test_rslora_scales_with_the_square_root_of_r():
    model = Tiny()
    state, product = adapter_state()
    recognizer.merge_lora(model, state, {**CONFIG, "use_rslora": True})
    assert torch.allclose(model.model.proj.weight, (4 / 2**0.5) * product)


@pytest.mark.parametrize(
    "change",
    [
        {"use_dora": True},
        {"peft_type": "IA3"},
        {"rank_pattern": {"proj": 8}},
        {"modules_to_save": ["lm_head"]},
        {"r": 0},
    ],
)
def test_adapters_that_cannot_be_merged_are_refused(change):
    state, _ = adapter_state()
    with pytest.raises(HandwritingError):
        recognizer.merge_lora(Tiny(), state, {**CONFIG, **change})


def test_an_adapter_for_another_model_is_refused():
    state, _ = adapter_state("base_model.model.model.missing")
    with pytest.raises(HandwritingError, match="no layer"):
        recognizer.merge_lora(Tiny(), state, CONFIG)
    state = {
        "base_model.model.model.proj.lora_A.weight": torch.ones(2, 5),
        "base_model.model.model.proj.lora_B.weight": torch.ones(4, 2),
    }
    with pytest.raises(HandwritingError, match="shape"):
        recognizer.merge_lora(Tiny(), state, CONFIG)


def test_unknown_adapter_tensors_are_refused():
    state, _ = adapter_state()
    state["base_model.model.lm_head.weight"] = torch.ones(1)
    with pytest.raises(HandwritingError, match="does not handle"):
        recognizer.merge_lora(Tiny(), state, CONFIG)


class FakeRecognizer(recognizer.Recognizer):
    """Reads each image as its own name; runs out of memory on big batches."""

    def __init__(self, limit):
        spec = HandwritingModel("k", "L", "", "b", None, "Read.")
        super().__init__(spec, None, None, "cpu")
        self.limit = limit
        self.batches = []

    def _generate(self, images):
        if len(images) > self.limit:
            raise torch.cuda.OutOfMemoryError("too big")
        self.batches.append(len(images))
        return [f"text {image}" for image in images]


def test_lines_are_read_in_order_in_batches():
    reader = FakeRecognizer(limit=100)
    done = []
    texts = reader.read(list(range(10)), on_done=done.append)
    assert texts == [f"text {n}" for n in range(10)]
    assert reader.batches == [8, 2]
    assert done == [8, 10]


def test_the_batch_shrinks_when_the_gpu_runs_out_of_memory():
    reader = FakeRecognizer(limit=3)
    assert reader.read(list(range(5))) == [f"text {n}" for n in range(5)]
    assert reader.batch_size == 2
    assert reader.batches == [2, 2, 1]


def test_running_out_of_memory_on_one_line_is_an_error():
    reader = FakeRecognizer(limit=0)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        reader.read([1])
