"""Model names, options and the result records."""

import json

import pytest

from textlab.features.hf_ocr import models


@pytest.mark.parametrize(
    "text",
    [
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "  Qwen/Qwen2.5-VL-3B-Instruct  ",
        "https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct",
        "huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/",
        "https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/main",
        "https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct?library=x",
    ],
)
def test_a_model_is_named_by_its_name_or_its_link(text):
    assert models.parse_repo_id(text) == "Qwen/Qwen2.5-VL-3B-Instruct"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "just-a-name",
        "a/b/c",
        "../etc/passwd",
        "owner/..",
        "a b/c",
        "o/n.git",
    ],
)
def test_other_text_is_refused(text):
    with pytest.raises(models.HfOcrError, match="owner/name"):
        models.parse_repo_id(text)


def test_options_check_their_values():
    options = models.HfOcrOptions("https://huggingface.co/org/model")
    assert options.repo_id == "org/model"
    assert options.to_dict() == {
        "repo_id": "org/model",
        "prompt": models.DEFAULT_PROMPT,
        "max_new_tokens": models.DEFAULT_MAX_NEW_TOKENS,
    }
    with pytest.raises(models.HfOcrError, match="token limit"):
        models.HfOcrOptions("org/model", max_new_tokens=0)
    with pytest.raises(models.HfOcrError, match="token limit"):
        models.HfOcrOptions(
            "org/model", max_new_tokens=models.MAX_NEW_TOKENS_LIMIT + 1
        )


def test_a_document_joins_its_pages():
    document = models.DocumentText(
        "a.pdf", [models.PageText(1, "one"), models.PageText(2, "two")]
    )
    assert document.text == "one\n\ntwo"


def test_the_model_used_is_cited_with_its_revision():
    model = models.ModelUsed.from_dict(
        {"repo_id": "org/model", "revision": "abc123"}
    )
    assert model.url == "https://huggingface.co/org/model"
    assert model.citation() == "org/model (revision abc123)"
    assert models.ModelUsed("org/model").citation() == "org/model"
    text = models.model_used_text(model, "")
    assert "org/model (revision abc123)" in text
    assert "Prompt: (none)" in text


def test_the_json_names_model_prompt_and_pages():
    document = models.DocumentText("a.png", [models.PageText(1, "Grüße")])
    data = json.loads(
        models.result_json(document, models.ModelUsed("org/m", "r1"), "Read.")
    )
    assert data == {
        "document": "a.png",
        "model": {"repo_id": "org/m", "revision": "r1"},
        "prompt": "Read.",
        "pages": [{"page": 1, "text": "Grüße"}],
    }
