"""Offline token-budget and recovery contracts; no models or GPU needed."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from textlab.features.translation import engine
from textlab.features.translation.chunking import (
    MAX_SPLIT_RETRIES,
    InputTooLongError,
    OutputTruncatedError,
    TranslationLimitError,
    chunk_text_for_translation,
    join_translations,
    split_for_retry,
    split_text,
    translate_lines,
)
from textlab.features.translation.hf_backend import generate_translations


class Tensor:
    def __init__(self, rows, texts=None):
        self.rows = rows
        self.texts = texts
        self.shape = (len(rows), max(map(len, rows), default=0))

    def to(self, device):
        return self

    def tolist(self):
        return self.rows


class Tokenizer:
    """Count UTF-8 bytes as tokens, plus BOS/EOS, to expose char heuristics."""

    model_max_length = 24
    eos_token_id = 2
    unk_token_id = 0

    def __init__(self):
        self.calls = []
        self.src_lang = None

    def __call__(self, value, **kwargs):
        self.calls.append((value, kwargs))
        assert kwargs.get("truncation") is False
        assert kwargs.get("add_special_tokens") is True
        if isinstance(value, str):
            return {"input_ids": [1] + list(value.encode()) + [2]}
        rows = [[1] + list(text.encode()) + [2] for text in value]
        width = max(map(len, rows))
        return {
            "input_ids": Tensor(
                [row + [0] * (width - len(row)) for row in rows],
                value,
            ),
        }

    def batch_decode(self, generated, **kwargs):
        return generated.texts

    def convert_tokens_to_ids(self, language):
        return 7


class Model:
    def __init__(self, finish=None):
        self.config = SimpleNamespace(max_position_embeddings=24)
        self.generation_config = SimpleNamespace(
            eos_token_id=2,
            forced_eos_token_id=2,
        )
        self.finish = finish or (lambda text: True)
        self.calls = []

    def generate(self, input_ids, **kwargs):
        assert kwargs["forced_eos_token_id"] is None
        self.calls.append((input_ids.texts, kwargs))
        rows = []
        outputs = []
        for text in input_ids.texts:
            complete = self.finish(text)
            count = kwargs["max_new_tokens"]
            row = [2, 8, 2] if complete else [2] + [8] * count
            rows.append(row)
            result = text.removeprefix("<2fr> ").strip().upper()
            outputs.append(result if complete else "DISCARDED PARTIAL")
        return Tensor(rows, outputs)


@pytest.fixture(autouse=True)
def fake_torch(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            inference_mode=nullcontext,
            cuda=SimpleNamespace(is_available=lambda: False),
        ),
    )


@pytest.mark.parametrize(
    "text",
    [
        "One sentence. Another sentence.  A third!",
        "Unübersehbare Wörter sind vollständig. 再见。再见。",
        "alpha\t beta\r\n gamma\n\n delta ",
        "alpha \x02TL_123\x03 beta \x02GL_0\x03 gamma",
    ],
)
def test_chunks_are_lossless_and_fit_measured_budget(text):
    def measure(value):
        return len(value.encode("utf-8")) + 2

    chunks = split_text(text, measure, 24)
    assert "".join(chunks) == text
    assert all(measure(chunk) <= 24 for chunk in chunks)
    for left, _right in zip(chunks, chunks[1:], strict=False):
        assert left[-1].isspace() or left[-1] in "。！？"
    assert join_translations(chunks, chunks) == text


def test_sentence_boundaries_preferred_and_cjk_spacing_preserved():
    assert split_text("One. Two words here.", len, 12) == [
        "One. ",
        "Two words ",
        "here.",
    ]
    source = "你好。世界。再见。"
    chunks = split_text(source, len, 4)
    assert chunks == ["你好。", "世界。", "再见。"]
    assert join_translations(chunks, chunks) == source


def test_indivisible_span_fails_without_cutting_word_or_placeholder():
    for text in (
        "extraordinary",
        "\x02TL_123456789\x03",
        "无空格且没有句号的长文本",
    ):
        with pytest.raises(InputTooLongError, match="No truncated"):
            split_text(text, len, 5)
    with pytest.raises(OutputTruncatedError, match="indivisible"):
        split_for_retry("extraordinary")


def test_compatibility_chunker_never_hard_splits_words():
    assert chunk_text_for_translation("one two three", max_chars=8) == [
        "one two ",
        "three",
    ]
    with pytest.raises(InputTooLongError):
        chunk_text_for_translation("indivisible", max_chars=3)
    assert chunk_text_for_translation(
        "one two",
        measure=lambda text: len(text) + 2,
        max_tokens=6,
    ) == ["one ", "two"]


def test_empty_layouts_and_output_cardinality():
    assert split_text("", len, 1) == []
    assert translate_lines(["", "\n\n", "   "], None) == ["", "\n\n", "   "]
    with pytest.raises(ValueError):
        split_text("text", len, 0)
    with pytest.raises(TranslationLimitError):
        translate_lines(["first", "second"], lambda texts: ["one"])
    with pytest.raises(TranslationLimitError):
        join_translations(["a", "b"], ["a"])


def test_hf_multibyte_input_is_split_by_tokens_not_characters():
    tokenizer, model = Tokenizer(), Model()
    source = "été été été été été été été été"
    progress = []
    result = generate_translations(
        model,
        tokenizer,
        [source],
        "cpu",
        batch_size=2,
        progress_cb=lambda done, total: progress.append((done, total)),
    )
    assert result == [source.upper()]
    batches = [texts for texts, _ in model.calls]
    # Batches share a token budget of batch_size full windows (2 x 24).
    assert all(
        len(texts) * (len(texts[0].encode()) + 2) <= 48 for texts in batches
    )
    assert all(
        len(text.encode()) + 2 <= 24 for batch in batches for text in batch
    )
    assert progress[-1][0] == progress[-1][1]


def test_madlad_prefix_and_special_tokens_are_in_every_chunk_budget():
    tokenizer, model = Tokenizer(), Model()
    source = "été été été été été"
    assert generate_translations(
        model,
        tokenizer,
        [source],
        "cpu",
        source_prefix="<2fr> ",
    ) == [source.upper()]
    sent = [text for texts, _ in model.calls for text in texts]
    assert len(sent) > 1
    assert all(text.startswith("<2fr> ") for text in sent)
    assert all(len(text.encode()) + 2 <= 24 for text in sent)


def test_smaller_model_window_overrides_the_default_budget():
    tokenizer, model = Tokenizer(), Model()
    tokenizer.model_max_length = 10**30
    model.config.max_position_embeddings = 12
    assert generate_translations(
        model,
        tokenizer,
        ["one two three four"],
        "cpu",
    ) == ["ONE TWO THREE FOUR"]
    assert all(
        len(text.encode()) + 2 <= 12
        for texts, _ in model.calls
        for text in texts
    )


def test_output_retry_discards_partial_and_retries_only_failed_items():
    tokenizer = Tokenizer()
    model = Model(finish=lambda text: len(text.split()) <= 2)
    notices = []
    result = generate_translations(
        model,
        tokenizer,
        ["one two three four", "fine"],
        "cpu",
        max_new_tokens=4,
        status_cb=notices.append,
    )
    assert result == ["ONE TWO THREE FOUR", "FINE"]
    assert notices and "Retrying" in notices[0]
    assert model.calls[0][0] == ["one two three four", "fine"]
    retried = [text for texts, _ in model.calls[1:] for text in texts]
    assert "fine" not in retried
    assert all(kwargs["max_new_tokens"] == 4 for _, kwargs in model.calls)


def test_natural_eos_at_limit_is_complete_but_decoder_start_is_not():
    assert generate_translations(
        Model(),
        Tokenizer(),
        ["one"],
        "cpu",
        max_new_tokens=2,
    ) == ["ONE"]
    with pytest.raises(OutputTruncatedError, match="indivisible"):
        generate_translations(
            Model(finish=lambda text: False),
            Tokenizer(),
            ["one"],
            "cpu",
            max_new_tokens=2,
        )


def test_output_retries_are_bounded():
    tokenizer, model = Tokenizer(), Model(finish=lambda text: False)
    tokenizer.model_max_length = 512
    model.config.max_position_embeddings = 512
    notices = []
    with pytest.raises(OutputTruncatedError, match="No partial"):
        generate_translations(
            model,
            tokenizer,
            ["word " * 64],
            "cpu",
            max_new_tokens=2,
            status_cb=notices.append,
        )
    assert len(notices) == MAX_SPLIT_RETRIES
    assert len(model.calls) == MAX_SPLIT_RETRIES + 1


def test_encoded_batch_is_checked_again_before_generation():
    class InconsistentTokenizer(Tokenizer):
        def __call__(self, value, **kwargs):
            encoded = super().__call__(value, **kwargs)
            if isinstance(value, list):
                encoded["input_ids"].shape = (len(value), 25)
            return encoded

    model = Model()
    with pytest.raises(InputTooLongError, match="rather than truncating"):
        generate_translations(
            model,
            InconsistentTokenizer(),
            ["one"],
            "cpu",
        )
    assert not model.calls


@pytest.mark.parametrize(
    "backend",
    [
        "nllb",
        "nllb-large",
        "madlad-3b",
        "opus-mt",
    ],
)
def test_single_and_batch_entrypoints_share_safe_generation(
    monkeypatch,
    backend,
):
    tokenizer, model = Tokenizer(), Model()
    for name in ("_load_nllb", "_load_madlad", "_load_marian"):
        monkeypatch.setattr(engine, name, lambda *args: (tokenizer, model))
    monkeypatch.setattr(engine, "resolve_batch_size", lambda backend: 2)
    text = "one two three four\n\nété été été été"
    fn = engine.make_translate_fn("deu_Latn", "fra_Latn", backend=backend)
    assert fn(text) == text.upper()
    assert fn.many([text, "", "\n"]) == [text.upper(), "", "\n"]
    if backend.startswith("nllb"):
        assert tokenizer.src_lang == "deu_Latn"
        assert model.calls[0][1]["forced_bos_token_id"] == 7


def test_short_input_needs_only_one_budget_measurement():
    calls = []

    def measure(text):
        calls.append(text)
        return len(text)

    assert split_text("a short paragraph", measure, 100) == [
        "a short paragraph",
    ]
    assert calls == ["a short paragraph"]


def test_512_token_cap_retains_input_under_the_old_character_limit():
    tokenizer, model = Tokenizer(), Model()
    tokenizer.model_max_length = 1024
    model.config.max_position_embeddings = 1024
    source = "été " * 150
    assert len(source) < 1200
    assert len(source.encode()) + 2 > 512
    assert generate_translations(
        model,
        tokenizer,
        [source],
        "cpu",
    ) == [source.upper()]
    chunks = [text for texts, _ in model.calls for text in texts]
    assert len(chunks) > 1
    assert all(len(text.encode()) + 2 <= 512 for text in chunks)


@pytest.mark.parametrize("backend", ["nllb", "madlad-3b", "opus-mt"])
def test_legacy_backend_functions_preserve_device_and_progress(
    monkeypatch,
    backend,
):
    tokenizer, model = Tokenizer(), Model()
    for name in ("_load_nllb", "_load_madlad", "_load_marian"):
        monkeypatch.setattr(engine, name, lambda *args: (tokenizer, model))
    monkeypatch.setattr(engine, "resolve_batch_size", lambda backend: 2)
    function = {
        "nllb": engine.translate_nllb,
        "madlad-3b": engine.translate_madlad,
        "opus-mt": engine.translate_opus_mt,
    }[backend]
    progress = []
    assert (
        function(
            "one two\nthree",
            "deu_Latn",
            "fra_Latn",
            device="cpu",
            progress_cb=lambda done, total: progress.append((done, total)),
        )
        == "ONE TWO\nTHREE"
    )
    assert progress[-1][0] == progress[-1][1]


def test_generation_errors_are_not_misreported_as_token_limits():
    class BrokenModel(Model):
        def generate(self, **kwargs):
            raise RuntimeError("device unavailable")

    with pytest.raises(RuntimeError, match="device unavailable"):
        generate_translations(BrokenModel(), Tokenizer(), ["one two"], "cpu")


def test_no_eos_configuration_fails_before_generation():
    tokenizer, model = Tokenizer(), Model()
    tokenizer.eos_token_id = None
    model.generation_config.eos_token_id = None
    with pytest.raises(OutputTruncatedError, match="no end-of-sequence"):
        generate_translations(model, tokenizer, ["one two"], "cpu")
    assert not model.calls


def test_status_callback_and_errors_propagate_through_factory(monkeypatch):
    tokenizer = Tokenizer()
    model = Model(finish=lambda text: len(text.split()) == 1)
    monkeypatch.setattr(engine, "_load_nllb", lambda *args: (tokenizer, model))
    monkeypatch.setattr(engine, "resolve_batch_size", lambda backend: 2)
    notices = []
    fn = engine.make_translate_fn(
        "deu_Latn",
        "fra_Latn",
        status_cb=notices.append,
    )
    assert fn("one two") == "ONE TWO"
    assert fn.many(["three four"]) == ["THREE FOUR"]
    assert len(notices) == 2
    with pytest.raises(InputTooLongError):
        fn("indivisible" * 30)


def test_sentences_are_translated_separately_and_rejoined_exactly():
    tokenizer, model = Tokenizer(), Model()
    tokenizer.model_max_length = 512
    model.config.max_position_embeddings = 512
    source = "First one.  Second (et al. 2020) here! Third, e.g. this."
    assert generate_translations(model, tokenizer, [source], "cpu") == [
        "FIRST ONE.  SECOND (ET AL. 2020) HERE! THIRD, E.G. THIS."
    ]
    sent = sorted(text for texts, _ in model.calls for text in texts)
    assert sent == sorted(
        [
            "First one.",
            "Second (et al. 2020) here!",
            "Third, e.g. this.",
        ]
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("One. Two.", ["One. ", "Two."]),
        ("See Fig. 3 and Eq. 2. Next.", ["See Fig. 3 and Eq. 2. ", "Next."]),
        (
            "Smith et al. Found it. J. Doe agreed.",
            ["Smith et al. Found it. ", "J. Doe agreed."],
        ),
        ("Values, i.e. Means. Done.", ["Values, i.e. Means. ", "Done."]),
        (
            "e.g. lowercase follows. Then.",
            ["e.g. lowercase follows. ", "Then."],
        ),
        ("你好。世界。", ["你好。", "世界。"]),
        ("no punctuation at all", ["no punctuation at all"]),
    ],
)
def test_sentence_slices_skip_abbreviations_and_are_lossless(text, expected):
    from textlab.features.translation.chunking import sentence_slices

    assert sentence_slices(text) == expected
    assert "".join(sentence_slices(text)) == text


def test_repeated_sentences_are_translated_once_and_cache_is_reused():
    tokenizer, model = Tokenizer(), Model()
    tokenizer.model_max_length = 512
    model.config.max_position_embeddings = 512
    cache = {}
    texts = ["Header text. Body one.", "Header  text. Body two."]
    assert generate_translations(
        model,
        tokenizer,
        texts,
        "cpu",
        cache=cache,
    ) == ["HEADER TEXT. BODY ONE.", "HEADER TEXT. BODY TWO."]
    sent = [text for texts, _ in model.calls for text in texts]
    assert sorted(sent) == ["Body one.", "Body two.", "Header text."]
    model.calls.clear()
    assert generate_translations(
        model,
        tokenizer,
        ["Body two. Header text."],
        "cpu",
        cache=cache,
    ) == ["BODY TWO. HEADER TEXT."]
    assert not model.calls


def test_short_sentences_share_one_length_sorted_batch():
    tokenizer, model = Tokenizer(), Model()
    words = ["a", "bbbb", "cc", "ddd"]
    assert generate_translations(
        model,
        tokenizer,
        words,
        "cpu",
        batch_size=2,
    ) == [word.upper() for word in words]
    assert [texts for texts, _ in model.calls] == [["bbbb", "ddd", "cc", "a"]]
