"""Offline contracts for lossless, fail-closed translation protection."""

import importlib
import re

import pytest

from textlab.features.translation.shield import (
    ProtectedContentError,
    shield,
    shielded_translate,
    shielded_translate_many,
    unshield,
)

MARKER = re.compile(r"\[(?:TL|GL)_[0-9A-F]{16}_[0-9]+\]")


def identity(text):
    return text


class BatchTranslator:
    def __init__(self, transform):
        self.transform = transform
        self.calls = []

    def __call__(self, text):
        raise AssertionError("The batch method should be used.")

    def many(self, texts):
        self.calls.append(texts)
        return self.transform(texts)


@pytest.mark.parametrize(
    "source",
    [
        "",
        " ",
        "\n\t",
        "ordinary prose",
        '`[label](https://example.test/private "title")`',
        "``a `tick` [link](url)``",
        "```python\nx = '[link](url)'\ny = '$math$'\n```",
        "````md\n```\n[link](url)\n```\n````",
        "~~~md\n```\n![alt](image.png)\n~~~",
        "```text\n[unclosed fence](url)",
        "$$[label](url) + x$$ and \\[a+b\\] and $x+y$ and \\(z\\)",
        '[label](https://example.test/a(b) "verbatim title")',
        "[label](<https://example.test/a b> 'title (with parentheses)')",
        r"[a [nested] label](folder/file\(1\).pdf (a title))",
        "[it's fine](https://example.test/it's-fine)",
        '[![alt [nested]](image.png "inner")](page.html "outer")',
        '[label `code [x]`](page.html "title ) ]")',
        '<a href="https://example.test/Bern" title="a > b">label</a>',
        "https://example.test ftp://example.test www.example.test",
        "/home/user/file.txt first.last+tag@example.test {user.name}",
        "%(name)s %5.2f %s %x",
    ],
)
def test_identity_roundtrip_is_lossless(source):
    masked, table = shield(source)
    assert isinstance(table, list)
    assert unshield(masked, table) == source
    assert shielded_translate(source, identity) == source
    assert all(marker.isascii() for marker in MARKER.findall(masked))
    assert all(
        not any(char.isspace() for char in marker)
        for marker in MARKER.findall(masked)
    )
    assert "\x02" not in masked and "\x03" not in masked


@pytest.mark.parametrize(
    "kind,source,glossary",
    [
        ("TL", "before `sensitive code` after", None),
        ("GL", "before sensitive after", {"sensitive": "EXACT target"}),
    ],
)
@pytest.mark.parametrize(
    "corrupt",
    [
        lambda marker: "",
        lambda marker: marker + marker,
        lambda marker: marker.lower(),
        lambda marker: marker[:-1],
        lambda marker: marker.replace("_", " ", 1),
        lambda marker: marker.replace("_0]", "_999]"),
        lambda marker: marker.replace("_0]", "_00]"),
        lambda marker: marker + "[TL_BAD]",
    ],
)
def test_corrupted_structure_and_glossary_markers_block_output(
    kind, source, glossary, corrupt
):
    def translate(masked):
        marker = next(
            item
            for item in MARKER.findall(masked)
            if item.startswith(f"[{kind}_")
        )
        return masked.replace(marker, corrupt(marker))

    with pytest.raises(ProtectedContentError) as error:
        shielded_translate(source, translate, glossary, fallback=False)
    assert isinstance(error.value, ValueError)
    assert "sensitive" not in str(error.value)
    assert "EXACT target" not in str(error.value)


@pytest.mark.parametrize(
    "generated",
    [
        "[TL_0123456789ABCDEF_0]",
        "[GL_0123456789ABCDEF_123]",
        "[TL_broken]",
        "[tl_0]",
        "[GL_",
        "\x02TL_0\x03",
    ],
)
def test_generated_markers_rejected_even_without_expected_protection(
    generated,
):
    masked, table = shield("ordinary prose")
    assert not table
    with pytest.raises(ProtectedContentError):
        unshield(masked + generated, table)
    with pytest.raises(ProtectedContentError):
        shielded_translate("ordinary prose", lambda _: generated, {})


def test_errors_distinguish_missing_duplicate_unknown_and_malformed():
    masked, table = shield("`private`")
    for output, message in (
        ("", "Missing"),
        (masked * 2, "Duplicate"),
        (masked.replace("_0]", "_42]"), "Unknown"),
        (masked.lower(), "Malformed"),
    ):
        with pytest.raises(ProtectedContentError, match=message):
            unshield(output, table)


@pytest.mark.parametrize(
    "source",
    [
        "[TL_0] [GL_0] \x02TL_0\x03 \x02GL_0\x03",
        "[TL_0123456789ABCDEF_0] [GL_0123456789ABCDEF_1]",
        "[tl_0] [GL_bad] [TL_unclosed\nordinary prose",
        "`[TL_0]` [label](https://example.test/[GL_0])",
    ],
)
def test_source_literal_markers_are_never_interpreted(source):
    masked, table = shield(source)
    assert unshield(masked, table) == source
    assert unshield(masked.upper(), table) == source.replace(
        "ordinary prose", "ORDINARY PROSE"
    ).replace("label", "LABEL")


def test_namespace_avoids_literal_source_collision(monkeypatch):
    module = importlib.import_module("textlab.features.translation.shield")
    values = iter(("0123456789abcdef", "fedcba9876543210"))
    monkeypatch.setattr(module, "token_hex", lambda _: next(values))
    source = "[TL_0123456789ABCDEF_0] `code`"
    masked, table = shield(source)
    assert "0123456789ABCDEF" not in masked
    assert unshield(masked, table) == source


def test_glossary_exact_count_case_and_longest_match():
    source = "University of Bern; BERN Bern berner sun sunday 无空格"
    glossary = {
        "Bern": "Berne",
        "University of Bern": "Université de Berne",
        "sun": "Soleil",
        "空格": "SPACE",
        "": "bad",
        " ": "bad",
    }
    assert shielded_translate(source, str.upper, glossary) == (
        "Université de Berne; Berne Berne BERNER Soleil SUNDAY 无SPACE"
    )
    assert (
        shielded_translate(
            source,
            identity,
            glossary,
            glossary_case_sensitive=True,
        )
        == "Université de Berne; BERN Berne berner Soleil sunday 无SPACE"
    )


def test_longest_glossary_term_wins_even_if_overlap_starts_later():
    assert (
        shielded_translate(
            "New York University",
            identity,
            {"New York": "NY", "York University": "YU"},
        )
        == "New YU"
    )


def test_glossary_targets_and_marker_like_keys_are_not_rematched():
    glossary = {
        "Bern": "[GL_0]",
        "GL": "bad",
        "TL": "also bad",
        "delete": "",
        "[": "opening",
    }
    assert shielded_translate("Bern Bern delete", identity, glossary) == (
        "[GL_0] [GL_0] "
    )


def test_every_occurrence_has_its_own_marker():
    seen = []

    def translate(masked):
        seen.extend(MARKER.findall(masked))
        return masked

    assert shielded_translate(
        "Bern Bern `Bern` `Bern`", translate, {"Bern": "Berne"}
    ) == ("Berne Berne `Bern` `Bern`")
    assert len(seen) == len(set(seen)) == 4


@pytest.mark.parametrize(
    "protected",
    [
        "`Bern`",
        "``Bern `Bern` ``",
        "```text\nBern\n```",
        "$$Bern$$",
        "$Bern$",
        r"\[Bern\]",
        r"\(Bern\)",
        "https://example.test/Bern",
        "/home/Bern/file",
        "Bern@example.test",
        '<a href="Bern" title="Bern > Bern">',
        '![Bern](Bern "Bern")',
        "{Bern}",
        "%(Bern)s",
    ],
)
def test_glossary_never_replaces_inside_protected_spans(protected):
    source = protected + "\nBern"
    assert shielded_translate(source, identity, {"Bern": "Berne"}) == (
        protected + "\nBerne"
    )


@pytest.mark.parametrize(
    "comment",
    [
        "<!-- page 1 -->",
        "<!-- Bern -->",
        "<!--\nBern [label](url) `code` $$math$$ <b>html</b>\n-->",
        "<!-- [TL_0] [GL_0123456789ABCDEF_0] -->",
        "<!-- unclosed Bern\ncomment",
    ],
)
def test_html_comments_are_shielded_whole_and_never_translated(comment):
    seen = []

    def translate(masked):
        seen.append(masked)
        return masked.upper()

    assert (
        shielded_translate(
            "before\n" + comment,
            translate,
            {"Bern": "Berne"},
        )
        == "BEFORE\n" + comment
    )
    assert len(MARKER.findall(seen[0])) == 1
    assert len(MARKER.findall(seen[0])[0]) == 23
    assert unshield(*shield(comment)) == comment


def test_html_comments_keep_link_label_brackets_opaque():
    source = '[Bern <!-- ] [ hidden -->](url "title")'
    assert shielded_translate(source, str.upper, {"Bern": "Berne"}) == (
        '[Berne <!-- ] [ hidden -->](url "title")'
    )


def test_adjacent_comments_do_not_hide_intervening_translatable_text():
    source = "<!-- page 1 --> Bern <!-- page 2 --> Bern"
    assert shielded_translate(source, str.upper, {"Bern": "Berne"}) == (
        "<!-- page 1 --> Berne <!-- page 2 --> Berne"
    )


def test_links_keep_url_title_image_alt_and_translate_visible_labels():
    source = (
        '[Bern `Bern` ![Bern](Bern "Bern")](Bern "Bern") '
        '<a title="Bern">Bern</a>'
    )
    assert shielded_translate(source, str.upper, {"Bern": "Berne"}) == (
        '[Berne `Bern` ![Bern](Bern "Bern")](Bern "Bern") '
        '<a title="Bern">Berne</a>'
    )


@pytest.mark.parametrize("texts", [[], [""], ["", " \t", "\n"]])
def test_empty_inputs_never_call_translator(texts):
    def fail(_):
        raise AssertionError("Empty inputs should not be translated.")

    assert shielded_translate_many(texts, fail) == texts
    for text in texts:
        assert shielded_translate(text, fail) == text


def test_batch_and_single_fallback_preserve_order_and_empty_layout():
    texts = ["", "Bern `x`", "\n", "bern {name}"]
    expected = ["", "Berne `x`", "\n", "Berne {name}"]
    translator = BatchTranslator(lambda values: [v.upper() for v in values])
    assert shielded_translate_many(texts, translator, {"Bern": "Berne"}) == (
        expected
    )
    assert len(translator.calls) == 1
    assert len(translator.calls[0]) == 2
    assert shielded_translate_many(texts, str.upper, {"Bern": "Berne"}) == (
        expected
    )
    assert shielded_translate_many(
        ["BERN Bern"],
        identity,
        {"Bern": "Berne"},
        glossary_case_sensitive=True,
    ) == ["BERN Berne"]


@pytest.mark.parametrize("output", [[], ["one"], ["one", "two", "three"]])
def test_batch_cardinality_mismatch_blocks_all_unaligned_outputs(output):
    translator = BatchTranslator(lambda _: output)
    with pytest.raises(ProtectedContentError, match="count mismatch") as err:
        shielded_translate_many(["", "first", "second"], translator)
    assert err.value.partial_results == ["", None, None]
    assert err.value.failed_indices == (1, 2)


@pytest.mark.parametrize("output", ["one", None, 42])
def test_invalid_batch_container_blocks_output(output):
    with pytest.raises(ProtectedContentError):
        shielded_translate_many(["source"], BatchTranslator(lambda _: output))


@pytest.mark.parametrize(
    "source,glossary",
    [
        (["`one`", "`two`"], None),
        (["Bern", "Bern"], {"Bern": "Berne"}),
    ],
)
def test_cross_unit_swaps_are_detected(source, glossary):
    translator = BatchTranslator(lambda values: values[::-1])
    with pytest.raises(ProtectedContentError, match="Unknown") as error:
        shielded_translate_many(source, translator, glossary, fallback=False)
    assert error.value.partial_results == [None, None]
    assert error.value.failed_indices == (0, 1)


def test_batch_retains_only_independently_verified_outputs():
    def translate(values):
        return [values[0].upper(), "corrupted", values[2].upper()]

    translator = BatchTranslator(translate)
    with pytest.raises(ProtectedContentError, match="Missing") as error:
        shielded_translate_many(
            ["", "first `x`", "private `y`", "last `z`"],
            translator,
            fallback=False,
        )
    assert error.value.partial_results == ["", "FIRST `x`", None, "LAST `z`"]
    assert error.value.failed_indices == (2,)
    assert "private" not in str(error.value)


@pytest.mark.parametrize("output", [None, 42, b"bytes"])
def test_non_text_unit_fails_clearly_and_keeps_verified_peers(output):
    with pytest.raises(ProtectedContentError, match="must be text"):
        shielded_translate("source", lambda _: output)
    translator = BatchTranslator(lambda _: [output, "SECOND"])
    with pytest.raises(ProtectedContentError) as error:
        shielded_translate_many(["first", "second"], translator)
    assert error.value.partial_results == [None, "SECOND"]


def reverse_markers(text, kind):
    prefix = f"[{kind}_"
    markers = iter(
        reversed(
            [
                marker
                for marker in MARKER.findall(text)
                if marker.startswith(prefix)
            ]
        )
    )
    return MARKER.sub(
        lambda match: next(markers)
        if match.group().startswith(prefix)
        else match.group(),
        text,
    )


@pytest.mark.parametrize(
    "source",
    [
        '[private](https://example.test "title")',
        '<b title="private">label</b>',
        "<b><i>private</i></b>",
        "`private` and `other`",
    ],
)
def test_reordered_structural_markers_are_rejected(source):
    masked, table = shield(source)
    with pytest.raises(ProtectedContentError, match="order") as error:
        unshield(reverse_markers(masked, "TL"), table)
    assert "private" not in str(error.value)
    with pytest.raises(ProtectedContentError, match="order"):
        shielded_translate(
            source, lambda text: reverse_markers(text, "TL"), fallback=False
        )


@pytest.mark.parametrize(
    "extra,message",
    [
        ("[TL_0123456789ABCDEF_42]", "Unknown"),
        ("[TL_malformed]", "Malformed"),
    ],
)
def test_order_check_does_not_accept_unknown_or_malformed_markers(
    extra, message
):
    masked, table = shield("[private](url)")
    with pytest.raises(ProtectedContentError, match=message):
        unshield(reverse_markers(masked, "TL") + extra, table)


@pytest.mark.parametrize(
    "source,expected",
    [
        ("Bern Zürich", "Z B"),
        ("[Bern Zürich](url)", "[Z B](url)"),
        ("[Bern](url) Zürich", "[Z](url) B"),
    ],
)
def test_glossary_markers_may_move_for_target_grammar(source, expected):
    assert (
        shielded_translate(
            source,
            lambda text: reverse_markers(text, "GL"),
            {"Bern": "B", "Zürich": "Z"},
        )
        == expected
    )


@pytest.mark.parametrize("blank", ["", " \t\n", "\u2003"])
def test_blank_output_cannot_erase_nonempty_source(blank):
    with pytest.raises(ProtectedContentError, match="Empty") as error:
        shielded_translate("private prose", lambda _: blank)
    assert "private" not in str(error.value)
    assert unshield(blank, []) == blank


@pytest.mark.parametrize("blank", ["", " \t\n", "\u2003"])
@pytest.mark.parametrize("batched", [True, False])
def test_blank_batch_outputs_keep_only_verified_peers(blank, batched):
    def translate(text):
        return blank if text == "private" else text.upper()

    translator = (
        BatchTranslator(lambda texts: [translate(text) for text in texts])
        if batched
        else translate
    )
    with pytest.raises(ProtectedContentError, match="Empty") as error:
        shielded_translate_many(["", "first", "private", "last"], translator)
    assert error.value.partial_results == ["", "FIRST", None, "LAST"]
    assert error.value.failed_indices == (2,)


def test_glossary_cannot_make_the_entire_restored_output_empty():
    with pytest.raises(ProtectedContentError, match="Empty"):
        shielded_translate("private", identity, {"private": ""})


def test_batch_structural_order_failures_keep_verified_outputs():
    translator = BatchTranslator(
        lambda texts: [
            texts[0].upper(),
            reverse_markers(texts[1], "TL"),
            texts[2].upper(),
        ]
    )
    with pytest.raises(ProtectedContentError, match="order") as error:
        shielded_translate_many(
            ["first `x`", "[private](url)", "last `z`"],
            translator,
            fallback=False,
        )
    assert error.value.partial_results == ["FIRST `x`", None, "LAST `z`"]
    assert error.value.failed_indices == (1,)


def test_legacy_positional_tables_remain_supported_and_validated():
    assert unshield("\x02TL_0\x03 [TL_1]", ["zero", "one"]) == "zero one"
    with pytest.raises(ProtectedContentError, match="order"):
        unshield("[TL_1] [TL_0]", ["zero", "one"])
    assert unshield("[TL_0]", ["[TL_1]"]) == "[TL_1]"
    for output in (
        "",
        "[TL_1]",
        "[TL_0][TL_0]",
        "[TL_bad]",
        "[TL_" + "9" * 5000 + "]",
    ):
        with pytest.raises(ProtectedContentError):
            unshield(output, ["private"])


def test_corrupted_markers_are_retranslated_between_protected_spans():
    calls = []

    def translate(text):
        calls.append(text)
        return MARKER.sub("", text).upper()  # model drops every marker

    assert (
        shielded_translate(
            "see `code` and [the docs](https://example.test)",
            translate,
        )
        == "SEE `code` AND [THE DOCS](https://example.test)"
    )
    # The retry never shows protected content to the model.
    assert all(
        "code" not in call and "example" not in call for call in calls[1:]
    )


def test_marker_dense_html_table_never_sends_markers_to_the_model():
    table = (
        "<table>"
        + "".join(
            f"<tr><td>cell {i}</td><td>{i}.5</td></tr>" for i in range(40)
        )
        + "</table>"
    )
    seen = []

    def many(texts):
        seen.extend(texts)
        return [text.upper() for text in texts]

    translator = BatchTranslator(many)
    assert shielded_translate_many(["intro", table], translator) == [
        "INTRO",
        table.replace("cell", "CELL"),
    ]
    assert not any(MARKER.search(text) for text in seen)
    assert "0.5" not in seen  # number-only cells are not translated


def test_glossary_targets_survive_segment_fallback():
    def translate(text):
        return MARKER.sub("", text).replace("city", "Stadt")

    assert (
        shielded_translate(
            "the city Bern and `x`",
            translate,
            {"Bern": "Berne"},
        )
        == "the Stadt Berne and `x`"
    )


def test_input_budget_error_from_marker_runs_falls_back_to_segments():
    from textlab.features.translation.chunking import InputTooLongError

    def many(texts):
        if any(MARKER.search(text) for text in texts):
            raise InputTooLongError("marker run too long")
        return [text.upper() for text in texts]

    assert shielded_translate_many(
        ["plain", "with `code`"],
        BatchTranslator(many),
    ) == ["PLAIN", "WITH `code`"]
