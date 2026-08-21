"""Sentence context must survive a hard-wrapped source.

A source that wraps a sentence across lines used to be translated one line at
a time: the model got half a clause, with no subject and no verb, and returned
a translation to match.
"""

import conftest_path  # noqa: F401

from core.translation.format import reflow_soft_wraps, translate_markdown

WRAPPED = (
    "Die Kommission hat beschlossen, dass die neuen\n"
    "Vorschriften erst im kommenden Jahr in Kraft\n"
    "treten sollen."
)


def test_a_sentence_split_over_lines_is_rejoined():
    assert reflow_soft_wraps(WRAPPED) == (
        "Die Kommission hat beschlossen, dass die neuen Vorschriften "
        "erst im kommenden Jahr in Kraft treten sollen."
    )


def test_a_break_after_a_finished_sentence_is_kept():
    """The conservative half of the rule: only an unfinished line is joined,
    so deliberately line-structured prose keeps its shape."""
    text = "Das Parlament stimmte zu.\nDie Sitzung wurde geschlossen."
    assert reflow_soft_wraps(text) == text

    # Blank lines separate paragraphs and are never crossed.
    two = f"{WRAPPED}\n\nEin zweiter Absatz."
    assert reflow_soft_wraps(two).count("\n\n") == 1


def test_a_word_hyphenated_across_the_break_is_put_back_together():
    assert reflow_soft_wraps("Ein Wort wurde ge-\ntrennt.") == "Ein Wort wurde getrennt."


def test_markdown_structure_keeps_its_own_lines():
    """None of these lines end in a full stop, so only the structural check
    stops them being swallowed into one paragraph."""
    for text in (
        "# Ein Titel\nEin Absatz",
        "- Ein Punkt\n- Noch einer",
        "> Ein Zitat\n> Noch eins",
        "| A | B |\n|---|---|",
        "Ein Absatz\n\n---\n\nNoch einer",
        "Ein Absatz\n    code_line(1)",
        "Zeile mit hartem Umbruch  \nNaechste Zeile",  # two trailing spaces
    ):
        assert reflow_soft_wraps(text) == text, text


def test_fenced_code_is_never_reflowed():
    """Nothing inside a fence ends in a full stop, so without fence tracking
    every line of code would be joined into one."""
    text = "Ein Absatz\n\n```python\nx = 1\ny = 2\n```"
    assert reflow_soft_wraps(text) == text


def test_reflow_is_idempotent_and_safe_on_edge_cases():
    assert reflow_soft_wraps(reflow_soft_wraps(WRAPPED)) == reflow_soft_wraps(WRAPPED)
    for text in ("", "no newlines here", "\n", "\n\n\n"):
        assert reflow_soft_wraps(text) == text


def test_translate_markdown_sends_whole_sentences_to_the_model():
    """The bug as reported: what actually reaches the translation model."""
    seen = []

    def fake_translate(text):
        seen.append(text)
        return text.upper()

    translate_markdown(f"{WRAPPED}\n\n- Ein Punkt", fake_translate)

    assert seen == [
        "Die Kommission hat beschlossen, dass die neuen Vorschriften "
        "erst im kommenden Jahr in Kraft treten sollen.",
        "Ein Punkt",
    ]


def test_subtitles_would_be_destroyed_which_is_why_callers_opt_in():
    """Documents the contract the .srt/.vtt dispatch relies on.

    A cue's index, timestamp and text have no sentence-ending punctuation
    between them, so reflowing collapses the whole cue into one line. Nothing
    in the helper can tell that apart from prose -- which is exactly why it is
    applied by the caller that knows the format, and not inside the engine.
    """
    cue = "1\n00:00:01,000 --> 00:00:04,000\nHello there, this is\na wrapped sentence."
    assert "\n" not in reflow_soft_wraps(cue)
