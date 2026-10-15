"""Checking, showing and exporting the citations in a summary."""

import pytest

from textlab.features.meeting_notes import citations
from textlab.features.meeting_notes.models import Segment

SEGMENTS = [
    Segment(i, f"line {i}", "Anna", start=float(60 * i)) for i in range(1, 21)
]


@pytest.mark.parametrize(
    "written,expected",
    [
        ("[3]", "[3]"),
        ("[3, 5]", "[3, 5]"),
        ("[5;3]", "[3, 5]"),
        ("[#3, #5]", "[3, 5]"),
        ("[Line 3]", "[3]"),
        ("[lines 3-5]", "[3, 4, 5]"),
        ("[5–3]", "[3, 4, 5]"),
        ("[1-20]", "[1, 20]"),  # wide ranges keep their ends
    ],
)
def test_citation_forms_are_normalized(written, expected):
    cited = citations.cite_summary(f"- point {written}", SEGMENTS)
    assert cited.text == f"- point {expected}"
    assert cited.invalid_citations == 0


def test_invalid_numbers_are_removed_and_counted():
    cited = citations.cite_summary("- a [3, 99].\n- b [42].", SEGMENTS)
    assert cited.text == "- a [3].\n- b."
    assert cited.invalid_citations == 2
    assert [s.id for s in cited.sources] == [3]


def test_sources_are_unique_and_in_transcript_order():
    cited = citations.cite_summary("- a [7]\n- b [2, 7]", SEGMENTS)
    assert [s.id for s in cited.sources] == [2, 7]


def test_links_and_checkboxes_are_not_citations():
    text = "- see [3](https://example.org)\n- [ ] todo [4]"
    cited = citations.cite_summary(text, SEGMENTS)
    assert cited.text == text
    assert [s.id for s in cited.sources] == [4]


def test_uncited_points_are_counted_and_marked():
    summary = (
        "**Decisions**\n"
        "- Ship on Friday [2]\n"
        "- Topic with cited sub-points:\n"
        "  - detail [3]\n"
        "- Nothing cited here\n"
        "1. Numbered, uncited\n"
        "Plain paragraph without a citation."
    )
    cited = citations.cite_summary(summary, SEGMENTS)
    assert cited.uncited_points == 2
    marked = citations.mark_uncited_points(cited.text, "(?)")
    assert "- Nothing cited here (?)" in marked
    assert "1. Numbered, uncited (?)" in marked
    assert "- Ship on Friday [2]\n" in marked
    assert "sub-points:\n" in marked


def test_format_citations_renders_each_marker():
    cited = citations.cite_summary("- a [2, 3]\n- b [5]", SEGMENTS)
    rendered = citations.format_citations(
        cited.text,
        cited.sources,
        lambda sources: "(" + ",".join(str(s.start) for s in sources) + ")",
    )
    assert rendered == "- a (120.0,180.0)\n- b (300.0)"


def test_footnotes_quote_the_cited_lines():
    cited = citations.cite_summary("- a [2]\n- b [2, 3]", SEGMENTS)
    assert citations.with_footnotes(cited) == (
        "- a [^2]\n- b [^2][^3]\n\n"
        "[^2]: [02:00] Anna: line 2\n"
        "[^3]: [03:00] Anna: line 3\n"
    )


def test_summary_without_citations_is_unchanged():
    cited = citations.cite_summary("- a\n- b", SEGMENTS)
    assert cited.sources == ()
    assert cited.uncited_points == 2
    assert citations.with_footnotes(cited) == "- a\n- b"
