"""Reading Grobid's TEI XML."""

from textlab.features.knowledge_graph.tei import (
    extract_metadata_fields,
    extract_plain_text_from_tei,
)


def test_the_metadata_of_a_paper(tei):
    meta = extract_metadata_fields(tei, "/papers/a.pdf", "P0001")
    assert meta["paper_id"] == "P0001"
    assert meta["pdf_path"] == "/papers/a.pdf"
    assert meta["title"] == "Graphs of Papers"
    # Authors without a first name are left out of the paper's authors.
    assert meta["authors"] == ["Ada Lovelace"]
    assert meta["n_authors"] == 1
    assert meta["abstract"] == "We draw graphs."
    assert meta["keywords"] == ["graphs", "papers"]
    assert meta["DOI"] == "10.1/ABC"
    assert meta["journal"] == "Journal of Tests"
    assert meta["publication_date"] == "2024-05-01"


def test_the_references_of_a_paper(tei):
    meta = extract_metadata_fields(tei, "a.pdf", "P0001")
    first, second = meta["citations"]
    assert first == {
        "citation_id": "b0",
        "title": "Cited Work",
        "authors": ["Alan M Turing", "Hopper"],
        "DOI": "10.2/XYZ",
        "year": "1950",
    }
    # Without an article title, the title of the book or journal is used.
    assert second["title"] == "A Book"
    assert second["authors"] == [] and second["year"] == ""
    assert meta["n_citations"] == 2
    assert meta["cited_dois"] == ["10.2/XYZ"]
    assert sorted(meta["cited_authors"]) == ["Alan M Turing", "Hopper"]


def test_the_plain_text_leaves_out_references_and_figures(tei):
    text = extract_plain_text_from_tei(tei)
    assert text == "First paragraph cites.\n\nSecond paragraph."


def test_a_document_without_body_has_no_text():
    tei = '<TEI xmlns="http://www.tei-c.org/ns/1.0"><text/></TEI>'
    assert extract_plain_text_from_tei(tei) == ""
