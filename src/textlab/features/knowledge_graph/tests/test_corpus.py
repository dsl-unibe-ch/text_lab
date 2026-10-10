"""The corpus folder: parsing papers, skipping known ones, the tables."""

import json

import pandas as pd
import pytest

from textlab.features.knowledge_graph import service
from textlab.features.knowledge_graph.corpus import table_row


@pytest.fixture
def papers(tmp_path):
    folder = tmp_path / "papers"
    folder.mkdir()
    for name in ("b.pdf", "a.pdf", "notes.txt"):
        (folder / name).write_bytes(b"%PDF")
    return folder


def parser(tei, fail=()):
    """A fake Grobid that records the PDFs it parses."""
    parsed = []

    def parse(pdf):
        parsed.append(pdf.name)
        if pdf.name in fail:
            raise RuntimeError("Grobid returned status 500")
        return tei

    parse.parsed = parsed
    return parse


def test_pdfs_and_corpus_names(papers):
    assert [p.name for p in service.list_pdfs(papers)] == ["a.pdf", "b.pdf"]
    assert service.corpus_folder_name(papers) == "papers_project_corpus"


def test_a_corpus_is_built_from_the_pdfs(papers, tei):
    corpus = papers.parent / "papers_project_corpus"
    updates = []
    summary = service.build_corpus(
        service.list_pdfs(papers),
        corpus,
        on_progress=updates.append,
        parse=parser(tei),
    )
    assert (summary.processed, summary.skipped, summary.errors) == (2, 0, 0)
    assert list(summary.table["paper_id"]) == ["P0001", "P0002"]
    assert list(summary.table["pdf_filename"]) == ["a.pdf", "b.pdf"]
    for name in ("fulltext.tei.xml", "fulltext.txt", "metadata.json"):
        assert (corpus / "P0001" / name).exists()
    for name in ("corpus_metadata.json", "corpus_table.csv"):
        assert (corpus / name).exists()
    assert len(service.read_records(corpus / service.TABLE_JSONL)) == 2
    assert updates[0].message == "Processing 1/2: a.pdf"
    assert updates[-1].message == "Building corpus metadata table..."
    assert updates[-1].fraction == 1.0


def test_known_papers_are_skipped_and_new_ones_numbered_on(papers, tei):
    corpus = papers.parent / "papers_project_corpus"
    service.build_corpus(service.list_pdfs(papers), corpus, parse=parser(tei))
    # A new paper that sorts first must not take over P0001.
    (papers / "0-new.pdf").write_bytes(b"%PDF")
    parse = parser(tei)
    summary = service.build_corpus(
        service.list_pdfs(papers), corpus, parse=parse
    )
    assert parse.parsed == ["0-new.pdf"]
    assert (summary.processed, summary.skipped) == (1, 2)
    paper_ids = dict(
        zip(
            summary.table["pdf_filename"],
            summary.table["paper_id"],
            strict=True,
        )
    )
    assert paper_ids == {
        "a.pdf": "P0001",
        "b.pdf": "P0002",
        "0-new.pdf": "P0003",
    }


def test_a_failed_paper_is_reported_and_retried_in_its_folder(papers, tei):
    corpus = papers.parent / "papers_project_corpus"
    pdfs = service.list_pdfs(papers)
    summary = service.build_corpus(
        pdfs, corpus, parse=parser(tei, fail={"a.pdf"})
    )
    assert (summary.processed, summary.errors) == (1, 1)
    error = (corpus / "P0001" / "error.txt").read_text(encoding="utf-8")
    assert error.startswith("Error processing a.pdf:\n")
    assert "Grobid returned status 500" in error
    assert list(summary.table["paper_id"]) == ["P0002"]

    parse = parser(tei)
    summary = service.build_corpus(pdfs, corpus, parse=parse)
    assert parse.parsed == ["a.pdf"]
    assert (corpus / "P0001" / "metadata.json").exists()
    assert not (corpus / "P0001" / "error.txt").exists()
    assert not (corpus / "P0003").exists()


def test_corpora_are_found_by_their_files(tmp_path):
    for name in ("x_project_corpus", "y_project_corpus", "other"):
        (tmp_path / name).mkdir()
    (tmp_path / "y_project_corpus" / service.TOPICS_JSONL).write_text("")
    assert [c.name for c in service.find_corpora(tmp_path)] == [
        "x_project_corpus",
        "y_project_corpus",
    ]
    found = service.find_corpora(tmp_path, required=service.TOPICS_JSONL)
    assert [c.name for c in found] == ["y_project_corpus"]
    assert service.find_corpora(tmp_path / "missing") == []


def test_an_empty_corpus_has_an_empty_table(tmp_path):
    table = service.build_corpus_table(tmp_path)
    assert isinstance(table, pd.DataFrame) and table.empty
    assert json.loads((tmp_path / "corpus_metadata.json").read_text()) == []


def test_a_table_row_joins_lists():
    row = table_row(
        {
            "paper_id": "P0001",
            "authors": [{"full_name": "Ada Lovelace"}, {"full_name": "B"}],
            "keywords": ["x", "y"],
            "doi": "10.1/a",
            "year": 2020,
            "filename": "/p/a.pdf",
            "citations": [
                {"title": " T1 ", "doi": "10.2/b", "authors": ["C"]},
                {"title": "", "authors": "not a list"},
            ],
        }
    )
    assert row["authors"] == "Ada Lovelace; B" and row["n_authors"] == 2
    assert row["keywords"] == "x; y"
    assert row["DOI"] == "10.1/a"
    assert row["publication_date"] == "2020"
    assert row["pdf_filename"] == "a.pdf"
    assert row["n_citations"] == 2
    assert row["cited_titles"] == "T1"
    assert row["cited_dois"] == "10.2/b"
    assert row["cited_authors"] == "C"


def test_the_topics_download_is_one_json_list(tmp_path):
    lines = [{"paper_id": "P0001"}, {"paper_id": "P0002"}]
    (tmp_path / service.TOPICS_JSONL).write_text(
        "".join(json.dumps(line) + "\n" for line in lines)
    )
    assert json.loads(service.topics_json(tmp_path)) == lines
