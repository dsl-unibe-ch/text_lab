"""A corpus: the papers of a PDF folder, parsed, in a folder of its own.

A corpus folder (``<papers folder>_project_corpus``, in a location the user
chooses) holds one ``P0001``, ``P0002``, ... folder per paper with
``fulltext.tei.xml``, ``fulltext.txt`` and ``metadata.json`` (or
``error.txt`` if Grobid failed), and the tables built from them:
``corpus_metadata.json``, ``corpus_table.csv``, ``corpus_table.jsonl`` and,
after topic extraction, ``corpus_table.with_topics.jsonl``.
"""

from __future__ import annotations

import json
import logging
import re
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from textlab.common.progress import Progress, ProgressCallback, no_progress
from textlab.features.knowledge_graph import grobid
from textlab.features.knowledge_graph.models import CorpusSummary
from textlab.features.knowledge_graph.tei import (
    extract_metadata_fields,
    extract_plain_text_from_tei,
)

LOGGER = logging.getLogger(__name__)

CORPUS_SUFFIX = "_project_corpus"
METADATA_FILE = "metadata.json"
ERROR_FILE = "error.txt"
CORPUS_METADATA_FILE = "corpus_metadata.json"
TABLE_CSV = "corpus_table.csv"
TABLE_JSONL = "corpus_table.jsonl"
TOPICS_JSONL = "corpus_table.with_topics.jsonl"

_PAPER_FOLDER = re.compile(r"P(\d+)")
_ERROR_HEADER = re.compile(r"Error processing (.+):\n")


def list_pdfs(folder: str | Path) -> list[Path]:
    """Return the PDFs directly in a folder, sorted by name.

    Args:
        folder: The papers folder.

    Returns:
        The PDF paths.
    """
    return sorted(Path(folder).glob("*.pdf"))


def corpus_folder_name(papers_folder: str | Path) -> str:
    """Return the name of the corpus folder for a papers folder.

    Args:
        papers_folder: The folder with the PDFs.

    Returns:
        ``"<folder name>_project_corpus"``.
    """
    return f"{Path(papers_folder).name}{CORPUS_SUFFIX}"


def find_corpora(
    location: str | Path, required: str | None = None
) -> list[Path]:
    """Return the corpus folders in a location, sorted by name.

    Args:
        location: A folder that contains ``*_project_corpus`` folders.
        required: A file name the corpus must contain, e.g.
            :data:`TABLE_JSONL`; any corpus folder if ``None``.

    Returns:
        The corpus folders; empty if the location is not a folder.
    """
    location = Path(location)
    if not location.is_dir():
        return []
    return [
        folder
        for folder in sorted(location.glob(f"*{CORPUS_SUFFIX}"))
        if required is None or (folder / required).exists()
    ]


def build_corpus(
    pdfs: list[Path],
    corpus: str | Path,
    on_progress: ProgressCallback = no_progress,
    parse: Callable[[Path], str] = grobid.process_pdf,
) -> CorpusSummary:
    """Parse new papers into a corpus folder and rebuild its tables.

    Papers already in the corpus (matched by PDF file name) are skipped, so
    the corpus can be extended by adding PDFs to the folder; a paper that
    failed before is tried again in its old folder. New papers get the
    next free ``P`` number. A paper Grobid cannot parse gets ``error.txt``
    with the traceback instead of ``metadata.json``.

    Args:
        pdfs: The papers.
        corpus: The corpus folder, created if needed.
        on_progress: Receives a :class:`Progress` update per paper.
        parse: Returns the TEI XML of a PDF (Grobid; replaced in tests).

    Returns:
        The counts and the rebuilt corpus table.
    """
    corpus = Path(corpus)
    corpus.mkdir(parents=True, exist_ok=True)
    done, failed = _known_papers(corpus)
    next_number = _next_paper_number(corpus)

    processed = skipped = errors = 0
    total = len(pdfs)
    for index, pdf in enumerate(pdfs, start=1):
        fraction = (index - 1) / total
        if pdf.name in done:
            on_progress(
                Progress(f"Skipping {pdf.name} (already processed)", fraction)
            )
            skipped += 1
            continue
        if pdf.name in failed:
            paper_dir = failed[pdf.name]
        else:
            paper_dir = corpus / f"P{next_number:04d}"
            next_number += 1
        paper_dir.mkdir(parents=True, exist_ok=True)

        on_progress(
            Progress(f"Processing {index}/{total}: {pdf.name}", fraction)
        )
        try:
            _process_paper(pdf, paper_dir, parse)
            processed += 1
        except Exception as exc:
            errors += 1
            on_progress(
                Progress(f"Error processing {pdf.name}: {exc}", index / total)
            )
            (paper_dir / ERROR_FILE).write_text(
                f"Error processing {pdf.name}:\n\n{traceback.format_exc()}",
                encoding="utf-8",
            )

    on_progress(Progress("Building corpus metadata table...", 1.0))
    table = build_corpus_table(corpus)
    return CorpusSummary(processed, skipped, errors, table)


def _process_paper(pdf: Path, paper_dir: Path, parse: Callable) -> None:
    """Parse one paper and write its files to its folder."""
    tei_xml = parse(pdf)
    (paper_dir / "fulltext.tei.xml").write_text(tei_xml, encoding="utf-8")
    (paper_dir / "fulltext.txt").write_text(
        extract_plain_text_from_tei(tei_xml), encoding="utf-8"
    )
    metadata = extract_metadata_fields(tei_xml, pdf, paper_dir.name)
    with open(paper_dir / METADATA_FILE, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    (paper_dir / ERROR_FILE).unlink(missing_ok=True)


def _paper_folders(corpus: Path) -> list[Path]:
    """Return the ``P<number>`` folders of a corpus, sorted by name."""
    return [
        folder
        for folder in sorted(corpus.glob("P*"))
        if folder.is_dir() and _PAPER_FOLDER.fullmatch(folder.name)
    ]


def _known_papers(corpus: Path) -> tuple[set[str], dict[str, Path]]:
    """Return the PDF names already parsed, and the folders of failed ones.

    The name of a parsed paper is in its ``metadata.json``; that of a
    failed one in the first line of its ``error.txt``.
    """
    done: set[str] = set()
    failed: dict[str, Path] = {}
    for folder in _paper_folders(corpus):
        metadata = folder / METADATA_FILE
        error = folder / ERROR_FILE
        if metadata.exists():
            try:
                with open(metadata, encoding="utf-8") as file:
                    done.add(Path(json.load(file).get("pdf_path", "")).name)
            except (OSError, ValueError, AttributeError):
                LOGGER.warning("Unreadable %s", metadata)
        elif error.exists():
            match = _ERROR_HEADER.match(error.read_text(encoding="utf-8"))
            if match:
                failed.setdefault(match.group(1), folder)
    done.discard("")
    return done, failed


def _next_paper_number(corpus: Path) -> int:
    """Return the number after the highest ``P`` folder of a corpus."""
    numbers = [
        int(_PAPER_FOLDER.fullmatch(folder.name).group(1))
        for folder in _paper_folders(corpus)
    ]
    return max(numbers, default=0) + 1


def build_corpus_table(corpus: str | Path) -> pd.DataFrame:
    """Collect the papers' metadata into the corpus tables.

    Writes ``corpus_metadata.json`` (every ``metadata.json``),
    ``corpus_table.csv`` and ``corpus_table.jsonl`` (one row per paper,
    lists joined with ``"; "``, citations as JSON).

    Args:
        corpus: The corpus folder.

    Returns:
        The corpus table.
    """
    corpus = Path(corpus)
    metadata_list = []
    for paper_dir in sorted(corpus.glob("P*")):
        metadata_file = paper_dir / METADATA_FILE
        if paper_dir.is_dir() and metadata_file.exists():
            with open(metadata_file, encoding="utf-8") as file:
                metadata_list.append(json.load(file))

    with open(corpus / CORPUS_METADATA_FILE, "w", encoding="utf-8") as file:
        json.dump(metadata_list, file, indent=2, ensure_ascii=False)
    LOGGER.info(
        "Wrote %s with %d papers", CORPUS_METADATA_FILE, len(metadata_list)
    )

    table = pd.DataFrame([table_row(meta) for meta in metadata_list])
    table.to_csv(corpus / TABLE_CSV, index=False)
    table.to_json(corpus / TABLE_JSONL, orient="records", lines=True)
    LOGGER.info(
        "Wrote %s and %s with %d rows", TABLE_CSV, TABLE_JSONL, len(table)
    )
    return table


def table_row(meta: dict[str, Any]) -> dict[str, Any]:
    """Return a paper's row of the corpus table.

    Args:
        meta: The paper's ``metadata.json``. Authors may be names or
            dictionaries with ``full_name``.

    Returns:
        The row.
    """
    authors = meta.get("authors", [])
    if authors and isinstance(authors[0], dict):
        authors_text = "; ".join(a.get("full_name", "") for a in authors)
    elif isinstance(authors, list):
        authors_text = "; ".join(authors)
    else:
        authors_text = str(authors)

    pdf_path = meta.get("pdf_path", "") or meta.get("filename", "")
    keywords = meta.get("keywords")
    citations = meta.get("citations", [])
    return {
        "paper_id": meta.get("paper_id", ""),
        "title": meta.get("title", ""),
        "authors": authors_text,
        "n_authors": len(authors) if isinstance(authors, list) else 0,
        "abstract": meta.get("abstract", ""),
        "keywords": "; ".join(keywords) if isinstance(keywords, list) else "",
        "DOI": meta.get("DOI", "") or meta.get("doi", ""),
        "journal": meta.get("journal", ""),
        "publication_date": (
            meta.get("publication_date", "") or str(meta.get("year", ""))
        ),
        "pdf_path": pdf_path,
        "pdf_filename": Path(pdf_path).name if pdf_path else "",
        "n_citations": meta.get("n_citations", 0) or len(citations),
        "citations": json.dumps(citations),
        "cited_titles": "; ".join(
            c.get("title", "").strip()
            for c in citations
            if c.get("title", "").strip()
        ),
        # Only the references that have a DOI.
        "cited_dois": "; ".join(
            c.get("doi", "") or c.get("DOI", "")
            for c in citations
            if c.get("doi") or c.get("DOI")
        ),
        "cited_authors": "; ".join(
            author
            for c in citations
            for author in (
                c.get("authors", [])
                if isinstance(c.get("authors"), list)
                else []
            )
        ),
    }


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Return the records of a JSON Lines file, such as the topics table.

    Args:
        path: The file.

    Returns:
        One dictionary per line.
    """
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def topics_json(corpus: str | Path) -> str:
    """Return the corpus table with topics as one JSON document.

    Args:
        corpus: A corpus folder with ``corpus_table.with_topics.jsonl``.

    Returns:
        The records as an indented JSON list, for download.
    """
    return json.dumps(read_records(Path(corpus) / TOPICS_JSONL), indent=4)
