"""Reading Grobid's TEI XML: a paper's metadata, citations and plain text."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

NS = {"tei": "http://www.tei-c.org/ns/1.0"}
_XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
_BIBL = ".//tei:sourceDesc/tei:biblStruct"


def extract_metadata_fields(
    tei_xml: str, pdf_path: str | Path, paper_id: str
) -> dict[str, Any]:
    """Return a paper's metadata, as stored in its ``metadata.json``.

    Args:
        tei_xml: Grobid's TEI XML of the paper.
        pdf_path: The paper's PDF.
        paper_id: The paper's identifier in the corpus, e.g. ``"P0001"``.

    Returns:
        ``paper_id``, ``pdf_path``, ``title``, ``authors`` (names),
        ``n_authors``, ``abstract``, ``keywords``, ``DOI``, ``journal``,
        ``publication_date``, ``citations`` (one dictionary per reference:
        ``citation_id``, ``title``, ``authors``, ``DOI``, ``year``),
        ``n_citations``, ``cited_dois`` and ``cited_authors`` (unique).
    """
    root = ET.fromstring(tei_xml.encode("utf-8"))
    metadata: dict[str, Any] = {
        "paper_id": paper_id,
        "pdf_path": str(pdf_path),
    }
    metadata["title"] = _text(
        root.find(".//tei:titleStmt/tei:title[@type='main']", NS)
    )

    # The paper's authors are in titleStmt in some documents, but usually
    # in sourceDesc.
    authors = _author_names(root.findall(".//tei:titleStmt/tei:author", NS))
    if not authors:
        authors = _author_names(
            root.findall(f"{_BIBL}/tei:analytic/tei:author", NS)
        )
    metadata["authors"] = authors
    metadata["n_authors"] = len(authors)

    metadata["abstract"] = _text(
        root.find(".//tei:abstract/tei:div/tei:p", NS)
    )
    metadata["keywords"] = [
        term.text.strip()
        for term in root.findall(".//tei:keywords/tei:term", NS)
        if term.text
    ]
    metadata["DOI"] = _text(root.find(".//tei:idno[@type='DOI']", NS))
    metadata["journal"] = _text(root.find(f"{_BIBL}/tei:monogr/tei:title", NS))
    date = root.find(
        f"{_BIBL}/tei:monogr/tei:imprint/tei:date[@type='published']", NS
    )
    metadata["publication_date"] = (
        date.get("when", "") if date is not None else ""
    )

    citations = [
        _citation(bibl)
        for bibl in root.findall(
            ".//tei:text/tei:back//tei:listBibl/tei:biblStruct", NS
        )
    ]
    metadata["citations"] = citations
    metadata["n_citations"] = len(citations)
    metadata["cited_dois"] = [c["DOI"] for c in citations if c["DOI"]]
    metadata["cited_authors"] = list(
        {author for c in citations for author in c["authors"]}
    )
    return metadata


def extract_plain_text_from_tei(tei_xml: str) -> str:
    """Return the body text of a paper, without references and figures.

    Args:
        tei_xml: Grobid's TEI XML of the paper.

    Returns:
        The paragraphs, whitespace collapsed, separated by blank lines;
        ``""`` if the document has no body.
    """
    root = ET.fromstring(tei_xml.encode("utf-8"))
    body = root.find(".//tei:text/tei:body", NS)
    if body is None:
        return ""

    # Reference markers lose their text but keep what follows them.
    for ref in body.findall(".//tei:ref", NS):
        if ref.text:
            ref.text = ""
        ref.tail = ref.tail or ""
    for parent in body.iter():
        for figure in list(parent.findall("tei:figure", NS)):
            parent.remove(figure)

    paragraphs = []
    for paragraph in body.findall(".//tei:p", NS):
        text = re.sub(r"\s+", " ", "".join(paragraph.itertext()).strip())
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


def _text(element: ET.Element | None) -> str:
    """Return an element's stripped text, or ``""``."""
    if element is None or not element.text:
        return ""
    return element.text.strip()


def _author_names(authors: list[ET.Element]) -> list[str]:
    """Return "first last" for the authors that have both names."""
    names = []
    for author in authors:
        person = author.find("tei:persName", NS)
        if person is None:
            continue
        first = person.find("tei:forename[@type='first']", NS)
        last = person.find("tei:surname", NS)
        if first is not None and last is not None:
            names.append(f"{first.text} {last.text}")
    return names


def _citation(bibl: ET.Element) -> dict[str, Any]:
    """Return one reference of the bibliography as a dictionary."""
    title = bibl.find(".//tei:analytic/tei:title[@type='main']", NS)
    if title is None:
        title = bibl.find(".//tei:monogr/tei:title", NS)
    date = bibl.find(
        ".//tei:monogr/tei:imprint/tei:date[@type='published']", NS
    )
    return {
        "citation_id": bibl.get(_XML_ID, ""),
        "title": _text(title),
        "authors": _cited_author_names(
            bibl.findall(".//tei:analytic/tei:author", NS)
        ),
        "DOI": _text(bibl.find(".//tei:idno[@type='DOI']", NS)),
        "year": date.get("when", "")[:4] if date is not None else "",
    }


def _cited_author_names(authors: list[ET.Element]) -> list[str]:
    """Return the names of a reference's authors, with middle names.

    Authors without a first name are listed by their last name.
    """
    names = []
    for author in authors:
        person = author.find("tei:persName", NS)
        if person is None:
            continue
        first = person.find("tei:forename[@type='first']", NS)
        middle = person.find("tei:forename[@type='middle']", NS)
        last = person.find("tei:surname", NS)
        if first is not None and last is not None:
            if middle is not None and middle.text:
                names.append(f"{first.text} {middle.text} {last.text}")
            else:
                names.append(f"{first.text} {last.text}")
        elif last is not None:
            names.append(last.text)
    return names
