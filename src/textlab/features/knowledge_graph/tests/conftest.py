"""Sample Grobid output and corpus records for the Knowledge Graph tests."""

import pytest

TEI = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt><title level="a" type="main"> Graphs of Papers </title>
      </titleStmt>
      <sourceDesc>
        <biblStruct>
          <analytic>
            <author><persName><forename type="first">Ada</forename>
              <surname>Lovelace</surname></persName></author>
            <author><persName><surname>NoFirstName</surname></persName>
            </author>
          </analytic>
          <monogr>
            <title level="j">Journal of Tests</title>
            <imprint><date type="published" when="2024-05-01"/></imprint>
          </monogr>
          <idno type="DOI">10.1/ABC</idno>
        </biblStruct>
      </sourceDesc>
    </fileDesc>
    <profileDesc>
      <abstract><div><p> We draw graphs. </p></div></abstract>
      <textClass><keywords><term>graphs</term><term>papers</term>
      </keywords></textClass>
    </profileDesc>
  </teiHeader>
  <text>
    <body>
      <div>
        <p>First   paragraph <ref type="bibr">[1]</ref> cites.</p>
        <figure><p>Figure text</p></figure>
        <p>Second paragraph.</p>
      </div>
    </body>
    <back>
      <listBibl>
        <biblStruct xml:id="b0">
          <analytic>
            <title level="a" type="main">Cited Work</title>
            <author><persName><forename type="first">Alan</forename>
              <forename type="middle">M</forename>
              <surname>Turing</surname></persName></author>
            <author><persName><surname>Hopper</surname></persName></author>
          </analytic>
          <monogr><imprint><date type="published" when="1950"/></imprint>
          </monogr>
          <idno type="DOI">10.2/XYZ</idno>
        </biblStruct>
        <biblStruct xml:id="b1">
          <monogr><title>A Book</title></monogr>
        </biblStruct>
      </listBibl>
    </back>
  </text>
</TEI>
"""


@pytest.fixture
def tei():
    return TEI


@pytest.fixture
def records():
    """Two papers with topics; P0001 cites P0002 and an outside paper."""
    return [
        {
            "paper_id": "P0001",
            "title": "Graphs of Papers",
            "authors": "Ada Lovelace; Grace Hopper",
            "DOI": "10.1/ABC",
            "n_citations": 2,
            "pdf_filename": "a.pdf",
            "cited_titles": "Second Paper; Outside Work",
            "citations": (
                '[{"title": "Second Paper", "DOI": ""},'
                ' {"title": "Outside Work", "DOI": "10.9/OUT"}]'
            ),
            "cited_authors": "Alan Turing; Grace Hopper",
            "topics": [
                {
                    "category": "Computer Science",
                    "label": "Graph Drawing",
                    "confidence": 0.9,
                },
                {
                    "category": "Computer Science",
                    "label": "Citation Networks",
                    "confidence": 0.4,
                },
            ],
        },
        {
            "paper_id": "P0002",
            "title": "Second Paper",
            "authors": "Grace Hopper",
            "DOI": "",
            "n_citations": 0,
            "pdf_filename": "b.pdf",
            "cited_titles": "",
            "citations": "[]",
            "cited_authors": "",
            "topics": [
                {
                    "category": "Computer Science",
                    "label": "Graph Drawing",
                    "confidence": 0.7,
                }
            ],
        },
    ]
