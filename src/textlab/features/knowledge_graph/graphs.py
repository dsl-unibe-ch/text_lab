"""Interactive graphs of a corpus with topics: per paper, or all papers.

Graphs are built with NetworkX and drawn with Pyvis (vis.js, embedded in
the page), both imported only when a graph is built. Records are the rows of
``corpus_table.with_topics.jsonl``; list fields may be ``"; "``-joined
strings or lists, and ``citations`` a JSON string or a list.

Each node has a ``node_type``; by type, its id and look:

- ``paper``: ``doi:<DOI>``, or the paper id without a DOI; blue dot.
- ``author``: ``author:<name>``; green triangle.
- ``category``: ``category:<name>``; purple box.
- ``topic``: ``topic:<label>``; red ellipse.
- ``cited_paper``: ``paper:<first 50 characters of the title>``; orange dot
  (purple in an ego graph when the cited paper is in the corpus).
- ``cited_author``: ``cited_author:<name>``; gold triangle.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

PAPER_COLOR = "#4A90E2"
AUTHOR_COLOR = "#50C878"
CATEGORY_COLOR = "#9B59B6"
TOPIC_COLOR = "#FF6B6B"
CITED_IN_CORPUS_COLOR = "#9B59B6"
CITED_EXTERNAL_COLOR = "#FFA500"
CITED_AUTHOR_COLOR = "#FFD700"
CITES_EDGE_COLOR = "#999999"
CITES_WORK_EDGE_COLOR = "#CCCCCC"

_EGO_OPTIONS = """
    {
      "physics": {
        "enabled": true,
        "stabilization": {
          "iterations": 200
        },
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 150,
          "springConstant": 0.04
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100
      }
    }
    """

_FULL_OPTIONS = """
    {
      "physics": {
        "enabled": true,
        "stabilization": {
          "iterations": 300
        },
        "barnesHut": {
          "gravitationalConstant": -15000,
          "centralGravity": 0.1,
          "springLength": 200,
          "springConstant": 0.02,
          "damping": 0.5
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "keyboard": true
      }
    }
    """


# ---------------------------------------------------------------------------
# Record fields
# ---------------------------------------------------------------------------


def _split(value: Any) -> list[Any]:
    """Return a list field: a ``;``-separated string split, or the list."""
    if isinstance(value, str):
        return [item.strip() for item in value.split(";") if item.strip()]
    if isinstance(value, list):
        return value
    return []


def _citations(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a record's citations, stored as JSON text or as a list."""
    value = record.get("citations", "")
    if isinstance(value, str) and value:
        try:
            value = json.loads(value)
        except ValueError:
            return []
    return value if isinstance(value, list) else []


def _doi(record: dict[str, Any]) -> str:
    """Return a record's DOI as stored, or ``""``."""
    return record.get("DOI", "") or record.get("doi", "")


def _author_name(author: Any) -> str:
    """Return an author's name; authors may be names or dictionaries."""
    return author if isinstance(author, str) else author.get("name", "Unknown")


def _title_dois(record: dict[str, Any]) -> dict[str, str]:
    """Return the DOI of each cited title (``""`` without one)."""
    dois = {}
    for citation in _citations(record):
        title = citation.get("title", "").strip()
        if title:
            dois[title] = (
                citation.get("DOI", "") or citation.get("doi", "")
            ).strip()
    return dois


def _titles(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return the records by lowercased title, to match cited titles."""
    by_title = {}
    for record in records:
        title = record.get("title", "").strip().lower()
        if title:
            by_title[title] = record
    return by_title


def _full_node_id(record: dict[str, Any]) -> str:
    """Return a paper's node id in the full graph (normalized DOI)."""
    doi = _doi(record).strip().lower()
    return f"doi:{doi}" if doi else record.get("paper_id", "unknown")


def _shorten(text: str, length: int) -> str:
    """Return text cut to ``length`` characters with an ellipsis."""
    return text[:length] + "..." if len(text) > length else text


def _network(graph: Any, height: str, options: str) -> Any:
    """Return a Pyvis network that draws a graph."""
    from pyvis.network import Network

    net = Network(
        height=height,
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
        directed=True,
        # The JavaScript is embedded, so the page works offline and Pyvis
        # does not copy its library into the working directory.
        cdn_resources="in_line",
    )
    net.from_nx(graph)
    net.set_options(options)
    return net


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------


def build_paper_ego_graph(
    paper_record: dict[str, Any],
    all_records: list[dict[str, Any]] | None = None,
    include_topics: bool = True,
    include_authors: bool = True,
    include_cited_papers: bool = True,
    include_cited_authors: bool = True,
) -> tuple[Any, Any]:
    """Build the graph around one paper.

    Args:
        paper_record: The paper.
        all_records: Every paper of the corpus, to mark cited papers that
            are in it.
        include_topics: Add the paper's topics and their categories.
        include_authors: Add the paper's authors.
        include_cited_papers: Add the papers it cites.
        include_cited_authors: Add the authors it cites.

    Returns:
        ``(graph, network)``: the ``networkx.DiGraph`` and the Pyvis
        network that draws it.
    """
    import networkx as nx

    graph = nx.DiGraph()
    paper_id = paper_record.get("paper_id", "unknown")
    title = paper_record.get("title", "Untitled")
    doi = _doi(paper_record)
    node_id = f"doi:{doi}" if doi else paper_id
    graph.add_node(
        node_id,
        label=paper_id,
        title=f"{title}\nDOI: {doi}" if doi else title,
        node_type="paper",
        color=PAPER_COLOR,
        size=30,
        shape="dot",
    )

    if include_authors:
        for author in _split(paper_record.get("authors", [])):
            name = _author_name(author)
            graph.add_node(
                f"author:{name}",
                label=name,
                title=f"Author: {name}",
                node_type="author",
                color=AUTHOR_COLOR,
                size=15,
                shape="triangle",
            )
            graph.add_edge(node_id, f"author:{name}", label="authored_by")

    if include_topics:
        _add_ego_topics(graph, node_id, paper_record.get("topics", []))

    if include_cited_papers:
        _add_ego_cited_papers(graph, node_id, paper_record, all_records)

    if include_cited_authors:
        for name in _split(paper_record.get("cited_authors", "")):
            author_id = f"cited_author:{name}"
            if author_id not in graph.nodes:
                graph.add_node(
                    author_id,
                    label=name,
                    title=f"Cited Author: {name}",
                    node_type="cited_author",
                    color=CITED_AUTHOR_COLOR,
                    size=12,
                    shape="triangle",
                )
            graph.add_edge(
                node_id,
                author_id,
                label="cites_work_by",
                color=CITES_WORK_EDGE_COLOR,
                style="dashed",
            )

    return graph, _network(graph, "600px", _EGO_OPTIONS)


def _add_ego_topics(
    graph: Any, node_id: str, topics: list[dict[str, Any]]
) -> None:
    """Add a paper's topics and their categories to its ego graph."""
    for topic in topics:
        category = topic.get("category", "General")
        label = topic.get("label", "unknown topic")
        confidence = topic.get("confidence", 0.5)

        category_id = f"category:{category}"
        if category_id not in graph.nodes:
            graph.add_node(
                category_id,
                label=category,
                title=f"Category: {category}",
                node_type="category",
                color=CATEGORY_COLOR,
                size=25,
                shape="box",
            )
        graph.add_edge(node_id, category_id, label="in_category")

        topic_id = f"topic:{label}"
        if topic_id not in graph.nodes:
            graph.add_node(
                topic_id,
                label=label,
                title=(
                    f"Topic: {label}\nCategory: {category}\n"
                    f"Confidence: {confidence:.2f}"
                ),
                node_type="topic",
                color=TOPIC_COLOR,
                size=10 + confidence * 15,
                shape="ellipse",
            )
        graph.add_edge(node_id, topic_id, label="has_topic", weight=confidence)
        graph.add_edge(
            topic_id, category_id, label="belongs_to", style="dashed"
        )


def _add_ego_cited_papers(
    graph: Any,
    node_id: str,
    record: dict[str, Any],
    all_records: list[dict[str, Any]] | None,
) -> None:
    """Add the papers a paper cites to its ego graph, marking the corpus."""
    corpus = _titles(all_records or [])
    dois = _title_dois(record)
    for cited_title in _split(record.get("cited_titles", "")):
        title = cited_title.strip()
        cited_id = f"paper:{title[:50]}"
        cited_doi = dois.get(cited_title, "")
        in_corpus = corpus.get(title.lower())
        if in_corpus is not None:
            cited_paper_id = in_corpus.get("paper_id", "unknown")
            color = CITED_IN_CORPUS_COLOR
            label = cited_paper_id
            hover = f"{title}\n(in corpus: {cited_paper_id})"
        else:
            color = CITED_EXTERNAL_COLOR
            label = _shorten(title, 30)
            hover = f"{title}\n(not in corpus)"
        if cited_doi:
            hover += f"\nDOI: {cited_doi}"

        if cited_id not in graph.nodes:
            graph.add_node(
                cited_id,
                label=label,
                title=hover,
                node_type="cited_paper",
                color=color,
                size=15,
                shape="dot",
            )
        graph.add_edge(
            node_id, cited_id, label="cites", color=CITES_EDGE_COLOR
        )


def build_full_corpus_graph(
    all_records: list[dict[str, Any]],
    include_topics: bool = True,
    include_authors: bool = True,
    include_cited_papers: bool = False,
    include_cited_authors: bool = False,
    min_topic_confidence: float = 0.0,
) -> tuple[Any, Any]:
    """Build the graph of all papers: shared authors and topics connect them.

    Papers are sized by their number of references, authors by their
    number of papers, categories and topics by how many papers have them.

    Args:
        all_records: The papers.
        include_topics: Add topics and categories.
        include_authors: Add authors (collaborations).
        include_cited_papers: Add citations: edges between papers of the
            corpus, and nodes for cited papers outside it.
        include_cited_authors: Add the cited authors.
        min_topic_confidence: Leave out topics with a lower confidence.

    Returns:
        ``(graph, network)``: the ``networkx.DiGraph`` and the Pyvis
        network that draws it.
    """
    import networkx as nx

    graph = nx.DiGraph()
    authors: dict[str, list[str]] = {}
    topics: dict[str, list[tuple[str, float, str]]] = {}
    categories: dict[str, list[str]] = {}

    for record in all_records:
        paper_id = record.get("paper_id", "unknown")
        title = record.get("title", "Untitled")
        doi = _doi(record).strip().lower()
        node_id = f"doi:{doi}" if doi else paper_id
        graph.add_node(
            node_id,
            label=paper_id,
            title=f"{title}\nDOI: {doi}" if doi else title,
            node_type="paper",
            color=PAPER_COLOR,
            size=15 + min(record.get("n_citations", 0) * 3, 30),
            shape="dot",
        )
        if include_authors:
            for author in _split(record.get("authors", [])):
                authors.setdefault(_author_name(author), []).append(node_id)
        if include_topics:
            for topic in record.get("topics", []):
                category = topic.get("category", "General")
                confidence = topic.get("confidence", 0.5)
                if confidence >= min_topic_confidence:
                    categories.setdefault(category, []).append(node_id)
                    topics.setdefault(
                        topic.get("label", "unknown topic"), []
                    ).append((node_id, confidence, category))

    if include_authors:
        for name, papers in authors.items():
            author_id = f"author:{name}"
            graph.add_node(
                author_id,
                label=name,
                title=f"Author: {name}\nPapers: {len(papers)}",
                node_type="author",
                color=AUTHOR_COLOR,
                size=10 + len(papers) * 5,
                shape="triangle",
            )
            for paper in papers:
                graph.add_edge(author_id, paper, label="authored")

    if include_topics:
        _add_full_topics(graph, categories, topics)
    if include_cited_papers:
        _add_full_cited_papers(graph, all_records)
    if include_cited_authors:
        _add_full_cited_authors(graph, all_records)

    return graph, _network(graph, "800px", _FULL_OPTIONS)


def _add_full_topics(
    graph: Any,
    categories: dict[str, list[str]],
    topics: dict[str, list[tuple[str, float, str]]],
) -> None:
    """Add the categories and topics of all papers to the full graph."""
    for category, papers in categories.items():
        category_id = f"category:{category}"
        graph.add_node(
            category_id,
            label=category,
            title=f"Category: {category}\nPapers: {len(papers)}",
            node_type="category",
            color=CATEGORY_COLOR,
            size=20 + len(papers) * 4,
            shape="box",
        )
        for paper in papers:
            graph.add_edge(paper, category_id, label="in_category")

    for label, uses in topics.items():
        topic_id = f"topic:{label}"
        average = sum(confidence for _, confidence, _ in uses) / len(uses)
        # A topic label is filed under the category of its first use.
        category = uses[0][2]
        category_id = f"category:{category}"
        graph.add_node(
            topic_id,
            label=label,
            title=(
                f"Topic: {label}\nCategory: {category}\n"
                f"Papers: {len(uses)}\nAvg confidence: {average:.2f}"
            ),
            node_type="topic",
            color=TOPIC_COLOR,
            size=10 + len(uses) * 2,
            shape="ellipse",
        )
        for paper, confidence, _ in uses:
            graph.add_edge(
                paper, topic_id, label="has_topic", weight=confidence
            )
        if category_id in graph.nodes:
            graph.add_edge(
                topic_id, category_id, label="belongs_to", style="dashed"
            )


def _add_full_cited_papers(graph: Any, records: list[dict[str, Any]]) -> None:
    """Add citations to the full graph.

    A cited paper of the corpus (matched by title) gets an edge to its
    node; any other cited paper gets a node of its own.
    """
    corpus = _titles(records)
    for record in records:
        citing_id = _full_node_id(record)
        dois = _title_dois(record)
        for cited_title in _split(record.get("cited_titles", "")):
            title = cited_title.strip()
            in_corpus = corpus.get(title.lower())
            if in_corpus is not None:
                cited_id = _full_node_id(in_corpus)
                if cited_id in graph.nodes:
                    graph.add_edge(
                        citing_id,
                        cited_id,
                        label="cites",
                        color=CITES_EDGE_COLOR,
                    )
                continue

            cited_id = f"paper:{title[:50]}"
            if cited_id not in graph.nodes:
                hover = f"{title}\n(not in corpus)"
                cited_doi = dois.get(cited_title, "")
                if cited_doi:
                    hover += f"\nDOI: {cited_doi}"
                graph.add_node(
                    cited_id,
                    label=_shorten(title, 25),
                    title=hover,
                    node_type="cited_paper",
                    color=CITED_EXTERNAL_COLOR,
                    size=10,
                    shape="dot",
                )
            graph.add_edge(
                citing_id, cited_id, label="cites", color=CITES_EDGE_COLOR
            )


def _add_full_cited_authors(graph: Any, records: list[dict[str, Any]]) -> None:
    """Add the cited authors to the full graph, sized by citing papers."""
    cited: dict[str, list[str]] = {}
    for record in records:
        citing_id = _full_node_id(record)
        for name in _split(record.get("cited_authors", "")):
            cited.setdefault(name, []).append(citing_id)

    for name, citing in cited.items():
        author_id = f"cited_author:{name}"
        if author_id not in graph.nodes:
            graph.add_node(
                author_id,
                label=name,
                title=f"Cited Author: {name}\nCited by {len(citing)} papers",
                node_type="cited_author",
                color=CITED_AUTHOR_COLOR,
                size=8 + len(citing) * 2,
                shape="triangle",
            )
        for paper in citing:
            graph.add_edge(
                paper,
                author_id,
                label="cites_work_by",
                color=CITES_WORK_EDGE_COLOR,
                style="dashed",
            )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def count_nodes(graph: Any, node_type: str) -> int:
    """Return how many nodes of a type a graph has.

    Args:
        graph: A graph from this module.
        node_type: E.g. ``"paper"`` or ``"author"``.

    Returns:
        The count.
    """
    return sum(
        1 for _, kind in graph.nodes(data="node_type") if kind == node_type
    )


def save_graph_html(network: Any, path: str | Path) -> str:
    """Save a graph as a self-contained HTML page and return the page.

    Args:
        network: The Pyvis network.
        path: Where to save it.

    Returns:
        The HTML.
    """
    html = network.generate_html()
    Path(path).write_text(html, encoding="utf-8")
    return html
