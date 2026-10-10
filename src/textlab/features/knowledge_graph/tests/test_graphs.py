"""Ego and full corpus graphs of sample records."""

from textlab.features.knowledge_graph import service


def edge_labels(graph):
    return sorted(label for _, _, label in graph.edges(data="label"))


def test_the_ego_graph_of_a_paper(records):
    graph, network = service.build_paper_ego_graph(
        records[0], all_records=records
    )
    center = graph.nodes["doi:10.1/ABC"]
    assert center["node_type"] == "paper" and center["label"] == "P0001"
    assert set(graph.nodes) == {
        "doi:10.1/ABC",
        "author:Ada Lovelace",
        "author:Grace Hopper",
        "category:Computer Science",
        "topic:Graph Drawing",
        "topic:Citation Networks",
        "paper:Second Paper",
        "paper:Outside Work",
        "cited_author:Alan Turing",
        "cited_author:Grace Hopper",
    }
    assert edge_labels(graph) == sorted(
        ["authored_by"] * 2
        + ["in_category"]
        + ["has_topic"] * 2
        + ["belongs_to"] * 2
        + ["cites"] * 2
        + ["cites_work_by"] * 2
    )
    # A cited paper of the corpus is shown by its paper id.
    in_corpus = graph.nodes["paper:Second Paper"]
    assert in_corpus["label"] == "P0002"
    assert "(in corpus: P0002)" in in_corpus["title"]
    outside = graph.nodes["paper:Outside Work"]
    assert outside["title"] == "Outside Work\n(not in corpus)\nDOI: 10.9/OUT"
    # Sized 10 + 0.9 * 15; Pyvis rounds the sizes down when it draws the
    # graph, and does so in the graph's own node attributes.
    assert graph.nodes["topic:Graph Drawing"]["size"] == 23
    assert service.count_nodes(graph, "author") == 2
    assert len(network.nodes) == graph.number_of_nodes()


def test_the_ego_graph_leaves_out_what_is_unchecked(records):
    graph, _ = service.build_paper_ego_graph(
        records[0],
        include_topics=False,
        include_authors=False,
        include_cited_papers=False,
        include_cited_authors=False,
    )
    assert list(graph.nodes) == ["doi:10.1/ABC"]


def test_the_full_graph_connects_shared_authors_and_topics(records):
    graph, _ = service.build_full_corpus_graph(records)
    assert set(graph.nodes) == {
        "doi:10.1/abc",
        "P0002",
        "author:Ada Lovelace",
        "author:Grace Hopper",
        "category:Computer Science",
        "topic:Graph Drawing",
        "topic:Citation Networks",
    }
    assert graph.nodes["doi:10.1/abc"]["size"] == 15 + 2 * 3
    assert graph.nodes["author:Grace Hopper"]["size"] == 10 + 2 * 5
    shared = graph.nodes["topic:Graph Drawing"]
    assert "Papers: 2\nAvg confidence: 0.80" in shared["title"]
    assert graph.has_edge("author:Grace Hopper", "P0002")
    assert graph.has_edge("P0002", "topic:Graph Drawing")
    assert service.count_nodes(graph, "paper") == 2


def test_the_full_graph_filters_topics_by_confidence(records):
    graph, _ = service.build_full_corpus_graph(
        records, include_authors=False, min_topic_confidence=0.5
    )
    assert "topic:Citation Networks" not in graph
    assert "topic:Graph Drawing" in graph


def test_the_full_graph_with_citations(records):
    graph, _ = service.build_full_corpus_graph(
        records,
        include_topics=False,
        include_authors=False,
        include_cited_papers=True,
        include_cited_authors=True,
    )
    # A paper of the corpus is linked, any other gets a node.
    assert graph.has_edge("doi:10.1/abc", "P0002")
    assert graph.nodes["paper:Outside Work"]["node_type"] == "cited_paper"
    assert graph.has_edge("doi:10.1/abc", "cited_author:Alan Turing")
    assert service.count_nodes(graph, "cited_author") == 2


def test_a_graph_is_saved_as_a_self_contained_page(records, tmp_path):
    _, network = service.build_paper_ego_graph(records[1])
    html = service.save_graph_html(network, tmp_path / "P0002.html")
    assert (tmp_path / "P0002.html").read_text(encoding="utf-8") == html
    assert "P0002" in html
    assert 'src="lib/' not in html  # the JavaScript is embedded
    assert not (tmp_path / "lib").exists()
