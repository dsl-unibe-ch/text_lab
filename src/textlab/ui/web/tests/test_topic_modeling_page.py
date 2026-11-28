"""The Topic Modeling page, with the analysis faked.

``run_topic_modeling`` returns two topics with what their algorithm's
library draws (pyLDAvis's data for LDA; the map, hierarchy, documents and
timeline for BERTopic; word clouds for Top2Vec) and the scores, and can be
held at a gate, to test cancelling; the rest runs for real: reading the
collection, the draft jobs it waits in, reading the settings, saving the
downloads and drawing each algorithm's views.
"""

import io
import re
import threading
import zipfile
from types import SimpleNamespace

import pandas as pd
import pytest

from textlab.common.progress import Progress, check_cancelled
from textlab.features.topic_modeling import service
from textlab.features.topic_modeling.models import (
    DocumentMap,
    Hierarchy,
    Notice,
    TopicOverview,
    TopicSummary,
)
from textlab.ui.web.jobs import JobState
from textlab.ui.web.pages import topic_charts
from textlab.ui.web.pages import topic_modeling as page
from textlab.ui.web.tests.conftest import BASE, wait_for
from textlab.ui.web.tests.test_job_pages import htmx

CSV = (
    b"Id,Answer,Year\n"
    b"1,taxes are too high in our town,2020\n"
    b"2,the football club won the cup,2021\n"
    b"3,a new budget for the school,2021\n"
)

BERTOPIC_OVERVIEW = TopicOverview(
    documents=3,
    outliers=0,
    topics=[
        TopicSummary(
            1,
            ["tax", "budget", "town"],
            [1.0, 0.5, 0.2],
            2,
            ["taxes <b>"],
            prevalence=0.667,
            time_words=["taxes, rise", "taxes, town"],
        ),
        TopicSummary(
            2,
            ["football", "cup"],
            [1.0, 0.8],
            1,
            ["the club"],
            prevalence=0.333,
            time_words=["", "football, cup"],
        ),
    ],
    similarity=[[1.0, 0.25], [0.25, 1.0]],
    positions=[[0.0, 0.5], [1.0, 0.5]],
    time_labels=["2020", "2021"],
    over_time=[[1, 1], [0, 1]],
    hierarchy=Hierarchy(
        order=[1, 0],
        links=[[[0.0, 0.75, 0.75, 0.0], [0.0, 0.0, 1.0, 1.0]]],
        height=0.75,
    ),
    document_map=DocumentMap(
        points=[[0.0, 0.0, 0], [0.5, 0.5, 0], [1.0, 1.0, 1]],
        excerpts=["taxes are", "a new budget", "the football"],
        documents=3,
    ),
)

LDA_OVERVIEW = TopicOverview(
    documents=3,
    outliers=0,
    topics=[
        TopicSummary(
            1,
            ["tax", "budget"],
            [1.0, 0.5],
            2,
            ["taxes <b>", "a new budget"],
            prevalence=0.6,
            mixtures=[[[0, 0.7], [1, 0.3]], [[0, 0.9]]],
            terms=[
                ["town", 4.0, 4.0, -3.0, 1.5],
                ["tax", 25.0, 30.0, -1.2, 0.4],
                ["budget", 9.0, 10.0, -2.3, 0.9],
            ],
        ),
        TopicSummary(
            2,
            ["football", "cup"],
            [1.0, 0.8],
            1,
            ["the club"],
            prevalence=0.4,
            mixtures=[[[1, 0.8], [0, 0.2]]],
            terms=[["cup", 12.0, 12.0, -0.9, 1.1]],
        ),
    ],
    similarity=[[1.0, 0.25], [0.25, 1.0]],
    positions=[[1.0, 0.375], [0.0, 0.625]],
    origin=[0.5, 0.375],
    salient=[["tax", 30.0], ["football", 20.0]],
    term_topics={"tax": [[0, 0.8], [1, 0.2]], "cup": [[1, 1.0]]},
)

TOP2VEC_OVERVIEW = TopicOverview(
    documents=3,
    outliers=0,
    topics=[
        TopicSummary(
            1,
            ["tax", "budget"],
            [1.0, 0.9],
            2,
            ["taxes"],
            prevalence=0.667,
            cloud=[["tax", 1.0], ["budget", 0.6], ["levy", 0.2]],
        ),
        TopicSummary(
            2,
            ["cup"],
            [1.0],
            1,
            ["the club"],
            prevalence=0.333,
            cloud=[["cup", 1.0]],
        ),
    ],
    similarity=[[1.0, 0.1], [0.1, 1.0]],
    positions=[[0.0, 0.5], [1.0, 0.5]],
)

#: The overview each algorithm's fake run returns.
OVERVIEWS = {
    service.Algorithm.BERTOPIC: BERTOPIC_OVERVIEW,
    service.Algorithm.LDA: LDA_OVERVIEW,
    service.Algorithm.TOP2VEC: TOP2VEC_OVERVIEW,
}


def zip_of(members):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


@pytest.fixture
def analysis(monkeypatch):
    """Fake the analysis; record how it was asked to run."""
    calls = SimpleNamespace(runs=[], gate=None, overview="auto")

    def run(table, config, *, source_name, on_progress, cancel):
        calls.runs.append(
            {
                "table": table,
                "config": config,
                "name": source_name,
                "cancel": cancel,
            }
        )
        on_progress(Progress("Running topic extraction...", None))
        if calls.gate is not None:
            while not calls.gate.wait(0.01):
                check_cancelled(cancel)
        return service.TopicModelingResult(
            algorithm=config.algorithm,
            enable_dtm=config.enable_dtm,
            topic_df=pd.DataFrame(
                {"Topic": [1, 2], "Count": [2, 1], "Keywords": ["tax", "cup"]}
            ),
            dashboard_assets={},
            evaluation_metrics={
                "Topic Diversity": 0.85,
                "Coherence (C_v)": 0.4321,
                "Coherence (U_mass)": None,
            },
            zip_bytes=zip_of(
                {
                    "document_topics.csv": "Answer,Dominant_Topic\n",
                    "topic_keywords.csv": "Topic,Keywords\n",
                    "run_configuration.txt": "settings",
                }
            ),
            notices=[Notice("info", "Read **3** documents.")],
            overview=OVERVIEWS[config.algorithm]
            if calls.overview == "auto"
            else calls.overview,
        )

    monkeypatch.setattr(service, "run_topic_modeling", run)
    return calls


def upload(client, name, data, **fields):
    return client.post(
        f"{BASE}/topic-modeling/collection",
        data=fields,
        files={"file": (name, data, "application/octet-stream")},
        headers=htmx(client),
    )


def collection_id(response):
    assert response.status_code == 200, response.text[:500]
    return re.search(r'name="collection" value="([0-9a-f]+)"', response.text)[
        1
    ]


def start(client, collection, **fields):
    return client.post(
        f"{BASE}/topic-modeling",
        data={"collection": collection, **fields},
        headers=htmx(client),
    )


def finished(runner, response):
    assert response.status_code == 200, response.text[:500]
    job_id = re.search(r'id="job-([0-9a-f]+)"', response.text).group(1)
    job = runner.get(job_id)
    wait_for(lambda: not job.active)
    return job


def card(client, job):
    return client.get(f"{BASE}/jobs/{job.id}").text


def test_the_page_offers_the_collection_and_the_algorithms(client):
    html = client.get(f"{BASE}/topic-modeling").text
    assert 'data-switch="algorithm"' in html
    assert 'hx-params="not file"' in html
    assert 'hx-params="file,collection"' in html
    assert 'accept=".csv,.xlsx,.zip"' in html
    assert re.search(r'name="algorithm" value="bertopic"[^>]*checked', html)
    for algorithm in service.Algorithm:
        assert algorithm.description in html
    assert "<option selected>English</option>" in html
    assert 'name="bt_reduction" value="umap" checked' in html
    assert "Find topics" in html


def test_a_table_is_read_and_its_columns_offered(client, runner):
    response = upload(client, "answers.csv", CSV)
    key = collection_id(response)
    html = response.text
    assert "answers.csv" in html and "3 rows · 3 columns" in html
    assert "<option selected>Answer</option>" in html
    assert 'name="date_column"' in html
    assert "<td>taxes are too high in our town</td>" in html
    assert runner.draft(key) is not None
    assert runner.jobs(page.PAGE) == []


def test_a_zip_of_texts_needs_no_column(client):
    response = upload(
        client, "texts.zip", zip_of({"a.txt": "one", "b.txt": "two"})
    )
    collection_id(response)
    assert "2 text files" in response.text
    assert 'name="text_column"' not in response.text


def test_another_upload_replaces_the_collection(client, runner):
    first = collection_id(upload(client, "a.csv", CSV))
    second = collection_id(upload(client, "b.csv", CSV, collection=first))
    assert runner.draft(first) is None
    assert runner.draft(second) is not None


def test_a_file_without_documents_is_refused(client, runner):
    response = upload(client, "empty.csv", b"Answer\n\n")
    assert response.status_code == 422
    assert "usable rows" in response.text
    response = upload(client, "notes.docx", b"x")
    assert response.status_code == 422


def test_topics_are_found_and_drawn(client, runner, analysis, released):
    key = collection_id(upload(client, "answers.csv", CSV))
    job = finished(
        runner,
        start(
            client,
            key,
            text_column="Answer",
            algorithm="lda",
            language="German",
            lda_topics="5",
            lda_passes="20",
            lda_stopwords=" town, club ",
            lda_bigrams="1",
            stability="1",
        ),
    )
    assert job.state is JobState.DONE, job.error_details
    (run,) = analysis.runs
    config = run["config"]
    assert run["name"] == "answers.csv"
    assert run["table"]["Answer"].tolist()[0].startswith("taxes")
    assert run["cancel"] is job.cancel
    assert (config.algorithm, config.language) == (
        service.Algorithm.LDA,
        "German",
    )
    assert (config.num_topics, config.passes) == (5, 20)
    assert config.custom_stopwords == "town, club"
    assert config.use_bigrams and config.evaluate_stability
    assert config.enable_dtm is False
    assert released == ["topic_modeling"]

    html = card(client, job)
    assert "Latent Dirichlet Allocation (LDA)" in html
    assert "2 topics" in html
    assert "Read <strong>3</strong> documents." in html
    # pyLDAvis's view: the topic bubbles, sized by their share of words.
    assert 'class="ldavis" data-ldavis ' in html
    assert re.search(
        r'<g class="ldavis__topic" data-ldavis-topic="0"[^>]*>\s*'
        r'<circle cx="85.61" cy="58.9" r="12.39" data-r="12.39"/>',
        html,
    )
    assert '<option value="0">Topic 1 · tax · budget</option>' in html
    assert "data-ldavis-lambda" in html
    # The most salient terms first, each spread over the topics.
    assert 'data-shares="0:0.8 1:0.2"' in html
    # The chosen topic's terms at lambda 1: by probability.
    terms = re.findall(
        r'data-logprob="([^"]+)" data-loglift="[^"]+" data-freq', html
    )
    assert terms[:3] == ["-1.2", "-2.3", "-3.0"]
    assert '<rect class="ldavis__total" width="100.0" height="10"/>' in html
    assert '<rect class="ldavis__freq" width="83.33" height="10"/>' in html
    assert "for topic 1 (60.0% of the words)" in html
    # The documents with their mixtures.
    assert re.search(
        r'<rect class="mixture__part mixture__part--this" x="0.0" '
        r'width="69.4" height="4">',
        html,
    )
    assert "Topic 1 70% · Topic 2 30%" in html
    assert "Topic 1 90% · Other topics 10%" in html
    assert "taxes &lt;b&gt;" in html
    assert "85%" in html and "0.432" in html and "Not computed" in html
    # No charts of the other algorithms.
    assert "topic-dot" not in html and "dendrogram" not in html

    names = [item.name for item in job.listed_downloads]
    assert names == [
        "answers_topics.zip",
        "answers_document_topics.csv",
        "answers_topic_keywords.csv",
        "answers_run_report.txt",
    ]


def test_the_same_collection_can_be_run_again(client, runner, analysis):
    key = collection_id(upload(client, "answers.csv", CSV))
    for topics in ("3", "4"):
        finished(
            runner,
            start(
                client,
                key,
                text_column="Answer",
                algorithm="lda",
                lda_topics=topics,
            ),
        )
    assert [run["config"].num_topics for run in analysis.runs] == [3, 4]
    assert runner.draft(key) is not None
    assert len(runner.jobs(page.PAGE)) == 2


def test_bertopic_settings_are_read(client, runner, analysis):
    key = collection_id(upload(client, "answers.csv", CSV))
    finished(
        runner,
        start(
            client,
            key,
            text_column="Answer",
            algorithm="bertopic",
            bt_clustering="kmeans",
            bt_clusters="7",
            bt_embedding="custom",
            bt_custom_model=" BAAI/bge-m3 ",
            bt_trust_code="1",
            bt_reduction="pca",
            bt_components="8",
            bt_lock_seed="1",
            bt_phrases="1",
            bt_min_df="2",
            over_time="1",
            date_column="Year",
            time_bins="12",
        ),
    )
    config = analysis.runs[0]["config"]
    assert config.clustering_algo == "KMeans"
    assert config.clustering_params == {"n_clusters": 7}
    assert config.num_topics is None
    assert (config.embedding_model_id, config.trust_remote_code) == (
        "BAAI/bge-m3",
        True,
    )
    assert config.dim_reduction_algo == "PCA"
    assert config.dim_params == {"n_components": 8, "random_state": 42}
    assert config.ngram_range == (1, 2) and config.min_df == 2
    assert config.reduce_frequent is False
    assert (config.enable_dtm, config.date_column, config.time_bins) == (
        True,
        "Year",
        12,
    )


def test_bertopic_defaults_find_the_number_of_topics(client, runner, analysis):
    key = collection_id(upload(client, "answers.csv", CSV))
    finished(
        runner,
        start(
            client,
            key,
            text_column="Answer",
            algorithm="bertopic",
            bt_count="auto",
            bt_min_size="5",
            bt_reduce_frequent="1",
            bt_lock_seed="1",
        ),
    )
    config = analysis.runs[0]["config"]
    assert config.num_topics == "auto"
    assert config.clustering_params == {
        "min_cluster_size": 5,
        "min_samples": 5,
    }
    assert config.embedding_model_id is None
    assert config.dim_reduction_algo == "UMAP"
    assert config.dim_params == {
        "n_components": 5,
        "n_neighbors": 15,
        "min_dist": 0.0,
        "random_state": 42,
    }


def test_topics_over_time_are_for_bertopic_only(client, runner, analysis):
    key = collection_id(upload(client, "answers.csv", CSV))
    finished(
        runner,
        start(
            client,
            key,
            text_column="Answer",
            algorithm="top2vec",
            t2v_count="set",
            t2v_topics="6",
            t2v_backend="doc2vec",
            t2v_speed="deep-learn",
            over_time="1",
            date_column="Year",
        ),
    )
    config = analysis.runs[0]["config"]
    assert (config.num_topics, config.top2vec_backend) == (6, "doc2vec")
    assert config.top2vec_speed == "deep-learn"
    assert config.enable_dtm is False and config.date_column is None


def test_without_an_overview_the_topic_table_is_shown(
    client, runner, analysis
):
    analysis.overview = None
    key = collection_id(upload(client, "answers.csv", CSV))
    job = finished(
        runner, start(client, key, text_column="Answer", algorithm="lda")
    )
    html = card(client, job)
    assert "topic-dot" not in html
    assert '<th scope="col">Keywords</th>' in html
    assert "Coherence (C_v)" in html


def test_a_run_can_be_cancelled(client, runner, analysis):
    analysis.gate = threading.Event()
    key = collection_id(upload(client, "answers.csv", CSV))
    response = start(client, key, text_column="Answer", algorithm="lda")
    job_id = re.search(r'id="job-([0-9a-f]+)"', response.text).group(1)
    job = runner.get(job_id)
    wait_for(lambda: len(analysis.runs) == 1)
    client.post(f"{BASE}/jobs/{job.id}/cancel", headers=htmx(client))
    wait_for(lambda: not job.active)
    assert job.state is JobState.CANCELLED
    assert job.downloads == []


@pytest.mark.parametrize(
    "fields,message",
    [
        ({"text_column": "Missing"}, "text column"),
        ({"algorithm": "word2vec"}, "algorithm"),
        ({"algorithm": "lda", "lda_topics": "1"}, "from 2 to 50"),
        ({"algorithm": "lda", "lda_topics": "2.5"}, "whole number"),
        ({"bt_embedding": "custom"}, "Name the embedding model"),
        ({"bt_min_dist": "2"}, "minimum distance must be a number"),
        (
            {"over_time": "1", "date_column": "Answer"},
            "another than the text column",
        ),
    ],
)
def test_bad_settings_are_explained(client, analysis, fields, message):
    key = collection_id(upload(client, "answers.csv", CSV))
    response = start(client, key, **{"text_column": "Answer", **fields})
    assert response.status_code == 422
    assert message in response.text
    assert analysis.runs == []


def test_a_run_needs_a_collection(client, analysis):
    response = start(client, "unknown", text_column="Answer")
    assert response.status_code == 422
    assert "Choose the collection first" in response.text


def view_of(overview, algorithm):
    return topic_charts.TopicsView(
        saved=service.SavedTopics(
            algorithm=algorithm,
            enable_dtm=False,
            columns=(),
            rows=(),
            metrics={},
            notices=(),
            overview=overview,
            files=(),
        ),
        config=service.TopicModelingConfig(
            algorithm=algorithm, language="English", text_column="Answer"
        ),
    )


def test_bertopic_is_drawn_with_its_own_charts(client, runner, analysis):
    key = collection_id(upload(client, "answers.csv", CSV))
    job = finished(
        runner,
        start(client, key, text_column="Answer", algorithm="bertopic"),
    )
    html = card(client, job)
    for tab in (
        "Topics",
        "Topic map",
        "Documents",
        "Hierarchy",
        "Similarity",
        "Over time",
        "Quality",
    ):
        assert f">{tab}</button>" in html
    # The keyword scores of each topic.
    assert '<h4 class="topic-card__title">tax · budget · town</h4>' in html
    assert '<rect width="50.0" height="4"/>' in html
    # The topic map.
    assert re.search(
        r'<g class="box topic-dot" data-region="[^"]+-topic-0">\s*'
        r'<circle cx="15.07" cy="50.0"',
        html,
    )
    # The documents, placed by meaning.
    assert re.search(
        r'<circle class="doc-point" cx="3.0" cy="97.0" r="0.6" '
        r'data-region="[^"]+-topic-0"><title>taxes are</title>',
        html,
    )
    assert 'class="doc-label"' in html
    # The hierarchy, the second topic first.
    assert "300.0,19.0 585.0,19.0 585.0,41.0 300.0,41.0" in html
    assert re.search(r">2 · football · cup<title>", html)
    # The heatmap in the hierarchy's order.
    assert "Topics 2 and 1: 0.25" in html
    # The topics over time, with the words of each interval.
    assert "Topic 1, 2020: 1 document (taxes, rise)" in html
    assert "Topic 2, 2020: 0 documents" in html
    assert 'class="trend" data-region=' in html
    assert "ldavis" not in html


def test_top2vec_is_drawn_with_word_clouds(client, runner, analysis):
    key = collection_id(upload(client, "answers.csv", CSV))
    job = finished(
        runner,
        start(client, key, text_column="Answer", algorithm="top2vec"),
    )
    html = card(client, job)
    assert '<span class="cloud__word cloud__word--5">tax</span>' in html
    assert '<span class="cloud__word cloud__word--3">budget</span>' in html
    assert '<span class="cloud__word cloud__word--1">levy</span>' in html
    assert re.search(r">Sizes</button>", html)
    sizes = re.findall(r'class="size__value">(\d+)<', html)
    assert sizes == ["2", "1"]
    assert ">Similarity</button>" in html
    assert ">Hierarchy</button>" not in html
    assert ">Over time</button>" not in html


def test_the_over_time_chart_shares_one_axis():
    view = view_of(BERTOPIC_OVERVIEW, service.Algorithm.BERTOPIC)
    chart = view.trends("job")
    first, second = chart.trends
    assert first.line == "44.0,12.0 588.0,12.0"
    assert second.line == "44.0,226.0 588.0,12.0"
    assert (second.peak, second.peak_at) == (1, "2021")
    assert [label for _, label in chart.ticks_y] == ["0", "1"]
    assert [anchor for _, _, anchor in chart.ticks_x] == ["start", "end"]
    assert view.mixed is False and view.kind == "bertopic"


def test_the_heatmap_follows_the_hierarchy():
    view = view_of(BERTOPIC_OVERVIEW, service.Algorithm.BERTOPIC)
    heatmap = view.heatmap("job")
    assert heatmap.size == 34
    assert heatmap.labels == ((19.0, 2), (29.0, 1))
    plain = view_of(TOP2VEC_OVERVIEW, service.Algorithm.TOP2VEC)
    assert plain.heatmap("job").labels == ((19.0, 1), (29.0, 2))


def test_the_lda_terms_rank_as_pyldavis_ranks_them():
    view = view_of(LDA_OVERVIEW, service.Algorithm.LDA)
    vis = view.ldavis("job")
    first = vis.topics[0]
    assert [term.term for term in first.terms] == ["tax", "budget", "town"]
    assert first.terms[0].shares == "0:0.8 1:0.2"
    assert first.r == topic_charts.bubble(0.6)
    assert vis.origin == (50.0, 58.9)
    assert vis.height > 100
    assert [label for label, *_ in vis.legend] == ["2%", "5%", "10%"]
    assert vis.salient[1].total_bar == 66.67


def test_a_mixture_bar_has_a_part_per_topic_and_the_rest():
    parts = topic_charts.mixture_bar([[1, 0.5], [0, 0.3]], 0, [1, 2])
    assert [(part.topic, part.this) for part in parts] == [
        (2, False),
        (1, True),
        ("Other", False),
    ]
    assert [part.x for part in parts] == [0.0, 50.0, 80.0]
    assert parts[0].width == 49.4


def test_round_axis_tops():
    assert [
        topic_charts.nice_ceiling(value) for value in (0, 1, 3, 7, 18)
    ] == [
        1,
        1,
        5,
        10,
        20,
    ]
    assert topic_charts.cloud_sizes([["a", 0.5], ["b", 0.5]]) == (
        ("a", 3),
        ("b", 3),
    )


def test_bubbles_stay_inside_the_map():
    for overview, algorithm in (
        (LDA_OVERVIEW, service.Algorithm.LDA),
        (BERTOPIC_OVERVIEW, service.Algorithm.BERTOPIC),
    ):
        for topic in view_of(overview, algorithm).topics("job"):
            assert topic.r <= topic.x <= 100 - topic.r
            assert topic.r <= topic.y <= 100 - topic.r
    assert topic_charts.map_margin([0.01]) == 8.0
    assert topic_charts.map_margin([1.0]) == 18.0
