"""The Topic Modeling service and worker, with the modeling replaced by fakes.

The pipeline and evaluation modules import BERTopic, Top2Vec and gensim,
which take half a minute to load, so these tests put fakes in their place.
"""

import io
import pickle
import sys
import zipfile
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

import textlab.features.topic_modeling as package
from textlab.common.jobs import WorkerError
from textlab.common.progress import Progress
from textlab.common.storage import Workspace
from textlab.features.topic_modeling import embeddings, service, worker
from textlab.features.topic_modeling.models import (
    Algorithm,
    Notice,
    TopicModelingConfig,
    TopicModelingResult,
)


def make_result(**changes):
    values = dict(
        algorithm=Algorithm.BERTOPIC,
        enable_dtm=False,
        topic_df=pd.DataFrame(
            {
                "Topic": ["Outlier", 1],
                "Count": [3, None],
                "Keywords": ["a, b", "c"],
            }
        ),
        dashboard_assets={"topic_barchart.html": "<html></html>"},
        evaluation_metrics={"Topic Diversity": 0.5, "Coherence (C_v)": None},
        zip_bytes=b"zip",
        notices=[Notice("warning", "careful")],
    )
    values.update(changes)
    return TopicModelingResult(**values)


# --- models --------------------------------------------------------------


def test_a_configuration_survives_the_trip_to_the_worker():
    config = TopicModelingConfig(
        algorithm=Algorithm.TOP2VEC,
        language="German",
        text_column="Text",
        ngram_range=(1, 2),
        clustering_params={"min_cluster_size": 5},
        evaluate_stability=True,
    )
    data = pickle.loads(pickle.dumps(config.to_dict()))
    assert TopicModelingConfig.from_dict(data) == config


def test_a_result_survives_the_trip_from_the_worker():
    result = make_result()
    rebuilt = TopicModelingResult.from_dict(result.to_dict(), b"zip")
    assert rebuilt.algorithm is Algorithm.BERTOPIC
    assert rebuilt.topic_df["Topic"].tolist() == ["Outlier", 1]
    assert rebuilt.topic_df.columns.tolist() == ["Topic", "Count", "Keywords"]
    assert rebuilt.notices == result.notices
    assert rebuilt.evaluation_metrics == result.evaluation_metrics


# --- loading -------------------------------------------------------------


def test_an_empty_table_is_refused():
    with pytest.raises(ValueError, match="usable rows"):
        service.load_table("data.csv", b"Text\n\n")


def test_fully_empty_rows_are_dropped():
    table = service.load_table("data.csv", b"Text,Year\nhello,2020\n,\n")
    assert len(table) == 1


def test_an_archive_without_text_files_is_refused():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("empty.txt", "   ")
    with pytest.raises(ValueError, match="non-empty .txt"):
        service.load_archive(buffer.getvalue())


# --- analysis ------------------------------------------------------------


@pytest.fixture
def modeling(monkeypatch):
    """Fake pipeline, evaluation and embedding model; record the calls."""
    calls = []

    def run_pipeline(df, config, **kwargs):
        calls.append(("pipeline", len(df), kwargs))
        return {
            "topic_df": pd.DataFrame(
                {"Topic": [1], "Count": [len(df)], "Keywords": ["tax"]}
            ),
            "docs_df": df.assign(Dominant_Topic=1),
            "dashboard_assets": {"topic_barchart.html": "<html>bars</html>"},
            "topic_keywords": [["tax"]],
        }

    def stability(df, config, keywords, **kwargs):
        calls.append(("stability", keywords, kwargs))
        return 0.75

    fake_pipeline = ModuleType("pipeline")
    fake_pipeline.run_topic_modeling_pipeline = run_pipeline
    fake_pipeline.evaluate_topic_stability = stability
    fake_evaluation = ModuleType("evaluation")
    fake_evaluation.evaluate_run = lambda run, texts, config: {
        "Topic Diversity": 1.0
    }
    for name, module in (
        ("pipeline", fake_pipeline),
        ("evaluation", fake_evaluation),
    ):
        monkeypatch.setitem(
            sys.modules, f"textlab.features.topic_modeling.{name}", module
        )
        monkeypatch.setattr(package, name, module, raising=False)

    model = SimpleNamespace(name="model")
    monkeypatch.setattr(
        embeddings, "load_embedding_model", lambda config: (model, "mini")
    )
    monkeypatch.setattr(
        embeddings,
        "long_document_notice",
        lambda *args: Notice("info", "long documents"),
    )
    monkeypatch.setattr(
        embeddings,
        "embed_documents",
        lambda m, texts, chunk_long_documents: ["vector"] * len(texts),
    )
    return calls


def bertopic_config(**changes):
    return TopicModelingConfig(
        algorithm=Algorithm.BERTOPIC,
        language="English",
        text_column="Text",
        **changes,
    )


def test_a_bertopic_run_embeds_once_and_reports_what_happened(modeling):
    table = pd.DataFrame(
        {
            "Text": ["a", "b", None, "c", "d"],
            "Date": [
                "2020-01-01",
                "not a date",
                "2020-02-01",
                "2021-01-01",
                "2022-01-01",
            ],
        }
    )
    config = bertopic_config(
        enable_dtm=True, date_column="Date", evaluate_stability=True
    )
    updates = []
    result = service.analyze_table(
        table, config, source_name="data.csv", on_progress=updates.append
    )

    # The row without text and the row without a date are dropped.
    (name, rows, kwargs), (stab, keywords, stab_kwargs) = modeling
    assert (name, rows) == ("pipeline", 3)
    assert len(kwargs["timestamps"]) == 3
    assert kwargs["precomputed_embeddings"] == ["vector"] * 3
    assert (
        stab_kwargs["precomputed_embeddings"]
        is kwargs["precomputed_embeddings"]
    )
    assert keywords == [["tax"]]

    assert [update.message for update in updates] == [
        "Loading embedding model 'all-MiniLM-L6-v2'...",
        "Encoding documents with the embedding model...",
        "Running topic extraction (Run 1/3)...",
        "Calculating Topic Coherence and Diversity...",
        "Running stability iterations 2-3 (unlocked seeds) and comparing "
        "topics...",
    ]
    assert result.notices == [
        Notice(
            "warning",
            "1 rows were skipped because 'Date' could not be parsed as a "
            "date/time.",
        ),
        Notice("info", "long documents"),
    ]
    assert result.evaluation_metrics == {
        "Topic Diversity": 1.0,
        "Topic Stability": 0.75,
    }
    names = zipfile.ZipFile(io.BytesIO(result.zip_bytes)).namelist()
    assert sorted(names) == [
        "document_topics.csv",
        "run_configuration.txt",
        "topic_barchart.html",
        "topic_keywords.csv",
    ]


def test_an_lda_run_loads_no_embedding_model(modeling, monkeypatch):
    def fail(config):
        raise AssertionError("LDA needs no embedding model")

    monkeypatch.setattr(embeddings, "load_embedding_model", fail)
    config = TopicModelingConfig(
        algorithm=Algorithm.LDA, language="English", text_column="Text"
    )
    table = pd.DataFrame({"Text": list("abcde")})
    result = service.analyze_table(table, config, source_name="data.csv")
    assert modeling[0][2]["precomputed_embeddings"] is None
    assert result.notices == []
    assert result.algorithm is Algorithm.LDA


# --- worker --------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    workspace = Workspace(tmp_path / "workspace")
    monkeypatch.setattr(service, "get_workspace", lambda: workspace)
    return workspace


def test_a_run_hands_the_table_to_the_worker(workspace, monkeypatch):
    requests = []

    def fake_run_worker(module, request, *, area, on_progress, cancel):
        requests.append((module, area))
        table = pd.read_pickle(request["table"])
        assert table["Text"].tolist() == ["a", "b"]
        assert request["config"]["algorithm"] == "bertopic"
        on_progress(Progress("Running topic extraction..."))
        with open(request["zip"], "wb") as file:
            file.write(b"the zip")
        return make_result().to_dict()

    monkeypatch.setattr(service, "run_worker", fake_run_worker)
    updates = []
    result = service.run_topic_modeling(
        pd.DataFrame({"Text": ["a", "b"]}),
        bertopic_config(),
        source_name="data.csv",
        on_progress=updates.append,
    )
    assert requests == [(service.WORKER_MODULE, "topic_modeling")]
    assert result.zip_bytes == b"the zip"
    assert [u.message for u in updates] == [
        "Starting topic modeling worker...",
        "Running topic extraction...",
    ]
    assert list(workspace.dir("topic_modeling").iterdir()) == []


def test_a_failure_users_can_act_on_keeps_its_message(workspace, monkeypatch):
    def fake_run_worker(module, request, **kwargs):
        raise WorkerError(
            "ValueError: too few documents",
            "Traceback",
            error_type="ValueError",
            error_message="too few documents",
        )

    monkeypatch.setattr(service, "run_worker", fake_run_worker)
    with pytest.raises(ValueError, match="^too few documents$"):
        service.run_topic_modeling(
            pd.DataFrame({"Text": ["a"]}), bertopic_config(), source_name="x"
        )


def test_other_failures_keep_the_worker_traceback(workspace, monkeypatch):
    def fake_run_worker(module, request, **kwargs):
        raise WorkerError("RuntimeError: CUDA", "Traceback", error_type="X")

    monkeypatch.setattr(service, "run_worker", fake_run_worker)
    with pytest.raises(WorkerError) as error:
        service.run_topic_modeling(
            pd.DataFrame({"Text": ["a"]}), bertopic_config(), source_name="x"
        )
    assert error.value.details == "Traceback"


def test_the_worker_writes_the_zip_and_returns_the_rest(tmp_path, monkeypatch):
    table_path = tmp_path / "table.pkl"
    pd.DataFrame({"Text": ["a"]}).to_pickle(table_path)
    seen = []

    def fake_analyze(table, config, *, source_name, on_progress):
        seen.append((len(table), config.algorithm, source_name))
        return make_result()

    monkeypatch.setattr(worker, "analyze_table", fake_analyze)
    data = worker.handle(
        {
            "table": str(table_path),
            "zip": str(tmp_path / "out.zip"),
            "source_name": "data.csv",
            "config": bertopic_config().to_dict(),
        },
        lambda update: None,
    )
    assert seen == [(1, Algorithm.BERTOPIC, "data.csv")]
    assert (tmp_path / "out.zip").read_bytes() == b"zip"
    assert "zip_bytes" not in data
