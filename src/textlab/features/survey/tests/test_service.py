"""The Survey service, with the batch reader replaced by fakes."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from textlab.common.progress import Progress
from textlab.features.survey import service


def make_template(warning=None):
    provenance = {"small_batch_warning": warning} if warning else {}
    return SimpleNamespace(
        control_count=6,
        rules={"q1": "single", "q2": "multiple"},
        provenance=provenance,
    )


@pytest.fixture
def batch_reader(monkeypatch):
    """Replace survey_batch with fakes that record their calls."""
    calls = []
    fake = service.survey_batch

    def prepare_template(files, *, label, progress, vl_session):
        calls.append(("prepare", list(files), label, vl_session))
        progress(0.5, "Synthesizing the blank form...")
        progress(1.5, "out of range")
        return make_template("few copies"), ["blank"]

    def read_document(path, template):
        calls.append(("read", Path(path).name))
        return SimpleNamespace(document=Path(path).name, export_directory=None)

    def to_form_groups(reading, template, document):
        document.groups = ["answers"]

    def slim_document(document):
        document.slim = True
        return document

    for name, function in {
        "prepare_template": prepare_template,
        "read_document": read_document,
        "answers_for_document": lambda reading, template: "csv",
        "safe_csv": lambda frame, path: Path(path).write_text(frame),
        "to_form_groups": to_form_groups,
        "survey_table_regions": lambda document, template: {"p1_r3"},
        "slim_document": slim_document,
        "drop_controls": lambda template, ids: len(ids),
        "rebuild_exports": lambda zip_bytes, template, readings, documents: (
            zip_bytes + b"+rebuilt",
            {"controls": 4, "documents": len(documents)},
        ),
    }.items():
        monkeypatch.setattr(fake, name, function)
    return calls


def test_the_form_is_learned_from_the_batch(batch_reader):
    updates = []
    session = object()
    batch = service.QuestionnaireBatch.learn(
        [Path("a.pdf"), Path("b.pdf")],
        on_progress=updates.append,
        vl_session=session,
    )
    assert batch_reader == [
        ("prepare", [Path("a.pdf"), Path("b.pdf")], True, session)
    ]
    assert updates == [
        Progress("Questionnaire layout — reading 2 file(s)...", 0.0),
        Progress("Questionnaire layout — Synthesizing the blank form...", 0.5),
        Progress("Questionnaire layout — out of range", 0.99),
        Progress("Questionnaire layout — 6 response controls in 2 answers"),
    ]
    assert batch.warning == "few copies"
    assert (batch.control_count, batch.answer_count) == (6, 2)


def test_a_questionnaire_is_read_into_its_folder(batch_reader, tmp_path):
    batch = service.QuestionnaireBatch(template=make_template())
    document = SimpleNamespace()
    skip = batch.read(Path("in/a.pdf"), document, tmp_path, "scans/a")
    assert skip == {"p1_r3"}
    assert batch.readings[0].export_directory == "scans/a"
    assert (tmp_path / service.ANSWERS_FILE).read_text() == "csv"
    assert document.groups == ["answers"]
    assert batch.warning is None

    batch.keep_document("scans/a", document)
    assert batch.documents == {"scans/a": document}
    assert document.slim


def test_dropping_controls_rebuilds_the_exports(batch_reader):
    batch = service.QuestionnaireBatch(
        template=make_template(), documents={"a": object()}
    )
    zip_bytes, removed, summary = batch.drop_controls(["c1", "c2"], b"zip")
    assert (zip_bytes, removed) == (b"zip+rebuilt", 2)
    assert summary == {"controls": 4, "documents": 1}
