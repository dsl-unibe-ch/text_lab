"""Topics from an LLM, with the model replaced by a fake client."""

import json
from types import SimpleNamespace

import pytest

from textlab.common.config import Settings
from textlab.features.knowledge_graph import service, topics


class FakeClient:
    """An OpenAI-compatible client that returns prepared answers."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.requests = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )

    def create(self, **request):
        self.requests.append(request)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        message = SimpleNamespace(content=answer)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


ANSWER = json.dumps(
    {
        "topics": [
            {
                "category": "CS",
                "label": "Graphs",
                "confidence": 2,
                "rationale": " Why. ",
            }
        ]
    }
)


def test_topics_are_cleaned():
    raw = (
        "```json\n"
        '{"topics": [{"label": " Graphs ", "confidence": "high"},'
        ' {"category": "CS", "label": ""}, "not a topic",'
        ' {"category": "CS", "label": "Trees", "confidence": -1}]}\n```'
    )
    assert topics.parse_topics(raw) == [
        {
            "category": "General",
            "label": "Graphs",
            "confidence": 0.0,
            "rationale": "",
        },
        {
            "category": "CS",
            "label": "Trees",
            "confidence": 0.0,
            "rationale": "",
        },
    ]


def test_text_around_the_json_and_too_many_topics():
    items = [{"label": f"T{i}", "confidence": 0.5} for i in range(10)]
    raw = "Here you go: " + json.dumps({"topics": items}) + " Done."
    assert [t["label"] for t in topics.parse_topics(raw)] == [
        f"T{i}" for i in range(topics.MAX_TOPICS)
    ]


@pytest.mark.parametrize("raw", ["no json", '{"labels": []}', "[1, 2]"])
def test_answers_without_topics_are_rejected(raw):
    with pytest.raises(ValueError):
        topics.parse_topics(raw)


def test_the_messages_hold_the_prompt_and_the_paper():
    system, user = topics.build_messages(" Title ", " Abstract ")
    assert system == {"role": "system", "content": topics.SYSTEM_PROMPT}
    assert user["content"] == "TITLE: Title\n\nABSTRACT_OR_TEXT:\nAbstract"


def test_a_failed_call_is_retried():
    client = FakeClient(RuntimeError("busy"), "answer")
    assert topics.run_llm([], client, model="m") == "answer"
    assert client.requests[0]["model"] == "m"
    assert len(client.requests) == 2


def test_the_last_failure_is_reported():
    client = FakeClient(*[RuntimeError("down")] * 3)
    with pytest.raises(RuntimeError, match="after 3 attempts: down"):
        topics.run_llm([], client)


def write_table(corpus, records):
    (corpus / service.TABLE_JSONL).write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )


def test_every_paper_with_an_abstract_gets_topics(tmp_path):
    write_table(
        tmp_path,
        [
            {"paper_id": "P0001", "title": "T", "abstract": "About graphs."},
            {"paper_id": "P0002", "title": "U", "abstract": None},
        ],
    )
    client = FakeClient(ANSWER)
    updates = []
    summary = service.extract_topics(
        tmp_path, client, "m", on_progress=updates.append
    )
    assert (summary.processed, summary.skipped) == (1, 1)
    assert summary.path == tmp_path / service.TOPICS_JSONL
    first, second = service.read_records(summary.path)
    assert first["topics"] == [
        {
            "category": "CS",
            "label": "Graphs",
            "confidence": 1.0,
            "rationale": "Why.",
        }
    ]
    assert "topics_ts" in first
    assert second["topics"] == [] and second["topics_note"] == "no_abstract"
    assert [u.message for u in updates] == [
        "Processing 1/2: P0001 - T...",
        "Skipping 2/2: P0002 (no abstract)",
    ]
    assert not (tmp_path / "corpus_table.with_topics.tmp.jsonl").exists()


def test_topics_need_a_corpus_table(tmp_path):
    with pytest.raises(FileNotFoundError):
        service.extract_topics(tmp_path, FakeClient())


def test_the_clients_point_at_ollama_and_gpustack(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:16000")
    assert str(service.ollama_client().base_url).startswith(
        "http://127.0.0.1:16000/v1"
    )
    settings = Settings(gpustack_url="https://gpustack.example/v1")
    monkeypatch.setattr(topics, "get_settings", lambda: settings)
    client = service.gpustack_client("secret")
    assert str(client.base_url).startswith("https://gpustack.example/v1")
    assert client.api_key == "secret"
