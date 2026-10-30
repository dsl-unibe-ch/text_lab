"""The Chat service, with the Ollama model replaced by fakes."""

import os
from types import SimpleNamespace

import fitz
import pytest

from textlab.common.progress import Progress
from textlab.common.storage import Workspace
from textlab.features.chat import (
    context_window,
    documents,
    generation,
    router,
    service,
)

# --- documents -----------------------------------------------------------


def test_documents_become_one_context_block():
    context, warnings = service.read_documents(
        [("notes.txt", b"Hello."), ("data.csv", b"a,b\n1,2\n")]
    )
    assert warnings == []
    assert "--- Start of file: notes.txt ---\nHello." in context
    assert "--- End of file: data.csv ---" in context
    assert "|---:|" in context  # the table as Markdown


def test_large_files_are_skipped_with_a_warning(monkeypatch):
    monkeypatch.setattr(documents, "MAX_FILE_MB", 0)
    context, warnings = service.read_documents([("big.txt", b"x")])
    assert context == ""
    assert warnings and "big.txt" in warnings[0]


def test_no_files_give_no_context():
    assert service.read_documents([]) == ("", [])


def pdf_bytes(text=""):
    with fitz.open() as doc:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
        return doc.tobytes()


def test_a_pdf_is_read_as_text():
    context, warnings = service.read_documents(
        [("paper.pdf", pdf_bytes("Results section"))]
    )
    assert warnings == []
    assert "Results section" in context


def test_a_scanned_pdf_is_skipped_with_a_warning():
    context, warnings = service.read_documents([("scan.pdf", pdf_bytes())])
    assert context == ""
    assert len(warnings) == 1 and "scan.pdf" in warnings[0]
    assert "OCR" in warnings[0]


def test_unreadable_files_are_skipped_with_a_warning():
    context, warnings = service.read_documents(
        [
            ("broken.pdf", b"not a pdf"),
            ("latin1.txt", "caf\xe9".encode("latin-1")),
            ("bad.json", b"{not json"),
            ("empty.txt", b"  \n"),
            ("notes.txt", b"Kept."),
        ]
    )
    assert "Kept." in context
    for name in ("broken.pdf", "latin1.txt", "bad.json", "empty.txt"):
        assert name not in context
        assert any(name in warning for warning in warnings)
    assert len(warnings) == 4


def test_unsupported_files_are_skipped_with_a_warning():
    context, warnings = service.read_documents([("image.png", b"\x89PNG")])
    assert context == ""
    assert warnings and "image.png" in warnings[0]


def test_tsv_and_json_files_are_read():
    context, warnings = service.read_documents(
        [
            ("data.tsv", b"name\tage\nAda\t36\n"),
            ("config.json", b'{"study": "pilot"}'),
        ]
    )
    assert warnings == []
    assert "| name" in context and "Ada" in context  # TSV as a table
    assert '{"study": "pilot"}' in context


def test_long_tables_are_truncated():
    rows = "".join(f"{i}\n" for i in range(documents.MAX_TABLE_ROWS + 5))
    context, _ = service.read_documents([("n.csv", ("n\n" + rows).encode())])
    assert f"First {documents.MAX_TABLE_ROWS} of" in context


def test_every_document_type_can_be_uploaded():
    assert set(service.DOCUMENT_TYPES) == {
        "pdf",
        "txt",
        "json",
        "csv",
        "tsv",
        "xls",
        "xlsx",
    }


# --- context window ------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", context_window.DEFAULT_CHAT_CONTEXT),
        ("65536", 65536),
        ("abc", context_window.DEFAULT_CHAT_CONTEXT),
        ("1024", context_window.MIN_CHAT_CONTEXT),
    ],
)
def test_the_context_window_can_be_configured(monkeypatch, value, expected):
    monkeypatch.setenv("TEXTLAB_CHAT_NUM_CTX", value)
    assert context_window.chat_context() == expected


def test_the_text_budget_leaves_room_for_the_answer(monkeypatch):
    monkeypatch.setenv("TEXTLAB_CHAT_NUM_CTX", "32768")
    budget = context_window.text_budget()
    assert 0 < budget
    assert (
        budget * context_window.ESTIMATE_MARGIN
        + context_window.RESPONSE_TOKENS
        <= 32768
    )


def test_history_keeps_the_newest_messages_that_fit():
    history = [
        {"role": "user", "content": "a" * 400},  # 100 tokens
        {"role": "assistant", "content": "b" * 400},
        {"role": "user", "content": "c" * 400},
    ]
    assert context_window.trim_history(history, 250) == history[1:]
    assert context_window.trim_history(history, 1000) == history
    assert context_window.trim_history(history, 50) == []


def usage(monkeypatch, history_tokens, document_tokens, budget=1000):
    monkeypatch.setattr(context_window, "text_budget", lambda: budget)
    history = [{"role": "user", "content": "h" * 4 * history_tokens}]
    return service.context_usage(
        history if history_tokens else [], "d" * 4 * document_tokens
    )


def test_the_usage_counts_the_conversation_and_the_documents(monkeypatch):
    result = usage(monkeypatch, 300, 200)
    assert (result.used_tokens, result.budget_tokens) == (500, 1000)
    assert result.history_tokens == 300
    assert result.fraction == 0.5
    assert result.window_tokens == context_window.chat_context()
    assert not (result.documents_split or result.history_trimmed)
    assert not result.suggest_new_chat


def test_a_nearly_full_conversation_suggests_a_new_chat(monkeypatch):
    assert usage(monkeypatch, 800, 0).suggest_new_chat
    trimmed = usage(monkeypatch, 800, 400)
    assert trimmed.history_trimmed and trimmed.suggest_new_chat


def test_long_documents_alone_do_not_suggest_a_new_chat(monkeypatch):
    # A new chat would send the documents again: it cannot help.
    assert not usage(monkeypatch, 0, 900).suggest_new_chat
    split = usage(monkeypatch, 100, 5000)
    assert split.documents_split
    assert not (split.history_trimmed or split.suggest_new_chat)
    # Read in parts, the conversation may use a quarter of the budget.
    crowded = usage(monkeypatch, 300, 5000)
    assert crowded.history_trimmed and crowded.suggest_new_chat


def test_texts_are_grouped_by_budget():
    texts = ["a" * 40, "b" * 40, "c" * 40, "d" * 400]  # 10, 10, 10, 100
    assert context_window.group_by_budget(texts, 25) == [
        texts[:2],
        [texts[2]],
        [texts[3]],
    ]


# --- answers -------------------------------------------------------------


def test_a_short_context_is_put_before_the_question(monkeypatch):
    sent = []

    def respond(model, messages, num_ctx):
        sent.append((model, messages, num_ctx))
        yield "Hi"
        yield " there"

    monkeypatch.setattr(service, "get_response_generator", respond)
    monkeypatch.setenv("TEXTLAB_CHAT_NUM_CTX", "16384")
    history = [{"role": "user", "content": "Earlier", "analysis": {}}]
    reply = "".join(service.answer("m", history, "What is it?", "### Context"))
    assert reply == "Hi there"
    model, messages, num_ctx = sent[0]
    assert (model, num_ctx) == ("m", 16384)
    assert messages == [
        {"role": "user", "content": "Earlier"},
        {
            "role": "user",
            "content": "### Context\n\nUser Question: What is it?",
        },
    ]


def test_old_turns_that_do_not_fit_are_left_out(monkeypatch):
    sent = []

    def respond(model, messages, num_ctx):
        sent.append(messages)
        yield "ok"

    monkeypatch.setattr(service, "get_response_generator", respond)
    monkeypatch.setattr(service, "text_budget", lambda: 300)
    history = [
        {"role": "user", "content": "old " * 400},  # 400 tokens
        {"role": "assistant", "content": "recent"},
    ]
    "".join(service.answer("m", history, "Q?"))
    assert sent[0] == [
        {"role": "assistant", "content": "recent"},
        {"role": "user", "content": "Q?"},
    ]


def test_a_context_that_fits_is_not_split(monkeypatch):
    sent = []

    def respond(model, messages, num_ctx):
        sent.append(messages)
        yield "ok"

    def fail(*args, **kwargs):
        raise AssertionError("the context was split")

    monkeypatch.setattr(service, "get_response_generator", respond)
    monkeypatch.setattr(service, "get_chunk_answer", fail)
    context = "x" * 40_000  # 10,000 tokens, within the default budget
    "".join(service.answer("m", [], "Summarize", context))
    assert context in sent[0][-1]["content"]


def test_a_long_context_is_answered_part_by_part(monkeypatch):
    partial = []
    windows = set()

    def chunk_answer(model, chunk, index, total, question, history, num_ctx):
        partial.append((chunk, index, total))
        windows.add(num_ctx)
        return f"answer {index}"

    def synthesis(model, answers, question, history, num_ctx):
        windows.add(num_ctx)
        yield " + ".join(answers)

    monkeypatch.setattr(service, "text_budget", lambda: 1)
    monkeypatch.setattr(
        service, "chunk_text", lambda text, chunk_size_tokens: ["one", "two"]
    )
    monkeypatch.setattr(service, "get_chunk_answer", chunk_answer)
    monkeypatch.setattr(service, "get_synthesis_generator", synthesis)
    updates = []
    reply = "".join(
        service.answer(
            "m", [], "Q?", "a long context", on_progress=updates.append
        )
    )
    assert reply == "answer 1 + answer 2"
    assert partial == [("one", 1, 2), ("two", 2, 2)]
    assert [u.message for u in updates] == [
        "Analyzing document part 1 of 2 (~3 tokens total)...",
        "Analyzing document part 2 of 2 (~3 tokens total)...",
        "Synthesizing responses from 2 chunks...",
    ]
    assert updates[-1] == Progress("Synthesizing responses from 2 chunks...")
    assert windows == {context_window.chat_context()}


def test_parts_fill_the_budget(monkeypatch):
    sizes = []

    def chunk_text(text, chunk_size_tokens):
        sizes.append(chunk_size_tokens)
        return ["part"]

    monkeypatch.setattr(service, "text_budget", lambda: 1000)
    monkeypatch.setattr(service, "chunk_text", chunk_text)
    monkeypatch.setattr(
        service, "get_chunk_answer", lambda *args: "partial answer"
    )
    monkeypatch.setattr(
        service,
        "get_synthesis_generator",
        lambda model, answers, question, history, num_ctx: iter(answers),
    )
    history = [{"role": "user", "content": "h" * 400}]  # 100 tokens
    "".join(service.answer("m", history, "q" * 40, "x" * 8000))
    assert sizes == [1000 - 100 - 10]


def test_partial_answers_too_long_together_are_combined(monkeypatch):
    combined = []

    def combine(model, answers, question, history, num_ctx):
        combined.append(list(answers))
        return "c" * 40  # 10 tokens

    def synthesis(model, answers, question, history, num_ctx):
        yield "|".join(answers)

    monkeypatch.setattr(service, "text_budget", lambda: 70)
    monkeypatch.setattr(
        service, "chunk_text", lambda text, chunk_size_tokens: ["1", "2", "3"]
    )
    monkeypatch.setattr(service, "get_chunk_answer", lambda *args: "a" * 120)
    monkeypatch.setattr(service, "get_combined_answer", combine)
    monkeypatch.setattr(service, "get_synthesis_generator", synthesis)
    reply = "".join(service.answer("m", [], "", "x" * 1000))
    # Three 30-token answers exceed the budget (69 tokens beside the
    # question): the first two are combined.
    assert combined == [["a" * 120, "a" * 120]]
    assert reply == "c" * 40 + "|" + "a" * 120


# --- generation ----------------------------------------------------------


def test_every_model_call_sets_the_context_window(monkeypatch):
    calls = []

    def chat(**request):
        calls.append(request)
        message = {"content": "text"}
        if request["stream"]:
            return iter([{"message": message}])
        return {"message": message}

    monkeypatch.setattr(generation.ollama, "chat", chat)
    history = [{"role": "user", "content": "h"}]
    assert "".join(generation.get_response_generator("m", history, 4)) == (
        "text"
    )
    assert generation.get_chunk_answer("m", "c", 1, 2, "q", [], 5) == "text"
    assert generation.get_combined_answer("m", ["a"], "q", [], 6) == "text"
    assert "".join(generation.get_synthesis_generator("m", ["a"], "q", [], 7))
    assert [c["options"]["num_ctx"] for c in calls] == [4, 5, 6, 7]
    assert all(c["messages"][0]["role"] == "system" for c in calls)


# --- routing -------------------------------------------------------------


def tool_call(arguments):
    function = {"name": "analyze_data", "arguments": arguments}
    return {"message": {"content": "", "tool_calls": [{"function": function}]}}


def test_a_data_question_goes_to_the_agents(monkeypatch):
    requests = []

    def chat(**request):
        requests.append(request)
        return tool_call('{"instruction": "Plot age"}')

    monkeypatch.setattr(router, "chat_no_think", chat)
    assert service.decide_tool_use("m", "plot age", "age: int") == (
        True,
        "Plot age",
    )
    # The same window as the answers, so routing does not reload the model.
    assert requests[0]["options"]["num_ctx"] == context_window.chat_context()


def test_an_empty_instruction_falls_back_to_the_message(monkeypatch):
    monkeypatch.setattr(
        router, "chat_no_think", lambda **request: tool_call({})
    )
    assert service.decide_tool_use("m", " plot it ", "") == (True, "plot it")


def test_plain_questions_and_errors_stay_in_the_chat(monkeypatch):
    monkeypatch.setattr(
        router,
        "chat_no_think",
        lambda **request: {"message": {"content": "Hello"}},
    )
    assert service.decide_tool_use("m", "hi", "") == (False, "")

    def fail(**request):
        raise RuntimeError("no tool support")

    monkeypatch.setattr(router, "chat_no_think", fail)
    assert service.decide_tool_use("m", "plot", "") == (False, "")


# --- exports -------------------------------------------------------------


def conversation():
    return [
        {"role": "user", "content": "Plot age"},
        {
            "role": "assistant",
            "content": "Done.",
            "analysis": {
                "artifacts": [
                    {
                        "filename": "hist.png",
                        "bytes": b"\x89PNG",
                        "code": "plt.hist()",
                        "tool_name": "plot_static_histogram",
                        "fig_json": None,
                    }
                ],
                "stats": [
                    {
                        "title": "t-test",
                        "result": "p = 0.1",
                        "code": "print(1)",
                    }
                ],
                "run_id": "chat-1",
            },
        },
    ]


def test_exports_include_charts_and_statistics():
    messages = conversation()
    assert service.has_analysis_plots(messages)
    markdown = service.format_chat_history(messages)
    assert "### User\nPlot age" in markdown
    assert "data:image/png;base64," in markdown
    assert "**t-test**" in markdown and "```python\nprint(1)" in markdown
    html = service.format_chat_history_html(messages)
    assert "<img src='data:image/png;base64," in html
    assert not service.has_analysis_plots(messages[:1])


# --- conversation folders and analysis -----------------------------------


def test_a_conversation_folder_is_private_and_removable(tmp_path, monkeypatch):
    workspace = Workspace(tmp_path / "workspace")
    monkeypatch.setattr(service, "get_workspace", lambda: workspace)
    folder = service.conversation_folder()
    assert os.path.dirname(folder) == str(workspace.dir("chat"))
    service.discard_conversation_folder(folder)
    assert not os.path.exists(folder)


def test_a_data_question_starts_an_agent_run(monkeypatch):
    started = []

    def start(request, *, data_path, run_prefix):
        started.append((request, data_path, run_prefix))
        return SimpleNamespace(run_id="chat-1")

    monkeypatch.setattr(service.visualization, "start_analysis", start)
    run = service.start_data_analysis("Plot age", "/w/data.csv", "m")
    ((request, data_path, prefix),) = started
    assert (request.model, request.prompt) == ("m", "Plot age")
    assert (data_path, prefix, run.run_id) == ("/w/data.csv", "chat", "chat-1")
