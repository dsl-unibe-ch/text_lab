"""The Chat service, with the Ollama model replaced by fakes."""

import os
from types import SimpleNamespace

from textlab.common.progress import Progress
from textlab.common.storage import Workspace
from textlab.features.chat import documents, router, service

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


def test_files_without_text_give_no_context():
    assert service.read_documents([("image.json", b"{}")]) == ("", [])
    assert service.read_documents([]) == ("", [])


# --- answers -------------------------------------------------------------


def test_a_short_context_is_put_before_the_question(monkeypatch):
    sent = []

    def respond(model, messages):
        sent.append((model, messages))
        yield "Hi"
        yield " there"

    monkeypatch.setattr(service, "get_response_generator", respond)
    history = [{"role": "user", "content": "Earlier", "analysis": {}}]
    reply = "".join(service.answer("m", history, "What is it?", "### Context"))
    assert reply == "Hi there"
    model, messages = sent[0]
    assert model == "m"
    assert messages == [
        {"role": "user", "content": "Earlier"},
        {
            "role": "user",
            "content": "### Context\n\nUser Question: What is it?",
        },
    ]


def test_a_long_context_is_answered_part_by_part(monkeypatch):
    partial = []

    def chunk_answer(model, chunk, index, total, question, history):
        partial.append((chunk, index, total))
        return f"answer {index}"

    def synthesis(model, answers, question, history):
        yield " + ".join(answers)

    monkeypatch.setattr(service, "MAX_CONTEXT_TOKENS", 1)
    monkeypatch.setattr(service, "chunk_text", lambda text: ["one", "two"])
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


# --- routing -------------------------------------------------------------


def tool_call(arguments):
    function = {"name": "analyze_data", "arguments": arguments}
    return {"message": {"content": "", "tool_calls": [{"function": function}]}}


def test_a_data_question_goes_to_the_agents(monkeypatch):
    monkeypatch.setattr(
        router,
        "chat_no_think",
        lambda **request: tool_call('{"instruction": "Plot age"}'),
    )
    assert service.decide_tool_use("m", "plot age", "age: int") == (
        True,
        "Plot age",
    )


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
