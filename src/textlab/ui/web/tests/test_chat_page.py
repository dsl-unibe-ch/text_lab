"""The Chat page: conversations, replies written as jobs, files, exports.

The model is faked (``service.respond``): it answers with the question in
Markdown, can be held at a gate while it shows a draft, and stops when the
reply is cancelled. Reading attachments, keeping tables, the context meter
and the exports run for real.
"""

import os
import re
import threading
from types import SimpleNamespace

import fitz
import pytest

from textlab.common.progress import Progress
from textlab.features.chat import service
from textlab.ui.web.jobs import JobState
from textlab.ui.web.render import shorten
from textlab.ui.web.tests.conftest import BASE, wait_for
from textlab.ui.web.tests.test_job_pages import htmx

MODELS = service.ModelOptions(
    ("small:8b", "big:30b"), "NVIDIA RTX 4090", False
)

CHARTS = {
    "artifacts": [
        {
            "filename": "scatter.json",
            "bytes": b"{}",
            "code": "px.scatter(df)",
            "r_code": "",
            "tool_name": "plot_interactive_scatter",
            "fig_json": '{"data": [], "layout": {}}',
        },
        {
            "filename": "hist.png",
            "bytes": b"\x89PNG fake",
            "code": "plt.hist(df.age)",
            "r_code": "hist(df$age)",
            "tool_name": "plot_static_histogram",
            "fig_json": None,
        },
    ],
    "stats": [
        {"title": "t-test", "result": "p = **0.04**", "code": "ttest()"}
    ],
    "run_id": "chat-1",
}


@pytest.fixture
def model(monkeypatch, tmp_path):
    """Fake the model; record what each reply was asked with."""
    calls = SimpleNamespace(asked=[], gate=None, reply=None, error=None)

    def respond(
        model,
        history,
        question,
        *,
        context="",
        table=None,
        on_progress,
        on_draft,
        cancel,
    ):
        calls.asked.append(
            SimpleNamespace(
                model=model,
                history=history,
                question=question,
                context=context,
                table=table,
            )
        )
        on_progress(Progress("Waiting for the model"))
        if calls.gate is not None:
            on_draft(service.Draft(thinking="Let me think", text="**So far**"))
            while not calls.gate.wait(0.01):
                if cancel.is_set():
                    return service.Reply(
                        model, "**So far**", stopped=True, seconds=1.0
                    )
        if calls.error is not None:
            raise calls.error
        if calls.reply is not None:
            return calls.reply
        return service.Reply(
            model,
            f"Answer to *{question}*",
            thinking="Let me think",
            seconds=2.0,
            thinking_seconds=1.0,
        )

    folders = tmp_path / "chat"

    def conversation_folder():
        folder = folders / f"conversation-{len(list(folders.glob('*')))}"
        folder.mkdir(parents=True)
        return str(folder)

    monkeypatch.setattr(service, "respond", respond)
    monkeypatch.setattr(service, "model_options", lambda: MODELS)
    monkeypatch.setattr(service, "server_ready", lambda attempts=1: True)
    monkeypatch.setattr(service, "conversation_folder", conversation_folder)
    return calls


def conversation_id(html):
    return re.search(r'data-page="chat:([0-9a-f]+)"', html).group(1)


def open_chat(client):
    response = client.get(f"{BASE}/chat")
    assert response.status_code == 200
    return conversation_id(response.text)


def send(client, chat, message, model="small:8b"):
    return client.post(
        f"{BASE}/chat/{chat}/messages",
        data={"message": message, "model": model},
        headers=htmx(client),
    )


def card_id(html):
    return re.search(r'id="job-([0-9a-f]+)"', html).group(1)


def answered(client, runner, response):
    assert response.status_code == 200, response.text[:500]
    job = runner.get(card_id(response.text))
    wait_for(lambda: not job.active)
    return job


def card(client, job):
    return client.get(f"{BASE}/jobs/{job.id}").text


def attach(client, chat, *files):
    return client.post(
        f"{BASE}/chat/{chat}/files",
        files=[("files", item) for item in files],
        headers=htmx(client),
    )


def pdf_bytes(text=""):
    with fitz.open() as doc:
        page_ = doc.new_page()
        if text:
            page_.insert_text((72, 72), text)
        return doc.tobytes()


# --- the page --------------------------------------------------------------


def test_the_page_starts_a_conversation(client, model):
    html = client.get(f"{BASE}/chat").text
    assert "What would you like to know?" in html
    assert 'name="message"' in html and "data-chat-composer" in html
    assert "<option selected>small:8b</option>" in html
    assert "<option>big:30b</option>" in html
    assert "New chat" in html and "Kept until the session ends." in html
    # Chat shows the disclaimer under the message box, not in a footer.
    assert 'class="footer"' not in html
    assert "check answers before you use them" in html


def test_a_server_that_does_not_answer_is_named(client, model, monkeypatch):
    monkeypatch.setattr(service, "server_ready", lambda attempts=1: False)
    html = client.get(f"{BASE}/chat").text
    assert "The language model server does not answer yet" in html
    response = send(client, conversation_id(html), "Hello?")
    assert response.status_code == 422
    assert "does not answer" in response.text


# --- messages and replies --------------------------------------------------


def test_a_message_is_answered(client, runner, model, released):
    chat = open_chat(client)
    response = send(client, chat, "What is Text Lab?")
    assert '<div class="message__text">What is Text Lab?</div>' in (
        response.text
    )
    assert "data-chat-answer" in response.text
    # The panel and the title come out of band.
    assert 'id="chat-panel-body" hx-swap-oob="true"' in response.text
    assert '<span id="chat-title" hx-swap-oob="true">What is Text Lab?' in (
        response.text
    )
    job = answered(client, runner, response)
    assert job.state is JobState.DONE and not job.listed
    assert job.page == f"chat:{chat}"
    assert released == ["llm:small:8b"]
    (asked,) = model.asked
    assert (asked.model, asked.history, asked.context, asked.table) == (
        "small:8b",
        [],
        "",
        None,
    )
    html = card(client, job)
    assert "Answer to <em>What is Text Lab?</em>" in html
    assert "Thought for 1 s" in html and "Let me think" in html
    assert 'data-copy-from="raw-' in html and "Again" in html
    assert "data-job-active" not in html


def test_the_conversation_goes_with_the_next_message(client, runner, model):
    chat = open_chat(client)
    answered(client, runner, send(client, chat, "First?"))
    answered(client, runner, send(client, chat, "Second?", model="big:30b"))
    assert model.asked[1].history == [
        {"role": "user", "content": "First?"},
        {"role": "assistant", "content": "Answer to *First?*"},
    ]
    assert model.asked[1].model == "big:30b"
    # The page shows both, and chooses the last model again.
    html = client.get(f"{BASE}/chat/{chat}").text
    assert html.index("First?") < html.index("Second?")
    assert "<option selected>big:30b</option>" in html
    assert "2 messages" in html


def test_a_reply_shows_its_draft_and_stops_keeping_it(client, runner, model):
    model.gate = threading.Event()
    chat = open_chat(client)
    response = send(client, chat, "Tell me a story")
    job = runner.get(card_id(response.text))
    wait_for(lambda: job.preview != "")
    html = card(client, job)
    assert "data-job-active" in html and "is-writing" in html
    assert "<strong>So far</strong>" in html and "Thought" in html
    # One reply at a time.
    refused = send(client, chat, "And another?")
    assert refused.status_code == 422
    assert "Wait for the answer" in refused.text

    client.post(f"{BASE}/jobs/{job.id}/cancel", headers=htmx(client))
    wait_for(lambda: not job.active)
    html = card(client, job)
    assert job.state is JobState.DONE and job.result.reply.stopped
    assert "Stopped." in html and "<strong>So far</strong>" in html
    # What was written stays in the conversation.
    model.gate = None
    answered(client, runner, send(client, chat, "Go on"))
    assert model.asked[-1].history[-1]["content"] == "**So far**"


def test_a_waiting_reply_names_the_job_before_it(client, runner, model):
    from textlab.ui.web.tests.test_jobs import Gate

    gate = Gate()
    other = runner.start(gate, title="interview.mp3", page="transcription")
    gate.started.wait(5)
    chat = open_chat(client)
    response = send(client, chat, "Hello?")
    assert 'Waits for "interview.mp3" to finish' in response.text
    assert "Cancel that job" in response.text
    gate.opened.set()
    answered(client, runner, response)
    assert other.state is JobState.DONE


def test_a_failed_reply_offers_to_try_again(client, runner, model):
    model.error = service.AnalysisError("The data analysis failed: boom")
    chat = open_chat(client)
    job = answered(client, runner, send(client, chat, "Plot it"))
    html = card(client, job)
    assert "No answer" in html and "boom" in html and "Try again" in html
    # The failed exchange is not part of the conversation for the model.
    model.error = None
    answered(client, runner, send(client, chat, "Hello"))
    assert model.asked[-1].history == []


@pytest.mark.parametrize(
    ("message", "model_name", "error"),
    [
        ("   ", "small:8b", "Write a message first."),
        ("Hi", "huge:70b", "Choose one of the models offered."),
    ],
)
def test_bad_messages_are_refused(client, model, message, model_name, error):
    response = send(client, open_chat(client), message, model=model_name)
    assert response.status_code == 422
    assert error in response.text


def test_a_message_too_long_for_the_model_is_refused(
    client, model, monkeypatch
):
    monkeypatch.setenv("TEXTLAB_CHAT_NUM_CTX", "8192")
    response = send(client, open_chat(client), "word " * 3000)
    assert response.status_code == 422
    assert "attach it instead" in response.text


def test_the_last_answer_can_be_written_again(client, runner, model):
    chat = open_chat(client)
    first = answered(client, runner, send(client, chat, "One?"))
    last = answered(client, runner, send(client, chat, "Two?"))
    ids = re.findall(r'id="exchange-([0-9a-f]+)"', card_page(client, chat))
    assert len(ids) == 2

    response = client.post(
        f"{BASE}/chat/{chat}/exchanges/{ids[1]}/again",
        data={"model": "big:30b"},
        headers=htmx(client),
    )
    again = answered(client, runner, response)
    assert f'id="exchange-{ids[1]}"' in response.text
    assert runner.get(last.id) is None and again.id != last.id
    assert model.asked[-1].model == "big:30b"
    assert [m["content"] for m in model.asked[-1].history] == [
        "One?",
        "Answer to *One?*",
    ]
    # Only the last answer can be written again.
    refused = client.post(
        f"{BASE}/chat/{chat}/exchanges/{ids[0]}/again",
        data={"model": "small:8b"},
        headers=htmx(client),
    )
    assert refused.status_code == 409
    assert runner.get(first.id) is not None


def card_page(client, chat):
    return client.get(f"{BASE}/chat/{chat}").text


# --- files -------------------------------------------------------------------


def test_files_are_read_once_and_sent_with_every_message(
    client, runner, model
):
    chat = open_chat(client)
    response = attach(client, chat, ("notes.txt", b"The budget is 40k."))
    assert response.status_code == 200, response.text[:300]
    assert "notes.txt" in response.text and ">Text<" in response.text
    assert 'id="chat-panel-body" hx-swap-oob="true"' in response.text
    answered(client, runner, send(client, chat, "What is the budget?"))
    context = model.asked[-1].context
    assert "--- Start of file: notes.txt ---\nThe budget is 40k." in context
    page_html = card_page(client, chat)
    assert '<ul class="message__files"' in page_html
    # Removed, the file is no longer sent.
    attachment = re.search(r"/files/([0-9a-f]+)/remove", page_html).group(1)
    removed = client.post(
        f"{BASE}/chat/{chat}/files/{attachment}/remove", headers=htmx(client)
    )
    assert "notes.txt" not in removed.text.split("chat-panel-body")[0]
    answered(client, runner, send(client, chat, "And now?"))
    assert model.asked[-1].context == ""


def test_a_table_is_kept_for_the_data_agents(client, runner, model):
    chat = open_chat(client)
    response = attach(client, chat, ("ages.csv", b"name,age\nAna,31\nBo,45\n"))
    assert ">Charts<" in response.text
    answered(client, runner, send(client, chat, "Plot the ages"))
    table = model.asked[-1].table
    assert table.name == "ages.csv"
    assert "age" in table.schema
    with open(table.path, "rb") as file:
        assert file.read().startswith(b"name,age")
    attachment = re.search(r"/files/([0-9a-f]+)/remove", response.text).group(
        1
    )
    client.post(
        f"{BASE}/chat/{chat}/files/{attachment}/remove", headers=htmx(client)
    )
    assert not os.path.exists(table.path)


def test_files_without_text_are_left_out_with_the_reason(client, model):
    chat = open_chat(client)
    response = attach(
        client, chat, ("scan.pdf", pdf_bytes()), ("paper.pdf", pdf_bytes("Hi"))
    )
    assert response.status_code == 200
    assert "probably because it is scanned" in response.text
    files = response.text.split("chat-panel-body")[0]
    assert "paper.pdf" in files and 'title="scan.pdf"' not in files


@pytest.mark.parametrize(
    ("files", "error"),
    [
        ([("report.docx", b"x")], "not a file this page reads"),
        ([(f"n{i}.txt", b"x") for i in range(5)], "4 files attached"),
    ],
)
def test_files_chat_cannot_take_are_refused(client, model, files, error):
    response = attach(client, open_chat(client), *files)
    assert response.status_code == 422
    assert error in response.text


def test_a_file_is_attached_once(client, model):
    chat = open_chat(client)
    attach(client, chat, ("notes.txt", b"x"))
    response = attach(client, chat, ("notes.txt", b"y"))
    assert response.status_code == 422
    assert "attached already" in response.text


# --- the data agents' charts -------------------------------------------------


def test_the_data_agents_charts_are_drawn(client, runner, model):
    model.reply = service.Reply(
        "small:8b",
        "Age is skewed.",
        analysis=CHARTS,
        activity=(("info", "Planning the charts"), ("warning", "A retry")),
        seconds=30.0,
    )
    chat = open_chat(client)
    job = answered(client, runner, send(client, chat, "Plot age"))
    html = card(client, job)
    assert f'data-chart="{BASE}/jobs/{job.id}/files/chart-1.json"' in html
    assert "plotly-3.7.0/plotly-strict.min.js" in html
    assert f'src="{BASE}/jobs/{job.id}/files/chart-2.png"' in html
    assert "px.scatter(df)" in html and "hist(df$age)" in html
    assert "t-test" in html and "<strong>0.04</strong>" in html
    assert "How the data agents worked: 2 steps" in html
    assert 'activity__step--warning">A retry</li>' in html
    figure = client.get(f"{BASE}/jobs/{job.id}/files/chart-1.json")
    assert figure.json() == {"data": [], "layout": {}}
    image = client.get(f"{BASE}/jobs/{job.id}/files/chart-2.png")
    assert image.headers["content-type"] == "image/png"
    # Charts are not listed as downloads; the conversation's HTML has them.
    assert job.listed_downloads == []
    panel = client.get(f"{BASE}/chat/{chat}/panel").text
    assert f"{BASE}/chat/{chat}/export.html" in panel


# --- conversations -----------------------------------------------------------


def test_conversations_are_kept_apart(client, runner, model):
    first = open_chat(client)
    answered(client, runner, send(client, first, "About cats?"))
    response = client.post(
        f"{BASE}/chat/new", data={"csrf_token": client.csrf}
    )
    assert response.status_code == 303
    second = response.headers["location"].rsplit("/", 1)[1]
    assert second != first
    # A new chat reuses the empty one.
    again = client.post(f"{BASE}/chat/new", data={"csrf_token": client.csrf})
    assert again.headers["location"].endswith(f"/chat/{second}")

    html = card_page(client, second)
    assert "About cats?" in html  # in the list of conversations
    assert 'class="exchange"' not in html
    assert f'href="{BASE}/chat/{first}"' in html
    # /chat shows the conversation used or started last.
    assert conversation_id(client.get(f"{BASE}/chat").text) == second


def test_a_deleted_conversation_takes_its_replies_and_files(
    client, runner, model
):
    chat = open_chat(client)
    attach(client, chat, ("ages.csv", b"a,b\n1,2\n"))
    job = answered(client, runner, send(client, chat, "Hi"))
    folder = model.asked[-1].table.path.rsplit("/", 1)[0]
    response = client.post(f"{BASE}/chat/{chat}/delete", headers=htmx(client))
    assert response.headers["HX-Redirect"] == f"{BASE}/chat"
    assert runner.get(job.id) is None
    assert not os.path.exists(folder)
    gone = client.get(f"{BASE}/chat/{chat}")
    assert gone.status_code == 303


def test_the_panel_shows_how_full_the_context_is(client, runner, model):
    chat = open_chat(client)
    assert "Context used" not in client.get(f"{BASE}/chat/{chat}/panel").text
    answered(client, runner, send(client, chat, "Hello"))
    panel = client.get(f"{BASE}/chat/{chat}/panel").text
    assert "Context used" in panel and "tokens" in panel
    assert f"{BASE}/chat/{chat}/export.md" in panel
    assert "export.html" not in panel


def test_a_conversation_is_downloaded(client, runner, model):
    chat = open_chat(client)
    attach(client, chat, ("notes.txt", b"Hello."))
    answered(client, runner, send(client, chat, "Summarize"))
    markdown = client.get(f"{BASE}/chat/{chat}/export.md")
    assert markdown.status_code == 200
    assert "text_lab_chat_" in markdown.headers["content-disposition"]
    assert "### User\nSummarize\n\n_Attached: notes.txt_" in markdown.text
    assert "### Assistant\nAnswer to *Summarize*" in markdown.text
    html = client.get(f"{BASE}/chat/{chat}/export.html")
    assert html.headers["content-type"].startswith("text/html")
    assert "Answer to *Summarize*" in html.text
    assert client.get(f"{BASE}/chat/{chat}/export.pdf").status_code == 404


def test_an_unknown_conversation_goes_back_to_chat(client, model):
    response = client.get(f"{BASE}/chat/0123456789ab")
    assert response.status_code == 303
    assert response.headers["location"] == f"{BASE}/chat"
    missing = send(client, "0123456789ab", "Hi")
    assert missing.status_code == 404


def test_titles_are_shortened_at_a_word():
    assert shorten("one two\nthree", 60) == "one two three"
    assert shorten("alpha beta gamma delta", 12) == "alpha beta…"
