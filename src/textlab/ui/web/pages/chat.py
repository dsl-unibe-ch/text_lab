"""The Chat page: a private chat with a local model about the user's files.

The session's conversations are kept in memory (:class:`ChatStore`, in
``app.state.chat``) with their attachments, until the user deletes them or
the session ends. Each message is answered by a job that is not listed: its
card (``chat/answer.html``) is the reply in the conversation. The job waits
while another job holds the GPU, shows the reply as the model writes it
(its thinking, the answer, or the data agents' steps) and can be stopped,
which keeps what was written (:func:`write_reply`).

Attachments are read once, when they are attached, and their text goes to
the model with every message while they stay attached. An attached table
is also saved in the conversation's folder in the workspace for the data
agents (``service.keep_table``); the folder goes with the conversation.
"""

from __future__ import annotations

import dataclasses
import mimetypes
import secrets
import threading
import time
from functools import partial
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from markupsafe import Markup
from starlette.concurrency import run_in_threadpool

from textlab.common import gpu_manager
from textlab.features.chat import service
from textlab.ui.web import uploads
from textlab.ui.web.auth import Checked, protected_router
from textlab.ui.web.jobs import (
    Download,
    Job,
    JobContext,
    JobRunner,
    JobState,
    content_disposition,
    render_card,
    runner_of,
)
from textlab.ui.web.render import is_htmx, renderer, root, shorten

router = protected_router()

#: The page's id. Its jobs are named by their conversation
#: (:func:`stream_page`), so that the page's event stream sends the cards of
#: the conversation shown only.
PAGE = "chat"

#: Characters of a conversation's title, taken from its first message.
TITLE_CHARS = 60

#: Least seconds between two updates of a reply being written. The reader
#: follows it word by word, so it updates more often than other previews.
DRAFT_SECONDS = 0.15

#: How many times a message checks that the model server answers, half a
#: second apart (it may still be starting when a session begins).
SERVER_ATTEMPTS = 4

#: File name endings of the files a conversation reads.
SUFFIXES = tuple(f".{kind}" for kind in service.DOCUMENT_TYPES)

#: Guards the creation of the session's chat store.
_STORE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Attachment:
    """A file attached to a conversation, read for the model.

    Attributes:
        id: Random id, used in links.
        document: The file's text.
        table: The file as a table the data agents can analyze, if it is
            one.
    """

    id: str
    document: service.Document
    table: service.DataTable | None = None


@dataclasses.dataclass
class Exchange:
    """A message of the user and the job that replies to it.

    Attributes:
        id: Random id, used in links.
        question: The message.
        files: The names of the files attached when it was sent.
        job_id: The job that writes the reply.
        asked: When it was sent (seconds since the epoch).
    """

    id: str
    question: str
    files: tuple[str, ...]
    job_id: str
    asked: float


@dataclasses.dataclass(eq=False)
class Conversation:
    """One conversation of the session.

    Attributes:
        id: Random id, used in links.
        created: When it was started (seconds since the epoch).
        model: The model of the last message, chosen again for the next.
        attachments: The files attached now, sent with every message.
        exchanges: The messages and their replies, oldest first.
        folder: The conversation's folder in the workspace, made for the
            first attached table; empty until then.
    """

    id: str
    created: float
    model: str = ""
    attachments: list[Attachment] = dataclasses.field(default_factory=list)
    exchanges: list[Exchange] = dataclasses.field(default_factory=list)
    folder: str = ""

    @property
    def title(self) -> str:
        """The first message, shortened; "New chat" before it."""
        if not self.exchanges:
            return "New chat"
        return shorten(self.exchanges[0].question, TITLE_CHARS)

    @property
    def updated(self) -> float:
        """When the last message was sent, or the conversation started."""
        return self.exchanges[-1].asked if self.exchanges else self.created

    @property
    def context(self) -> str:
        """The attached documents' text, as the model reads it."""
        return service.document_context(
            [attachment.document for attachment in self.attachments]
        )

    @property
    def table(self) -> service.DataTable | None:
        """The first attached table, which the data agents analyze."""
        return next(
            (item.table for item in self.attachments if item.table), None
        )

    def exchange(self, exchange_id: str) -> Exchange | None:
        """Return one of the exchanges, or ``None``."""
        return next(
            (item for item in self.exchanges if item.id == exchange_id), None
        )


class ChatStore:
    """The session's conversations, kept in memory until the session ends."""

    def __init__(self) -> None:
        """Start without conversations."""
        self.lock = threading.RLock()
        self._conversations: dict[str, Conversation] = {}

    def get(self, conversation_id: str) -> Conversation | None:
        """Return a conversation, or ``None`` if there is none by that id."""
        with self.lock:
            return self._conversations.get(conversation_id)

    def all(self) -> list[Conversation]:
        """Return the conversations, the most recently used first."""
        with self.lock:
            found = list(self._conversations.values())
        return sorted(found, key=lambda item: item.updated, reverse=True)

    def new(self) -> Conversation:
        """Return an empty conversation: one there is, or a new one."""
        with self.lock:
            for conversation in self._conversations.values():
                if not conversation.exchanges and not conversation.attachments:
                    return conversation
            conversation = Conversation(secrets.token_hex(6), time.time())
            self._conversations[conversation.id] = conversation
            return conversation

    def remove(self, conversation_id: str) -> Conversation | None:
        """Forget a conversation and return it, or ``None``."""
        with self.lock:
            return self._conversations.pop(conversation_id, None)


def store_of(request: Request) -> ChatStore:
    """Return the session's chat store, made on first use.

    Args:
        request: The current request.

    Returns:
        The store, in ``app.state.chat``.
    """
    state = request.app.state
    with _STORE_LOCK:
        store = getattr(state, "chat", None)
        if store is None:
            store = state.chat = ChatStore()
    return store


def conversation_or_404(
    request: Request, conversation_id: str
) -> Conversation:
    """Return a conversation of the session, or answer 404."""
    conversation = store_of(request).get(conversation_id)
    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "This conversation no longer exists. It may have been "
                "deleted; reload the page."
            ),
        )
    return conversation


def stream_page(conversation: Conversation) -> str:
    """Return the page id of a conversation's jobs, such as "chat:3f2a"."""
    return f"{PAGE}:{conversation.id}"


# ---------------------------------------------------------------------------
# Replies
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Chart:
    """A chart of the data agents, as the reply shows it.

    Attributes:
        label: What the chart is, such as "Histogram".
        file: The job's download that holds it.
        interactive: Whether it is a Plotly figure (JSON) rather than an
            image.
        code: Python code that draws it.
        r_code: R code that draws it, if any.
    """

    label: str
    file: str
    interactive: bool
    code: str = ""
    r_code: str = ""


@dataclasses.dataclass(frozen=True)
class Answer:
    """A finished reply, as its card shows it.

    Attributes:
        reply: The service's reply.
        charts: The data agents' charts.
        stats: The data agents' statistical results: dictionaries with
            ``title``, ``result`` (Markdown), ``code`` and ``r_code``.
    """

    reply: service.Reply
    charts: tuple[Chart, ...] = ()
    stats: tuple[dict[str, Any], ...] = ()


def write_reply(
    model: str,
    history: list[dict[str, Any]],
    question: str,
    context: str,
    table: service.DataTable | None,
    job: JobContext,
) -> Answer:
    """Reply to a message; the work of a reply's job.

    Args:
        model: The Ollama model.
        history: The conversation before the message.
        question: The message.
        context: The attached documents' text.
        table: The attached table the data agents may analyze.
        job: The job's context.

    Returns:
        The reply, with the data agents' charts offered as the job's
        downloads (not listed), which the card draws.
    """
    reply = service.respond(
        model,
        history,
        question,
        context=context,
        table=table,
        on_progress=job.report,
        on_draft=partial(job.preview, every=DRAFT_SECONDS),
        cancel=job.cancel,
    )
    analysis = reply.analysis or {}
    charts = []
    for index, artifact in enumerate(analysis.get("artifacts", []), 1):
        chart = offer_chart(job, index, artifact)
        if chart is not None:
            charts.append(chart)
    return Answer(reply, tuple(charts), tuple(analysis.get("stats", [])))


def offer_chart(
    job: JobContext, index: int, artifact: dict[str, Any]
) -> Chart | None:
    """Offer one of the data agents' charts as a download of the job.

    Args:
        job: The reply's job.
        index: The chart's number in the reply, from 1.
        artifact: The chart (``Artifact.to_dict()``).

    Returns:
        The chart as the card shows it, or ``None`` for a chart without
        content.
    """
    label = service.tool_label(artifact.get("tool_name", "")) or (
        f"Chart {index}"
    )
    figure = artifact.get("fig_json")
    if figure:
        name, data, media_type = (
            f"chart-{index}.json",
            figure.encode("utf-8"),
            "application/json",
        )
    elif artifact.get("bytes"):
        suffix = Path(artifact.get("filename", "")).suffix.lower() or ".png"
        name, data = f"chart-{index}{suffix}", artifact["bytes"]
        media_type = mimetypes.guess_type(name)[0] or "image/png"
    else:
        return None
    job.offer(Download(name, label, media_type, data=data, listed=False))
    return Chart(
        label,
        name,
        interactive=bool(figure),
        code=artifact.get("code", ""),
        r_code=artifact.get("r_code", ""),
    )


def reply_of(runner: JobRunner, exchange: Exchange) -> service.Reply | None:
    """Return the finished reply to an exchange, or ``None``."""
    job = runner.get(exchange.job_id)
    if job is None or job.state is not JobState.DONE:
        return None
    return job.result.reply


def history(
    runner: JobRunner, exchanges: list[Exchange]
) -> list[dict[str, Any]]:
    """Return exchanges as the messages the model reads.

    Exchanges without a finished reply with text (failed, cancelled,
    stopped before a word) are left out.

    Args:
        runner: The app's job runner.
        exchanges: The exchanges, oldest first.

    Returns:
        The messages; the data agents' replies keep their ``analysis``,
        for the exports.
    """
    messages: list[dict[str, Any]] = []
    for exchange in exchanges:
        reply = reply_of(runner, exchange)
        if reply is None or not reply.text:
            continue
        messages.append({"role": "user", "content": exchange.question})
        messages.append(reply.message())
    return messages


def active_job(runner: JobRunner, conversation: Conversation) -> Job | None:
    """Return the job of a reply still waiting or being written."""
    for exchange in conversation.exchanges:
        job = runner.get(exchange.job_id)
        if job is not None and job.active:
            return job
    return None


def ask(
    request: Request,
    conversation: Conversation,
    question: str,
    model: str,
    *,
    again: Exchange | None = None,
) -> Exchange:
    """Send a message: start the job that replies to it.

    Args:
        request: The current request.
        conversation: The conversation.
        question: The message.
        model: The Ollama model.
        again: An exchange to answer again, the conversation's last; its
            reply is removed and a new one written.

    Returns:
        The exchange, new or answered again.

    Raises:
        UploadError: If a reply of the conversation is still written.
    """
    runner = runner_of(request)
    store = store_of(request)
    with store.lock:
        if active_job(runner, conversation) is not None:
            raise uploads.UploadError(
                "Wait for the answer, or stop it, before you send the next "
                "message."
            )
        earlier = [
            item for item in conversation.exchanges if item is not again
        ]
        exchange_id = again.id if again else secrets.token_hex(6)
        job = runner.create(
            title=f"Chat: {shorten(question, TITLE_CHARS)}",
            page=stream_page(conversation),
            gpu=gpu_manager.LLM,
            ollama_model=model,
            card_template="chat/answer.html",
            listed=False,
            page_data={
                "conversation": conversation.id,
                "exchange": exchange_id,
            },
        )
        runner.submit(
            job,
            partial(
                write_reply,
                model,
                history(runner, earlier),
                question,
                conversation.context,
                conversation.table,
            ),
        )
        files = tuple(item.document.name for item in conversation.attachments)
        if again is None:
            exchange = Exchange(
                exchange_id, question, files, job.id, time.time()
            )
            conversation.exchanges.append(exchange)
        else:
            runner.remove(again.job_id)
            again.files, again.job_id = files, job.id
            exchange = again
        conversation.model = model
    return exchange


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def model_choice(form_value: Any, options: service.ModelOptions) -> str:
    """Return the model a form chose, checked against the offered ones.

    Raises:
        UploadError: If no model is offered, or another was sent.
    """
    if not options.models:
        raise uploads.UploadError(
            "No language models are set up for Chat. Please tell support."
        )
    model = str(form_value or options.models[0])
    if model not in options.models:
        raise uploads.UploadError("Choose one of the models offered.")
    return model


async def check_server() -> None:
    """Refuse a message while the model server does not answer."""
    if not await run_in_threadpool(service.server_ready, SERVER_ATTEMPTS):
        raise uploads.UploadError(
            "The language model server (Ollama) does not answer. Try again "
            "in a minute; if it keeps failing, send ollama.log from the "
            "session's folder to support."
        )


def panel_context(
    request: Request, conversation: Conversation
) -> dict[str, Any]:
    """Return what the side panel shows of the session's conversations.

    Args:
        request: The current request.
        conversation: The conversation shown.

    Returns:
        The template values: the conversation, the session's
        conversations, the models offered, how full the model's context
        is, and whether the conversation has messages and charts to
        export.
    """
    runner = runner_of(request)
    messages = history(runner, conversation.exchanges)
    return {
        "conversation": conversation,
        "conversations": store_of(request).all(),
        "options": service.model_options(),
        "usage": service.context_usage(messages, conversation.context),
        "has_messages": bool(messages),
        "has_charts": service.has_analysis_plots(messages),
    }


def render_panel(
    request: Request, conversation: Conversation, *, oob: bool = False
) -> str:
    """Render the side panel, out of band if ``oob`` is set."""
    return renderer(request).render(
        request,
        "chat/panel.html",
        oob=oob,
        **panel_context(request, conversation),
    )


def render_exchange(
    request: Request, conversation: Conversation, exchange: Exchange
) -> str:
    """Render a message and its reply's card."""
    job = runner_of(request).get(exchange.job_id)
    card = Markup(render_card(request, job)) if job is not None else ""
    return renderer(request).render(
        request,
        "chat/exchange.html",
        conversation=conversation,
        exchange=exchange,
        card=card,
    )


def render_files(
    request: Request,
    conversation: Conversation,
    problems: list[str] | None = None,
) -> str:
    """Render the attached files, the side panel and any problems."""
    files = renderer(request).render(
        request,
        "chat/files.html",
        conversation=conversation,
        problems=problems or [],
    )
    return files + render_panel(request, conversation, oob=True)


def show(request: Request, conversation: Conversation) -> HTMLResponse:
    """Return the page with a conversation."""
    context = panel_context(request, conversation)
    models = context["options"].models
    if conversation.model in models:
        model = conversation.model
    else:
        model = models[0] if models else ""
    return renderer(request).response(
        request,
        "chat/page.html",
        page_id=PAGE,
        stream_page=stream_page(conversation),
        exchanges=[
            Markup(render_exchange(request, conversation, exchange))
            for exchange in conversation.exchanges
        ],
        model=model,
        ready=service.server_ready(1),
        accept=",".join(SUFFIXES),
        formats=uploads.format_names(SUFFIXES),
        max_files=service.MAX_FILES,
        max_file_mb=service.MAX_FILE_MB,
        problems=[],
        oob=False,
        **context,
    )


def go_to(request: Request, path: str) -> Response:
    """Send the browser to a page: by htmx's header, or a redirect."""
    url = f"{root(request)}{path}"
    if is_htmx(request):
        return Response(status_code=204, headers={"HX-Redirect": url})
    return RedirectResponse(url, status_code=303)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/chat")
def page(request: Request) -> HTMLResponse:
    """Show the most recent conversation, or a new one.

    Args:
        request: The current request.

    Returns:
        The page.
    """
    store = store_of(request)
    conversations = store.all()
    return show(request, conversations[0] if conversations else store.new())


@router.get("/chat/{conversation_id}")
def conversation_page(request: Request, conversation_id: str) -> Response:
    """Show a conversation of the session.

    Args:
        request: The current request.
        conversation_id: The conversation's id.

    Returns:
        The page, or a redirect to the Chat page if the conversation no
        longer exists.
    """
    conversation = store_of(request).get(conversation_id)
    if conversation is None:
        return RedirectResponse(f"{root(request)}/chat", status_code=303)
    return show(request, conversation)


@router.post("/chat/new")
def new_conversation(request: Request, session: Checked) -> Response:
    """Start a new conversation, or go to the empty one there is.

    Args:
        request: The current request.
        session: The session, with its CSRF token checked.

    Returns:
        A redirect to the conversation.
    """
    return go_to(request, f"/chat/{store_of(request).new().id}")


@router.post("/chat/{conversation_id}/messages")
async def send_message(
    request: Request, conversation_id: str, session: Checked
) -> HTMLResponse:
    """Send a message and start its reply.

    Args:
        request: The current request.
        conversation_id: The conversation's id.
        session: The session, with its CSRF token checked.

    Returns:
        The message with its reply's card, to add to the conversation, and
        the side panel out of band.

    Raises:
        UploadError: For an empty or too long message, a model not
            offered, a reply still being written, or a model server that
            does not answer.
    """
    conversation = conversation_or_404(request, conversation_id)
    form = await request.form()
    question = str(form.get("message") or "").strip()
    if not question:
        raise uploads.UploadError("Write a message first.")
    usage = service.context_usage([{"role": "user", "content": question}])
    if usage.fraction > 1:
        raise uploads.UploadError(
            "This message is longer than the model can read at once. Save "
            "the text as a .txt file and attach it instead: long files are "
            "read in parts."
        )
    model = model_choice(form.get("model"), service.model_options())
    await check_server()
    exchange = ask(request, conversation, question, model)
    return HTMLResponse(
        render_exchange(request, conversation, exchange)
        + render_panel(request, conversation, oob=True)
    )


@router.post("/chat/{conversation_id}/exchanges/{exchange_id}/again")
async def answer_again(
    request: Request,
    conversation_id: str,
    exchange_id: str,
    session: Checked,
) -> HTMLResponse:
    """Answer the last message again, with the model chosen now.

    Args:
        request: The current request.
        conversation_id: The conversation's id.
        exchange_id: The exchange's id; it must be the last.
        session: The session, with its CSRF token checked.

    Returns:
        The message with its new reply's card, in place of the old.

    Raises:
        HTTPException: If the exchange is not the conversation's last
            (409) or does not exist (404).
    """
    conversation = conversation_or_404(request, conversation_id)
    exchange = conversation.exchange(exchange_id)
    if exchange is None:
        raise HTTPException(status_code=404, detail="No such message.")
    if conversation.exchanges[-1] is not exchange:
        raise HTTPException(
            status_code=409,
            detail="Only the last answer can be written again.",
        )
    form = await request.form()
    model = model_choice(form.get("model"), service.model_options())
    await check_server()
    ask(request, conversation, exchange.question, model, again=exchange)
    return HTMLResponse(
        render_exchange(request, conversation, exchange)
        + render_panel(request, conversation, oob=True)
    )


@router.post("/chat/{conversation_id}/files")
async def attach_files(
    request: Request, conversation_id: str, session: Checked
) -> HTMLResponse:
    """Attach files to a conversation and read them.

    Files that give no text (scanned PDFs, empty or unreadable files) are
    not attached; the answer says why.

    Args:
        request: The current request.
        conversation_id: The conversation's id.
        session: The session, with its CSRF token checked.

    Returns:
        The attached files, with the problems of any left out, and the
        side panel out of band.

    Raises:
        UploadError: For no file, a type Chat does not read, more files
            than a conversation can hold, or a file attached already.
    """
    conversation = conversation_or_404(request, conversation_id)
    # More files than a conversation takes are refused below, with a
    # message saying so; the form's own limit only keeps out abuse.
    form = await uploads.read_form(request, max_files=4 * service.MAX_FILES)
    found = uploads.files(form, "files")
    if not found:
        raise uploads.UploadError("Choose a file first.")
    attached = {item.document.name for item in conversation.attachments}
    for upload in found:
        uploads.check_suffix(upload, SUFFIXES)
        if uploads.display_name(upload) in attached:
            raise uploads.UploadError(
                f"{uploads.display_name(upload)} is attached already."
            )
    if len(conversation.attachments) + len(found) > service.MAX_FILES:
        raise uploads.UploadError(
            f"A conversation can have {service.MAX_FILES} files attached at "
            "a time. Remove one first."
        )
    problems = []
    for upload in found:
        data = await upload.read()
        problem = await run_in_threadpool(
            attach, request, conversation, uploads.display_name(upload), data
        )
        if problem:
            problems.append(problem)
    return HTMLResponse(render_files(request, conversation, problems))


def attach(
    request: Request, conversation: Conversation, name: str, data: bytes
) -> str:
    """Read a file and attach it to a conversation, if it has text.

    Args:
        request: The current request.
        conversation: The conversation.
        name: The file's name.
        data: Its content.

    Returns:
        Why the file was not attached, or ``""`` if it was.
    """
    document = service.read_document(name, data)
    if document.problem:
        return document.problem
    table = None
    if service.is_table(name):
        with store_of(request).lock:
            if not conversation.folder:
                conversation.folder = service.conversation_folder()
        table = service.keep_table(name, data, conversation.folder)
    with store_of(request).lock:
        conversation.attachments.append(
            Attachment(secrets.token_hex(6), document, table)
        )
    return ""


@router.post("/chat/{conversation_id}/files/{attachment_id}/remove")
def remove_file(
    request: Request,
    conversation_id: str,
    attachment_id: str,
    session: Checked,
) -> HTMLResponse:
    """Remove an attached file from a conversation.

    Args:
        request: The current request.
        conversation_id: The conversation's id.
        attachment_id: The attachment's id.
        session: The session, with its CSRF token checked.

    Returns:
        The attached files, and the side panel out of band.
    """
    conversation = conversation_or_404(request, conversation_id)
    with store_of(request).lock:
        removed = [
            item
            for item in conversation.attachments
            if item.id == attachment_id
        ]
        conversation.attachments = [
            item
            for item in conversation.attachments
            if item.id != attachment_id
        ]
    for item in removed:
        if item.table is not None:
            service.discard_table(item.table)
    return HTMLResponse(render_files(request, conversation))


@router.get("/chat/{conversation_id}/panel")
def panel(request: Request, conversation_id: str) -> HTMLResponse:
    """Return the side panel, after a reply changed what it shows.

    Args:
        request: The current request.
        conversation_id: The conversation's id.

    Returns:
        The panel.
    """
    conversation = conversation_or_404(request, conversation_id)
    return HTMLResponse(render_panel(request, conversation))


@router.post("/chat/{conversation_id}/delete")
def delete_conversation(
    request: Request, conversation_id: str, session: Checked
) -> Response:
    """Delete a conversation with its replies, files and charts.

    Args:
        request: The current request.
        conversation_id: The conversation's id.
        session: The session, with its CSRF token checked.

    Returns:
        A redirect to the Chat page.
    """
    conversation = store_of(request).remove(conversation_id)
    if conversation is not None:
        runner = runner_of(request)
        for exchange in conversation.exchanges:
            runner.cancel(exchange.job_id)
            runner.remove(exchange.job_id)
        if conversation.folder:
            service.discard_conversation_folder(conversation.folder)
    return go_to(request, "/chat")


def export_messages(
    runner: JobRunner, conversation: Conversation
) -> list[dict[str, Any]]:
    """Return a conversation's messages for its exports.

    The files attached to a message are named after it.
    """
    messages = []
    for exchange in conversation.exchanges:
        reply = reply_of(runner, exchange)
        if reply is None or not reply.text:
            continue
        question = exchange.question
        if exchange.files:
            question += f"\n\n_Attached: {', '.join(exchange.files)}_"
        messages.append({"role": "user", "content": question})
        messages.append(reply.message())
    return messages


@router.get("/chat/{conversation_id}/export.{kind}")
def export(request: Request, conversation_id: str, kind: str) -> Response:
    """Download a conversation, as Markdown or as HTML with its charts.

    Args:
        request: The current request.
        conversation_id: The conversation's id.
        kind: ``md`` or ``html``.

    Returns:
        The file, as an attachment.

    Raises:
        HTTPException: For another kind (404).
    """
    conversation = conversation_or_404(request, conversation_id)
    messages = export_messages(runner_of(request), conversation)
    stamp = time.strftime(
        "%Y-%m-%d_%H-%M", time.localtime(conversation.updated)
    )
    if kind == "md":
        content = service.format_chat_history(messages)
        media_type = "text/markdown; charset=utf-8"
    elif kind == "html":
        content = service.format_chat_history_html(messages)
        media_type = "text/html; charset=utf-8"
    else:
        raise HTTPException(status_code=404, detail="No such export.")
    return Response(
        content,
        media_type=media_type,
        headers={
            "Content-Disposition": content_disposition(
                f"text_lab_chat_{stamp}.{kind}"
            )
        },
    )
