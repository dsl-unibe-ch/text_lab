# Chat

A private chat with a local model on the session's Ollama server, about the
user's own files: attached documents (PDF, text, tables) are read into the
model's context, and questions about an attached table that need charts or
statistics are answered by the Visualization feature's agents. Used by the
Chat page of both interfaces.

## Pipeline

`service.respond` replies to a message; the web page calls it in a job.
For each message:

1. **Attachments** (`documents`): PDFs, text and JSON files and tables
   (CSV, TSV, Excel) are read one by one (`read_document`; tables as
   Markdown, at most 100 rows) and put into one context block
   (`document_context`). Files over 10 MB, files that cannot be read and
   PDFs without a text layer (scanned) give a `problem` instead of text,
   so the model is never asked about a document it cannot see. The pages
   read the attachments once and reuse their text while they stay
   attached. A table is also kept on disk for the agents (`keep_table`).
2. **The model**: pulled from the registry if the server does not have
   it, with its download's progress.
3. **Routing** (`router`, only with a table attached): the model is asked,
   as a tool call, whether the message needs charts or statistics. If so,
   `service.start_data_analysis` runs the Visualization agents on the table
   in the background; `respond` follows their steps (`Draft.activity`), and
   their result becomes the reply, with its charts and statistics.
4. **Answer** (`generation`): otherwise the model answers, streamed, with
   the context in front of the question. The stream gives `Piece`s, the
   model's thinking (reasoning models think first) apart from the answer,
   so a page can show both as they are written (`Draft`). Earlier turns
   that no longer fit are left out, oldest first. Context too long for one
   request is answered part by part, and the partial answers are combined
   into the streamed answer (in groups first, if they do not fit
   together).
5. **Stopping**: every request takes a `cancel` event. The stream is read
   in a thread of its own, so a stop takes effect within a quarter of a
   second, even while the model is still loading; the request is closed
   when its next piece arrives, which makes Ollama stop writing.
   `respond` keeps what was written (`Reply.stopped`); stopped agents
   finish their current step first.
6. **Export** (`exports`): the conversation as Markdown (charts as images
   where possible) or as HTML with interactive charts.

## Context window

Every request to the model, including the router's, sets Ollama's context
window (`num_ctx`) to `context_window.chat_context()`: 32768 tokens, or
`TEXTLAB_CHAT_NUM_CTX`. Without it, Ollama uses its default window, often
only a few thousand tokens, and silently keeps only the end of a longer
request: asked to summarize a long PDF, the model would see only its last
pages and summarize those.

`context_window.text_budget()` is the text one request may hold: the
window minus 4096 tokens for the answer and 1000 for instructions, divided
by 1.3 because the four-characters-per-token estimate undercounts many
languages. With the default window that is about 21,000 tokens (roughly
15,000 English words). A document that does not fit is split into parts of
that size, less the question and the most recent turns (at most a quarter
of the budget).

`context_window.context_usage(history, context)` estimates how much of the
budget the next message uses (the conversation plus the attachments, which
are sent with every message). The page shows it as a meter in the sidebar
and suggests a new chat when the conversation fills 75% of the budget or
its oldest messages no longer fit. Long attachments alone do not trigger
the suggestion, as a new chat would send them again.

All chat requests use the same window because Ollama reloads the model
whenever `num_ctx` changes. The Visualization agents use their own window
(`TEXTLAB_AGENT_NUM_CTX`), so the model is reloaded when a data analysis
starts and when the chat continues afterwards.

## Public API

Interfaces use `service.py`:

| Name | Purpose |
|---|---|
| `respond(model, history, question, context=, table=, on_progress=, on_draft=, cancel=)` | The reply to a message (`Reply`: text, thinking, the agents' charts and statistics, stopped, timings), from the agents or the chat model; `Draft`s while it is written. Raises `AnalysisError` when the agents fail |
| `reply(model, history, question, context, on_progress=, cancel=)` | The chat model's reply, streamed as `Piece`s (thinking or answer) |
| `answer(model, history, question, context, on_progress=)` | The answer's text, streamed (the Streamlit page) |
| `read_document(name, data)`, `document_context(documents)` | One attachment read (`Document`: text, or the `problem`), and the context block of several |
| `read_documents(files)` | Context block and warnings from `(name, bytes)` pairs |
| `DOCUMENT_TYPES`, `MAX_FILES`, `MAX_FILE_MB` | What a conversation can have attached |
| `keep_table(name, data, folder)`, `discard_table(table)` | Keep an attached table for the agents (`DataTable`: path and column summary), each in a folder of its own |
| `context_usage(history, context)` | `ContextUsage`: estimated share of the context window the next message uses |
| `decide_tool_use(model, text, schema, chat_history=)` | `(use_tools, instruction)` |
| `prepare_data_file`, `start_data_analysis` | Keep the attached table; answer with the agents (the Streamlit page) |
| `conversation_folder()`, `discard_conversation_folder(folder)` | The conversation's private folder |
| `format_chat_history`, `format_chat_history_html`, `has_analysis_plots` | Exports |
| `model_options()` | `ModelOptions`: the models for this session's GPU, the GPU, whether large models are offered |
| `server_ready(attempts)` | Whether the Ollama server accepts connections |
| `model_installed`, `pull_model(model, on_progress=)`, `is_model_loaded` | Model handling |

Messages are dictionaries with `role` and `content`; an assistant turn
answered by the agents also has `analysis`
(`visualization.models.AnalysisResult.to_payload()`).

## Layout

```
chat/
├── service.py     # the API above
├── documents.py   # reading attachments
├── context_window.py  # num_ctx and the text budget of a request
├── generation.py  # streamed answers, long contexts in parts
├── router.py      # chat or data analysis
├── exports.py     # Markdown and HTML exports
└── tests/
```

## Files written

| What | Where | Removed |
|---|---|---|
| The attached tables and the charts made from them | `chat/conversation-*/table-*` in the job workspace (the web page; the Streamlit page keeps its one table in `chat/conversation-*`) | When the table is removed or the conversation deleted (the Streamlit page: a new chat), and always when the session ends |

Documents attached as context are read in memory and not written to disk
by Chat. Their text is kept in the page's memory (the web server's, or
Streamlit's session state) while they stay attached.

## Configuration

| Variable | Effect |
|---|---|
| `TEXTLAB_CHAT_NUM_CTX` | Context window (tokens) of every chat request: 32768 if unset, at least 8192. Raise it on large GPUs to read longer documents in one pass; lower it if the model does not fit in GPU memory |

The models offered are listed in `common/models.json`; the server address
comes from `OLLAMA_HOST`.

## Tests

In `tests/`, without a model:

- `test_service.py`: reading attachments (PDF, scanned PDF, TSV, JSON,
  unreadable files, one by one), the context window, text budget and
  usage meter, answers with short and long context, thinking apart from
  the answer, stopping (also while the model loads), routing, `respond`
  with the chat model and the agents (downloading the model, stopping,
  failing agents), kept tables, exports, conversation folders, starting an
  analysis, the models per GPU.
- `test_chat_page.py`: the Streamlit page imports only the service. The
  web page's tests are in `ui/web/tests/test_chat_page.py`.
