# Chat

A private chat with a local model on the session's Ollama server, about the
user's own files: attached documents (PDF, text, tables) are read into the
model's context, and questions about an attached table that need charts or
statistics are answered by the Visualization feature's agents. Used by the
Chat page.

## Pipeline

For each message:

1. **Attachments** (`documents`): PDFs, text and JSON files and tables
   (CSV, TSV, Excel) are read into one context block (tables as Markdown,
   at most 100 rows). Files over 10 MB, files that cannot be read and PDFs
   without a text layer (scanned) are skipped with a warning, so the model
   is never asked about a document it cannot see. The page reads the
   attachments once and reuses their text while they stay attached.
2. **Routing** (`router`, only with a table attached): the model is asked,
   as a tool call, whether the message needs charts or statistics. If so,
   `service.start_data_analysis` runs the Visualization agents on the table
   in the background, and their result becomes the assistant's turn,
   with its charts and statistics.
3. **Answer** (`generation`): otherwise the model answers, streamed, with
   the context in front of the question. Earlier turns that no longer fit
   are left out, oldest first. Context too long for one request is
   answered part by part, and the partial answers are combined into the
   streamed answer (in groups first, if they do not fit together).
4. **Export** (`exports`): the conversation as Markdown (charts as images
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

All chat requests use the same window because Ollama reloads the model
whenever `num_ctx` changes. The Visualization agents use their own window
(`TEXTLAB_AGENT_NUM_CTX`), so the model is reloaded when a data analysis
starts and when the chat continues afterwards.

## Public API

Interfaces use `service.py`:

| Name | Purpose |
|---|---|
| `read_documents(files)` | Context block and warnings from `(name, bytes)` pairs |
| `DOCUMENT_TYPES` | File extensions the page accepts as attachments |
| `answer(model, history, question, context, on_progress=)` | The streamed answer |
| `decide_tool_use(model, text, schema, chat_history=)` | `(use_tools, instruction)` |
| `prepare_data_file`, `start_data_analysis` | Keep the attached table; answer with the agents |
| `conversation_folder()`, `discard_conversation_folder(folder)` | The conversation's private folder |
| `format_chat_history`, `format_chat_history_html`, `has_analysis_plots` | Exports |
| `model_installed`, `pull_model`, `is_model_loaded` | Model handling for the page |

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
| The attached table and the charts made from it | `chat/conversation-*` in the job workspace | When the user starts a new chat, and always when the session ends |

Documents attached as context are read in memory and not written to disk.
Their text is kept in the page's session state while they stay attached,
and dropped when the user starts a new chat.

## Configuration

| Variable | Effect |
|---|---|
| `TEXTLAB_CHAT_NUM_CTX` | Context window (tokens) of every chat request: 32768 if unset, at least 8192. Raise it on large GPUs to read longer documents in one pass; lower it if the model does not fit in GPU memory |

The models offered are listed in `common/models.json`; the server address
comes from `OLLAMA_HOST`.

## Tests

In `tests/`, without a model:

- `test_service.py`: reading attachments (PDF, scanned PDF, TSV, JSON,
  unreadable files), the context window and text budget, answers with
  short and long context, routing, exports, conversation folders, starting
  an analysis.
- `test_chat_page.py`: the page imports only the service.
