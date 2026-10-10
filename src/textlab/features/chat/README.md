# Chat

A private chat with a local model on the session's Ollama server, about the
user's own files: attached documents (PDF, text, tables) are read into the
model's context, and questions about an attached table that need charts or
statistics are answered by the Visualization feature's agents. Used by the
Chat page.

## Pipeline

For each message:

1. **Attachments** (`documents`): PDFs, text files and tables are read into
   one context block (tables as Markdown, at most 100 rows); files over
   10 MB are skipped with a warning.
2. **Routing** (`router`, only with a table attached): the model is asked,
   as a tool call, whether the message needs charts or statistics. If so,
   `service.start_data_analysis` runs the Visualization agents on the table
   in the background, and their result becomes the assistant's turn,
   with its charts and statistics.
3. **Answer** (`generation`): otherwise the model answers, streamed, with
   the context in front of the question. Context larger than the model's
   budget (`common.ollama.MAX_CONTEXT_TOKENS`) is answered part by part,
   and the partial answers are combined into the streamed answer.
4. **Export** (`exports`): the conversation as Markdown (charts as images
   where possible) or as HTML with interactive charts.

## Public API

Interfaces use `service.py`:

| Name | Purpose |
|---|---|
| `read_documents(files)` | Context block and warnings from `(name, bytes)` pairs |
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

## Configuration

No settings of its own. The models offered are listed in
`common/models.json`; the server address comes from `OLLAMA_HOST`.

## Tests

In `tests/`, without a model:

- `test_service.py`: reading attachments, answers with short and long
  context, routing, exports, conversation folders, starting an analysis.
- `test_chat_page.py`: the page imports only the service.
