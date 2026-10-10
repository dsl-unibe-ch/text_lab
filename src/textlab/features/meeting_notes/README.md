# Meeting Notes

Turns a recording or a transcript into structured notes with a local LLM
served by the session's Ollama server. The recording is transcribed with
the [transcription](../transcription/README.md) feature first.

## Pipeline

1. **Transcribe** (From Audio tab only) through
   `transcription.service.run_transcription`, in a worker process, so the
   GPU is free for Ollama afterwards.
2. **Prepare the text**: the segment table becomes speaker-labeled text
   (`transcript_csv_to_speaker_text`); names the user gives the speakers are
   applied (`apply_speaker_labels`) and described to the model
   (`build_speaker_context`).
3. **Summarize** according to the chosen mode (`SUMMARY_MODES`: general,
   meeting notes, interview, academic abstract, lecture notes):
   - a transcript that fits the context budget (`needs_chunking` is false)
     in one streamed request (`get_summary_stream`);
   - a longer one map-reduce style: notes per chunk
     (`get_partial_notes`), then a streamed synthesis
     (`get_synthesis_stream`).
4. **Export** the Markdown document with summary and transcript
   (`format_summary_document`).

Requests use a low temperature and disable "thinking", retrying without the
`think` argument when a model or server does not support it. The prompts
contain no model-specific handling, so any Ollama model can be used.

## Public API

Everything is in `service.py`:

| Name | Purpose |
|---|---|
| `SUMMARY_MODES` | The summary types, with labels, descriptions and prompts |
| `needs_chunking(text)` | Whether a transcript must be summarized in chunks |
| `get_summary_stream(model, text, mode, speaker_context, output_language)` | Stream a single-pass summary |
| `get_partial_notes(..., on_progress=)` | Notes per chunk, with progress |
| `get_synthesis_stream(model, partial_notes, mode, output_language)` | Stream the final summary from the notes |
| `transcript_csv_to_speaker_text`, `apply_speaker_labels`, `extract_unique_speakers`, `build_speaker_context` | Transcript helpers |
| `format_summary_document(...)` | The downloadable Markdown document |

`output_language` is a language name such as `"English"`, or `None` to
answer in the transcript's language.

## Files written

None. Transcription writes only to the workspace (see the transcription
README); summaries stay in memory until the user downloads them.

## Configuration

The Ollama server address comes from `OLLAMA_HOST`, set by the launch
script. The models offered are listed in `common/models.json`.

## Tests

`tests/test_service.py`, with Ollama faked: transcript helpers, the
chunking threshold, the language instruction in prompts, one request per
chunk with progress, streaming, and the retry without `think`.
