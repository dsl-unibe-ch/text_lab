# Meeting Notes

Turns a recording or a transcript into structured notes with a local LLM
served by the session's Ollama server. Each point in the notes cites the
transcript lines it is based on, so readers can check it. The recording is
transcribed with the [transcription](../transcription/README.md) feature
first.

## Pipeline

1. **Transcribe** (From Audio tab only) through
   `transcription.service.run_transcription`, in a worker process, so the
   GPU is free for Ollama afterwards.
2. **Prepare the transcript** as numbered segments (`transcript.py`): a
   segment table becomes one `Segment` per row with speaker and timestamps
   (`segments_from_table`); plain text one per line, with very long lines
   split into sentences (`segments_from_text`). Names the user gives the
   speakers are applied (`rename_speakers`) and described to the model
   (`build_speaker_context`).
3. **Summarize** according to the chosen mode (`SUMMARY_MODES`: general,
   meeting notes, interview, academic abstract, lecture notes). The model
   sees each line as `[12] Anna: ...` and is told to cite line numbers after
   every point:
   - a transcript that fits in one request (`needs_chunking` is false) in
     one streamed request (`get_summary_stream`);
   - a longer one map-reduce style: cited notes per chunk of whole lines
     (`get_partial_notes`), then a streamed synthesis that keeps the
     citations (`get_synthesis_stream`). If the notes are too long for one
     request, consecutive notes are merged first.
4. **Check the citations** (`citations.py`, `cite_summary`): citations are
   normalized to `[12, 15]`, numbers that name no line are removed and
   counted, and bullet points without a citation are counted. The parser
   accepts the forms models write: `[12]`, `[12, 15]`, `[12-15]`,
   `[#12]`, `[Line 12]`.
5. **Show and export**: the page replaces citations with the segments'
   labels (`format_citations`, `segment_label`: the start time, or
   `line 12` for text without timestamps) and marks uncited points
   (`mark_uncited_points`). Downloads turn citations into Markdown footnotes
   that quote the cited lines (`with_footnotes`, `format_summary_document`).
   The transcript shown and downloaded carries the same labels
   (`transcript_text`).

Requests use a low temperature and disable "thinking", retrying without the
`think` argument when a model or server does not support it. The prompts
contain no model-specific handling, so any Ollama model can be used.

## Context window

Every request sets `num_ctx`. Without it, Ollama uses its default window
(as small as 4096 tokens, depending on version and GPU) and silently drops
the part of the prompt that does not fit, so the summary of a long meeting
would be based on part of it.

- A single-pass request uses the smallest of 8k, 16k and 32k tokens (up to
  the maximum) that holds the prompt plus 4096 tokens for the answer.
  Token counts are estimated at four characters per token and multiplied
  by 1.3, because that undercounts many languages other than English.
- A transcript that does not fit in the maximum is chunked; all map,
  merge and synthesis requests then use the maximum, so Ollama loads the
  model once. Ollama reloads a model whenever `num_ctx` changes.

## Public API

Interfaces import from `service.py` only; it re-exports the names below
from `models.py`, `transcript.py` and `citations.py` (see `__all__`).

| Name | Purpose |
|---|---|
| `Segment`, `CitedSummary` | A transcript line; a summary with checked citations |
| `SUMMARY_MODES` | The summary types, with labels, descriptions and prompts |
| `needs_chunking(segments)` | Whether a transcript must be summarized in chunks |
| `get_summary_stream(model, segments, mode, speaker_context, output_language)` | Stream a single-pass summary |
| `get_partial_notes(..., on_progress=)` | Cited notes per chunk, with progress |
| `get_synthesis_stream(model, partial_notes, mode, output_language)` | Stream the final summary from the notes |
| `segments_from_table`, `segments_from_text`, `speakers`, `rename_speakers`, `build_speaker_context` | Reading transcripts and naming speakers |
| `segment_label`, `transcript_text`, `word_count` | Showing transcripts |
| `cite_summary`, `format_citations`, `mark_uncited_points`, `with_footnotes` | Checking and showing citations |
| `format_summary_document(...)` | The downloadable Markdown document |

`output_language` is a language name such as `"English"`, or `None` to
answer in the transcript's language.

## Files written

None. Transcription writes only to the workspace (see the transcription
README); summaries stay in memory until the user downloads them.

## Configuration

The Ollama server address comes from `OLLAMA_HOST`, set by the launch
script. The models offered are listed in `common/models.json`.

| Variable | Meaning |
|---|---|
| `TEXTLAB_SUMMARY_NUM_CTX` | Largest context window (tokens) for summary requests: 32768 if unset. Raise it on large GPUs to summarize longer meetings in one pass; lower it if the model does not fit in GPU memory |

## Tests

With Ollama faked:

- `tests/test_transcript.py`: reading tables and text into segments,
  sentence splitting, speaker naming, labels.
- `tests/test_citations.py`: the citation forms accepted, removal of
  invalid citations, uncited points, rendering and footnotes.
- `tests/test_service.py`: the context window chosen for each request,
  chunking by whole lines, merging notes that do not fit, citation
  instructions in every prompt, progress, streaming, the retry without
  `think`, and the downloadable document.
