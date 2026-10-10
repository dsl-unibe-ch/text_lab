# Data Handling

Text Lab promises its users that files they upload are processed in their
own session and not left behind. This page describes how the code keeps that
promise and what a developer must do to keep it true.

## Where data may go

| Data | Location | Removed |
|---|---|---|
| Uploads, intermediate files, generated charts | The job workspace (`textlab.common.storage`) | By the feature when a run ends, and always when the session ends |
| Results shown to the user | Streamlit session state | When the page is reloaded or closed, or the session ends |
| Output the user deliberately saves | A location the user chooses (Knowledge Graph output folder) | When the user deletes it |
| Logs | The session's job folder in the user's home directory | When the user deletes it |
| Model files the user downloads (custom Topic Modeling embeddings) | The user's Hugging Face cache | When the user deletes them |

Nothing else may be written: not to the home directory, not to the source
tree, and not to shared folders on research storage.

## The job workspace

At the start of a session the launch script creates one private folder for
the job and exports its path as `TEXT_LAB_WORKDIR`:

- It is created in the job's `$TMPDIR`, or in `TEXT_LAB_WORKDIR_BASE` when
  the site configuration sets it.
- It has owner-only permissions (`700`).
- A `trap` in the launch script deletes it when the session ends, including
  when the job is cancelled or reaches its time limit.
- On UBELIX, Slurm's epilog also deletes `$TMPDIR` (`/scratch/local/<job>`)
  after every job, so even a killed job leaves nothing behind.

The launch script also points `TMPDIR` inside the container at
`$TEXT_LAB_WORKDIR/tmp`. Code that uses Python's `tempfile` module with its
defaults, Ollama and Streamlit's own temporary files therefore all land in
the workspace.

### Using it in code

```python
from textlab.common.storage import get_workspace

workspace = get_workspace()

# A folder for a feature area, created on first use (owner-only):
jobs_dir = workspace.dir("ocr")

# A folder removed when the block ends, also on errors:
with workspace.temp_dir("visualization") as run_dir:
    ...

# A uniquely named folder that lives for a whole session, for example a
# chat session's uploads; removed with the workspace at the latest:
session_dir = workspace.make_temp_dir("chat", prefix="chat-")
```

Each feature uses its own area name, so features cannot delete each other's
files. Worker processes (`textlab.common.jobs`) exchange their request,
progress and result through a job folder in the caller's area, removed when
the run ends; the transcription feature stages uploads the same way
(`staged_uploads`, `staged_zip`), translation writes the PDF pages it
sends to OCR to `translation/ocr-*`, OCR keeps each run in `ocr/job-*`,
`ocr/batch-*` or, for manual engine selection, `ocr/manual-*`, Survey
writes the blank form's images and rebuilt exports to `survey/*`, and Topic
Modeling writes the table it hands to its worker, and the result ZIP, to
`topic_modeling/run-*`.
Temporary folders are removed even when a tool left read-only files in them
(`storage.remove_tree`). Without `TEXT_LAB_WORKDIR` (tests, scripts run by
hand), the workspace is a private folder under the system temporary
directory.

A workspace folder that already exists but belongs to another user is
refused with `WorkspaceError`, so a shared `/tmp` cannot be used to read
another user's files.

## What the tests enforce

`tests/test_data_footprint.py` scans the source code and fails if:

- code builds a path from the home directory (`Path.home()`,
  `expanduser("~")`, `$HOME`, `$XDG_CACHE_HOME`), or
- a string names the UBELIX site (`/storage/`, `unibe.ch`, `ubelix`) instead
  of reading it from the site configuration.

The few legitimate uses are listed in the test with their reason, for
example the Topic Modeling cache for model files. An allow-list entry that
no longer matches anything also fails the test, so the lists stay accurate.

`src/textlab/common/tests/test_storage.py` checks the workspace itself:
permissions, cleanup of temporary folders on errors, and the refusal of
folders owned by someone else.

## When you add or change a feature

- Write temporary files only through `get_workspace()`, in the feature's own
  area. Prefer `temp_dir()` so files disappear as soon as they are not
  needed.
- Do not log file contents or model output wholesale; logs stay in the
  user's home directory after the session.
- If the feature sends data anywhere outside the session (like the
  Knowledge Graph's GPUStack option), say so in the feature's user guide
  page and in the README.
- Update the privacy section of the feature's user guide page and the table
  in the README when what is stored, or where, changes.
