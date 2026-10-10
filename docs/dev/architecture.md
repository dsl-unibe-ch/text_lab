# Architecture

Text Lab is a Streamlit app started by an Open OnDemand job inside an
Apptainer image. This page describes how the code is organized and the rules
that keep it maintainable.

## Layers

The code is split into a backend and user interfaces:

- **Backend** (`textlab.features`, `textlab.common`): the processing logic.
  It takes plain inputs (file paths, bytes, option objects) and returns plain
  results. It never renders anything and never reads Streamlit's session
  state.
- **User interfaces** (`textlab.ui`): today the Streamlit app. The pages
  collect input, call the backend and display the result.

Because the backend does not depend on Streamlit, the same functions can
later serve a web frontend (for example one built with HTMX) or command-line
tools that run as Slurm batch jobs, without copying any logic.

## Repository layout

The target layout of the repository:

```
text_lab/
├── manifest.yml  form.yml  submit.yml.erb  view.html.erb  icon.png
├── template/                  # Open OnDemand job scripts (+ dev.env.example)
├── deploy/
│   ├── site.env               # site configuration (reference copy)
│   ├── container/             # Apptainer definition (text_lab.def)
│   └── sbatch/                # batch job templates (planned)
├── scripts/                   # developer scripts (test_on_node.sbatch)
├── pyproject.toml             # tool configuration: ruff, pytest, import-linter
├── src/textlab/
│   ├── common/                # shared backend: config, storage, GPU, safety
│   ├── features/              # backend, one package per feature
│   │   ├── transcription/
│   │   ├── meeting_notes/
│   │   ├── ocr/
│   │   ├── survey/
│   │   ├── translation/
│   │   ├── topic_modeling/
│   │   ├── visualization/
│   │   ├── chat/
│   │   └── knowledge_graph/
│   ├── cli.py                 # `textlab` command (planned)
│   └── ui/
│       └── streamlit/         # Home.py, pages/, components/, assets/
├── tests/                     # cross-cutting tests only
└── docs/                      # user guide and developer guide
```

## Rules

These rules apply to all code in `src/textlab/`.

1. **The backend never imports a user interface.** Nothing in
   `textlab.features` or `textlab.common` imports `streamlit` or
   `textlab.ui`. An import-linter contract in `pyproject.toml` enforces this
   on every CI run.
2. **Plain inputs and outputs.** Backend functions take paths, bytes and
   option data classes and return result data classes. They do not touch
   session state or display anything.
3. **Progress and cancellation are parameters.** Long-running functions
   accept a progress callback and a cancel event instead of drawing progress
   widgets. Each interface passes its own callback.
4. **Settings come from one place.** Paths, hosts and other site-specific
   values come from the site configuration (`deploy/site.env`) and are read
   through `textlab.common.config`, never hardcoded in feature code. See
   [Deployment](deployment.md).
5. **Files are written through one workspace.** Temporary files go to the
   per-job workspace from `textlab.common.storage`, which is private and
   removed when the session ends, so nothing a user uploads stays behind.
   See [Data handling](data-handling.md).
6. **Heavy work runs in worker subprocesses**, started with
   `python -m <module>`, so a crash or a GPU out-of-memory error does not
   take down the app.
7. **Heavy libraries are imported lazily** (inside functions), so modules can
   be imported and tested without a GPU or the full image.
8. **Pages stay thin.** A page holds layout, widgets, session state and calls
   to the backend; anything else belongs in the backend.

Rules 1, 4 and 5 are checked automatically: rule 1 by import-linter, rules 4
and 5 by `tests/test_data_footprint.py`. Rules 3 and 6 have shared modules in
`textlab.common`: `progress` (progress updates and cancellation) and `jobs`
(worker processes). LLM features share `textlab.common.ollama`.

## Feature packages

Each feature package under `src/textlab/features/` holds:

| File | Purpose |
|---|---|
| `service.py` | The public backend functions the interfaces call |
| `models.py` | Data classes for options and results |
| `worker.py` | Subprocess entry point, for features that run heavy models |
| `cli.py` | Command-line interface for batch jobs (placeholder for now) |
| `README.md` | Developer notes for the feature |
| `tests/` | Tests of this feature |

A feature may split its service into more modules when it grows (OCR has one
adapter per engine, for example), but interfaces only import from
`service.py` and `models.py`.

The feature README covers the following; `features/transcription/README.md`
is a complete example:

- **Purpose**: what the feature does, in two or three sentences.
- **Pipeline**: the processing steps, from input to output.
- **Public API**: the functions in `service.py` and their data classes.
- **Files written**: every file or directory the feature creates, and when
  it is removed.
- **Configuration**: environment variables and settings it reads.
- **Tests**: what is covered and which markers the tests use.

## A refactored feature: transcription

Transcription was the first feature refactored and shows how the pieces fit.

```
features/transcription/
├── service.py         # run_transcription(), transcribe_files(), staging
├── models.py          # TranscriptionOptions, Transcript, TranscriptionResult
├── audio.py           # decoding, language detection, VAD
├── formats.py         # exports (CSV, ELAN, SRT, VTT) and readers
├── whisper_models.py  # model choice per language
├── worker.py          # python -m textlab.features.transcription.worker
├── cli.py             # batch command (placeholder)
├── README.md
└── tests/
```

What happens when a user transcribes a recording on the Transcribe page:

1. The page builds `TranscriptionOptions` from its widgets and writes the
   upload to the job workspace with `staged_uploads()`.
2. It calls `run_transcription(files, options, on_progress=status)`, where
   `status` is a `StatusBox` from `ui/streamlit/components/progress.py`.
3. `run_transcription` hands the request to `common.jobs.run_worker`, which
   starts `python -m textlab.features.transcription.worker` with a job
   folder in the workspace.
4. The worker runs `transcribe_files()`, the WhisperX pipeline, and reports
   progress through a file the parent polls and passes to `status`.
5. The worker writes its result and exits, which releases all GPU memory.
   The page receives a `TranscriptionResult`, the staged files are deleted,
   and the page shows the transcript and the downloads from `formats`.

A batch job would call `transcribe_files()` directly, and a web frontend
would call `run_transcription()` with its own progress callback; neither
needs the Streamlit page.

### When a feature runs in the app process: translation

Not every feature uses a worker process. Translation keeps its model loaded
in the Streamlit process between clicks, because the text editor must
answer in a second or two and a worker would reload the model every time.
Its GPU memory is released through `common.gpu_manager` when another
feature needs the GPU. The service is the same kind of API either way:
`translation.service.translate_text()` and `translate_documents()` take
options and an `on_progress` callback, and the page imports nothing else
from the feature. Use a worker when a feature's models are only needed for
one run; keep it in-process when keeping a model loaded is the point.

### Workers in another environment: OCR

Some engines need dependencies that conflict with the app's, so the image
has separate conda environments for them (`common/container.py` names
them). Their workers still start with `python -m textlab.features...`, with
that environment's interpreter: `common.jobs.worker_environment(python)`
makes `textlab` importable and puts the environment's programs and
libraries first. Such a worker must not import anything from `textlab`
beyond the standard library and its own environment, since the app's
packages are not installed there. The PaddleOCR-VL worker of the OCR
feature is the example; it also stays running for a whole batch
(`ocr.vl_session.VLWorkerSession`), because loading its weights takes
longer than recognizing a short document.

### A large page

A page too large for one file keeps the Streamlit script in `pages/` and
moves its parts to a package of the same name in `ui/streamlit/`: the OCR
page has `ui/streamlit/ocr/` with the result tabs, the response review and
its session state. These modules are UI code like the page itself.

## Migration status

All code now lives in `src/textlab/`; the old `src/core/`, `src/pages/` and
`src/tools/` folders are gone. "Moved" means the modules sit in their feature
package with only import and path updates: the page still holds backend
logic, and the module names are still the old ones. "Refactored" means the
feature follows the rules above.

| Feature | Backend | Page | Status |
|---|---|---|---|
| Transcription | `features/transcription/` | `Transcribe.py` | Refactored |
| Meeting Notes | `features/meeting_notes/` | `Meeting_Notes_Generator.py` | Refactored |
| OCR | `features/ocr/` | `OCR.py` | Automatic pipeline refactored; manual engines moved |
| Survey | `features/survey/` | part of `OCR.py` (partly hidden) | Moved |
| Translation | `features/translation/` | `Translate.py` | Refactored |
| Topic Modeling | `features/topic_modeling/` | `Topic_Modeling.py` | Moved |
| Visualization | `features/visualization/` | `Visualize_Data.py` | Moved |
| Chat | `features/chat/` | `Chat.py` | Moved |
| Knowledge Graph | `features/knowledge_graph/` | `Knowledge_Graph.py` | Moved |

Backend paths are relative to `src/textlab/`, pages to
`src/textlab/ui/streamlit/pages/`. Shared code (settings, workspace, worker
processes, progress, Ollama, GPU management, upload and HTML safety, model
and language configuration) is in `src/textlab/common/`.

Known issues to resolve during the refactor:

- The PaddleOCR 2 worker of manual engine selection and the MCP server are
  still started by file path, and `gpu_manager` recognizes leftover workers
  by file or module name; they move to `common.jobs` when manual OCR engines
  (phase 6b) and Visualization are refactored.
- Manual OCR engine selection runs its engines in the UI
  (`ui/streamlit/ocr/legacy.py`, not linted) until phase 6b.
- The home page still names the University of Bern and UBELIX in its text
  (allow-listed in `tests/test_data_footprint.py`).
- Translation calls the OCR feature for scanned PDF pages
  (`translation/documents/pdf_extract.py`, through `ocr.service`), and the
  OCR batch reads questionnaires through `survey.survey_batch`; Survey gets
  its own service in phase 6b.
- About 190 emojis remain in the pages; they are removed as each feature is
  refactored, keeping functional symbols such as checkbox glyphs.

Code that is moved but not refactored is excluded from ruff
(`extend-exclude` in `pyproject.toml`); refactoring a feature removes its
entries.
