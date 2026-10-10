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
├── template/                  # Open OnDemand job scripts
├── deploy/
│   ├── container/             # Apptainer definition (text_lab.def)
│   └── sbatch/                # batch job templates (planned)
├── scripts/                   # developer scripts (test_on_node.sbatch)
├── pyproject.toml             # tool configuration: ruff, pytest, import-linter
├── src/textlab/
│   ├── common/                # backend code shared by several features
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
   values are read from the environment in one settings module, never
   hardcoded in feature code.
5. **Files are written through one workspace.** Temporary files go to a
   per-job workspace that is created with private permissions and removed
   when the session ends, so nothing a user uploads stays behind.
6. **Heavy work runs in worker subprocesses**, started with
   `python -m <module>`, so a crash or a GPU out-of-memory error does not
   take down the app.
7. **Heavy libraries are imported lazily** (inside functions), so modules can
   be imported and tested without a GPU or the full image.
8. **Pages stay thin.** A page holds layout, widgets, session state and calls
   to the backend; anything else belongs in the backend.

Rules 3 to 5 depend on shared modules (`config`, `storage`, `progress`,
`jobs`) that are added in a later step of the refactor.

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

The feature README covers:

- **Purpose**: what the feature does, in two or three sentences.
- **Pipeline**: the processing steps, from input to output.
- **Public API**: the functions in `service.py` and their data classes.
- **Files written**: every file or directory the feature creates, and when
  it is removed.
- **Configuration**: environment variables and settings it reads.
- **Tests**: what is covered and which markers the tests use.

## Migration status

| Feature | Backend today | Page today | Status |
|---|---|---|---|
| Transcription | `src/core/transcribe_*.py` | `src/pages/Transcribe.py` | Not migrated |
| Meeting Notes | `src/core/summarize_engine.py` | `src/pages/Meeting_Notes_Generator.py` | Not migrated |
| OCR | `src/core/` (OCR modules) | `src/pages/OCR.py` | Not migrated |
| Survey | `src/core/form_extract.py`, `survey_*.py` | part of `src/pages/OCR.py` | Not migrated |
| Translation | `src/core/translation/` | `src/pages/Translate.py` | Not migrated |
| Topic Modeling | `src/core/topic_modeling/` | `src/pages/Topic_Modeling.py` | Not migrated |
| Visualization | `src/core/visualization/` | `src/pages/Visualize_Data.py` | Not migrated |
| Chat | `src/core/chat_engine.py` | `src/pages/Chat.py` | Not migrated |
| Knowledge Graph | `src/core/kg_engine.py` | `src/pages/Knowledge_Graph.py` | Not migrated |

Each feature package already has a `README.md` listing the files that will
move into it.
