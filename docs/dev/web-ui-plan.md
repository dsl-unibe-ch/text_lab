# Web UI migration plan

!!! note "Status"
    Plan agreed on 26 November 2026; phase 0 done the same day (see
    [Phase 0 results](#phase-0-results)). Phase 1 built the foundation, the
    launcher and the redeployment files, and the mockups of Home and
    Transcription were agreed on 27 November (see
    [Phase 1 progress](#phase-1-progress)). Phase 2, the pilot, was tried
    in the sandbox and committed on 27 November (see
    [Phase 2 progress](#phase-2-progress)). Phase 3 started with Meeting
    Notes; Translation, OCR and Topic Modeling followed, each tried in the
    sandbox and committed, and Chat was built on 27 November (see
    [Phase 3 progress](#phase-3-progress)). The work happens on the local
    branch `feat/web-ui`, one commit per phase or page, and reaches `main`
    when the whole UI is done. Update this page when a phase ends or a
    decision changes.

Text Lab's user interface moves from Streamlit to a web UI built with
[FastAPI](https://fastapi.tiangolo.com/), Jinja2 templates and
[HTMX](https://htmx.org/), with a fresh, modern and responsive design. The
backend (`features/`, `common/`) stays as it is: since version 3.0.0 every
page calls only its feature's `service.py`, and the new pages do the same.

## Why

- **Long runs tie up the page.** A Streamlit page runs a transcription or
  an OCR batch inside the page script: the tab has to stay open until the
  run ends, and reloading the page loses its result.
- **Two runs can break each other.** Starting a feature releases the GPU
  for it (`gpu_manager.prepare_gpu`), which stops the GPU workers of other
  features, including a run still going in another browser tab.
- **Every click reruns the page script**, which makes pages slow and their
  state fragile (session-state keys, `st.rerun`).
- **The look is Streamlit's.** Layout, components and behaviour on small
  screens can hardly be changed beyond what Streamlit offers.
- **Pages cannot be tested.** Tests can only read the page sources; the
  routes of a web app can be called and checked with FastAPI's test client.

## Decisions

| Question | Decision | Reason |
|---|---|---|
| How users switch | **At once**, when every page exists in the new UI. Streamlit stays selectable for one release as a rollback, then it is removed | A fresh design next to Streamlit pages would feel like two apps, and keeping two UIs doubles every UI change |
| How it is built | **Page by page on one local branch** (`feat/web-ui`), committed per phase or page, merged into `main` once every page exists; then one release and one HPC redeploy | One developer, no reviewers to wait for; production is stable and `main` sees no changes meanwhile (should a fix land on `main`, merge it into the branch) |
| Design | **Fresh**: modern, responsive, easy to use, accessible | |
| Long runs | **Background jobs**, one GPU job at a time, results kept for the session | Users can leave or reload a page, and runs can no longer take the GPU from each other |
| First pages | **Home and Transcription** | Transcription has the cleanest service: a worker process, progress reports and cancelling |

Considered and rejected:

- **Both UIs in one server.** Streamlit 1.61's `st.App` can mount other
  routes, so FastAPI pages could replace Streamlit pages one at a time. That
  only helps a page-by-page switch, which mixes two designs.
- **Streamlit as a permanent second option**: two UIs to keep working for
  every change.

During the migration the Streamlit pages get bug fixes only. New behaviour
goes into the services, where both UIs use it, or into the new pages.

## Architecture

```
src/textlab/ui/
├── launch.py          # starts the UI that TEXT_LAB_UI names
├── streamlit/         # the current UI, removed after the switch
└── web/
    ├── app.py         # app factory: routes, middleware, static files
    ├── auth.py        # login with the session password, cookie, CSRF
    ├── jobs.py        # job queue, progress, cancelling, SSE
    ├── uploads.py     # streams uploads into the job workspace
    ├── render.py      # templates, icons, links from the base path
    ├── settings.py    # the server's settings
    ├── server.py      # uvicorn, started by the launcher
    ├── pages/         # one module per page: its routes; the navigation
    ├── templates/     # Jinja2: base layout, components, one folder per page
    ├── static/
    │   ├── css/       # design system: tokens, layout, components
    │   ├── js/        # app.js (live updates, forms), theme.js
    │   └── vendor/    # htmx, icons, later plotly.js (pinned)
    ├── tests/
    └── README.md      # how a page is built
```

### How a page works

1. `GET /transcription` returns the whole page: the base layout
   (navigation, header, job indicator) around the page's template.
2. The form is sent with HTMX (`hx-post`, multipart). The route checks the
   options, streams the upload into the workspace, submits a job and returns
   a job card for the page's result area.
3. The page follows its jobs over server-sent events (SSE), in one stream
   for all of them (`/events?page=<id>`; [phase 0](#phase-0-results) shows
   why). `static/js/app.js` opens it while one of the session's jobs waits
   or runs, and closes it when the server says that none does. Each event
   carries the changed job cards and the job indicator as out-of-band
   fragments; the card of a finished job holds its result.
4. Downloads are plain links (`/jobs/<id>/files/<name>`).

Rules, checked by tests as for the Streamlit pages today:

- A page module imports only the features' `service` modules, and uses
  only the names they export in `__all__`. What a page needs beyond that
  goes into the service.
- No processing in routes or templates.
- No inline scripts: JavaScript lives in files under `static/`.

### Server

- uvicorn with one worker process: logins, jobs and results live in
  memory, and each Open OnDemand session has one user.
- Routes that call services are plain `def` functions, which FastAPI runs
  in a thread pool. The heavy work already runs in worker processes or in
  Ollama.

## Design

- **A design system in plain CSS**, typographic rather than boxy: ink on
  paper, structure from rules and a grid instead of cards and shadows,
  the red of the Text Lab logo as the one accent. IBM Plex Sans carries
  the text (condensed for titles) and IBM Plex Mono the labels, numbers,
  times and file names; both are vendored (SIL Open Font License), so no
  fonts are loaded from elsewhere. Custom properties for colours, type,
  spacing; light and dark themes that follow the system setting, with a
  switch. The first draft used system fonts and rounded cards, and looked
  like every other web app; the second (27 November 2026) gave the UI a
  character of its own.
- **Responsive layout**: the navigation is a numbered index of the tools,
  grouped by what they work on (audio, documents, text, data), on the
  left on wide screens and behind a menu button on narrow ones. A status
  bar on top shows when the session ends (Slurm's `SLURM_JOB_END_TIME`),
  the jobs, the theme and signing out. Text keeps a readable width; pages
  work from a phone to a wide monitor.
- **Shared components**: page masthead, forms as sheets of numbered
  steps, file drop zone (drag and drop, the chosen files with their
  sizes), options with their explanations, job records with status and
  progress (current step, elapsed time, cancel), result tabs, file
  listings for downloads, notices (information, warning, error with
  details for support), "Models used" lists, job indicator.
- **Icons**: few, from a vendored open-source SVG set (Lucide, ISC
  licence), for arrows, downloads, playing and closing. No emojis.
- **Accessibility**: WCAG 2.1 AA contrast, everything usable with the
  keyboard, visible focus, labelled fields, progress announced to screen
  readers, reduced motion respected.
- **No build step and no CDN.** htmx 2.0.11 (the current release on npm,
  where htmx 4 is tagged `next`, not `latest`), the Lucide icons (1.71.0,
  only those used), the IBM Plex fonts (5.3.0) and later plotly.js are
  vendored in `static/vendor/`.
  Each is pinned and checked against the registry's checksum, with its
  version, source and licence in `static/vendor/README.md`. Browsers fetch
  nothing from outside the cluster. htmx's SSE extension is not used: the
  page's one stream opens and closes with the session's active jobs, which
  a few lines of `app.js` handle.
- **Mockups first.** Static mockups of Home and Transcription are agreed
  before the pilot pages are built.

## Jobs and the GPU

- **A job is one run**: a recording or a ZIP batch to transcribe, a
  document or a batch to read, a topic model to fit. A page submits it and
  shows its progress; the user can leave the page or reload it and find the
  result again.
- **One GPU job at a time.** Further jobs wait in order, and their card
  says so ("waiting, 1 job ahead"). The runner calls
  `gpu_manager.prepare_gpu(<feature>)` when a job starts, not when it is
  submitted, so a waiting job never stops a running one.
- **Short GPU requests** (a chat message, the translation editor, a meeting
  summary) also wait for a running job; the page says which job is running
  and offers to cancel it. If this proves too strict, measure whether a
  small LLM fits on the GPU next to a running job. The translation editor
  and Chat run each request as a job that is not listed (`listed=False`)
  and show it in place (their own `card_template`): a request held open
  while a model loads could pass the proxy's timeout, and a job reports
  progress and can be stopped.
- **Progress** comes from the services' `on_progress` callbacks and reaches
  the page over SSE, in the page's one event stream, which is open only
  while one of its jobs runs or waits. After a reload, the page reconnects
  to its jobs.
- **Cancelling**: a waiting job is removed; a running one gets its
  service's `cancel` event. Only Transcription and Topic Modeling take
  `cancel` today; the other services get it when their page is built
  (`common.jobs.run_worker` already stops workers).
- **Results** stay in memory in the job, as they stay in Streamlit's
  session state today, until the user removes them or the session ends.
  Uploads are staged in the workspace and removed when the job ends, so
  the privacy promises of the user guide stay true.
- **The job indicator** in the header lists running and waiting jobs and
  recent results, on every page.

## Security

- **Login** with the session password shown on the Open OnDemand dashboard
  (`TOKEN`), compared in constant time, then a signed session cookie
  (itsdangerous): `HttpOnly`, `Secure`, `SameSite=Strict`, limited to the
  session's path (`/node/<host>/<port>/`).
- **The user named by the proxy** (`X-Forwarded-User`) is checked too: a
  request naming another user is refused. The header cannot replace the
  password: the server listens on the node's network so that the proxy
  reaches it, and a direct connection can send any header.
- **Connect without copying the password** (decided in phase 1): for the
  web UI, the dashboard's *Connect* button posts the password to the login
  route; only the session's owner sees the card that holds it. The card
  needs to know the interface, which `before.sh.erb` reads from the
  configuration (`template/config.sh`) and passes as `text_lab_ui`
  (`conn_params`). Streamlit sessions show the password to copy, as now.
  A login posted from another origin than the portal is refused.
- **A CSRF token** on every request that changes something (HTMX sends it
  as a header). All Open OnDemand apps share one origin, so the browser
  alone does not keep other pages from sending requests to Text Lab.
- **A Content Security Policy**: scripts only from the app's static files,
  no inline scripts, htmx's `eval` turned off.
- **Uploads**: a size limit (10 GB, as now), safe file names
  (`common.upload_safety`), streamed to the workspace, never held in memory
  whole.
- **Output**: templates escape everything; text from models (Markdown,
  HTML) is shown as text or sanitized (`common.html_safety`).

## Deployment and the switch

- **One HPC redeploy, at the switch.** `template/script.sh.erb` starts
  `python -m textlab.ui.launch` instead of Streamlit. The launcher starts
  the UI that `TEXT_LAB_UI` names (`streamlit` by default, or `web`), with
  today's port, base path and upload limit, so production behaves as
  before, and names its log (`streamlit.log` or `textlab.log`). For
  releases without the launcher the script starts Streamlit directly, so
  rollbacks keep working. The *Connect* change goes into the same
  redeploy; [Deployment](deployment.md#the-launcher-and-the-connect-button)
  lists the files and the order.
- **Sandbox**: `TEXT_LAB_UI=web` in `template/dev.env` shows the new UI,
  and `export TEXT_LAB_WEB_DEV=1` its developer pages under `/dev`: the
  component gallery, the mockups and a demo job.
- **The switch**: the release with the whole UI, the HPC redeploy of its
  Open OnDemand files, then `TEXT_LAB_UI=web` in the live `site.env`.
  Rolling back means setting it back, without another redeploy.
- **Removal**: one release later, the launcher starts only the web UI, and
  Streamlit leaves the image at its next rebuild (`locks/main.txt` and
  `text_lab.def`).

## Phases

| Phase | Content | Done when |
|---|---|---|
| 0. Spike (done) | A minimal FastAPI app in the sandbox, behind the Open OnDemand proxy: base path, login cookie, SSE progress, a 2 GB upload streamed to the workspace, a download | All of it works, or a fallback is chosen (HTMX polling every second instead of SSE) |
| 1. Foundation (done) | The launcher and the redeployment files (deployed at the switch). In `ui/web/`: app factory, login, CSRF, CSP, layout, design system and components, job runner (queue, SSE, cancelling), upload and download helpers, error pages, test harness. Mockups of Home and Transcription | An empty app with login, navigation and a demo job runs in the sandbox, with tests, and the mockups are agreed |
| 2. Pilot (done) | Home and Transcription: a recording or a ZIP batch, language and model, voice activity detection, speaker identification with speaker counts, the audio player with word highlighting, every download | Everything the Streamlit page does works, with tests, and the UI test passes |
| 3. Pages (in progress) | One commit each, in this order: Meeting Notes, Translation, OCR (one page with a switcher, one commit per tool: Standard (the automatic pipeline), Choose an engine (the legacy engines), then historic handwriting and Hugging Face models together; survey extraction left out), Topic Modeling, Chat, Visualize Data, Knowledge Graph | As in phase 2, with the page's user guide drafted |
| 4. Switch | The branch merged into `main`; the new user guide replaces the old one; a release, the HPC redeploy, `TEXT_LAB_UI=web` in production | Users work in the new UI; Streamlit can still be selected as a rollback |
| 5. Removal | `ui/streamlit/` and its tests are removed; Streamlit leaves the image | One release after the switch, without a rollback |

### What is rebuilt

The Streamlit code to replace, about 8,400 lines in all:

| Page | Lines | Notes |
|---|---|---|
| Home | 120 | |
| Transcription | 900 | with the audio player |
| Meeting Notes | 1,270 | transcribes first, then summarizes with citations |
| Translation | 900 | text editor with a model kept loaded, documents |
| OCR | 2,310 | automatic mode, results, review, legacy engines, historic handwriting, Hugging Face models |
| Topic Modeling | 1,060 | charts, editable tables |
| Chat | 520 | streamed answers, documents, charts |
| Visualize Data | 420 | analysis in the background, charts |
| Knowledge Graph | 710 | Grobid, graph view |
| Shared | 200 | login, disclaimer, GPU notices, progress box |

## Phase 0 results

On 26 November 2026 the spike (`scripts/web_ui_spike/`, kept until
phase 1 replaces it) ran in a Slurm job on a GPU node and was tested in
Chrome through the Open OnDemand proxy. Everything the plan depends on
works; one finding changes the design.

| Check | Result |
|---|---|
| Base path | Pages, scripts and style sheets work under `/node/<host>/<port>/`, with the app mounted under that path and every link built from it |
| Login | The session cookie works behind the proxy with `HttpOnly`, `Secure`, `SameSite=Strict` and the session's path, and scripts cannot read it. A plain form posts through the proxy, so *Connect* can post the password too |
| CSRF and CSP | Requests without the token are refused. The strict Content Security Policy causes no violations with htmx 2.0.11 (`includeIndicatorStyles` and `allowEval` off) |
| Live progress | The proxy does not buffer server-sent events: 40 events in 20 seconds arrived on time (spread 0.05 s). With keep-alive comments every 15 seconds, streams lasted over 3 minutes through 50-second gaps without reconnecting. A job reattached after a page reload. Polling works as well |
| Closed pages | The server notices a closed page at its next write to the stream, within 15 seconds with the keep-alive comments, so no stream is left open |
| Uploads | 100 MB, 1,100 MB and 2,200 MB passed the proxy at 50 to 120 MB/s, so no limit applies up to at least 2.2 GB. The server's memory stayed at about 80 MB: uploads go to disk |
| Downloads | A streamed 1,100 MB download arrived complete (68 MB/s); a file name with umlauts was saved correctly |

The proxy forwards the backend's own host and scheme
(`Host: gnode35.hpc.unibe.ch:47983`, `http`), with the original ones in
`X-Forwarded-Host` (`ondemand.hpc.unibe.ch`) and `X-Forwarded-Proto`
(`https`), and the logged-in user in `X-Forwarded-User`.

Consequences, already worked into the sections above:

- **The browser speaks HTTP/1.1 to the proxy**, so it opens at most 6
  connections to `ondemand.hpc.unibe.ch` at a time, for all its tabs on
  the site together, other Open OnDemand apps included. An open event
  stream holds one for as long as it is open. In the spike, three streams,
  two uploads and a download held all six, and polling requests waited:
  the 10-second polling job made 8 requests, against 11 when it ran
  alone. Nothing failed. So each page has **one
  event stream** for all its jobs and the job indicator, open only while a
  job runs or waits, and uploads run one at a time.
- **Links are built from the base path**, never from the request's host
  and scheme, which are the backend's.
- **`X-Forwarded-User` is an extra check**, not a replacement for the
  password (see [Security](#security)).
- **Uploads** are spooled to `TMPDIR`, which the launch script already
  places in the private workspace, then copied to their job's folder
  (1.6 seconds for 2.2 GB).

## Phase 1 progress

Built on 26 November 2026, with tests (`src/textlab/ui/web/tests/`,
`tests/test_launch.py`):

- The launcher and the redeployment files (see
  [Deployment](deployment.md#the-launcher-and-the-connect-button)).
- The app factory under the base path, with the Content Security Policy,
  the check of `X-Forwarded-User` and the upload limit (also for uploads
  that do not announce their size).
- Login with the session password (*Connect* posts it), the session
  cookie, CSRF tokens; every route but the login requires a session.
- The layout (the index of tools, the status bar with the session's end,
  the job indicator and the theme switch, the footer with the
  disclaimer), the design system and the shared components; light and
  dark; narrow and wide screens. The design was redone on 27 November
  after the first review (see [Design](#design)).
- The job runner: a queue, one job at a time, `prepare_gpu` at the start,
  cancelling, progress, results and downloads; short GPU requests
  (`exclusive()`); the event stream and the job cards.
- Error pages and notices, with an error id for unexpected errors.
- Developer pages under `/dev`: the component gallery, the mockups of Home
  and Transcription, and a demo job.

The mockups were agreed on 27 November, with their choices: the language
is detected in the job, from the recording's first 30 seconds, instead of
when the file is chosen (the web page uploads the file only when the job
is submitted); the result shows the player and the transcript together,
with the plain text and the models in tabs; every output file downloads on
its own, besides the ZIP of all of them. The mockups were removed when the
pages were built.

Still to do from phase 1, with phase 2's sandbox test: try the foundation
with `TEXT_LAB_UI=web` (login through *Connect*, the demo job, cancelling,
a reload while a job runs, narrow screens) and without it (Streamlit as
before). The spike (`scripts/web_ui_spike/`) is removed. The release and
the HPC redeployment wait for the switch (phase 4).

## Phase 2 progress

Built on 27 November 2026, with tests
(`src/textlab/ui/web/tests/test_transcription_page.py`, and in
`features/transcription/tests/` for the service's new parts):

- **Home** (`pages/home.py`): what Text Lab is, the tools, the session's
  jobs (updated live from the page's event stream) and where to get help.
  A notice says which tools are not built yet; it goes away by itself when
  every tool is `ready`.
- **Transcription** (`pages/transcription.py`), with the three uses of the
  Streamlit page:
    - *One recording*: a job that prepares the recording for the browser
      (an MP3 and its waveform, `service.prepare_playback`), transcribes it
      and offers every export, and a ZIP of all of them with a WAV copy for
      ELAN, as before.
    - *A ZIP of recordings*: checked when it arrives (it must hold
      recordings), transcribed with the same settings, one ZIP with a folder
      per recording; skipped recordings are listed.
    - *Review a transcript*: a recording and a transcript table, uploaded
      or named by their paths on the cluster. It needs no GPU, so it runs
      beside the queue (`JobRunner.submit(..., at_once=True)`) and does not
      wait for a running transcription.
- **The player**, in `app.js`: the browser's own `<audio>` element plays
  the MP3 from the job's files (with range requests, so long recordings
  start at once and seeking works); the waveform is drawn from levels
  measured on the server. The passage and the word being spoken are
  marked, the transcript follows, clicking a time, a word or the waveform
  plays from there, and the waveform works as a slider from the keyboard.
  Unlike the Streamlit player, it plays the whole recording, not the first
  30 minutes.
- **The service** (`features/transcription/service.py`) exports what pages
  use in `__all__`, with new helpers: the passages for a player
  (`turns`, `turns_from_table`), the recording for the browser
  (`prepare_playback`), the exports on disk (`save_transcript_files`,
  `save_zip`), the recordings in a ZIP (`zip_recordings`) and language
  names. The rules test now also checks names read as `service.x`.
- The user guide's draft: `docs/next/transcription.md`, left out of the
  published site (`exclude_docs` in `mkdocs.yml`).

A test runs a recording through the page with the real pipeline (Whisper
"tiny" on a generated tone, in the image, marked `container` and `slow`):
the worker, ffmpeg, the MP3 and the ZIP with its WAV work together. The
sandbox test passed on 27 November, and phase 2 was committed.

## Phase 3 progress

### Meeting Notes

Built on 27 November 2026, with tests
(`src/textlab/ui/web/tests/test_meeting_notes_page.py`, and in
`features/meeting_notes/tests/` for `summarize`):

- **From a recording**: one job transcribes the recording (with the
  Transcription page's settings: language, speaker counts), then releases
  the GPU for the language model (`JobContext.use_gpu`) and summarizes.
  The transcription has the whole GPU: the job's language model is loaded
  only for the second step. **From a transcript**: a segment table, a text
  file or pasted text; the job names its model (`ollama_model`), so the
  GPU release keeps it loaded.
- **While it runs**, the card shows the summary as the model writes it
  (`JobContext.preview`, at most once a second). Cancelling stops the
  model between tokens.
- **The result**: the summary rendered from Markdown on the server
  (`render.markdown`: raw HTML escaped, no images, links in a new tab),
  with its citations as the passages' times. With a recording, the
  player's controls sit above the tabs and every citation plays its
  passage; without one, a citation shows its line in the Transcript tab.
  Sources, the transcript (with the player, as on Transcription), and the
  downloads of the Streamlit page.
- **Speakers** are named *Speaker 1*, *Speaker 2* by order of appearance,
  as on Transcription, in the summary, its sources and the transcript.
  **Summarize again** (below a result) gives them names and chooses
  another type, model or language: a new job from the same transcript,
  with the recording linked into it, no new transcription. The Streamlit
  page's "Clear GPU memory" is gone: jobs release the GPU themselves.
- **A failed summary** keeps the transcript as the result, with the error
  and the form to summarize again open, as the Streamlit page's retry did.
- `summarize()` in the meeting notes service does the single-pass or
  chunked summary in one call, with progress, the text so far and
  cancelling; `summary_models()` and `server_ready()` also moved out of
  the page. The player became macros shared by both pages
  (`templates/components/player.html`).
- The user guide's draft: `docs/next/meeting_notes.md`.
- Fixed with it: the Menu button showed on wide screens, where the index
  is always visible, so it did nothing there; it now shows only on narrow
  screens, where it opens the index (a test checks the rule order).
- `markdown-it-py` is named in `text_lab.def` (it was in the lock already,
  through `rich`), so it stays if `rich` goes; no rebuild is needed.

Fixed after the first sandbox run: the form's language model field was
called `model`, which the Transcription page's settings (reused here)
read as another Whisper model, so the transcription tried to load the
language model. The field is now `llm`, Meeting Notes reads the
transcription settings without another model
(`read_options(form, other_model=False)`), and a test checks it.

Still to do: try Meeting Notes in the sandbox against the Streamlit page
(both sources, each summary type, a long recording that is summarized in
chunks, citations that play, summarizing again with names, a failure such
as a model that is not installed, cancelling while it writes). Committed
on 27 November (b165b39).

### Translation

Built on 27 November 2026, with tests
(`src/textlab/ui/web/tests/test_translation_page.py`, with faked models
and the real service, and in `features/translation/tests/` for the
service's new parts). The Streamlit page's look and its limits were one of
the reasons for the new UI, so the page was designed anew rather than
ported:

- **One form for both uses**: the languages (From, a swap button, To)
  and two settings shared by text and documents, folded to one line each
  (*Engine* with each engine's description, the model and formality for a
  language model; *Glossary* as rows of term and translation). The
  switcher shows the editor or the documents' steps.
- **The editor**: the text on the left, the translation on the right.
  There is no *Load model* step any more: the first translation with an
  engine loads it and says so. Each translation is a job that is not
  listed (see [Jobs and the GPU](#jobs-and-the-gpu)); a short one comes
  back with the answer to the request (the route waits up to 1.5 s), a
  longer one follows over the event stream, with the previous translation
  dimmed until the new one is ready. A new translation stops and drops the
  one before. **Stop** cancels; a translation that waits names the job it
  waits for and offers to cancel it. *Detect language* is the default
  source: it is detected in the job, and the menu then names it. Ctrl +
  Enter translates, **Translate as I type** translates after a pause (the
  choice is remembered in the browser), the swap button translates the
  translation back, **Copy** copies it. Up to 20,000 characters (longer
  texts go through Documents as `.txt`). The page keeps the text and the
  settings in memory for the session, so a reload shows them again.
- **Documents**: files and ZIP archives, translated by a job with
  cancelling. With *Detect language*, each document's language is
  detected; with a chosen language, that one is used (the Streamlit
  page's separate check box is gone). The result lists each document with
  its language, its files, the PDF checks, and *Compare source and
  translation*: the passages side by side in the page (the first 300; the
  side-by-side file has all). With several files, the record's listing
  offers the ZIP of all; with one, the file itself.
- **The service** (`features/translation/service.py`): `cancel` for
  `translate_text` and `translate_documents` (the document code now lets
  `CancelledError` through where it ignored progress errors or blocked an
  output), each document's pairs in `DocumentResults.pairs`, documents
  without a source language detected or reported, and `languages`,
  `language_name`, `ollama_models`, `server_ready`,
  `SUPPORTED_EXTENSIONS`, `PDF_OUTPUTS` for pages. The engine table gives
  each backend a short `name` and a `description`; the Streamlit page's
  labels are unchanged.
- **The job runner**: `listed=False` jobs stay out of the job indicator
  and Home; `card_template` gives a job its own card; card templates also
  get the `running` job.
- Two Lucide icons added (`arrow-left-right`, `copy`), from the same
  release.
- The user guide's draft: `docs/next/translate.md`.

Still to do: try Translation in the sandbox against the Streamlit page (a
text with each engine, detection, the glossary, swapping, translating as
you type, a long text that follows over the event stream, a translation
while a transcription runs, Stop; documents of each type, a ZIP, a mixed
batch with detection, PDFs with each output and LaTeX, cancelling, a PDF
whose output is blocked; narrow screens and the dark theme), then commit
it. The sandbox test passed on 27 November.

### OCR

One page with a switcher for the four tools (decided on 27 November), one
commit per tool. Survey extraction is left out (see
[Open questions](#open-questions)).

**Standard** (the automatic pipeline; first called Automatic, renamed
after the first review), built on 27 November 2026, with tests
(`src/textlab/ui/web/tests/test_ocr_page.py`, with the pipeline faked and
the rest of the service real, and in `features/ocr/tests/`):

- **One drop zone** for a PDF, an image or a ZIP; a ZIP is read as a batch
  (the Streamlit page's two workflows become one). Options: searchable PDF
  (on by default, also for batches) with the language for placing words,
  highest quality, describe figures. The survey options are gone.
- **The result** shows each page beside its text: the regions outlined on
  the page image (an SVG of `<rect>`s, coloured by type, with a legend),
  and the page's text block by block in reading order, each labelled with
  its type in the same colour. Pointing at an outline or a block marks
  both; clicking an outline brings its text into view. Pages one at a
  time, with a page choice and arrows. Tabs for the tables (each with its
  CSV), the figures (with generated descriptions apart from the printed
  caption), and the models used, for citation. Check boxes as read
  (filled, empty, uncertain), with a notice of uncertain marks. Every
  download in the record's file listing, *Everything* first.
- **A batch** gives the ZIP of the Streamlit page and an overview of its
  files: pages, how many from the text layer, what was found, time.
- **The service** (`features/ocr/service.py`): `cancel` for
  `recognize_document` and `recognize_batch` (the pipeline checks between
  pages and files; the PaddleOCR-VL worker, resident or one-shot, is
  killed at once), `BatchResult.files`, and `save_document`, which writes
  a result's downloads, page images, figures and table CSVs into the job's
  outputs and returns the pages, regions (boxes, sanitized table HTML,
  LaTeX, figure files, descriptions, marks), tables, figures and models to
  show. The page keeps only that, not the document with its images.
- Region colours as tokens (`--region-*`, from the existing hues, so they
  follow the theme); the Lucide `arrow-left` icon added.
- The user guide's draft: `docs/next/ocr.md`.
- Fixed after the first sandbox run: a ZIP batch took 16 minutes, because
  the resident PaddleOCR-VL worker stalled on its first file until the
  stall limit (900 s) stopped it; the one-shot fallback then read the five
  pages in 43 s. The worker inherited the session's `OMP_NUM_THREADS=16`:
  the Streamlit OCR page had set it to 1 for its whole process, which the
  web UI does not do, and the workers only set it if it was unset (after
  numpy had read it). `common.container.paddle_environment` now sets one
  thread in the environment both Paddle workers start with.
  The language of the searchable PDF is labelled as what it is: the
  language for placing words (Tesseract's installed packs), not the
  recognition's.

Standard was tried in the sandbox and committed on 27 November.

**Choose an engine** (the Streamlit page's "legacy engines" expander),
built on 27 November 2026, with tests (`test_ocr_page.py`, with the
engines' recognition faked and the rest real, and `test_engines.py`,
`test_manual.py` in `features/ocr/tests/`):

- **The form**: one drop zone (a PDF, an image, or a ZIP read as a batch,
  as in Standard), the engines as choices with what each is good at, and
  the chosen engine's settings: the language (EasyOCR 86, PaddleOCR 106,
  by name) or what to read (GLM-OCR's three modes); OlmOCR has none. An
  engine that cannot run is shown switched off with the reason (GLM-OCR
  without the Ollama server). The button names the engine.
- **The result** shows each page beside its text, as in Standard: the
  lines (PaddleOCR) or blocks (EasyOCR) an engine found are outlined on
  the page image the engine read, linked to the text; GLM-OCR's tables are
  shown as tables (sanitized) with a CSV each, and its Markdown rendered;
  OlmOCR's one Markdown text is shown without a page. A copy button per
  page. The downloads: *Everything* (the Streamlit ZIP, now with table
  CSVs), the text and the engine's JSON. The Streamlit page showed the
  engines' drawn preview images instead; the web UI does not draw them.
- **A batch** gives the result ZIP and an overview: pages, characters and
  time per file.
- **The engines** (`features/ocr/engines/`): each page keeps the image it
  was read from and its outlined lines (`TextLine`); PaddleOCR's worker
  also returns the straightened page, as PaddleOCR turns and unwarps
  pages before it finds lines and gives their outlines on that image.
  Adapters give a `description`, `markdown`, `ollama_model` (so GLM-OCR's
  model stays loaded when the GPU is released) and `problem()`.
  Cancelling: between pages for EasyOCR and GLM-OCR; PaddleOCR's worker
  and OlmOCR's pipeline are killed at once with the processes they started
  (`run_process` gives each a process group; OlmOCR starts vLLM).
- **The service**: `cancel` for `recognize_with_engine` and
  `recognize_batch_with_engine`, which now returns an `EngineBatch` (the
  ZIP and the overview; the Streamlit page takes its ZIP), and
  `save_engine_run`, which writes the downloads, page images (TIFF and BMP
  as PNG) and table CSVs and returns the pages to show.
- Shared with Standard: the pager bar (`ocr/macros.html`); the `markdown`
  filter; nested switches re-apply the switches inside them; the
  `dropzone` macro takes an `id`, for two drop zones on one page.
- Checked against the image: PaddleOCR 3.3.3 runs its document
  preprocessor (orientation and UVDoc unwarping) by default, so its
  outlines belong to `doc_preprocessor_res.output_img`, which the worker
  now returns. A smoke run on a GPU node (27 November) read a drawn page
  with EasyOCR and PaddleOCR through the service and `save_engine_run`:
  every outline sits on its text, PaddleOCR's on its straightened page,
  where the text had moved about 50 px from the original (outlines on
  the original would have been off by as much). Unwarping also bends a
  flat page slightly, as on the Streamlit page.
- The guide's draft: the *Choose an engine* section of `docs/next/ocr.md`.

Choose an engine was tried in the sandbox and committed on 27 November.

**Historic handwriting** and **Hugging Face model**, built on 27 November
2026 as one step (the user's choice), with tests
(`test_ocr_handwriting_page.py` and `test_ocr_huggingface_page.py`, with
the workers faked and the rest real, and in each feature's `tests/`):

- **Page modules of their own**: `pages/ocr_handwriting.py` and
  `pages/ocr_huggingface.py` hold each tool's route and work; the OCR page
  draws their forms with what their `form_context()` gives. The switcher's
  "Soon" mark is gone, as all four tools are built.
- **Historic handwriting**: one drop zone (PDF, image or ZIP) and the
  configured models as choices, each with its description and the models
  it reads with; the line detector is named. The result shows each page
  beside its numbered lines, each outlined on the page (`<rect>`s in the
  full page's pixels over the page scaled down to 1,600 px) and linked to
  its text, then the models used, for citation. Downloads: everything
  with PAGE XML, the text, the JSON with boxes. A batch gives the ZIP and
  an overview: pages and lines per file.
- **Hugging Face model**: one drop zone, the model's name or link, the
  prompt and the token limit. The user's downloaded models are listed
  with their size: *Use* fills the model field, *Delete* removes one
  (refused while a Hugging Face job of the session waits or runs). The
  result shows each page beside the text as the model wrote it (models
  answer in plain text, Markdown or HTML alike), and the model, revision
  and prompt. A batch gives the ZIP and an overview: pages and characters
  per file.
- **The services**: `cancel` for `transcribe_document`/`transcribe_batch`
  and `read_document`/`read_batch` (between files and steps; the workers
  are stopped at once by `common.jobs.run_worker`); per-file overviews in
  both `BatchRun`s; `save_transcription` and `save_run`, which write the
  downloads and page images and return what to show. Handwriting runs now
  keep the page images, scaled down. The Hugging Face service re-exports
  the user's model folder (`stored_models`, `delete_model`, ...), so the
  page imports only the service.
- **Shared**: `common/outputs.py` (`SavedFile`, `OutputFolder`,
  `web_image`), used by the save functions of all three features (the OCR
  service re-exports `SavedFile`); `uploads.format_names` for the drop
  zones' lists of file types.
- The guide's draft: the last two sections of `docs/next/ocr.md`.

Both were tried in the sandbox and committed together on 27 November.

### Topic Modeling

Built on 27 November 2026, without a mockup (the user asked for the
migration straight away), with tests (`test_topic_modeling_page.py` with
the analysis faked, `features/topic_modeling/tests/test_overview.py`, and
the service tests):

- **The collection first**: choosing a table or a ZIP of text files
  uploads it at once; it is read, and its size, columns (the likeliest
  text column chosen) and first rows come back into the form, with the
  topics-over-time option (timestamp column, intervals; BERTopic only).
  The upload waits in a draft job (`JobRunner.draft`, new); each run
  copies it into a job of its own, so the same collection can run again
  with other settings, each run a record of its own (the Streamlit page
  replaced its result).
- **The form**: the algorithms as choices with what each does, the
  language, each algorithm's settings in a switch (BERTopic's grouping and
  number of topics in nested switches; its language model, keywords,
  dimension reduction and outliers folded), and the stability check.
- **The result**, drawn in SVG from the run's numbers (the libraries'
  Plotly and pyLDAvis charts need inline scripts and a CDN, which the web
  UI does not allow; they stay in the ZIP): the topics beside a map where
  alike topics lie close (dots and topics linked as OCR's outlines and
  text), each topic's keywords with weight bars and a few example
  documents; a similarity heatmap; each topic over time as a small chart
  to its own peak; the quality scores with what each says. Downloads:
  everything with the interactive charts, the documents with their topic,
  the topics with their keywords, and the run's settings and scores.
- **The feature**: a new `overview.py` builds what is drawn the same way
  for every algorithm, from the keyword weights each engine now keeps and
  one vector per topic each engine now gives
  (`bertopic_topic_vectors`, `lda_topic_vectors`,
  `top2vec_topic_vectors`); classical multidimensional scaling places the
  topics. `TopicModelingResult.overview` crosses the worker boundary;
  `summarize_table` and `save_result` serve the page. A failing overview
  keeps the rest of the run, with a notice.
- The guide's draft: `docs/next/topic_modeling.md`.

Tried in the sandbox and committed on 27 November (the user liked the
look and the options).

**Second version of the result** (27 November, after the user's review:
the three algorithms looked the same, while their libraries draw them
differently): each algorithm is now drawn as its own library draws it.

- **LDA, as pyLDAvis**: the intertopic distance map (bubbles placed and
  sized by pyLDAvis, by their share of the words, with its PC axes and
  size legend) beside the terms: the 30 most salient of the collection
  until a topic is chosen, then the chosen topic's 30 most relevant, the
  grey bar a term's overall frequency, the red its frequency in the topic;
  the lambda slider re-ranks them, and pointing at a term sizes the
  bubbles by its spread over the topics (`app.js`). A Documents tab shows
  each topic's documents with their mixture of topics as a thin bar. The
  data is pyLDAvis's own, prepared once for its dashboard and for the page
  (topics in model order, so the numbers match the tables).
- **BERTopic**: its keyword score bar charts per topic, the topic map, the
  documents placed by meaning (UMAP of the embeddings, as
  `visualize_documents`; a sample of 1,500), the hierarchy (a dendrogram,
  average linkage of the topic similarity), the heatmap in the
  hierarchy's order, and the topics over time on one chart with each
  interval's words in the tooltips (BERTopic's own topics over time,
  computed once for its chart and the page).
- **Top2Vec**: its word clouds, the topic sizes, the topic map, the
  documents placed by its document vectors, the heatmap.
- **Emphasis, not a colour per topic** (data-viz guidance: categorical
  colours cap at three for scatter-like charts): grey marks, the topic
  pointed at or chosen in the accent, the rest faded; one `data-region`
  per topic across charts and lists. A muted grey token, `--ink-4`, for
  reference marks, checked with the data-viz validator (3:1 on the sheet,
  apart from the accent with colour-vision deficiency, both themes).
- **Code**: `features/topic_modeling/overview.py` builds the shared parts
  and each library's (`ldavis_parts`, the document map, the hierarchy, the
  words per interval, the clouds, the mixtures); the engines give the
  inputs (`prepare_ldavis`, `lda_document_topics`, `topics_over_time_frame`,
  `topics_over_time_rows`, `top2vec_topic_words`, document vectors).
  `ui/web/pages/topic_charts.py` holds the drawing (`TopicsView`, moved out
  of the page module); `templates/topic_modeling/view_<algorithm>.html`
  and `parts/`.
- Checked with real runs on a CPU node (all three algorithms, a made-up
  corpus of four themes), rendered with the page's templates and looked
  at as images: LDA's map, BERTopic's documents, hierarchy, heatmap and
  over time. Fixed from that look: the size legend moved below the map,
  bubbles kept clear of the edges, the heatmap sized by its topics.

Tried in the sandbox and committed on 27 November.

### Chat

Built on 27 November 2026, without a mockup (the user asked for a modern
chat interface in place of Streamlit's), with tests (`test_chat_page.py`
with the model faked; the service's new parts in
`features/chat/tests/test_service.py`):

- **A conversation** as modern chat apps show it: the messages on the
  right in a tinted block (with the files they were sent with), the
  replies as text, the message box kept at the foot of the window (Enter
  sends, Shift+Enter starts a new line; it grows with the text), the model
  chosen beside Send. Streamlit's sidebar became a side panel: the
  session's conversations, the context meter (with the same notes as
  before), the downloads, deleting the conversation; on narrow screens it
  slides in.
- **Several conversations per session** (the Streamlit page had one, lost
  on a reload): kept in the server's memory (`app.state.chat`) until they
  are deleted or the session ends; *New chat* reuses an empty one.
- **Replies are jobs** that are not listed (`chat/answer.html`), as the
  translation editor's: they wait while another job holds the GPU (and
  say which, with *Cancel that job*), and reach the page word by word over
  the event stream (`context.preview(draft, every=0.15)`). A reasoning
  model's **thinking** shows apart from the answer, folded, with its
  newest words, and *Thought for 12 s* when done. **Stop** stops the model
  at once (the stream is read in a thread of its own, so even while the
  model loads) and keeps what was written. **Again** writes the last
  answer again, with the model chosen now. **Copy** gives the answer as
  Markdown; code blocks have their own.
- **Files**: attached with the button or dropped anywhere on the page, up
  to four, read once (`read_document`); a file without text is not
  attached, and the page says why. Tables are kept for the data agents,
  each in a folder of its own (`keep_table`; one folder for all would let
  a second table overwrite the first).
- **The data agents** answer data questions as before (the model
  decides); the reply shows their steps as they work, then the summary,
  the charts, the statistics and the code. Interactive charts are drawn
  with plotly.js 3.7.0, vendored as its strict bundle (no `unsafe-eval`),
  the version the image's plotly.py writes for; its style rules come from
  a style sheet read from the bundle (`scripts/plotly_css.py`), since its
  own `<style>` element would be blocked. Visualize Data will use the
  same.
- **The feature**: `service.respond` replies to a message (pulling the
  model with progress, routing, the agents or the chat model, stopping);
  the stream gives the model's thinking apart from its text (`Piece`);
  `model_options`, `server_ready`, `discard_table`. The Streamlit page
  works as before (`answer`).
- **The job runner**: previews can be any object and come more often
  (`every=`); `page_data` holds what a card needs besides the job; the
  event stream sends the job indicator only when it changed (a reply
  written word by word changes the runner's version many times a second);
  a page can name its jobs otherwise than by its id (`stream_page`), so
  the stream sends the cards of the conversation shown only, and the page
  skips cards it has in the same version.
- The guide's draft: `docs/next/chat.md`.

Tried in the sandbox and committed on 27 November. The user's review
asked for two changes, made the same day:

- **The width**: the page used the content width of the other pages
  (78rem), which left the right part of a wide window empty. Chat now
  takes the whole width beside the index: the conversation in a wider
  column (60rem) in the middle, the side panel the height of the window
  against the right edge, as the index is against the left. Below 1280px
  the panel slides in behind the Conversations button.
- **Following the answer**: the page did not keep the answer being
  written in view. It now follows it with a `ResizeObserver` on the
  conversation, whatever changes its size, while a reply is written and
  for a few seconds after (its charts); scrolling up stops that, and
  scrolling back to the end or sending a message starts it again. Scroll
  anchoring is off on the page.

The width was right in the sandbox, the following still not, and the
message box stood in the middle of the window with nothing below it. The
cause of both: the conversation's element had the class `transcript`,
which the audio player's list of passages already had, with a height
limit (28rem) and its own scrolling. The conversation was a box of that
height scrolling inside itself: the page could not follow it, and the
message box came right after it. It is the `thread` now, and a test
checks that the Chat section of the style sheet shares no class with the
rest (`test_chat_styles_have_class_names_of_their_own`). Before the first
message, the welcome stands in the middle of the thread, and the message
box at the foot of the window.

Tried in the sandbox and committed on 27 November.

### Visualize Data

Built on 27 November 2026, without a mockup, with tests
(`test_visualize_data_page.py` with the agents faked; the service's new
parts in `features/visualization/tests/test_service.py`):

- **One form**, as Topic Modeling's: the table is read as soon as it is
  chosen, with the agents' own readers, and kept in a draft job while the
  user writes the request; every analysis links it into a job of its own,
  so the same table can be asked about again (Streamlit kept one result,
  lost on a new run or a reload). A new table replaces the one the form
  names; after a reload the form names none, so the session keeps the
  newest three tables.
- **The columns** come back as a table to tick (Streamlit's column
  profile and multiselect in one): the type the agents will give each
  column (number, date, category, or text, from the same rule as their
  summary, `plot_data.column_kind`), how full it is, its distinct values
  and what they are like; *All* and *None*, and a count. The first rows
  are in a second tab. All columns ticked, or none, is the whole table, of
  which the agents are then told nothing (Streamlit sent every column's
  name when all were selected).
- **Requests that name the table's columns** are suggested under the
  request box (a distribution, a comparison of groups, the relation of
  two numbers, an association, a word cloud; `example_requests`), and
  *What the agents can do* replaces Streamlit's capabilities box. The model
  of the last analysis is offered first.
- **An analysis is a job** in the GPU queue. The agents run in the job's
  thread (`service.analyze`, new; Streamlit polled a thread of their own),
  and each of their steps reaches the card as it is taken (the job's
  preview, shown by a template of the page: `preview_template`, new in the
  runner), the newest in view. Cancelling stops them after their current
  step. A missing model is pulled with its progress, and can be
  cancelled meanwhile.
- **The result**: what was asked, the summary, the statistics (open), the
  charts with their Python and R code (a chart without R code says why),
  and the agents' steps folded away; the downloads are the HTML report and
  the ZIP (`service.save_result`, which also writes each chart as a file
  of the job for the page to draw). Charts and statistics are drawn by
  macros Chat now shares (`components/analysis.html`), with the styles
  in a section of their own.
- **Shared**: `common.model_config.model_options` and
  `common.ollama.pull_model` (from Chat), `render.shorten` (from Chat),
  `uploads.copy_into` (from Topic Modeling). Every job card carries
  `data-version`, and the page skips any card it has in the same version
  (Chat's rule, now for all pages), so a finished analysis's charts are
  not drawn again when a new job opens the event stream. Boxes the reader
  opened stay open on any page (`data-keep-open`).
- The rules test now checks every section of the style sheets: a class is
  styled in one section only.
- The guide's draft: `docs/next/visualize_data.md`.

Still to do: try it in the sandbox; then commit. Next: Knowledge Graph.

## Testing

- **Page tests** with FastAPI's test client (httpx is in the image), in
  `src/textlab/ui/web/tests/`: every route renders, forms refuse bad input,
  a job runs from submission to download with a fake service, nothing is
  reachable without logging in, and requests without the CSRF token are
  refused.
- **Import rules**: pages use only the feature services; import-linter
  already keeps the backend from importing any UI.
- **Manual checks**: each page lists in this plan what to try in the sandbox, taken
  from the features of the Streamlit page it replaces.
- Browser tests (Playwright) are not planned: the image has no browser.

## Documentation

- The developer guide and the feature READMEs change with each page's
  commit, as usual.
- The user guide describes what production runs. Each new page's guide is
  written with the page under `docs/next/`, which the published site leaves out
  (`exclude_docs` in `mkdocs.yml`), and replaces the current guide at the
  switch.

## Risks

| Risk | Mitigation |
|---|---|
| The browser opens at most 6 connections to the proxy at a time (HTTP/1.1), for all its tabs on the site together | One event stream per page, open only while jobs are active; one upload at a time |
| Features are lost on the way | A checklist per page, from the Streamlit page; the switch waits for every page |
| New widgets are needed: editable tables (OCR review, Topic Modeling), charts, streamed chat, the graph view, the audio player | HTML tables with inputs, or a vendored grid library; plotly.js vendored; SSE; pyvis output in a frame; the player's own JavaScript ported |
| Most services cannot be cancelled yet | Added with each page |
| Features change in the Streamlit pages during the migration | Streamlit pages get bug fixes only |

## Out of scope

- The command-line and batch tools (`cli.py`, `deploy/sbatch/`) stay
  placeholders.
- More than one user per server: each Open OnDemand session keeps its own
  app.
- Translating the interface: English only, as now.

## Open questions

None at the moment. Decided on 27 November: survey extraction (the
experimental check box of automatic OCR, with its review of the detected
form) is left out of the new UI; the survey feature stays in the backend
for now.
