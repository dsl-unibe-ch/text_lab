# Web UI

Text Lab's web UI: [FastAPI](https://fastapi.tiangolo.com/), Jinja2
templates and [htmx](https://htmx.org/). It replaces the Streamlit UI page by
page; until the switch, sessions run it only with `TEXT_LAB_UI=web` (see
`docs/dev/web-ui-plan.md`).

## Running it

In a sandbox app, set in `template/dev.env`:

```bash
TEXT_LAB_SRC="$HOME/ondemand/dev/text_lab/src"
TEXT_LAB_UI=web
export TEXT_LAB_WEB_DEV=1   # the developer pages under /dev
```

Launch the sandbox app and press **Connect to Text Lab**: the button posts
the session password, and you are signed in. The server's log is
`textlab.log` in the session's folder.

The launcher (`textlab/ui/launch.py`) starts the server with the session's
port, base path (`/node/<host>/<port>/`) and the portal's host name;
`server.py` runs uvicorn with one process.

## How it is built

| Module | What it does |
|---|---|
| `app.py` | Builds the app: the pages under the base path, security headers, the check of the proxy's user header, the upload limit, error pages |
| `auth.py` | Login with the session password, the session cookie, CSRF tokens; `LoggedIn`, `Checked` and `protected_router()` for routes |
| `jobs.py` | The job runner (one job at a time, GPU release at start, cancelling), job cards, downloads, the event stream |
| `uploads.py` | Reading forms with files, checking them, saving them into a job |
| `render.py` | Templates, icons, links from the base path, number formats, Markdown from models (`markdown()`: raw HTML escaped, no images) |
| `settings.py` | The server's settings (`TOKEN`, `USER`, `TEXT_LAB_WEB_DEV`) |
| `pages/` | One module per page (`home`, `transcription`, `meeting_notes`, `translation`, `ocr`, `placeholder` for the tools not built yet, `dev`), and `TOOLS`, the navigation |
| `templates/` | `base.html` (layout), `components/macros.html` and `components/player.html` (the audio player), `jobs/`, one folder per page (`pages/` for Home and the placeholders, `transcription/`, `meeting_notes/`, `translation/`, `ocr/`); `mockups/` and `dev/` for the developer pages |
| `static/` | `css/` (`fonts.css`, `tokens.css`, `layout.css`, `components.css`), `js/` (`app.js`, `theme.js`), `vendor/` (htmx, IBM Plex fonts, Lucide icons; see its README) |

### Links

The proxy serves the app under the session's path, and passes requests on
with the node's own host and scheme. Build every link from `root` (in
templates `{{ root }}/...`, in Python `render.root(request)`), never with
`request.url_for`. Static files go through `static('css/x.css')`, which adds
a version so browsers can cache them.

### Login and CSRF

Every route needs a session: use `protected_router()` for a page's router.
Routes that change something take a `Checked` parameter, which also checks
the CSRF token. htmx sends the token with every request (`hx-headers` on
`<body>`); a plain form carries it in a hidden `csrf_token` field. A test
fails if any route but the login is reachable without a session.

### Jobs

A page runs a feature as a job, so users can leave the page and runs never
compete for the GPU:

```python
@router.post("/transcription")
async def submit(request: Request, session: Checked) -> HTMLResponse:
    form = await uploads.read_form(request, max_files=1)
    upload = uploads.single_file(form, "file", suffixes=AUDIO_EXTENSIONS)
    runner = runner_of(request)
    job = runner.create(
        title=uploads.display_name(upload),
        page="transcription",
        gpu=gpu_manager.TRANSCRIBE,
        result_template="transcription/result.html",
    )
    try:
        path = await run_in_threadpool(uploads.save, upload, job.inputs)
    except BaseException:
        runner.discard(job)
        raise
    runner.submit(job, partial(transcribe, path, options))
    return HTMLResponse(render_card(request, job))
```

The work gets a `JobContext`: pass `context.report` as the service's
`on_progress` and `context.cancel` as its `cancel`, write files into
`context.outputs` and offer them with `context.offer(Download(...))`. Its
return value is the job's `result`, which the result template shows.
Uploads in `job.inputs` are deleted when the job ends.

The page lists its jobs with `render_card` and sets `page_id`. `app.js`
opens the page's event stream while one of the session's jobs waits or runs
and swaps in the cards and the job indicator as they change (on Home also
its list of jobs). Short GPU requests outside jobs (a chat message) use
`runner.exclusive()`.

Work with two steps on the GPU releases it again between them with
`context.use_gpu(feature, ollama_model)` (Meeting Notes: transcribing,
then summarizing). A job that uses an Ollama model names it in
`runner.create(..., ollama_model=...)`, so the release keeps it loaded.
Output that grows while the work runs, such as a summary, is shown on the
card with `context.preview(text_so_far)`; updates reach the card at most
once a second.

Short work without the GPU, which should not wait for an hour-long job,
starts at once beside the queue: `runner.submit(job, work, at_once=True)`
(the review of a transcript).

A short request on the GPU that a page shows in place rather than as a
record, such as a translation in Translation's editor, is a job too, so it
waits its turn and its progress reaches the page over the event stream
(a request held open for minutes could time out at the proxy). Create it
with `listed=False`, which keeps it out of the job indicator and Home, and
`card_template=` for its own card (the editor's output pane); that
template must keep the card's id, `job-<id>`, and `data-job-active` while
the job is active. A route can wait a moment for such a job and answer
with its result, so short requests come back at once. Files a result uses but the card should not
list, such as the recording a player plays, are offered with
`Download(..., listed=False)`; they are fetched from the same link.

### Errors

Raise `HTTPException` or `uploads.UploadError` with a message for the user.
For htmx requests the answer is a notice that `app.js` puts into the page;
otherwise an error page. Unexpected exceptions are logged with an error id,
which the user sees.

## Rules

Checked by `tests/test_rules.py`:

- Pages import from the features only their `service` modules, and only the
  names those export in `__all__` (also when they read them as
  `service.name`, or under another name such as
  `from textlab.features.transcription import service as transcription`).
- No inline scripts, event handlers or styles in templates: the Content
  Security Policy blocks them. Behaviour goes into `static/js/app.js`,
  wired through `data-` attributes; sizes and colours into the style
  sheets.
- No emojis. Icons come from `static/vendor/lucide/` through
  `icon("name")`; a test fails for an icon that is not vendored, or one
  vendored but unused.
- Colours, spacing and type come from the tokens in `static/css/tokens.css`,
  in light and dark.

Also: no processing in routes or templates (it belongs in the service), and
no files fetched from outside the cluster (no CDN).

## Design

Typographic: ink on paper, rules and a grid rather than boxes and
shadows, the logo's red as the one accent. Titles in IBM Plex Sans,
condensed; labels, numbers, times and file names in IBM Plex Mono, small
capitals for labels (the `label` class). Forms are sheets of numbered
steps, jobs are records in a ledger, downloads are file listings. Keep to
these patterns, and to the tokens in `static/css/tokens.css`, rather than
adding cards, shadows or icons.

The developer pages (`/dev`, with `TEXT_LAB_WEB_DEV=1`) show the component
gallery, the mockups of pages before they are built (removed once a page
exists), and a demo job that runs through the job runner without a model.

### The audio player

The macros in `templates/components/player.html` make a player: `controls`
(an `<audio>` element that loads the job's MP3 only when played, the time,
a waveform of levels measured on the server) and `transcript` (passages,
`.turn`, and words, `.w`, with their times in `data-s` and `data-e`). Put
both inside one element with `data-player`. `app.js` marks the passage and
word being spoken, keeps the passage in view, plays from any element with
`data-seek` inside it (a time, a word, a citation in Meeting Notes' summary)
and treats the waveform as a slider. Transcription puts the player in a
tab; Meeting Notes puts the controls above its tabs, so citations in the
summary play the recording while it shows.

Without a recording, Meeting Notes' citations carry `data-goto`: `app.js`
selects the tab of the transcript line with that `data-line` in the same
result and marks it.

### Pages beside their text

OCR's result shows a page image with the regions outlined on it: an SVG
over the image, its `viewBox` the image's size in pixels, one `<rect>` per
region (attributes, not styles, so the Content Security Policy allows
them), coloured by type through the `rt-<type>` classes and the
`--region-*` tokens. An outline and the block of its text share a
`data-region`; `app.js` marks both while either is pointed at, and a click
on an outline brings its text into view. Pages show one at a time in a
`data-pager`. Page images and figures are job files (`listed=False`).

## Tests
## Tests

`tests/` here, run with the other tests (see `docs/dev/testing.md`):
login and CSRF (`test_auth.py`), pages, headers and errors (`test_app.py`),
the job runner (`test_jobs.py`), jobs through the web with the demo job
(`test_job_pages.py`), uploads (`test_uploads.py`), the Transcription page
and Home with a faked backend (`test_transcription_page.py`), Meeting Notes
with a faked model (`test_meeting_notes_page.py`), Translation with
faked models and the real service (`test_translation_page.py`), OCR with
the pipeline faked (`test_ocr_page.py`) and the rules
(`test_rules.py`). The fake transcription backend is the `backend` fixture
in `conftest.py`. The launcher and the Open OnDemand files are tested in
`tests/test_launch.py` at the repository root.
