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
| `render.py` | Templates, icons, links from the base path, number formats |
| `settings.py` | The server's settings (`TOKEN`, `USER`, `TEXT_LAB_WEB_DEV`) |
| `pages/` | One module per page, and `TOOLS`, the navigation |
| `templates/` | `base.html` (layout), `components/macros.html`, `jobs/`, one folder per page; `mockups/` and `dev/` for the developer pages |
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
and swaps in the cards and the job indicator as they change. Short GPU
requests outside jobs (a chat message) use `runner.exclusive()`.

### Errors

Raise `HTTPException` or `uploads.UploadError` with a message for the user.
For htmx requests the answer is a notice that `app.js` puts into the page;
otherwise an error page. Unexpected exceptions are logged with an error id,
which the user sees.

## Rules

Checked by `tests/test_rules.py`:

- Pages import from the features only their `service` modules, and only the
  names those export in `__all__`.
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
gallery, the mockups of the pages before they are built, and a demo job that
runs through the job runner without a model.

## Tests

`tests/` here, run with the other tests (see `docs/dev/testing.md`):
login and CSRF (`test_auth.py`), pages, headers and errors (`test_app.py`),
the job runner (`test_jobs.py`), jobs through the web with the demo job
(`test_job_pages.py`), uploads (`test_uploads.py`) and the rules
(`test_rules.py`). The launcher and the Open OnDemand files are tested in
`tests/test_launch.py` at the repository root.
