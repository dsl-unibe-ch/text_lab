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
| `render.py` | Templates, icons, links from the base path, number formats, Markdown from models (`markdown()`, also a template filter: raw HTML escaped, no images) |
| `settings.py` | The server's settings (`TOKEN`, `USER`, `TEXT_LAB_WEB_DEV`) |
| `pages/` | One module per page (`home`, `transcription`, `meeting_notes`, `translation`, `ocr` with its tools `ocr_handwriting` and `ocr_huggingface` in modules of their own, `topic_modeling`, `placeholder` for the tools not built yet, `dev`), and `TOOLS`, the navigation |
| `templates/` | `base.html` (layout), `components/macros.html` and `components/player.html` (the audio player), `jobs/`, one folder per page (`pages/` for Home and the placeholders, `transcription/`, `meeting_notes/`, `translation/`, `ocr/`, `topic_modeling/`); `mockups/` and `dev/` for the developer pages |
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
its list of jobs). A page whose jobs are named otherwise than by its id
sets `stream_page`: Chat names each reply's job by its conversation
(`chat:<id>`), so the stream sends the cards of the conversation shown
only. Short GPU requests outside jobs can use `runner.exclusive()`; the
pages so far make them jobs instead (see below).

Work with two steps on the GPU releases it again between them with
`context.use_gpu(feature, ollama_model)` (Meeting Notes: transcribing,
then summarizing). A job that uses an Ollama model names it in
`runner.create(..., ollama_model=...)`, so the release keeps it loaded.
Output that grows while the work runs, such as a summary, is shown on the
card with `context.preview(text_so_far)`; updates reach the card at most
once a second. The preview can be any object the card template knows
(Chat's `Draft`: thinking, answer, the agents' steps), and
`context.preview(value, every=0.15)` updates more often, for a reply read
as it is written.

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
`Download(..., listed=False)`; they are fetched from the same link. What
the card template needs besides the job goes into
`runner.create(..., page_data={...})`, such as the conversation and the
message a chat reply belongs to (for its Again button).

### Chat

Each message is answered by a job of its own (`listed=False`, its card
`chat/answer.html` being the reply in the conversation), as the
translation editor's: it waits while another job holds the GPU and says
which, and the reply reaches the page word by word over the event stream.
**Stop** cancels the job; the service stops the model within a quarter of
a second and the job ends *done*, with what was written. The session's
conversations live in `app.state.chat` (`pages/chat.py`, `ChatStore`),
with their attachments (read once, their text kept in memory) and their
exchanges (message, files, job). A reply's history is built from the
finished replies before it.

`app.js` sends with Enter (Shift+Enter for a new line), grows the message
box, turns Send into Stop while a reply is active, attaches files dropped
anywhere on the page, keeps the reply being written in view (unless the
reader scrolled up), keeps a thinking or steps box open across updates if
the reader opened it, adds Copy to code blocks, and asks for the side panel
again when a reply ends. When an event stream opens, it sends every card
of the conversation; the page skips cards it has in the same version
(`data-version`, `htmx:oobBeforeSwap`), so finished replies and their
charts are not drawn again. The page leaves out the layout's footer
(`{% block footer %}`) and shows the disclaimer under the message box.

The data agents' interactive charts are Plotly figures, offered as the
reply job's files (`chart-<n>.json`, not listed) and drawn by `app.js`
with the vendored plotly.js (`static/vendor/plotly-3.7.0/`, the strict
bundle, loaded only on a page with such a chart). Its style rules come
from `plotly.css`, linked as `#plotly.js-style-global.no-inline-styles`,
since plotly.js's own `<style>` element would be blocked; see
`static/vendor/README.md`. The charts take the page's colours and type,
also when the theme changes. Static charts are images.

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
`data-pager`, with the bar of `ocr/macros.html`. Page images and figures
are job files (`listed=False`).

The engine tool's result is built the same way: the text lines or blocks
an engine found are `<polygon>`s (engines give corners, not boxes), each
linked to its line of text; tables an engine wrote are shown sanitized,
and Markdown (OlmOCR, GLM-OCR) through the `markdown` filter.

The OCR page holds four forms in one sheet: the tool switcher shows one,
and the engine form is a switch of its own, showing the chosen engine's
settings. Historic handwriting and Hugging Face model have page modules
of their own (`pages/ocr_handwriting.py`, `pages/ocr_huggingface.py`)
with their routes and work; the OCR page draws their forms with what their
`form_context()` gives. Handwriting outlines its lines with `<rect>`s in
the full page's pixels over a page image scaled down for display. The
Hugging Face form lists the user's downloaded models
(`ocr/hf_models.html`): *Use* fills the model field (`data-fill`), and
*Delete* sends only the model's name (`hx-vals`, `hx-params`) and swaps
the list. Switches can be nested; each re-applies the switches
inside it, so the controls of hidden parts stay disabled. Two drop zones
of the same name on one page need their own `id` (the `dropzone` macro's
`id` argument).

### Charts from data

Topic Modeling draws its charts itself, in SVG, from the numbers the run
returns (`TopicOverview`): the libraries' Plotly and pyLDAvis charts load
their code from a CDN and run inline scripts, which the Content Security
Policy does not allow; they stay in the download. Each algorithm is drawn
as its own library draws it (`templates/topic_modeling/view_<algorithm>.html`
with the parts in `parts/`): LDA as pyLDAvis (a bubble map beside the
chosen topic's terms, ranked by relevance with the lambda slider, which
`app.js` re-ranks), BERTopic with its keyword bar charts, topic map,
document map, hierarchy, heatmap and topics over time, Top2Vec with word
clouds and sizes. `pages/topic_charts.py` works out the geometry
(`TopicsView`), so the templates only place it; bars are SVG `<rect>`s of
the width given, as style attributes are not allowed.

The charts use emphasis, not a colour per topic: everything in grey, the
topic pointed at or chosen in the accent. Marks of one topic share a
`data-region` across the charts and lists, which `app.js` marks together;
`:has()` fades the rest. The muted grey for reference marks (`--ink-4`)
was checked with the data-viz validator: 3:1 on the sheet and apart from
the accent with colour-vision deficiency, in both themes.

### A file chosen before the settings

Topic Modeling needs a table's columns before the run. The drop zone sits
in a wrapper that posts the file as soon as it is chosen (`hx-trigger=
"change"`, `hx-params="file,collection"`); the page reads it, keeps it in
a draft job (`JobRunner.draft`) and answers with its columns and first
rows. The form itself posts everything but the file (`hx-params="not
file"`); each run copies the collection into a job of its own, so the
same collection can run again with other settings.

## Tests

`tests/` here, run with the other tests (see `docs/dev/testing.md`):
login and CSRF (`test_auth.py`), pages, headers and errors (`test_app.py`),
the job runner (`test_jobs.py`), jobs through the web with the demo job
(`test_job_pages.py`), uploads (`test_uploads.py`), the Transcription page
and Home with a faked backend (`test_transcription_page.py`), Meeting Notes
with a faked model (`test_meeting_notes_page.py`), Translation with
faked models and the real service (`test_translation_page.py`), OCR with
the pipeline and the engines faked (`test_ocr_page.py`), historic
handwriting and Hugging Face models with their workers faked
(`test_ocr_handwriting_page.py`, `test_ocr_huggingface_page.py`), Topic
Modeling with the analysis faked (`test_topic_modeling_page.py`), Chat
with the model faked (`test_chat_page.py`) and the rules
(`test_rules.py`). The fake transcription backend is the `backend` fixture
in `conftest.py`. The launcher and the Open OnDemand files are tested in
`tests/test_launch.py` at the repository root.
