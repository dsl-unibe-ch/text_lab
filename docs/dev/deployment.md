# Deployment

Text Lab is an Open OnDemand interactive app. A session is a Slurm job that
runs `template/script.sh.erb`, which starts Ollama and the Streamlit app
inside an Apptainer image.

## What lives where

| Part | Where | Who changes it |
|---|---|---|
| Open OnDemand files (`manifest.yml`, `form.yml`, `submit.yml.erb`, `view.html.erb`, `icon.png`, `template/`) | Deployed by the HPC team from the repository | Needs a redeployment by the HPC team |
| Site configuration (`site.env`) | Reference copy in `deploy/site.env`; live copy on shared storage | The maintainers, at any time |
| Application code (`src/`) | A release folder on shared storage | The maintainers, at any time |
| Apptainer image (`.sif`) | Built from `deploy/container/text_lab.def`, stored on shared storage | The maintainers, at any time |
| Models (Ollama, Whisper, Hugging Face, OCR) | Model stores on shared storage | The maintainers, at any time |

The deployed launch script contains a single site-specific value: the path of
the live site configuration (`TEXT_LAB_SITE_ENV`). Everything else is read
from that file, so changing code, the image, models or settings never needs
the HPC team.

On UBELIX, the shared folder is
`/storage/research/dsl_shared/solutions/ondemand/text_lab`
(`TEXT_LAB_BASE`):

```
text_lab/
├── site.env              # live site configuration
├── current -> releases/vX.Y.Z
├── releases/             # one checkout per released version
├── container/            # images (.sif) and the model stores (models/)
└── logs/                 # session logs of versions before 3.0.0
```

## How a session starts

1. The user fills in the launch form (`form.yml`) and Open OnDemand submits
   a Slurm job (`submit.yml.erb`).
2. On the compute node, Open OnDemand copies `template/` into the session's
   job folder in the user's home directory and runs `before.sh.erb` (port
   and session token) and then `script.sh.erb`.
3. The launch script loads `dev.env` if one is next to it (sandbox apps
   only), then the site configuration, and checks that the release, the
   image and the model stores it names exist.
4. It resolves `current` to the release folder once, so the session keeps
   the version it started with even if `current` changes meanwhile.
5. It creates the private job workspace, with a trap that deletes it when
   the job ends, and picks random ports for Ollama and Grobid.
6. It starts the image with the model stores and the workspace bound, runs
   `ollama serve` and, once Ollama answers, the Streamlit app
   (`textlab/ui/streamlit/Home.py`) from the release folder.
7. `after.sh` waits for the app's port; the session card then shows
   **Connect**. The app checks the session token on every page
   (`auth.py`).

The top of `output.log` in the job folder lists the site configuration,
source tree, image, workspace and ports the session used.

## What needs whom

| You want to | You change | HPC team needed? |
|---|---|---|
| Release new application code | A new release folder, then `current` | No |
| Roll back to an earlier version | `current` | No |
| Use a new image | Build it, copy it to `container/`, set `TL_CONTAINER` in `site.env` | No |
| Add or update a model | The model store (see [Model stores](#model-stores)) | No |
| Change a setting (paths, endpoints, approved models) | `site.env` | No |
| Change the launch form (GPU types, partitions, QoS, time limits) | `form.yml`, `submit.yml.erb` | Yes |
| Change how a session starts (binds, ports, Ollama options) | `template/` | Yes |
| Move the site configuration | `TEXT_LAB_SITE_ENV` in `template/script.sh.erb` | Yes |

Changes the HPC team deploys should be rare; prefer a setting in
`site.env` that the launch script reads.

## The site configuration

`deploy/site.env` is a shell file sourced by the launch script. It sets:

- **Locations** used by the launch script: the base folder on shared storage,
  the release to serve (`TEXT_LAB_SRC`), the image (`TL_CONTAINER`), the
  model stores, extra paths to bind into the image, the portal host name,
  and optionally where the job workspace goes (`TEXT_LAB_WORKDIR_BASE`).
- **App settings**, exported into the container and read by
  `textlab.common.config`: the Hugging Face token file, the Swiss German
  Whisper models, the Grobid image, the GPUStack endpoint and the Ollama
  models used for OCR enrichment and surveys.

Every assignment has the form `NAME="${NAME:-default}"`, so a value set
earlier (by `template/dev.env`) wins. `tests/test_site_config.py` checks
this, and that every setting the app reads is exported by the file.

### Changing a setting

1. Edit `deploy/site.env` in your working tree.
2. Test it in a sandbox app (see below), which uses the repository copy.
3. Commit the change.
4. Copy the file to its live location, `$TEXT_LAB_BASE/site.env`. New
   sessions use it; running sessions are not affected.

### Adding a setting

1. Add a `SettingSpec` and a field to `textlab.common.config`.
2. Export it in `deploy/site.env` with a comment saying what it is for.
3. Read it in the code with `get_settings()`, or `get_settings().require()`
   where the feature cannot work without it.

### Deploying on another cluster

1. Build the image from `deploy/container/text_lab.def` (or copy it) and
   get a Grobid image if the Knowledge Graph is wanted.
2. Create the shared folder with `container/`, `releases/` and the model
   stores, and fill the stores (see [Model stores](#model-stores)).
3. Copy `deploy/site.env`, change the locations and settings, and put it
   where the deployed launch script expects it (or change
   `TEXT_LAB_SITE_ENV` at the top of `template/script.sh.erb` before
   handing the app to the HPC team).
4. Adjust `form.yml` (cluster name, partitions, QoS and GPU types) for the
   cluster. If `$TMPDIR` is not set per job, or is too small, set
   `TEXT_LAB_WORKDIR_BASE`.
5. Make a release (see [Releases](#releases)) and ask the HPC team to
   deploy the Open OnDemand files.

## Running your working tree in a sandbox app

`template/dev.env` (gitignored) overrides values before the site
configuration is read. Copy `template/dev.env.example` to `template/dev.env`
in your sandbox checkout and set what you want to test:

| To test | Set in `dev.env` | Leave unset |
|---|---|---|
| Your code with the live configuration | `TEXT_LAB_SRC` (your `src/`) | `TEXT_LAB_SITE_ENV` |
| Your code and a changed `site.env` | `TEXT_LAB_SRC`, `TEXT_LAB_SITE_ENV` (the repository's `deploy/site.env`) | |
| A new image with the released code | `TL_CONTAINER` | `TEXT_LAB_SRC` |
| A new image with your code | `TEXT_LAB_SRC`, `TL_CONTAINER` | |
| Exactly what production runs (before a release goes live) | Nothing: remove `dev.env` | Everything |

Open OnDemand copies `template/` into the session's job folder, and the
launch script loads `dev.env` from there. Check the top of `output.log` to
confirm which source tree, configuration and image the session used.

The sandbox runs whatever is checked out in the working tree, so stay on the
branch you are testing.

## Model stores

Models are not part of the image: they live in stores on shared storage,
bound into the image at start (`HOST_*_DIR` in `site.env`). Some features
read files from outside `TEXT_LAB_BASE`, also set in `site.env`.

| Store | Holds | Used by | Where the code names the models |
|---|---|---|---|
| `models/ollama` | Ollama models | Chat, Visualize Data, Meeting Notes, Translation (LLM), OCR enrichments and GLM-OCR, surveys, Knowledge Graph (local) | `common/models.json`; `TEXTLAB_VISION_MODEL`, `TEXTLAB_APPROVED_SURVEY_MODELS`, `TEXTLAB_GLM_OCR_MODEL` in `site.env` |
| `models/whisper` | Whisper checkpoints (`large-v3-turbo.pt`, ...) | Transcription, Meeting Notes | `transcription/whisper_models.py` |
| `models/whisperx`, `models/torch` | WhisperX and PyTorch caches | Transcription | WhisperX defaults |
| `models/huggingface` | Hugging Face cache: alignment models, diarization, translation models, olmOCR, Topic Modeling embeddings | Transcription, Translation, OCR (olmOCR), Topic Modeling | `transcription/service.py`, `translation/engine.py`, `topic_modeling/models.py` |
| `models/easyocr` | EasyOCR detection and recognition models | OCR (EasyOCR) | `ocr/engines/easy_ocr.py` |
| `models/paddleocr`, `models/paddlex` | PaddleOCR and PaddleX models, including PaddleOCR-VL | OCR (automatic pipeline, PaddleOCR) | `ocr/` |
| `TEXT_LAB_CUSTOM_WHISPER_DIR` | Swiss German Whisper models | Transcription (Swiss German) | `transcription/whisper_models.py` |
| `TEXT_LAB_HF_TOKEN_FILE` | Hugging Face token for the gated diarization model | Transcription (speakers) | `site.env` |

### Who can write

The launch script binds the stores read-write, so a library downloads a
missing model into its store, but only for a user who may write there; for
everyone else the download fails and the feature reports an error. Normal
users can only read the stores. On UBELIX (checked in October 2026):

- `ollama` and `whisper` are writable by the maintainers' group
  (`rs_dsl_shared`);
- the other stores are writable only by the user who created them.

The stores' default ACLs give new files no permissions for other users, so
a model a maintainer downloads is unreadable for everyone else until its
permissions are opened (step 3 below).

### Adding or updating a model

1. Add the model to the code or configuration that offers it (for Ollama,
   `common/models.json` or a setting in `site.env`).
2. As a user who can write to the store, download it:
    - Ollama: start a Text Lab session and choose the model in Chat, which
      pulls it; or run `ollama pull <model>` inside the image with
      `OLLAMA_MODELS` pointing to the store.
    - Hugging Face, Whisper, EasyOCR, PaddleOCR: use the feature once in a
      session with the model selected; the library downloads it into the
      bound store.
3. Make the new files readable by everyone, for example
   `chmod -R a+rX "$TEXT_LAB_BASE/container/models/<store>"`.
4. Release the code change as usual.

A fresh installation fills its stores the same way, one feature at a time.
A single command that downloads everything the code names is planned (see
`cli.py`); until it exists, the table above is the list to work through.

## Logs

Each session writes `streamlit.log` and `ollama.log` next to Open OnDemand's
`output.log`, in the session's job folder in the user's home directory. Users
find the folder through the **Session ID** link on their session card and
send the logs with a problem report (see the user guide,
[Reporting a Problem](../launch.md#reporting-a-problem)).

## Releases

The site configuration serves `$TEXT_LAB_BASE/current/src`. `current` is a
symbolic link to a release folder, a checkout of a tagged version, so that a
release or a rollback is a change of the link. Sessions resolve the path at
start, so running sessions keep the version they started with.

### Making a release

1. Merge the changes into `main` and check that CI passes. Merging
   publishes the documentation site, including the user guide.
2. Run the tests on a compute node (`scripts/test_on_node.sbatch`).
3. Tag the commit (`vX.Y.Z`) and create a GitHub release with notes.
4. Check out the tag into a new release folder, readable by everyone:

    ```bash
    cd "$TEXT_LAB_BASE/releases"
    git clone --depth 1 --branch vX.Y.Z \
        https://github.com/dsl-unibe-ch/text_lab.git vX.Y.Z
    chmod -R a+rX,go-w vX.Y.Z
    ```

5. If `deploy/site.env` changed, copy it to `$TEXT_LAB_BASE/site.env`.
6. Try the release in a sandbox app with `TEXT_LAB_SRC` set to the new
   folder's `src/`.
7. Switch new sessions to it:

    ```bash
    ln -sfn releases/vX.Y.Z "$TEXT_LAB_BASE/current"
    ```

8. If the Open OnDemand files changed, ask the HPC team to deploy them from
   the tag.

### Rolling back

Point `current` back at the previous release folder
(`ln -sfn releases/vX.Y.W "$TEXT_LAB_BASE/current"`). New sessions use it
at once; running sessions keep their version. Keep at least the previous
release folder until the new one has run without problems. If the release
also changed `site.env`, restore the previous copy too.

If a release needed the HPC team (changed Open OnDemand files), rolling it
back needs them as well: they redeploy the files of the previous tag.

### A new image

1. Change `deploy/container/text_lab.def` and build the image on a compute
   node.
2. Copy it to `$TEXT_LAB_BASE/container/` under a new name (keep the old
   image: running sessions use it).
3. Run the tests with the new image (`TL_CONTAINER` in
   `scripts/test_on_node.sbatch`) and try it in a sandbox app (`TL_CONTAINER`
   in `dev.env`).
4. Set `TL_CONTAINER` in `site.env`, commit, and copy `site.env` to its live
   location. Rolling back means restoring the previous value.

## First release of this layout (3.0.0)

Versions before 3.0.0 have the path of their code
(`$TEXT_LAB_BASE/src_main280526/src`) in the deployed launch script and
read no site configuration. Version 3.0.0 is the first to read `site.env`
and serve `current`, so the switch needs the HPC team once:

1. Prepare the shared folder without touching the running deployment: the
   release folder `releases/v3.0.0`, `current` pointing to it, and
   `site.env`. The old launch script reads none of them.
2. Rehearse in a sandbox app without `dev.env`: it then runs exactly what
   production will run.
3. Ask the HPC team to deploy the Open OnDemand files from tag `v3.0.0`.
   Merge to `main` close to that date, since merging publishes the new user
   guide.
4. Start a session on the production app and check `output.log` and each
   feature.
5. To roll back, the HPC team redeploys the files of `v2.2.1`, which still
   serve `src_main280526`; keep that folder until 3.0.0 has run without
   problems for a while.
6. Afterwards, remove `src_main280526` and the old versions' session logs
   (`logs/`). `ollama_tmp/` and `run_olmocr.sh` in the shared folder are
   used by no version in the repository; check with their owners before
   removing them.
