# Deployment

Text Lab is an Open OnDemand interactive app. A session is a Slurm job that
runs `template/script.sh.erb`, which starts Ollama and the Streamlit app
inside an Apptainer image.

## What lives where

| Part | Where | Who changes it |
|---|---|---|
| Open OnDemand files (`manifest.yml`, `form.yml`, `submit.yml.erb`, `view.html.erb`, `template/`) | Deployed by the HPC team from the repository | Needs a redeployment by the HPC team |
| Site configuration (`site.env`) | Reference copy in `deploy/site.env`; live copy on shared storage | The maintainers, at any time |
| Application code (`src/`) | A release folder on shared storage | The maintainers, at any time |
| Apptainer image (`.sif`) | Built from `deploy/container/text_lab.def`, stored on shared storage | The maintainers, at any time |
| Models (Ollama, Whisper, Hugging Face, OCR) | Model stores on shared storage | The maintainers, at any time |

The deployed launch script contains a single site-specific value: the path of
the live site configuration (`TEXT_LAB_SITE_ENV`). Everything else is read
from that file, so changing code, the image, models or settings never needs
the HPC team.

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
4. Copy the file to its live location,
   `/storage/research/dsl_shared/solutions/ondemand/text_lab/site.env` on
   UBELIX. New sessions use it; running sessions are not affected.

### Adding a setting

1. Add a `SettingSpec` and a field to `textlab.common.config`.
2. Export it in `deploy/site.env` with a comment saying what it is for.
3. Read it in the code with `get_settings()`, or `get_settings().require()`
   where the feature cannot work without it.

### Deploying on another cluster

Copy `deploy/site.env`, change the locations and settings, and put it where
the deployed launch script expects it (or change `TEXT_LAB_SITE_ENV` at the
top of `template/script.sh.erb` before handing the app to the HPC team).
Adjust `form.yml` (cluster name, partitions, QoS and GPU types) for the
cluster. If `$TMPDIR` is not set per job, or is too small, set
`TEXT_LAB_WORKDIR_BASE`.

## Running your working tree in a sandbox app

`template/dev.env` (gitignored) overrides values before the site
configuration is read. Copy `template/dev.env.example` to `template/dev.env`
in your sandbox checkout and set:

- `TEXT_LAB_SRC` to the `src/` folder of your working tree,
- `TEXT_LAB_SITE_ENV` to the repository's `deploy/site.env`, to test changes
  to it before copying it to shared storage,
- optionally `TL_CONTAINER` to a newly built image.

Open OnDemand copies `template/` into the session's job folder, and the
launch script loads `dev.env` from there. The top of `output.log` shows the
site configuration, source tree, image and workspace the session used.

The sandbox runs whatever is checked out in the working tree, so stay on the
branch you are testing.

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
start, so running sessions keep the version they started with. This layout
is set up when the refactor is released.
