# Developer Guide

This part of the documentation is for people who work on the Text Lab code:
maintaining features, adding new ones, or deploying the app on a cluster.
The rest of the site is the user guide.

!!! note "Refactor in progress"
    Text Lab is being restructured so that each feature's processing logic
    (the backend) is separate from the Streamlit pages (the frontend). The
    pages describe the target structure and say where the code is today.
    The [migration status](architecture.md#migration-status) table shows how
    far each feature has moved.

## Contents

- [Architecture](architecture.md): how the repository is organized, the
  rules that keep backend and frontend apart, and what a feature package
  contains.
- [Testing](testing.md): where tests live, how to run them on a compute
  node, and what runs automatically on GitHub.
- [Deployment](deployment.md): what the HPC team deploys and what you
  control, the site configuration, sandbox apps, logs and releases.
- [Data handling](data-handling.md): where user data may be written, the
  job workspace, and the tests that keep the privacy promise true.

## Repository at a glance

| Path | What it holds |
|---|---|
| `manifest.yml`, `form.yml`, `submit.yml.erb`, `view.html.erb`, `template/` | Open OnDemand app files: the launch form and the job script that starts the app |
| `src/textlab/features/` | Backend, one package per feature |
| `src/textlab/common/` | Backend code shared by several features |
| `src/textlab/ui/streamlit/` | The Streamlit app: `Home.py`, `pages/`, `auth.py`, `assets/` |
| `deploy/site.env` | Site configuration: paths, images, model stores and settings for one cluster |
| `deploy/container/` | Apptainer definition of the image that holds all dependencies |
| `deploy/sbatch/` | Batch job templates (planned) |
| `scripts/` | Developer scripts, such as running the tests on a compute node |
| `tests/` | Cross-cutting tests; feature tests live in each feature's `tests/` folder |
| `docs/` | This site: user guide and developer guide |

## Running the app from your working tree

A sandbox Open OnDemand app can serve your working tree instead of the
production release: copy `template/dev.env.example` to `template/dev.env`
(gitignored) and set the paths in it. [Deployment](deployment.md) explains
the details, along with the site configuration and releases.

The app imports the `textlab` package, so `src/` must be on `PYTHONPATH`. The
launch script sets it inside the container; for tests, `pyproject.toml` does.

## Conventions

- Code follows PEP 8 (79-character lines) and is checked with `ruff`.
- Every module, class and public function has a docstring in Google style,
  saying what it takes, what it returns and which files it writes.
- No emojis in code, comments, log messages or the user interface.
- Documentation changes with the code: a change that affects behavior,
  structure or deployment updates the README, the user guide and this
  developer guide in the same commit.
- Heavy work (models, containers, test suites) runs on compute nodes, never
  on login nodes.
