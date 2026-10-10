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

## Repository at a glance

| Path | What it holds |
|---|---|
| `manifest.yml`, `form.yml`, `submit.yml.erb`, `view.html.erb`, `template/` | Open OnDemand app files: the launch form and the job script that starts the app |
| `src/textlab/` | The Python package, organized by feature (target structure) |
| `src/core/`, `src/pages/`, `src/Home.py` | Code not yet migrated into `src/textlab/` |
| `deploy/container/` | Apptainer definition of the image that holds all dependencies |
| `deploy/sbatch/` | Batch job templates (planned) |
| `scripts/` | Developer scripts, such as running the tests on a compute node |
| `tests/` | Tests; feature tests move next to their feature during the refactor |
| `docs/` | This site: user guide and developer guide |

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
