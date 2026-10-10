# Testing

## Where tests live

- **Feature tests** live next to their feature, in
  `src/textlab/features/<feature>/tests/`.
- **Cross-cutting tests** live in `tests/`: checks that no user data is
  written outside the job workspace and that no site-specific values are
  hardcoded (`test_data_footprint.py`), and that the site configuration
  provides every setting (`test_site_config.py`).

Shared code has its tests in `src/textlab/common/tests/`. Each `tests/`
folder is a package (it has an `__init__.py`), so a test can import helpers
from another test module by its full name, for example
`from textlab.features.translation.tests.test_translation_pdf import make_pdf`.

## Markers

Tests that need more than plain Python are marked, so they can be selected
or skipped:

| Marker | Meaning |
|---|---|
| `gpu` | Needs a CUDA GPU |
| `container` | Needs the Text Lab image (models, binaries such as Tesseract) |
| `ollama` | Needs a running Ollama server |
| `slow` | Takes more than about a minute |

Unregistered markers are an error (`--strict-markers`), so add new ones to
`pyproject.toml` first.

## Running the tests

The tests run inside the Apptainer image on a compute node, never on a login
node. `scripts/test_on_node.sbatch` wraps the command.

### One-time setup

The image does not include pytest. Install it once into your home directory,
from a compute node:

```bash
SIF=/storage/research/dsl_shared/solutions/ondemand/text_lab/container/text_lab_210526.sif
apptainer exec --env PYTHONNOUSERSITE=1 "$SIF" \
  /opt/conda/envs/text_lab_main/bin/python -m pip install \
  --target "$HOME/.cache/text_lab_pytest" pytest
```

Warnings from pip's dependency resolver during this install are harmless.

### As a batch job

From the repository root:

```bash
sbatch --qos=job_gratis scripts/test_on_node.sbatch
sbatch --qos=job_gratis scripts/test_on_node.sbatch -m "not slow"
```

The output goes to `textlab-tests-<jobid>.out` in the repository root.
Arguments after the script name are passed to pytest.

### In an interactive session

From an `srun` or `salloc` session on a compute node:

```bash
bash scripts/test_on_node.sbatch -k translation
```

The script refuses to run outside a Slurm allocation. `TL_CONTAINER` selects
a different image, for example a newly built one.

## Lint and architecture checks

These are lightweight and also run on GitHub for every push:

```bash
ruff check .            # PEP 8, docstrings, import order, likely bugs
ruff format --check .   # formatting
lint-imports            # backend never imports a user interface
```

`lint-imports` needs `src` on the Python path (`PYTHONPATH=src`). Code that
has been moved but not refactored yet is excluded from ruff
(`extend-exclude` in `pyproject.toml`); each refactored feature is removed
from that list.

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request:

- ruff lint and format checks
- a syntax check of all Python files
- the import-linter contracts
- a strict build of this documentation site

GitHub's machines have no GPU and no image, so they will run only the tests
without markers. That job is added once the tests have moved into the
feature packages.
