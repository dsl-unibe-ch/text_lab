#!/usr/bin/env bash
# Run the Text Lab test suite inside the Apptainer image on a compute node.
#
# Never run this on a login (submit) node. Either submit it as a batch job
# from the repository root:
#
#   sbatch --qos=job_gratis scripts/test_on_node.sbatch
#   sbatch --qos=job_gratis scripts/test_on_node.sbatch -m "not slow"
#
# or run it directly from inside an interactive allocation (srun/salloc):
#
#   bash scripts/test_on_node.sbatch -k translation
#
# All arguments are passed through to pytest. The image has no pytest, so it
# is installed once into PYTEST_DEPS (see docs/dev/testing.md).
#
# Environment overrides:
#   TL_CONTAINER  path to the .sif to test against
#   PYTEST_DEPS   directory holding the pytest installation

#SBATCH --job-name=textlab-tests
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=textlab-tests-%j.out

set -euo pipefail

# The container path moves into the site configuration in a later phase.
TL_CONTAINER="${TL_CONTAINER:-/storage/research/dsl_shared/solutions/ondemand/text_lab/container/text_lab_210526.sif}"
CONTAINER_PYTHON="/opt/conda/envs/text_lab_main/bin/python"
PYTEST_DEPS="${PYTEST_DEPS:-$HOME/.cache/text_lab_pytest}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Error: not inside a Slurm allocation. Submit with sbatch or run" \
       "from an srun/salloc session on a compute node." >&2
  exit 1
fi

if [[ ! -d "$PYTEST_DEPS" ]]; then
  echo "Error: pytest is not installed in $PYTEST_DEPS." >&2
  echo "See 'One-time setup' in docs/dev/testing.md." >&2
  exit 1
fi

# Under sbatch the script runs from a spool copy, so use the submit directory.
cd "${SLURM_SUBMIT_DIR:-$PWD}"

echo "Repository: $PWD"
echo "Container:  $TL_CONTAINER"
echo "Job:        $SLURM_JOB_ID on $(hostname)"

# PYTHONNOUSERSITE keeps packages from ~/.local out of the container
# environment; they break its transformers import.
apptainer exec --nv \
  --bind /storage \
  --env PYTHONNOUSERSITE=1 \
  --env PYTHONPATH="$PYTEST_DEPS" \
  "$TL_CONTAINER" \
  "$CONTAINER_PYTHON" -m pytest -p no:cacheprovider "$@"
