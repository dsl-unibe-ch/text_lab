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
# The image, the model stores and the folders to bind come from the site
# configuration, as for the app, so the tests see the same environment.
#
# Environment overrides:
#   TEXT_LAB_SITE_ENV  site configuration (default: deploy/site.env here)
#   TL_CONTAINER       path to the .sif to test against
#   PYTEST_DEPS        directory holding the pytest installation

#SBATCH --job-name=textlab-tests
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=textlab-tests-%j.out

set -euo pipefail

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

TEXT_LAB_SITE_ENV="${TEXT_LAB_SITE_ENV:-$PWD/deploy/site.env}"
# shellcheck source=/dev/null
source "$TEXT_LAB_SITE_ENV"

echo "Repository: $PWD"
echo "Site:       $TEXT_LAB_SITE_ENV"
echo "Container:  $TL_CONTAINER"
echo "Job:        $SLURM_JOB_ID on $(hostname)"

# The model stores, mounted where the image expects them (as in
# template/script.sh.erb), for the tests that run real models.
binds=(
  --bind "$HOST_WHISPER_DIR:/opt/whisper"
  --bind "$HOST_WHISPERX_DIR:/opt/whisperx"
  --bind "$HOST_TORCH_DIR:/opt/torch"
  --bind "$HOST_HF_DIR:/opt/huggingface"
)
if [[ -n "${TEXT_LAB_EXTRA_BINDS:-}" ]]; then
  binds+=(--bind "$TEXT_LAB_EXTRA_BINDS")
fi

# PYTHONNOUSERSITE keeps packages from ~/.local out of the container
# environment; they break its transformers import.
apptainer exec --nv \
  "${binds[@]}" \
  --env PYTHONNOUSERSITE=1 \
  --env PYTHONPATH="$PYTEST_DEPS" \
  --env HF_HOME=/opt/huggingface \
  "$TL_CONTAINER" \
  "$CONTAINER_PYTHON" -m pytest -p no:cacheprovider "$@"
