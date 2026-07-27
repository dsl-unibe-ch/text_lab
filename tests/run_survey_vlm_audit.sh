#!/usr/bin/env bash

# Run the survey audit with TextLab's container dependencies and an isolated
# Ollama server. Invoke this from any directory on an allocated GPU node.
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: bash tests/run_survey_vlm_audit.sh INPUT OUTPUT [audit options]" >&2
    exit 2
fi

AUDIT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDIT_REPO_DIR="$(cd "$AUDIT_SCRIPT_DIR/.." && pwd)"
AUDIT_SIF="${TL_CONTAINER:-$AUDIT_REPO_DIR/text_lab.sif}"
AUDIT_OLLAMA_DIR="${TEXTLAB_OLLAMA_MODELS_DIR:-/storage/research/dsl_shared/solutions/ondemand/text_lab/container/models/ollama}"
AUDIT_PADDLEX_DIR="${TEXTLAB_PADDLEX_DIR:-/storage/research/dsl_shared/solutions/ondemand/text_lab/container/models/paddlex}"
AUDIT_PADDLEOCR_DIR="${TEXTLAB_PADDLEOCR_DIR:-/storage/research/dsl_shared/solutions/ondemand/text_lab/container/models/paddleocr}"
AUDIT_PORT="${TEXTLAB_AUDIT_OLLAMA_PORT:-$((20000 + RANDOM % 10000))}"
AUDIT_HOST="127.0.0.1:$AUDIT_PORT"
AUDIT_LOG="/tmp/textlab_survey_audit_${USER}_${AUDIT_PORT}.log"

if [ ! -f "$AUDIT_SIF" ]; then
    echo "TextLab container not found: $AUDIT_SIF" >&2
    exit 2
fi
if ! command -v apptainer >/dev/null 2>&1; then
    echo "apptainer is required to run the TextLab audit container" >&2
    exit 2
fi

AUDIT_CONTAINER_ARGS=(
    --nv
    --bind /storage:/storage
    --bind "$AUDIT_OLLAMA_DIR:/opt/ollama:rw"
    --env "OLLAMA_MODELS=/opt/ollama/models"
    --env "OLLAMA_HOST=$AUDIT_HOST"
    --env "OLLAMA_FLASH_ATTENTION=true"
    --env "PADDLE_PDX_CACHE_HOME=$AUDIT_PADDLEX_DIR"
    --env "PADDLEX_HOME=$AUDIT_PADDLEX_DIR"
    --env "PADDLEOCR_CACHE=$AUDIT_PADDLEOCR_DIR"
)

apptainer exec "${AUDIT_CONTAINER_ARGS[@]}" "$AUDIT_SIF" ollama serve >"$AUDIT_LOG" 2>&1 &
AUDIT_SERVER_PID=$!
trap 'kill "$AUDIT_SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
    if curl -fsS "http://$AUDIT_HOST/api/tags" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$AUDIT_SERVER_PID" 2>/dev/null; then
        echo "Ollama failed to start; see $AUDIT_LOG" >&2
        exit 1
    fi
    sleep 1
done
if ! curl -fsS "http://$AUDIT_HOST/api/tags" >/dev/null 2>&1; then
    echo "Timed out waiting for Ollama; see $AUDIT_LOG" >&2
    exit 1
fi

apptainer exec \
    "${AUDIT_CONTAINER_ARGS[@]}" \
    --env "PYTHONPATH=$AUDIT_REPO_DIR/src:$AUDIT_REPO_DIR/tests" \
    --pwd "$AUDIT_REPO_DIR" \
    "$AUDIT_SIF" \
    /opt/conda/envs/text_lab_main/bin/python \
    tests/audit_survey_vlm.py \
    "$@" \
    --base-url "http://$AUDIT_HOST"
