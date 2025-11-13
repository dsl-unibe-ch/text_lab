#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# CONFIGURATION
###############################################################################

# --- Path to your SIF file ---
SIF_FILE="/storage/research/dsl_shared/solutions/ondemand/text_lab/container/text_lab_121125.sif"

# --- Host path for Ollama models ---
HOST_OLLAMA_DIR="/storage/research/dsl_shared/solutions/ondemand/text_lab/container/models/ollama"

# --- Container path this directory is mapped to ---
CONT_OLLAMA_DIR="/opt/ollama"

# --- A unique name for the Apptainer instance ---
INSTANCE_NAME="ollama_pull_service"

# --- List of models to pull ---
MODELS_TO_PULL=(
    "gpt-oss:20b"
    "deepseek-r1:8b"
    "qwen3-coder:30b"
    "gemma3:27b"
    "qwen3:4b"
)

###############################################################################
# SCRIPT LOGIC
###############################################################################

# --- Cleanup function to stop the container instance ---
# This trap ensures the instance is stopped when the script exits,
# whether it finishes successfully (EXIT) or is interrupted (INT, TERM).
cleanup() {
    echo ""
    echo "-----------------------------------------------------------------"
    echo "Stopping container instance '$INSTANCE_NAME'..."
    apptainer instance stop "$INSTANCE_NAME"
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# 1. Check for SIF file
if [ ! -f "$SIF_FILE" ]; then
    echo "FATAL: Container file not found at $SIF_FILE"
    exit 1
fi

# 2. Ensure the host model directory exists
mkdir -p "$HOST_OLLAMA_DIR/models"

# 3. Force-stop any old instances, just in case
echo "Cleaning up any old instances..."
apptainer instance stop -f "$INSTANCE_NAME" > /dev/null 2>&1 || true

# 4. Start the new container instance
echo "Starting container instance '$INSTANCE_NAME'..."
apptainer instance start \
    --nv \
    --bind "$HOST_OLLAMA_DIR:$CONT_OLLAMA_DIR" \
    --env OLLAMA_MODELS="$CONT_OLLAMA_DIR/models" \
    "$SIF_FILE" \
    "$INSTANCE_NAME"

# 5. Start 'ollama serve' in the background *inside* the instance
echo "Starting 'ollama serve' inside the instance..."
apptainer exec "instance://$INSTANCE_NAME" ollama serve &

# Give the server time to start up
echo "Waiting 10 seconds for the server to initialize..."
sleep 10

# 6. Loop and pull models using the running instance
echo "Starting to pull ${#MODELS_TO_PULL[@]} models..."
echo "-----------------------------------------------------------------"

for model in "${MODELS_TO_PULL[@]}"; do
    echo "Pulling model: $model"
    
    # Run 'ollama pull' *inside* the running instance
    apptainer exec "instance://$INSTANCE_NAME" \
        ollama pull "$model"
    
    if [ $? -ne 0 ]; then
        echo "WARNING: Failed to pull $model. Continuing with next model..."
    else
        echo "Successfully pulled $model"
    fi
    echo "-----------------------------------------------------------------"
done

echo "Ollama model pull script finished."
# The 'trap' will automatically call the 'cleanup' function now.