OLM_VER="0.11.6"
OLLAMA_DIR="../container/models/ollama"
TMP="/tmp/ollama_pkgs"

set -e

mkdir -p "$TMP" "$OLLAMA_DIR"
pushd "$TMP"
curl -fL "https://github.com/ollama/ollama/releases/download/v${OLM_VER}/ollama-linux-amd64.tgz" -o ollama.tgz
tar -xzf ollama.tgz
ls -al
popd
cp -r "$TMP/lib/ollama/*" "$OLLAMA_DIR"
rm -rf "$TMP"

echo ">>> Starting Ollama server in the background"
bin/ollama serve > /tmp/ollama-build.log 2>&1 &
OLLAMA_PID=$!

for m in \
    gemma3:12b \
    gemma3:27b \
    deepseek-r1:8b \
    deepseek-r1:14b \
    deepseek-r1:70b \
    llama3.2:latest \
    llama3.1:latest \
    qwen2.5:32b; do
    echo "---> pulling Ollama model: ${m}"
    bin/ollama pull "${m}" || echo "WARNING: pull failed for ${m}"
done

# Shut the Server Down
kill "$OLLAMA_PID" 2>/dev/null || true
wait "$OLLAMA_PID" 2>/dev/null || true
echo ">>> Ollama server stopped"
