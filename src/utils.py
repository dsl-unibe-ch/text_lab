import streamlit as st
import subprocess
import socket
import shutil
import time
import sys
import os

OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434
OLLAMA_MODELS = os.getenv("OLLAMA_MODELS", "/opt/ollama/models")

def _port_open():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((OLLAMA_HOST, OLLAMA_PORT)) == 0

def ensure_ollama_server():
    # 0. Fast path ─ already up?
    if _port_open():
        return

    # 1. Make sure we have a writable models dir
    os.makedirs(OLLAMA_MODELS, exist_ok=True)
    env = os.environ.copy()
    # Always use the unified models path
    env["OLLAMA_MODELS"] = os.getenv("OLLAMA_MODELS", OLLAMA_MODELS)
    # Force CUDA backend to avoid silent CPU fallback
    env.setdefault("OLLAMA_LLM_LIBRARY", "cuda")
    # Make sure at least one GPU is visible
    env.setdefault("CUDA_VISIBLE_DEVICES", os.getenv("CUDA_VISIBLE_DEVICES", "0"))
    # (Optional) help dynamic linker find CUDA runtime in some setups
    env.setdefault("LD_LIBRARY_PATH", f"/usr/local/cuda/lib64:{env.get('LD_LIBRARY_PATH', '')}")

    # 2. Spawn the daemon *once*
    if shutil.which("ollama") is None:
        st.error("`ollama` binary not found in the container.")
        st.stop()

    st.info("Starting Ollama daemon…")
    subprocess.Popen(["ollama", "serve"],
                     stdout=sys.stdout, stderr=sys.stderr,
                     env=env)

    # 3. Wait (max 30 s) until the TCP port answers
    for _ in range(60):
        if _port_open():
            st.success("Ollama daemon is ready.")
            return
        time.sleep(0.5)

    st.error("Ollama daemon failed to start - check model path and logs.")
    st.stop()