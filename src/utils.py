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
    env["OLLAMA_LLM_LIBRARY"] = "cuda"
    # If CUDA_VISIBLE_DEVICES is missing or empty, expose GPU 0
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    env["CUDA_VISIBLE_DEVICES"] = (cvd if cvd else "0")
    # Let the daemon find both the backend and CUDA runtime
    env["LD_LIBRARY_PATH"] = (
        f'{env.get("LD_LIBRARY_PATH", "")}'
        f':/usr/local/lib/ollama'
        f':/usr/local/cuda/lib64'
        f':/usr/local/cuda/targets/x86_64-linux/lib'
    )

    # 2. Spawn the daemon *once*
    if shutil.which("ollama") is None:
        st.error("`ollama` binary not found in the container.")
        st.stop()

    subprocess.Popen(["ollama", "serve"],
                     stdout=sys.stdout, stderr=sys.stderr,
                     env=env)

    # 3. Wait (max 30 s) until the TCP port answers
    for _ in range(60):
        if _port_open():
            return
        time.sleep(0.5)

    st.error("Ollama daemon failed to start - check model path and logs.")
    st.stop()