import streamlit as st
import subprocess
import socket
import shutil
import time
import ollama
import sys
import os

st.set_page_config(
    page_title="Ollama Chat Interface",
    layout="centered",
    initial_sidebar_state="expanded"
)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from auth import check_token

check_token()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "127.0.0.1")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", 11434))
OLLAMA_MODELS = os.getenv("OLLAMA_MODELS", "/tmp/ollama_models")

# ------------------------------
# 1. Server setup 
# ------------------------------
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
    env.setdefault("OLLAMA_MODELS", OLLAMA_MODELS)

    # 2. Spawn the daemon *once*
    if shutil.which("ollama") is None:
        st.error("`ollama` binary not found in the container.")
        st.stop()

    st.info("Starting Ollama daemon…")
    subprocess.Popen(["ollama", "serve", "--addr", f"{OLLAMA_HOST}:{OLLAMA_PORT}"],
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


def extract_model_name(entry):
    if hasattr(entry, 'model') and isinstance(getattr(entry, 'model'), str):
        return entry.model
    elif isinstance(entry, dict) and "name" in entry:
        return entry["name"]
    elif isinstance(entry, str):
        return entry
    elif isinstance(entry, (tuple, list)) and len(entry) > 0:
        return entry[0]
    else:
        return str(entry)

# ------------------------------
# 2. Generating responses
# ------------------------------
def get_response_generator(model_name, prompt):
    def response_generator():
        for chunk in ollama.generate(model=model_name, prompt=prompt, stream=True):
            if chunk.done:
                break
            yield chunk.response

    return response_generator


def generate_response(messages, model_name):
    prompt = ""
    for msg in messages:
        if msg["role"] == "user":
            prompt += f"User: {msg['content']}\n"
        else:
            prompt += f"Assistant: {msg['content']}\n"
    prompt += "Assistant:"

    response_generator = get_response_generator(model_name, prompt)

    with st.chat_message("assistant"):
        final_response = st.write_stream(response_generator)

    return final_response.strip()


# ------------------------------
# 3. Main UI
# ------------------------------
def main():
    st.markdown(
        """
        <style>
            .main {
                max-width: 800px;
                margin: 0 auto;
            }
            [data-testid="stChatMessage"] {
                border: 1px solid #3f3f3f;
                padding: 1rem;
                border-radius: 0.5rem;
                margin: 0.5rem 0;
            }
            [data-testid="stChatMessage"]:has(div:has-text("User:")) {
                background: #313131;
            }
            [data-testid="stChatMessage"]:has(div:has-text("Assistant:")) {
                background: #1e1e1e;
            }
            .block-container {
                padding-top: 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    ensure_ollama_server()

    # Sidebar
    st.sidebar.title("Model Selection")
    available_models_in_ui = [
        "llama3.2:latest",
        "llama3.1:latest",
        "gemma3:12b",
        "gemma3:27b",
        "deepseek-r1:8b",
        "deepseek-r1:14b",
        "deepseek-r1:70b",
        "qwen2.5:32b",
    ]

    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = available_models_in_ui[0]

    st.session_state["selected_model"] = st.sidebar.selectbox(
        "Select a model:",
        options=available_models_in_ui,
        index=available_models_in_ui.index(st.session_state["selected_model"])
    )

    if st.sidebar.button("🗑️ Start New Chat"):
        st.session_state["messages"] = []
        st.rerun()

    st.sidebar.markdown(
        """
        ---
        ⚠️ **Disclaimer**

        The selected AI models may produce inaccurate, misleading, or inappropriate responses, including hallucinated content. Please verify any critical information independently.

        The University of Bern is **not responsible** for the output generated by the models. Use at your own discretion.
        """,
        unsafe_allow_html=True
    )

    model_name = st.session_state["selected_model"]

    # Get local models
    try:
        models_dict = ollama.list()
        local_models = models_dict["models"]
        local_model_names = [extract_model_name(m) for m in local_models]
    except Exception as e:
        st.error(f"Error listing locally available models: {str(e)}")
        local_model_names = []

    # Pull model if needed
    if model_name not in local_model_names:
        st.write("\n")
        st.write("\n")
        st.info(f"Model '{model_name}' not found locally. Pulling it now. This might take several minutes and is only done once. Please stay on the page if you wish to pull the model")
        try:
            ollama.pull(model=model_name)
            st.success(f"Successfully pulled '{model_name}'.")
        except Exception as e:
            st.error(f"Error pulling model '{model_name}': {str(e)}")

    st.title("Ollama Chat Interface")

    # Chat history state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display previous chat
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input and response
    user_text = st.chat_input("Type your message...")

    if user_text:
        st.session_state["messages"].append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        with st.spinner("Thinking..."):
            assistant_reply = generate_response(st.session_state["messages"], model_name)

        st.session_state["messages"].append({"role": "assistant", "content": assistant_reply})

if __name__ == "__main__":
    main()
