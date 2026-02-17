import streamlit as st
import ollama
import sys
import os
import subprocess  # <--- Added to run nvidia-smi
from ollama import ResponseError

st.set_page_config(
    page_title="Ollama Chat Interface",
    layout="centered",
    initial_sidebar_state="expanded"
)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from auth import check_token
from utils import ensure_ollama_server

check_token()

# ------------------------------
# 1. Server & Hardware setup 
# ------------------------------

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

# --- NEW FUNCTION: Detect GPU Name ---
def get_gpu_name():
    """
    Returns the name of the GPU (e.g., 'NVIDIA A100-SXM4-80GB', 'NVIDIA GeForce RTX 4090')
    """
    try:
        # Run nvidia-smi to query the GPU name
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], 
            encoding="utf-8"
        )
        return result.strip()
    except Exception:
        # Fallback if nvidia-smi fails or no GPU found
        return "Unknown/CPU"

# ------------------------------
# 2. Generating responses
# ------------------------------
# ... (No changes to response_generator or generate_response functions) ...

def get_response_generator(model_name, messages):
    def response_generator():
        try:
            stream = ollama.chat(model=model_name, messages=messages, stream=True)
            for chunk in stream:
                yield chunk["message"]["content"]
        except ResponseError as e:
            status = getattr(e, "status_code", "?")
            msg = str(e)
            st.error(f"Ollama ResponseError (status={status})")
            st.code(msg)
    return response_generator


def generate_response(messages, model_name):
    response_generator = get_response_generator(model_name, messages)
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
            .main { max-width: 800px; margin: 0 auto; }
            [data-testid="stChatMessage"] { border: 1px solid #3f3f3f; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; }
            [data-testid="stChatMessage"]:has(div:has-text("User:")) { background: #313131; }
            [data-testid="stChatMessage"]:has(div:has-text("Assistant:")) { background: #1e1e1e; }
            .block-container { padding-top: 1rem; }
        </style>
        """,
        unsafe_allow_html=True
    )

    ensure_ollama_server()

    # --- NEW LOGIC: GPU Detection & Model Filtering ---
    current_gpu = get_gpu_name()
    
    # Define models categorized by capability requirements
    small_models = [
        "gemma3:27b",
        "ministral-3:14b"
    ]
    large_models = [
        "qwen3-next:80b",
        "qwen3-coder-next:latest"
    ]

    # Check if we are on a powerful GPU (A100, H100, H200)
    is_high_memory_gpu = any(x in current_gpu for x in ["A100", "H100", "H200"])

    if is_high_memory_gpu:
        # High-end GPU: Show everything
        available_models_in_ui = small_models + large_models
        gpu_badge = f"🚀 **High-Performance Mode** detected ({current_gpu})"
    else:
        # Consumer GPU (RTX) or unknown: Show only small models
        available_models_in_ui = small_models
        gpu_badge = f"⚠️ **Standard Mode** detected ({current_gpu}). Large models are hidden. Use other GPUs for full access."

    # Sidebar
    st.sidebar.title("Model Selection")
    
    # Show the user which hardware is detected
    st.sidebar.info(gpu_badge)

    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = available_models_in_ui[0]

    # Safety check: if session state has a model that is no longer available (e.g. from a previous run), reset it
    if st.session_state["selected_model"] not in available_models_in_ui:
        st.session_state["selected_model"] = available_models_in_ui[0]

    st.session_state["selected_model"] = st.sidebar.selectbox(
        "Select a model:",
        options=available_models_in_ui,
        index=available_models_in_ui.index(st.session_state["selected_model"])
    )
    
    # ... (Rest of your UI code remains exactly the same) ...

    if st.sidebar.button("🗑️ Start New Chat"):
        st.session_state["messages"] = []
        st.rerun()
    
    # ... (Disclaimer and rest of main function) ...
    st.sidebar.markdown(
        """
        ---
        ⚠️ **Disclaimer**
        The selected AI models may produce inaccurate, misleading, or inappropriate responses...
        """,
        unsafe_allow_html=True
    )
    
    model_name = st.session_state["selected_model"]
    
    # ... (Pull logic and Chat Interface loop) ...
    try:
        models_dict = ollama.list()
        local_models = models_dict["models"]
        local_model_names = [extract_model_name(m) for m in local_models]
    except Exception as e:
        st.error(f"Error listing locally available models: {str(e)}")
        local_model_names = []

    if model_name not in local_model_names:
        st.write("\n\n")
        st.info(f"Model '{model_name}' not found locally. Pulling it now...")
        try:
            ollama.pull(model=model_name)
            st.success(f"Successfully pulled '{model_name}'.")
        except Exception as e:
            st.error(f"Error pulling model '{model_name}': {str(e)}")

    st.title("Ollama Chat Interface")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

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