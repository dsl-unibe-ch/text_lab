import streamlit as st
import ollama
import sys
import os
import subprocess
import fitz  
import pandas as pd
from io import BytesIO
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

def get_gpu_name():
    """
    Returns the name of the GPU (e.g., 'NVIDIA A100-SXM4-80GB', 'NVIDIA GeForce RTX 4090')
    """
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], 
            encoding="utf-8"
        )
        return result.strip()
    except Exception:
        return "Unknown/CPU"
    
def is_model_loaded(model_name):
    """
    Checks if the specific model is already loaded in Ollama's VRAM.
    """
    try:
        # ollama.ps() returns a list of currently running models
        running_models = ollama.ps()
        
        # Check if our model is in that list
        # We check 'name' and allow for tag variations (e.g., 'gemma3:27b' vs 'gemma3:27b:latest')
        for model in running_models.get('models', []):
            running_name = model.get('name', '')
            if running_name == model_name or running_name.startswith(model_name + ":"):
                return True
        return False
    except Exception:
        # If the API fails, we assume it's NOT loaded to be safe
        return False

# ------------------------------
# 2. File Processing Functions
# ------------------------------

def read_pdf(file_bytes):
    """Extract text from a PDF file using PyMuPDF (fitz)."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = []
        for page in doc:
            text.append(page.get_text())
        return "\n".join(text)
    except Exception as e:
        return f"[Error reading PDF: {str(e)}]"

def read_txt(file_bytes):
    """Extract text from a plain text file."""
    try:
        return file_bytes.decode("utf-8")
    except Exception as e:
        return f"[Error reading TXT: {str(e)}]"

def read_tabular(file_bytes, file_name):
    """Read CSV or Excel and return a markdown string representation."""
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(BytesIO(file_bytes))
        else:
            df = pd.read_excel(BytesIO(file_bytes))
        
        # Limit to first 100 rows to avoid token explosion
        if len(df) > 100:
            return f"[Dataset truncated. First 100 of {len(df)} rows]:\n" + df.head(100).to_markdown()
        return df.to_markdown()
    except Exception as e:
        return f"[Error reading Table: {str(e)}]"

def process_uploaded_files(uploaded_files):
    """
    Process list of uploaded files and return a single context string.
    Enforces size and count limits.
    """
    if not uploaded_files:
        return ""

    file_context = "### User Uploaded File Content:\n"
    total_size_mb = 0
    
    for uploaded_file in uploaded_files:
        # 1. Size Check (Limit to 10MB per file to be safe)
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > 10:
            st.warning(f"File '{uploaded_file.name}' is too large ({file_size_mb:.1f}MB). Skipped (Max 10MB).")
            continue
        
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name.lower()
        
        extracted_text = ""
        if file_name.endswith('.pdf'):
            extracted_text = read_pdf(file_bytes)
        elif file_name.endswith('.txt'):
            extracted_text = read_txt(file_bytes)
        elif file_name.endswith(('.csv', '.xlsx', '.xls')):
            extracted_text = read_tabular(file_bytes, file_name)
        
        if extracted_text:
            file_context += f"\n--- Start of file: {uploaded_file.name} ---\n"
            file_context += extracted_text
            file_context += f"\n--- End of file: {uploaded_file.name} ---\n"

    return file_context

# ------------------------------
# 3. Generating responses
# ------------------------------

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
# 4. Main UI
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

    # --- GPU Detection & Model Filtering ---
    current_gpu = get_gpu_name()
    small_models = ["gemma3:27b", "ministral-3:14b"]
    large_models = ["qwen3-next:80b", "qwen3-coder-next:latest"]

    is_high_memory_gpu = any(x in current_gpu for x in ["A100", "H100", "H200"])

    if is_high_memory_gpu:
        available_models_in_ui = small_models + large_models
        gpu_badge = f"🚀 **High-Performance Mode** detected ({current_gpu})"
    else:
        available_models_in_ui = small_models
        gpu_badge = f"⚠️ **Standard Mode** detected ({current_gpu}). Large models are hidden."

    # Sidebar
    st.sidebar.title("Model Selection")
    st.sidebar.info(gpu_badge)

    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = available_models_in_ui[0]
    if st.session_state["selected_model"] not in available_models_in_ui:
        st.session_state["selected_model"] = available_models_in_ui[0]

    st.session_state["selected_model"] = st.sidebar.selectbox(
        "Select a model:",
        options=available_models_in_ui,
        index=available_models_in_ui.index(st.session_state["selected_model"])
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 Upload Context")
    
    # --- File Uploader Widget ---
    uploaded_files = st.sidebar.file_uploader(
        "Attach files (Max 4)", 
        type=["pdf", "txt", "csv", "xlsx"], 
        accept_multiple_files=True
    )
    
    # Enforce file count limit
    if uploaded_files and len(uploaded_files) > 4:
        st.sidebar.error("Maximum 4 files allowed. Please remove some.")
        uploaded_files = uploaded_files[:4]

    if st.sidebar.button("🗑️ Start New Chat"):
        st.session_state["messages"] = []
        st.rerun()

    st.sidebar.markdown(
        """
        ---
        ⚠️ **Disclaimer**
        The selected AI models may produce inaccurate, misleading, or inappropriate responses.
        """,
        unsafe_allow_html=True
    )

    model_name = st.session_state["selected_model"]

    # Pull logic...
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
        # 1. Process files if they exist
        context_text = ""
        if uploaded_files:
            with st.spinner("Processing files..."):
                context_text = process_uploaded_files(uploaded_files)
        
        # 2. Construct final message content
        if context_text:
            full_prompt = f"{context_text}\n\nUser Question: {user_text}"
            display_text = f"**[Uploaded {len(uploaded_files)} file(s)]**\n\n{user_text}"
        else:
            full_prompt = user_text
            display_text = user_text

        # 3. Add user message to history
        st.session_state["messages"].append({"role": "user", "content": display_text})
        with st.chat_message("user"):
            st.markdown(display_text)

        # 4. Generate response
        last_msg_obj = st.session_state["messages"][-1]
        original_content = last_msg_obj["content"]
        last_msg_obj["content"] = full_prompt
        
        # --- DYNAMIC SPINNER LOGIC ---
        # Check if model is already in VRAM to give the user a heads-up
        if is_model_loaded(model_name):
            spinner_text = "Thinking..."
        else:
            spinner_text = f"🚀 Loading **{model_name}** into GPU memory... This first run may take 1-2 minutes."

        with st.spinner(spinner_text):
            assistant_reply = generate_response(st.session_state["messages"], model_name)
        
        # Restore original content for display history
        last_msg_obj["content"] = original_content

        st.session_state["messages"].append({"role": "assistant", "content": assistant_reply})

if __name__ == "__main__":
    main()