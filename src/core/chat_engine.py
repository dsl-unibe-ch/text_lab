import os
import subprocess
import socket
import time
import fitz  
import pandas as pd
from io import BytesIO
import ollama

# --- Server Settings ---
OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434
OLLAMA_MODELS = os.getenv("OLLAMA_MODELS", "/opt/ollama/models")

def _port_open():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((OLLAMA_HOST, OLLAMA_PORT)) == 0

def check_ollama_server():
    """
    Checks if the Ollama server is reachable.
    Returns True if successful, False if it times out.
    """
    for i in range(20):
        if _port_open():
            return True
        time.sleep(0.5)
    return False

# --- Hardware & Model Checks ---

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
    """Returns the name of the GPU (e.g., 'NVIDIA A100-SXM4-80GB')"""
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], 
            encoding="utf-8"
        )
        return result.strip()
    except Exception:
        return "Unknown/CPU"
    
def is_model_loaded(model_name):
    """Checks if the specific model is already loaded in Ollama's VRAM."""
    try:
        running_models = ollama.ps()
        for model in running_models.get('models', []):
            running_name = model.get('name', '')
            if running_name == model_name or running_name.startswith(model_name + ":"):
                return True
        return False
    except Exception:
        return False

# --- File Processing ---

def read_pdf(file_bytes):
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = [page.get_text() for page in doc]
        return "\n".join(text)
    except Exception as e:
        return f"[Error reading PDF: {str(e)}]"

def read_txt(file_bytes):
    try:
        return file_bytes.decode("utf-8")
    except Exception as e:
        return f"[Error reading TXT: {str(e)}]"

def read_tabular(file_bytes, file_name):
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(BytesIO(file_bytes))
        else:
            df = pd.read_excel(BytesIO(file_bytes))
        
        if len(df) > 100:
            return f"[Dataset truncated. First 100 of {len(df)} rows]:\n" + df.head(100).to_markdown()
        return df.to_markdown()
    except Exception as e:
        return f"[Error reading Table: {str(e)}]"

def process_uploaded_files(uploaded_files):
    """
    Process list of uploaded files.
    Returns: (context_text, list_of_warnings)
    """
    if not uploaded_files:
        return "", []

    file_context = "### User Uploaded File Content:\n"
    warnings = []
    has_valid_text = False
    
    for uploaded_file in uploaded_files:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > 10:
            warnings.append(f"File '{uploaded_file.name}' is too large ({file_size_mb:.1f}MB). Skipped (Max 10MB).")
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
            has_valid_text = True
            file_context += f"\n--- Start of file: {uploaded_file.name} ---\n"
            file_context += extracted_text
            file_context += f"\n--- End of file: {uploaded_file.name} ---\n"

    if not has_valid_text:
        return "", warnings

    return file_context, warnings

# --- Generation ---

def get_response_generator(model_name, messages):
    """Yields chunks of the response from Ollama."""
    stream = ollama.chat(model=model_name, messages=messages, stream=True)
    for chunk in stream:
        yield chunk["message"]["content"]