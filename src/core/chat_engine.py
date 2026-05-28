import os
import subprocess
import socket
import time
from io import BytesIO
from typing import List, Dict, Tuple, Generator, Any, Union

import fitz  
import pandas as pd
import ollama

# --- Server Configuration ---
ENV_HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")

if ":" in ENV_HOST:
    _clean_host = ENV_HOST.replace("http://", "").replace("https://", "")
    OLLAMA_HOST = _clean_host.split(":")[0]
    OLLAMA_PORT = int(_clean_host.split(":")[1])
else:
    OLLAMA_HOST = ENV_HOST.replace("http://", "").replace("https://", "")
    OLLAMA_PORT = 11434

OLLAMA_MODELS = os.getenv("OLLAMA_MODELS", "/opt/ollama/models")

# --- Context / Token Management ---
CHARS_PER_TOKEN = 4          # rough approximation for token counting
MAX_CONTEXT_TOKENS = 75_000  # trigger chunked processing above this (~300K chars)
CHUNK_SIZE_TOKENS = 14_000   # tokens per chunk sent to the LLM (~56K chars)

# --- System Prompt ---
SYSTEM_PROMPT = (
    "You are a helpful, knowledgeable, and concise AI assistant. "
    "Answer questions accurately and honestly. "
    "If you are unsure about something, say so rather than guessing. "
    "When analysing documents or data provided by the user, focus on the content given "
    "and clearly indicate when you are drawing on your own knowledge versus the provided material."
)


def is_port_open(host: str, port: int) -> bool:
    """
    Check if a specific network port is open and accepting connections.

    Args:
        host (str): The hostname or IP address.
        port (int): The port number to check.

    Returns:
        bool: True if the port is open, False otherwise.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def check_ollama_server() -> bool:
    """
    Verify that the Ollama server is reachable on the configured host and port.
    Attempts to connect up to 20 times with a 0.5-second delay between attempts.

    Returns:
        bool: True if the server is reachable, False if it times out.
    """
    for _ in range(20):
        if is_port_open(OLLAMA_HOST, OLLAMA_PORT):
            return True
        time.sleep(0.5)
    return False


# --- Hardware & Model Checks ---

def extract_model_name(entry: Union[str, Dict[str, Any], Any]) -> str:
    """
    Safely extract the model name from various Ollama API response formats.

    Args:
        entry: The model entry object, dictionary, or string.

    Returns:
        str: The extracted model name.
    """
    if hasattr(entry, 'model') and isinstance(getattr(entry, 'model'), str):
        return entry.model
    if isinstance(entry, dict) and "name" in entry:
        return entry["name"]
    if isinstance(entry, str):
        return entry
    if isinstance(entry, (tuple, list)) and len(entry) > 0:
        return str(entry[0])
    return str(entry)


def get_gpu_name() -> str:
    """
    Query nvidia-smi to retrieve the name of the installed GPU.

    Returns:
        str: The GPU name (e.g., 'NVIDIA A100-SXM4-80GB') or 'Unknown/CPU' on failure.
    """
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            encoding="utf-8",
            stderr=subprocess.DEVNULL
        )
        return result.strip()
    except Exception:
        return "Unknown/CPU"


def is_model_loaded(model_name: str) -> bool:
    """
    Check if a specific language model is currently loaded in Ollama's VRAM.

    Args:
        model_name (str): The target model identifier.

    Returns:
        bool: True if loaded, False otherwise.
    """
    try:
        running_models = ollama.ps()
        for model in running_models.get('models', []):
            running_name = model.get('name', '')
            if running_name == model_name or running_name.startswith(f"{model_name}:"):
                return True
        return False
    except Exception:
        return False


# --- File Processing ---

def read_pdf(file_bytes: bytes) -> str:
    """
    Extract text content from a PDF file.

    Args:
        file_bytes (bytes): The raw bytes of the PDF file.

    Returns:
        str: The extracted text or an error message.
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text_blocks = [page.get_text() for page in doc]
        return "\n".join(text_blocks)
    except Exception as e:
        return f"[Error reading PDF: {str(e)}]"


def read_txt(file_bytes: bytes) -> str:
    """
    Decode a plain text file.

    Args:
        file_bytes (bytes): The raw bytes of the text file.

    Returns:
        str: The decoded string or an error message.
    """
    try:
        return file_bytes.decode("utf-8")
    except Exception as e:
        return f"[Error reading TXT: {str(e)}]"


def read_tabular(file_bytes: bytes, file_name: str) -> str:
    """
    Parse CSV or Excel files into a Markdown table representation.
    Truncates to the first 100 rows to prevent context overflow.

    Args:
        file_bytes (bytes): The raw bytes of the tabular file.
        file_name (str): The name of the file.

    Returns:
        str: A Markdown-formatted table or an error message.
    """
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(BytesIO(file_bytes))
        else:
            df = pd.read_excel(BytesIO(file_bytes))

        if len(df) > 100:
            return f"[Dataset truncated. First 100 of {len(df)} rows]:\n{df.head(100).to_markdown()}"
        return df.to_markdown()
    except Exception as e:
        return f"[Error reading Table: {str(e)}]"


def process_uploaded_files(uploaded_files: List[Any]) -> Tuple[str, List[str]]:
    """
    Process a list of uploaded files via Streamlit's file_uploader and compile their context.

    Args:
        uploaded_files (list): A list of Streamlit UploadedFile objects.

    Returns:
        tuple: A tuple containing the compiled context string and a list of warning messages.
    """
    if not uploaded_files:
        return "", []

    file_context = "### User Uploaded File Content:\n"
    warnings = []
    has_valid_text = False

    for uploaded_file in uploaded_files:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > 10:
            warnings.append(f"File '{uploaded_file.name}' exceeds the 10MB limit ({file_size_mb:.1f}MB). Skipped.")
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


# --- Generation & Formatting ---

def get_response_generator(model_name: str, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
    """
    Stream responses from the Ollama chat endpoint.

    Args:
        model_name (str): The language model to use.
        messages (list): The conversation history.

    Yields:
        str: Incremental text chunks from the language model.
    """
    system_message = {"role": "system", "content": SYSTEM_PROMPT}
    stream = ollama.chat(model=model_name, messages=[system_message] + messages, stream=True)
    for chunk in stream:
        yield chunk["message"]["content"]


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a string using a character-based heuristic."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def chunk_text(text: str, chunk_size_tokens: int = CHUNK_SIZE_TOKENS) -> List[str]:
    """
    Split text into chunks of approximately chunk_size_tokens each.
    Prefers splitting on paragraph breaks to preserve coherence.

    Args:
        text (str): The text to split.
        chunk_size_tokens (int): Target token count per chunk.

    Returns:
        List[str]: Non-empty text chunks.
    """
    chunk_size_chars = chunk_size_tokens * CHARS_PER_TOKEN
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size_chars
        if end < len(text):
            # prefer paragraph break, fall back to line break
            bp = text.rfind('\n\n', start, end)
            if bp == -1:
                bp = text.rfind('\n', start, end)
            if bp != -1 and bp > start:
                end = bp
        chunks.append(text[start:end].strip())
        start = end
    return [c for c in chunks if c]


def get_chunk_answer(
    model_name: str,
    chunk: str,
    chunk_index: int,
    total_chunks: int,
    user_question: str,
    chat_history: List[Dict[str, str]],
) -> str:
    """
    Send a single document chunk to the LLM and return a partial answer (non-streaming).

    Args:
        model_name (str): The language model to use.
        chunk (str): The document fragment for this call.
        chunk_index (int): 1-based index of this chunk.
        total_chunks (int): Total number of chunks.
        user_question (str): The user's original question.
        chat_history (list): Conversation history *excluding* the current user turn.

    Returns:
        str: The model's partial answer for this chunk.
    """
    chunk_prompt = (
        f"You are analyzing part {chunk_index} of {total_chunks} of a document.\n\n"
        f"--- Document Part {chunk_index}/{total_chunks} ---\n{chunk}\n"
        f"--- End of Part {chunk_index}/{total_chunks} ---\n\n"
        f"Based only on the content above, provide a partial answer to:\n{user_question}"
    )
    system_message = {"role": "system", "content": SYSTEM_PROMPT}
    messages = [system_message] + chat_history + [{"role": "user", "content": chunk_prompt}]
    response = ollama.chat(model=model_name, messages=messages, stream=False)
    if isinstance(response, dict):
        return response["message"]["content"]
    return response.message.content


def get_synthesis_generator(
    model_name: str,
    partial_answers: List[str],
    user_question: str,
    chat_history: List[Dict[str, str]],
) -> Generator[str, None, None]:
    """
    Stream a final synthesized answer from the LLM, combining all per-chunk partial answers.

    Args:
        model_name (str): The language model to use.
        partial_answers (list): Collected answers from each chunk.
        user_question (str): The user's original question.
        chat_history (list): Conversation history *excluding* the current user turn.

    Yields:
        str: Incremental text chunks of the synthesized answer.
    """
    parts = "\n\n".join(
        f"--- Answer from Part {i + 1} ---\n{ans}" for i, ans in enumerate(partial_answers)
    )
    synthesis_prompt = (
        f"A long document was split into {len(partial_answers)} parts. "
        f"Each part was analyzed separately to answer: \"{user_question}\"\n\n"
        f"{parts}\n\n"
        f"--- End of Partial Answers ---\n\n"
        f"Now synthesize all of the above into one comprehensive, well-structured final answer."
    )
    system_message = {"role": "system", "content": SYSTEM_PROMPT}
    messages = [system_message] + chat_history + [{"role": "user", "content": synthesis_prompt}]
    stream = ollama.chat(model=model_name, messages=messages, stream=True)
    for chunk in stream:
        if isinstance(chunk, dict):
            yield chunk["message"]["content"]
        else:
            yield chunk.message.content


def format_chat_history(messages: List[Dict[str, str]]) -> str:
    """
    Convert the session state messages into a clean, readable Markdown string suitable for export.

    Args:
        messages (list): The conversation history.

    Returns:
        str: The formatted Markdown document.
    """
    formatted_text = "# Text Lab Chat Export\n\n"
    for msg in messages:
        if msg["role"] == "user":
            formatted_text += f"### User\n{msg['content']}\n\n---\n\n"
        elif msg["role"] == "assistant":
            formatted_text += f"### Assistant\n{msg['content']}\n\n---\n\n"
    return formatted_text