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


def message_text(message: Any) -> str:
    """Return the textual content of an Ollama chat message as a string.

    Ollama >= 0.28 splits reasoning ("thinking") models' output into a separate
    ``thinking`` field, leaving ``content`` set to ``None`` on reasoning-only
    chunks/turns. Older Ollama returned the reasoning inline in ``content`` (as
    ``<think>`` text), so ``content`` was always a string. Callers that stream or
    concatenate content must therefore treat ``None`` as an empty string.
    """
    if message is None:
        return ""
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    return content or ""


def _normalize_tool_call(tc: Any) -> Dict[str, Any]:
    """Convert a ToolCall object (or dict) into a plain dict."""
    if isinstance(tc, dict):
        return tc
    # Pydantic ToolCall → plain dict
    fn = tc.function if hasattr(tc, "function") else tc.get("function", {})
    if not isinstance(fn, dict):
        fn = {
            "name": getattr(fn, "name", ""),
            "arguments": getattr(fn, "arguments", {}),
        }
    return {"function": fn}


def _normalize_message(msg: Any) -> Dict[str, Any]:
    """Convert a Message object (or dict) into a plain dict for the Ollama
    messages list.

    The ollama Python SDK (≥ 0.4) returns ``Message`` Pydantic objects from
    ``ollama.chat()``. These objects support ``__getitem__`` but NOT ``.get()``,
    so downstream code that calls ``msg.get("tool_calls")`` breaks. Converting
    to a plain dict fixes that and ensures the message can be serialised back
    into the next ``ollama.chat()`` call.
    """
    if isinstance(msg, dict):
        return msg
    # Use model_dump() if available (Pydantic v2), otherwise build manually.
    if hasattr(msg, "model_dump"):
        return msg.model_dump(exclude_none=True)
    d: Dict[str, Any] = {
        "role": getattr(msg, "role", "assistant"),
        "content": getattr(msg, "content", None) or "",
    }
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        d["tool_calls"] = [_normalize_tool_call(tc) for tc in tool_calls]
    return d


def _normalize_response(response: Any) -> Dict[str, Any]:
    """Convert a ChatResponse object into the dict shape the codebase expects:
    ``{"message": {...}, "model": ..., ...}``.
    """
    if isinstance(response, dict):
        # Still normalise the inner message, just in case.
        if "message" in response and not isinstance(response["message"], dict):
            response["message"] = _normalize_message(response["message"])
        return response
    # Pydantic ChatResponse → plain dict
    msg = getattr(response, "message", None)
    return {
        "message": _normalize_message(msg) if msg is not None else {},
        "model": getattr(response, "model", ""),
        "done": getattr(response, "done", True),
    }


def chat_no_think(
    model: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]] | None = None,
) -> Any:
    """Non-streaming ``ollama.chat`` with reasoning disabled.

    Structured tool-routing/agent calls need a fast, deterministic tool decision,
    not chain-of-thought. With thinking enabled (the default for reasoning models
    on Ollama >= 0.28) these calls waste the context window on reasoning, bloat
    multi-turn history with stale ``thinking`` blocks, and frequently stop before
    emitting a real tool call. ``think=False`` avoids that.

    Falls back gracefully when the installed client/model/server does not accept
    the ``think`` argument.
    """
    kwargs: Dict[str, Any] = {"model": model, "messages": messages}
    if tools is not None:
        kwargs["tools"] = tools
    try:
        return _normalize_response(ollama.chat(think=False, **kwargs))
    except TypeError:
        # Older ollama-python without the ``think`` keyword.
        return _normalize_response(ollama.chat(**kwargs))
    except Exception:
        # Model/server rejected ``think`` (e.g. a non-reasoning model) — retry plain.
        return _normalize_response(ollama.chat(**kwargs))


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
        models = (
            running_models.get("models", [])
            if isinstance(running_models, dict)
            else getattr(running_models, "models", [])
        )
        for model in (models or []):
            running_name = extract_model_name(model)
            if running_name == model_name or running_name.startswith(f"{model_name}:"):
                return True
        return False
    except Exception:
        return False


def get_loaded_models() -> List[str]:
    """
    Return the names of all models currently resident in Ollama's VRAM.

    Returns:
        List[str]: Model name strings, or an empty list if none are loaded
        or the Ollama server is unreachable.
    """
    try:
        running = ollama.ps()
        models = (
            running.get("models", [])
            if isinstance(running, dict)
            else getattr(running, "models", [])
        )
        return [extract_model_name(m) for m in models]
    except Exception:
        return []


def unload_model(model_name: str) -> bool:
    """
    Evict a single model from Ollama's VRAM by requesting a keep_alive of 0.

    This sends a minimal generate request with ``keep_alive=0``, which
    instructs Ollama to release the model immediately after responding.

    Args:
        model_name (str): The model identifier to unload.

    Returns:
        bool: True if the request was accepted, False on any error.
    """
    try:
        ollama.generate(model=model_name, prompt="", keep_alive=0)
        return True
    except Exception:
        return False


def unload_all_models() -> List[str]:
    """
    Evict every model currently loaded in Ollama's VRAM.

    Returns:
        List[str]: Names of the models that were unloaded. Empty if none
        were loaded or the server was unreachable.
    """
    loaded = get_loaded_models()
    for name in loaded:
        unload_model(name)
    return loaded


def warm_model(model_name: str) -> bool:
    """
    Pre-load a model into Ollama's VRAM by sending an empty generate request
    with a non-zero keep_alive, so the next real request is instant.

    Args:
        model_name (str): The model identifier to warm.

    Returns:
        bool: True if the request was accepted, False on any error.
    """
    try:
        ollama.generate(model=model_name, prompt="", keep_alive=300)
        return True
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
        text = message_text(chunk["message"] if isinstance(chunk, dict) else chunk.message)
        if text:
            yield text


# --- Image generation as a first-class chat tool -----------------------------
# Exposing generate_image to the model (not just the keyword router) lets it fire
# in conversation — e.g. when the user confirms ("yes") an image the model just
# offered to make. The system hint also stops models from role-playing a fake
# `dalle.text2im`/JSON tool call or claiming they cannot generate images.

IMAGE_TOOL_SYSTEM_HINT = (
    "You have a REAL image-generation tool named `generate_image` that renders images locally. "
    "When the user actually wants an image NOW — asks you to create, draw, generate, paint, design, or "
    "produce an image/picture/visual/poster/logo/scene/portrait, or confirms (e.g. 'yes') an image you just "
    "offered to make — you MUST call the `generate_image` tool with a single detailed `prompt` describing the "
    "scene. Never claim you are unable to generate images, and never write a tool call as plain text or JSON — "
    "use the actual tool. "
    "BUT if the user is only discussing or asking a QUESTION about images or prompts (e.g. 'which prompt would "
    "you use for...', 'how would you describe...', 'can you explain...') rather than requesting an image right "
    "now, answer in plain text and do NOT call the tool. Answer normally for everything else."
)

GENERATE_IMAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_image",
        "description": (
            "Generate/create/draw an image locally from a text description. Call this whenever the "
            "user asks to create, generate, draw, paint, or produce an image, picture, visual, poster, "
            "logo, scene, etc., including when they confirm a picture you offered to make."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "A detailed, self-contained visual description of the image to create, derived "
                        "from the user's request and the conversation: subject, style, setting, mood, colours."
                    ),
                }
            },
            "required": ["prompt"],
        },
    },
}


def _extract_image_tool_prompt(message) -> str | None:
    """Return the `prompt` argument if `message` contains a generate_image tool call."""
    import json as _json

    tool_calls = (
        message.get("tool_calls") if isinstance(message, dict)
        else getattr(message, "tool_calls", None)
    )
    if not tool_calls:
        return None
    for tc in tool_calls:
        fn = tc["function"] if isinstance(tc, dict) else tc.function
        name = fn["name"] if isinstance(fn, dict) else fn.name
        if name != "generate_image":
            continue
        args = fn["arguments"] if isinstance(fn, dict) else fn.arguments
        if isinstance(args, str):
            try:
                args = _json.loads(args)
            except Exception:
                args = {}
        prompt = (args or {}).get("prompt", "")
        if isinstance(prompt, str) and prompt.strip():
            return prompt.strip()
    return None


def stream_chat_with_image_tool(
    model_name: str,
    messages: List[Dict[str, str]],
    offer_image: bool = False,
) -> Tuple[str, Any]:
    """Stream a chat reply while offering the model the generate_image tool.

    Returns a ``(kind, payload)`` tuple:
      * ``("image", prompt)``  — the model chose to generate an image; run the pipeline.
      * ``("text", generator)`` — a normal streamed text answer (yield to the UI).

    A single streaming call: if the model opens with a generate_image tool call we
    catch it; otherwise we stream its text unchanged (no extra latency). Falls back
    to a plain stream if the model does not support tool calling.
    """
    def _plain_stream() -> Generator[str, None, None]:
        system_message = {"role": "system", "content": SYSTEM_PROMPT}
        for chunk in ollama.chat(model=model_name, messages=[system_message] + messages, stream=True):
            content = message_text(chunk["message"] if isinstance(chunk, dict) else chunk.message)
            if content:
                yield content

    if not offer_image:
        return "text", _plain_stream()

    def _content_of(msg):
        return message_text(msg)

    try:
        system_message = {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + IMAGE_TOOL_SYSTEM_HINT}
        stream = ollama.chat(
            model=model_name,
            messages=[system_message] + messages,
            tools=[GENERATE_IMAGE_TOOL],
            stream=True,
        )
        for chunk in stream:
            message = chunk["message"] if isinstance(chunk, dict) else chunk.message
            prompt = _extract_image_tool_prompt(message)
            if prompt:
                return "image", prompt
            content = _content_of(message)
            if content:
                # Real text answer begins here; emit this chunk then the remainder.
                def _continue(first_chunk, remaining):
                    yield first_chunk
                    for c in remaining:
                        m = c["message"] if isinstance(c, dict) else c.message
                        cc = _content_of(m)
                        if cc:
                            yield cc
                return "text", _continue(content, stream)
        return "text", iter(())
    except Exception:
        # Model likely lacks tool support — fall back to a normal streamed reply.
        return "text", _plain_stream()


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
    return message_text(response["message"] if isinstance(response, dict) else response.message)


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
        text = message_text(chunk["message"] if isinstance(chunk, dict) else chunk.message)
        if text:
            yield text


# --- Tool Routing (Data Analysis Supervisor) ---

ROUTER_SYSTEM_PROMPT = (
    "You are a routing supervisor for a chat assistant. The user is chatting and has "
    "uploaded a tabular dataset (a schema is provided below). Decide whether answering "
    "the user's latest message requires generating plots/charts or running statistical "
    "analysis (correlations, t-tests, ANOVA, regression) on that dataset.\n\n"
    "- If the request requires visualisations or statistics on the dataset, call the "
    "`analyze_data` tool with a clear, self-contained instruction derived from the user's "
    "request and conversation.\n"
    "- If the request is general conversation, a factual question, or can be answered from "
    "the dataset text already in context WITHOUT producing plots or statistical tests, do "
    "NOT call any tool and simply reply normally.\n\n"
    "Only call `analyze_data` when visual or statistical output is genuinely needed."
)

ANALYZE_DATA_TOOL = {
    "type": "function",
    "function": {
        "name": "analyze_data",
        "description": (
            "Run data visualisation and/or statistical analysis on the user's uploaded "
            "dataset. Use this only when the user's request requires plots, charts, or "
            "statistical tests."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": (
                        "A clear, self-contained analysis instruction derived from the "
                        "user's request (e.g. 'Plot an interactive scatter of age vs income "
                        "and run a correlation between them')."
                    ),
                }
            },
            "required": ["instruction"],
        },
    },
}


def decide_tool_use(
    model_name: str,
    user_text: str,
    schema_text: str,
    chat_history: List[Dict[str, str]] | None = None,
) -> Tuple[bool, str]:
    """
    Ask the model (acting as a router/supervisor) whether the user's latest message
    requires data-analysis tools (plots / statistics) on the uploaded dataset.

    Args:
        model_name: The Ollama model to use for routing. Must support tool calling.
        user_text: The user's latest message.
        schema_text: A compact schema/summary of the uploaded dataset.
        chat_history: Prior conversation turns (excluding the current user turn).

    Returns:
        Tuple[bool, str]: (use_tools, instruction). When use_tools is False the
        instruction is an empty string. On any error, falls back to (False, "").
    """
    history = chat_history or []
    router_messages = (
        [{"role": "system", "content": ROUTER_SYSTEM_PROMPT}]
        + history
        + [
            {
                "role": "user",
                "content": (
                    f"Dataset schema:\n{schema_text}\n\n"
                    f"User message: {user_text}"
                ),
            }
        ]
    )

    try:
        response = chat_no_think(
            model=model_name,
            messages=router_messages,
            tools=[ANALYZE_DATA_TOOL],
        )
    except Exception:
        # Router failed (e.g. model lacks tool support) — fall back to plain chat.
        return False, ""

    message = response["message"] if isinstance(response, dict) else response.message
    tool_calls = (
        message.get("tool_calls") if isinstance(message, dict) else getattr(message, "tool_calls", None)
    )
    if not tool_calls:
        return False, ""

    for tool_call in tool_calls:
        fn = tool_call["function"] if isinstance(tool_call, dict) else tool_call.function
        name = fn["name"] if isinstance(fn, dict) else fn.name
        if name != "analyze_data":
            continue
        args = fn["arguments"] if isinstance(fn, dict) else fn.arguments
        if isinstance(args, str):
            import json
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        instruction = (args or {}).get("instruction", "").strip()
        if not instruction:
            instruction = user_text.strip()
        return True, instruction

    return False, ""


def _artifact_to_markdown(artifact: Dict[str, Any], index: int) -> str:
    """
    Render a single plot artifact as embeddable Markdown.

    Static images (PNG) are embedded inline as base64 data URIs. Interactive Plotly
    charts (stored as JSON) are converted to a static PNG via kaleido when available;
    otherwise a note and the reproducible source code are included instead.
    """
    import base64

    tool_label = artifact.get("tool_name", "") or f"Plot {index + 1}"
    md = f"**{tool_label.replace('_', ' ').title()}**\n\n"
    fig_json = artifact.get("fig_json")
    img_bytes = artifact.get("bytes")
    png_b64 = None

    if fig_json:
        try:
            import plotly.io as pio
            fig = pio.from_json(fig_json)
            png_bytes = fig.to_image(format="png")  # requires kaleido
            png_b64 = base64.b64encode(png_bytes).decode("ascii")
        except Exception:
            png_b64 = None
    elif img_bytes and artifact.get("filename", "").lower().endswith((".png", ".jpg", ".jpeg")):
        png_b64 = base64.b64encode(img_bytes).decode("ascii")

    if png_b64:
        md += f"![{tool_label}](data:image/png;base64,{png_b64})\n\n"
    elif fig_json:
        md += (
            "_Interactive chart — not embeddable as a static image in this export. "
            "Reproduce it with the code below._\n\n"
        )

    code = artifact.get("code")
    if code:
        md += f"```python\n{code}\n```\n\n"
    return md


def _has_analysis_plots(messages: List[Dict[str, Any]]) -> bool:
    """Return True if any assistant turn produced plot artifacts."""
    for msg in messages:
        analysis = msg.get("analysis") if isinstance(msg, dict) else None
        if analysis and analysis.get("artifacts"):
            return True
    return False


def format_chat_history_html(messages: List[Dict[str, Any]]) -> str:
    """
    Convert the conversation into a self-contained HTML document.

    Interactive Plotly charts are rendered using Plotly's own ``fig.to_html``
    (which generates correct embedding code). Plotly.js is bundled inline on
    the first interactive chart so the file works without an internet connection.
    Static images are embedded as base64 data URIs.

    Args:
        messages (list): The conversation history.

    Returns:
        str: A complete HTML document string.
    """
    import base64
    import html as html_lib

    parts: List[str] = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Text Lab Chat Export</title>",
        "<style>",
        "body{font-family:system-ui,Arial,sans-serif;max-width:900px;margin:2rem auto;"
        "padding:0 1rem;line-height:1.5;color:#1e1e1e;}",
        ".turn{border:1px solid #ddd;border-radius:8px;padding:1rem 1.25rem;margin:1rem 0;}",
        ".user{background:#f5f5f5;}.assistant{background:#fff;}",
        ".role{font-weight:600;margin-bottom:.5rem;color:#555;}",
        "img{max-width:100%;height:auto;}",
        "pre{background:#f0f0f0;padding:.75rem;border-radius:6px;overflow-x:auto;}",
        "h4{margin-top:1.25rem;}",
        "</style>",
        "</head><body>",
        "<h1>Text Lab Chat Export</h1>",
    ]

    # Track whether Plotly.js has been bundled yet.
    # First interactive chart includes it inline; subsequent charts omit it to
    # avoid duplication. This keeps the file self-contained and offline-capable.
    plotlyjs_included = False

    for msg in messages:
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        content_html = html_lib.escape(msg.get("content", "")).replace("\n", "<br>")
        parts.append(f"<div class='turn {role}'>")
        parts.append(f"<div class='role'>{role.title()}</div>")
        parts.append(f"<div>{content_html}</div>")

        analysis = msg.get("analysis") if role == "assistant" else None
        if analysis:
            artifacts = analysis.get("artifacts", [])
            if artifacts:
                parts.append("<h4>Generated Visualisations</h4>")
            for artifact in artifacts:
                label = html_lib.escape(
                    (artifact.get("tool_name", "") or "Plot").replace("_", " ").title()
                )
                parts.append(f"<p><strong>{label}</strong></p>")
                fig_json = artifact.get("fig_json")
                if fig_json:
                    try:
                        import plotly.io as pio
                        fig = pio.from_json(fig_json)
                        # Bundle plotlyjs inline on the first chart so the exported
                        # file is self-contained. Subsequent charts skip it.
                        include_js = True if not plotlyjs_included else False
                        chart_html = fig.to_html(
                            full_html=False,
                            include_plotlyjs=include_js,
                        )
                        plotlyjs_included = True
                        parts.append(chart_html)
                    except Exception:
                        parts.append("<p><em>Could not render interactive chart.</em></p>")
                else:
                    img_bytes = artifact.get("bytes")
                    if img_bytes:
                        b64 = base64.b64encode(img_bytes).decode("ascii")
                        parts.append(
                            f"<img src='data:image/png;base64,{b64}' alt='{label}'>"
                        )
                code = artifact.get("code")
                if code:
                    parts.append(f"<details><summary>View source code</summary>"
                                 f"<pre>{html_lib.escape(code)}</pre></details>")

            stats_results = analysis.get("stats", [])
            if stats_results:
                parts.append("<h4>Statistical Analysis Results</h4>")
                for item in stats_results:
                    parts.append(
                        f"<p><strong>{html_lib.escape(item.get('title', 'Result'))}</strong></p>"
                    )
                    result_html = html_lib.escape(item.get("result", "")).replace("\n", "<br>")
                    parts.append(f"<div>{result_html}</div>")
                    if item.get("code"):
                        parts.append(
                            f"<details><summary>View source code</summary>"
                            f"<pre>{html_lib.escape(item['code'])}</pre></details>"
                        )

        parts.append("</div>")

    parts.append("</body></html>")
    return "".join(parts)


def format_chat_history(messages: List[Dict[str, str]]) -> str:
    """
    Convert the session state messages into a clean, readable Markdown string suitable for export.

    Assistant turns that produced data analysis embed their plots (as base64 images)
    and statistical results so the exported document is self-contained.

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
            formatted_text += f"### Assistant\n{msg['content']}\n\n"

            analysis = msg.get("analysis")
            if analysis:
                artifacts = analysis.get("artifacts", [])
                if artifacts:
                    formatted_text += "#### Generated Visualisations\n\n"
                    for idx, artifact in enumerate(artifacts):
                        formatted_text += _artifact_to_markdown(artifact, idx)

                stats_results = analysis.get("stats", [])
                if stats_results:
                    formatted_text += "#### Statistical Analysis Results\n\n"
                    for item in stats_results:
                        formatted_text += f"**{item.get('title', 'Result')}**\n\n"
                        formatted_text += f"{item.get('result', '')}\n\n"
                        if item.get("code"):
                            formatted_text += f"```python\n{item['code']}\n```\n\n"

            formatted_text += "---\n\n"
    return formatted_text