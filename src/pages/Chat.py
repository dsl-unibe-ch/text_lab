import streamlit as st
import ollama
import sys
import os
import asyncio
import datetime
import pathlib
import tempfile
import threading
import time
import uuid
from ollama import ResponseError
from PIL import Image
import plotly.io as pio

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
favicon_path = os.path.join(src_dir, "assets", "text_lab_logo.png")

favicon = Image.open(favicon_path)

st.set_page_config(
    page_title="Ollama Chat Interface",
    page_icon=favicon,
    layout="centered",
    initial_sidebar_state="expanded"
)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from auth import check_token
from core.chat_engine import (
    check_ollama_server,
    get_gpu_name,
    is_model_loaded,
    extract_model_name,
    process_uploaded_files,
    get_response_generator,
    format_chat_history,
    format_chat_history_html,
    _has_analysis_plots,
    estimate_tokens,
    chunk_text,
    get_chunk_answer,
    get_synthesis_generator,
    decide_tool_use,
    stream_chat_with_image_tool,
    unload_all_models,
    warm_model,
    MAX_CONTEXT_TOKENS,
)

from core.model_config import get_available_models, is_high_memory_gpu

# --- Data-analysis tool integration (reuses the Visualisation MAS, unchanged) ---
from core.visualization.viz_agent import run_analysis
from core.visualization.viz_config import MAX_ROWS, get_tool_label
from core.visualization.viz_utils import save_data_file, get_fast_data_preview
from core.visualization.plot_data import get_all_columns_summary_impl

# --- Image-generation tool integration (local Ideogram 4 via a persistent MCP) ---
from core.image_gen import image_engine
from core.image_gen.image_engine import ImageBlockedError
from core.image_gen.magic_prompt import aspect_ratio_from_size, expand_prompt_ollama
from core.image_gen.image_config import (
    DEFAULT_PRESET,
    DEFAULT_SIZE,
    IMAGE_TIMEOUT_SECONDS,
    PRESET_LABELS,
    SIZE_OPTIONS,
    is_image_gen_installed,
    is_image_gen_supported,
    model_repo_for_gpu,
)

_SRC_DIR = pathlib.Path(__file__).resolve().parent.parent
MCP_SERVER_SCRIPT = str(_SRC_DIR / "core" / "visualization" / "mcp_server.py")
IMAGE_MCP_SERVER_SCRIPT = str(_SRC_DIR / "core" / "image_gen" / "mcp_server.py")
ARTIFACTS_DIR = str(_SRC_DIR / "mcp_artifacts")
IMAGE_ARTIFACTS_DIR = str(_SRC_DIR / "mcp_artifacts" / "images")
ANALYSIS_TIMEOUT_SECONDS = 600
TABULAR_EXTENSIONS = (".csv", ".tsv", ".xls", ".xlsx", ".json")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(IMAGE_ARTIFACTS_DIR, exist_ok=True)
# Keep the MCP subprocess and Streamlit in agreement on where PNGs are written.
os.environ.setdefault("IMAGE_GEN_ARTIFACTS_DIR", IMAGE_ARTIFACTS_DIR)

check_token()


def _get_session_data_dir() -> str:
    """Return a persistent per-session temp dir for uploaded data files."""
    data_dir = st.session_state.get("chat_data_dir")
    if not data_dir or not os.path.isdir(data_dir):
        data_dir = tempfile.mkdtemp(prefix="chat-", dir=ARTIFACTS_DIR)
        st.session_state["chat_data_dir"] = data_dir
    return data_dir


def _ensure_data_file(uploaded_files) -> tuple[str | None, str | None, str | None]:
    """
    Persist the first uploaded tabular file to disk so the analysis tools can read it.

    Returns (data_file_path, file_name, schema_text) or (None, None, None) when no
    tabular file is present.
    """
    if not uploaded_files:
        return None, None, None

    tabular = next(
        (f for f in uploaded_files if f.name.lower().endswith(TABULAR_EXTENSIONS)),
        None,
    )
    if tabular is None:
        return None, None, None

    file_id = (tabular.name, tabular.size)
    if st.session_state.get("chat_data_file_id") == file_id and st.session_state.get("chat_data_path"):
        return (
            st.session_state["chat_data_path"],
            st.session_state["chat_data_name"],
            st.session_state.get("chat_data_schema", ""),
        )

    run_dir = _get_session_data_dir()
    data_file_path = save_data_file(tabular.getvalue(), tabular.name, run_dir)
    try:
        schema_text = get_all_columns_summary_impl(data_file_path)
    except Exception as e:
        schema_text = f"[Could not summarise dataset: {e}]"

    st.session_state["chat_data_file_id"] = file_id
    st.session_state["chat_data_path"] = data_file_path
    st.session_state["chat_data_name"] = tabular.name
    st.session_state["chat_data_schema"] = schema_text
    return data_file_path, tabular.name, schema_text


def _read_plot_artifacts(plots: list[dict]) -> list[dict]:
    """Read plot files produced by the MAS into serialisable artifacts for the chat."""
    artifacts: list[dict] = []
    for item in plots:
        path = item.get("path", "")
        if not path or not os.path.exists(path):
            continue
        filename = os.path.basename(path)
        with open(path, "rb") as f:
            file_bytes = f.read()
        artifact = {
            "filename": filename,
            "bytes": file_bytes,
            "code": item.get("code", ""),
            "tool_name": item.get("tool_name", ""),
            "fig_json": None,
        }
        if filename.endswith(".json"):
            try:
                artifact["fig_json"] = file_bytes.decode("utf-8")
            except Exception:
                continue
        artifacts.append(artifact)
    return artifacts


def _render_analysis_payload(payload: dict, run_id: str) -> None:
    """Render an assistant analysis turn: plots, then statistical results."""
    artifacts = payload.get("artifacts", [])
    stats_results = payload.get("stats", [])

    if artifacts:
        for idx, artifact in enumerate(artifacts):
            tool_label = get_tool_label(artifact.get("tool_name", ""))
            if tool_label:
                st.markdown(f"**{tool_label}**")
            if artifact.get("fig_json"):
                fig = pio.from_json(artifact["fig_json"])
                st.plotly_chart(fig, use_container_width=True, key=f"chatplot_{run_id}_{idx}")
            else:
                st.image(artifact["bytes"], caption=artifact["filename"])
            if artifact.get("code"):
                with st.expander(f"View Source Code: {tool_label or artifact['filename']}"):
                    st.code(artifact["code"], language="python")

    if stats_results:
        for s_idx, item in enumerate(stats_results):
            with st.expander(item.get("title", "Statistical Result"), expanded=False):
                st.markdown(item.get("result", ""))
                if item.get("code"):
                    st.code(item["code"], language="python")


def _start_chat_analysis_thread(
    instruction: str,
    data_file_path: str,
    file_name: str,
    model_name: str,
) -> None:
    """Run the visualisation MAS in a daemon thread so the chat UI stays responsive."""
    cancel_event = threading.Event()
    live_logs: list[tuple[str, str]] = []
    thread_result: dict = {"status": "running", "result": None, "artifacts": [], "error": None}
    run_id = f"chat-{uuid.uuid4().hex[:8]}"

    def _worker() -> None:
        try:
            head_df = get_fast_data_preview(data_file_path, file_name, nrows=5)
            head_str = head_df.to_string() if head_df is not None else "(preview unavailable)"
            messages = [
                {
                    "role": "user",
                    "content": (
                        f"User Request: {instruction}\n\n"
                        f"Data Head:\n{head_str}\n\n"
                        f"Note: datasets larger than {MAX_ROWS:,} rows will be truncated."
                    ),
                }
            ]

            def _log_cb(log_type: str, msg: str) -> None:
                live_logs.append((log_type, msg))

            live_logs.append(("info", "Starting Supervisor Agent..."))
            analysis_result = asyncio.run(
                asyncio.wait_for(
                    run_analysis(
                        messages,
                        data_file_path,
                        model_name,
                        MCP_SERVER_SCRIPT,
                        log_callback=_log_cb,
                        cancel_event=cancel_event,
                    ),
                    timeout=ANALYSIS_TIMEOUT_SECONDS,
                )
            )
            thread_result["result"] = analysis_result
            thread_result["artifacts"] = _read_plot_artifacts(analysis_result.get("plots", []))
            thread_result["status"] = "cancelled" if cancel_event.is_set() else "done"
        except asyncio.TimeoutError:
            thread_result["status"] = "timeout"
        except Exception as e:
            thread_result["error"] = str(e)
            thread_result["status"] = "error"

    st.session_state["chat_tool_cancel"] = cancel_event
    st.session_state["chat_tool_logs"] = live_logs
    st.session_state["chat_tool_result"] = thread_result
    st.session_state["chat_tool_run_id"] = run_id
    st.session_state["chat_tool_instruction"] = instruction
    st.session_state["chat_tool_title"] = "Analysing your data..."
    st.session_state["chat_tool_state"] = "running"

    threading.Thread(target=_worker, daemon=True).start()


def _caption_summary(caption: str) -> str:
    """Short, human-readable preview of the generated JSON caption for the UI."""
    import json as _json

    try:
        data = _json.loads(caption)
        hld = data.get("high_level_description")
        if isinstance(hld, str) and hld.strip():
            return hld.strip()[:200]
    except Exception:
        pass
    return caption[:120]


def _start_image_gen_thread(
    prompt: str,
    width: int,
    height: int,
    sampler_preset: str,
    seed: int,
    low_vram: bool = False,
    active_model: str = "",
) -> None:
    """Run image generation in a daemon thread so the chat UI stays responsive.

    Results are stored in the same session-state slots the data-analysis run uses,
    so `_render_tool_run_section` renders and archives the turn unchanged: the
    generated PNG is packaged as a standard analysis artifact (bytes + tool_name).
    """
    cancel_event = threading.Event()
    live_logs: list[tuple[str, str]] = []
    thread_result: dict = {"status": "running", "result": None, "artifacts": [], "error": None}
    run_id = f"img-{uuid.uuid4().hex[:8]}"

    def _restore_chat_model() -> None:
        """Low-VRAM only: tear down the image backend to free the ~20 GB diffusion
        pipeline (it stays resident in the MCP subprocess and would otherwise starve
        the chat model of VRAM, hanging its reload), then warm the chat model back."""
        if not low_vram:
            return
        try:
            live_logs.append(("info", "Unloading the image model to free GPU memory..."))
            image_engine.shutdown_backend()
        except Exception:
            pass
        if active_model:
            try:
                live_logs.append(("info", f"Reloading chat model ({active_model})..."))
                warm_model(active_model)
            except Exception:
                pass

    def _worker() -> None:
        try:
            # Expand the free-form prompt into Ideogram's required JSON caption
            # FIRST, while the chat model is still warm — this reuses the loaded
            # model (no extra VRAM) and must happen before any low-VRAM unload.
            # Ideogram gray-blocks raw text, so a good caption is essential.
            caption = prompt
            if active_model:
                live_logs.append(("info", "Rewriting your prompt into a structured caption..."))
                try:
                    caption = expand_prompt_ollama(
                        prompt, active_model, aspect_ratio_from_size(width, height)
                    )
                    live_logs.append(("info", f"Caption ready: {_caption_summary(caption)}"))
                except Exception as exc:  # never block generation on the rewrite step
                    live_logs.append(("warning", f"Prompt rewrite failed ({exc}); using the raw prompt."))
                    caption = prompt

            if low_vram:
                live_logs.append(("info", "Freeing GPU memory for image generation..."))
                unload_all_models()
            live_logs.append(("info", "Loading the image model (first run may take ~1 minute)..."))
            artifact = image_engine.generate_image(
                caption,
                IMAGE_MCP_SERVER_SCRIPT,
                width=width,
                height=height,
                sampler_preset=sampler_preset,
                seed=seed,
                timeout=float(IMAGE_TIMEOUT_SECONDS),
            )
            if cancel_event.is_set():
                thread_result["status"] = "cancelled"
                _restore_chat_model()
                return

            _restore_chat_model()

            path = artifact["path"]
            with open(path, "rb") as f:
                image_bytes = f.read()
            thread_result["artifacts"] = [{
                "filename": os.path.basename(path),
                "bytes": image_bytes,
                "code": "",
                "tool_name": "generate_image",
                "fig_json": None,
            }]
            thread_result["result"] = {
                "summary": f"Here's the image for: *{prompt}*",
                "stats": [],
            }
            thread_result["status"] = "done"
        except ImageBlockedError:
            thread_result["status"] = "blocked"
            _restore_chat_model()
        except Exception as e:
            thread_result["error"] = str(e)
            thread_result["status"] = "error"
            _restore_chat_model()

    st.session_state["chat_tool_cancel"] = cancel_event
    st.session_state["chat_tool_logs"] = live_logs
    st.session_state["chat_tool_result"] = thread_result
    st.session_state["chat_tool_run_id"] = run_id
    st.session_state["chat_tool_instruction"] = prompt
    st.session_state["chat_tool_title"] = "Generating image..."
    st.session_state["chat_tool_state"] = "running"

    threading.Thread(target=_worker, daemon=True).start()


def _render_tool_run_section() -> bool:
    """
    Poll the running analysis thread, render its live activity log, and on completion
    append the assistant turn to history. Returns True while still running.
    """
    thread_result: dict = st.session_state.get("chat_tool_result", {})
    live_logs: list = st.session_state.get("chat_tool_logs", [])
    status: str = thread_result.get("status", "running")
    is_complete = status != "running"

    status_title = st.session_state.get("chat_tool_title", "Analysing your data...")
    with st.chat_message("assistant"):
        with st.status(
            status_title, expanded=(not is_complete),
            state="running" if not is_complete else "complete",
        ):
            for log_type, msg in list(live_logs):
                if log_type == "error":
                    st.error(msg)
                elif log_type == "warning":
                    st.warning(msg)
                else:
                    st.write(msg)
            if not live_logs:
                st.caption("Starting agents...")

    if status == "running":
        time.sleep(1)
        st.rerun()
        return True

    run_id = st.session_state.get("chat_tool_run_id", "chat-unknown")
    if status == "done":
        result = thread_result.get("result", {}) or {}
        summary = result.get("summary", "") or "Analysis complete."
        payload = {
            "artifacts": thread_result.get("artifacts", []),
            "stats": result.get("stats", []),
            "run_id": run_id,
        }
        st.session_state["messages"].append(
            {"role": "assistant", "content": summary, "analysis": payload}
        )
    elif status == "timeout":
        st.session_state["messages"].append(
            {"role": "assistant", "content": (
                f"The analysis exceeded the {ANALYSIS_TIMEOUT_SECONDS // 60}-minute limit. "
                "Try a simpler request or a smaller dataset."
            )}
        )
    elif status == "cancelled":
        st.session_state["messages"].append(
            {"role": "assistant", "content": "Analysis was cancelled."}
        )
    elif status == "blocked":
        st.session_state["messages"].append(
            {"role": "assistant", "content": (
                "The image model's built-in safety filter blocked this prompt, even after a "
                "retry. This filter is known to over-trigger — try rephrasing (more neutral, "
                "concrete wording usually helps) or adjust the seed and try again."
            )}
        )
    elif status == "error":
        st.session_state["messages"].append(
            {"role": "assistant", "content": f"An error occurred while processing your request: {thread_result.get('error', '')}"}
        )

    st.session_state["chat_tool_state"] = "idle"
    st.rerun()
    return False


def _ollama_messages(msgs: list) -> list:
    """Strip messages down to role/content for the Ollama API (drops UI-only keys)."""
    return [{"role": m["role"], "content": m["content"]} for m in msgs]


def main():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
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

    # UI check for server status
    if not check_ollama_server():
        st.error("Could not connect to Ollama server.")
        st.info("Please check the log file: text_lab/ollama_server.log")
        st.stop()

    # --- GPU Detection & Model Filtering ---
    current_gpu = get_gpu_name()
    available_models_in_ui = get_available_models(current_gpu)

    if is_high_memory_gpu(current_gpu):
        gpu_badge = f"**High-Performance Mode** detected ({current_gpu})"
    else:
        gpu_badge = f" **Standard Mode** detected ({current_gpu}). Large models are hidden."

    if not available_models_in_ui:
        st.error("No models are configured. Please check src/config/models.json.")
        st.stop()

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
        type=["pdf", "txt", "csv", "tsv", "xls", "xlsx", "json"], 
        accept_multiple_files=True
    )
    
    # Enforce file count limit
    if uploaded_files and len(uploaded_files) > 4:
        st.sidebar.error("Maximum 4 files allowed. Please remove some.")
        uploaded_files = uploaded_files[:4]

    # Persist any tabular upload so the data-analysis tools can read it from disk.
    data_file_path, data_file_name, data_schema = _ensure_data_file(uploaded_files)
    if data_file_path:
        st.sidebar.success(
            f"📊 Data tools enabled for **{data_file_name}**. "
            "Ask for plots or statistics and I'll analyse it."
        )

    # --- Image generation controls (local Ideogram 4) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎨 Image Generation")
    image_gen_installed = is_image_gen_installed()
    image_gen_ok = image_gen_installed and is_image_gen_supported(current_gpu)

    # Defaults used when the feature is unavailable or the controls are hidden.
    image_mode = False
    image_width, image_height = SIZE_OPTIONS[DEFAULT_SIZE]
    image_preset_key = DEFAULT_PRESET
    image_seed = 0
    image_unload_chat_model = False

    if not image_gen_installed:
        st.sidebar.caption("Unavailable in this build (the `ideogram4` package is missing).")
    elif not image_gen_ok:
        st.sidebar.caption(f"Needs a GPU with ≥20 GB VRAM. Current: {current_gpu}.")
    else:
        # Pick the weights repo for this GPU (fp8 on high-memory cards, nf4 else)
        # and hand it to the MCP subprocess, which inherits this process's env.
        selected_image_repo = model_repo_for_gpu(current_gpu)
        os.environ["TEXT_LAB_IMAGEGEN_MODEL_REPO"] = selected_image_repo
        # On 80+ GB A100/H100 nodes, FP8 and the chat model can coexist. Keep the
        # low-VRAM unload behavior for smaller cards, with an override for admins.
        image_unload_chat_model = (
            os.environ.get("TEXT_LAB_IMAGEGEN_UNLOAD_CHAT", "").strip() == "1"
            or not is_high_memory_gpu(current_gpu)
        )
        if is_high_memory_gpu(current_gpu):
            st.sidebar.caption(
                "✨ Using the higher-quality Ideogram 4 (fp8) model, enabled by this "
                "GPU's larger memory."
            )
        else:
            st.sidebar.caption(
                f"⚠️ Image generation is slow on this GPU ({current_gpu}): with limited "
                "VRAM the chat model is unloaded and reloaded around each image, so "
                "expect a wait of a few minutes per generation."
            )
        image_mode = st.sidebar.toggle(
            "🖼️ Generate image",
            value=st.session_state.get("image_mode", False),
            help=(
                "On: every message is turned into an image. "
                "Off: I still generate an image when a message clearly asks for one "
                "(e.g. \"draw a cat\", \"create a visual of a mountain\")."
            ),
        )
        st.session_state["image_mode"] = image_mode

        size_labels = list(SIZE_OPTIONS.keys())
        size_label = st.sidebar.selectbox(
            "Image size", size_labels,
            index=size_labels.index(st.session_state.get("image_size_label", DEFAULT_SIZE)),
        )
        st.session_state["image_size_label"] = size_label
        image_width, image_height = SIZE_OPTIONS[size_label]

        preset_labels = list(PRESET_LABELS.keys())
        default_preset_label = next(
            (k for k, v in PRESET_LABELS.items() if v == DEFAULT_PRESET), preset_labels[0]
        )
        preset_label = st.sidebar.selectbox(
            "Quality", preset_labels,
            index=preset_labels.index(st.session_state.get("image_preset_label", default_preset_label)),
        )
        st.session_state["image_preset_label"] = preset_label
        image_preset_key = PRESET_LABELS[preset_label]

        image_seed = int(st.sidebar.number_input(
            "Seed", min_value=0, value=int(st.session_state.get("image_seed", 0)), step=1,
            help="Same seed + prompt + settings reproduces the same image.",
        ))
        st.session_state["image_seed"] = image_seed

    if st.sidebar.button("🗑️ Start New Chat"):
        st.session_state["messages"] = []
        for key in (
            "chat_data_file_id", "chat_data_path", "chat_data_name", "chat_data_schema",
            "chat_tool_state", "chat_tool_result", "chat_tool_logs", "chat_tool_run_id",
            "chat_tool_instruction", "chat_tool_cancel", "chat_tool_title",
        ):
            st.session_state.pop(key, None)
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
        models_list = models_dict.get("models", []) if isinstance(models_dict, dict) else getattr(models_dict, 'models', [])
        local_model_names = [extract_model_name(m) for m in models_list]
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
            if msg.get("analysis"):
                _render_analysis_payload(msg["analysis"], msg["analysis"].get("run_id", "hist"))

    # If an analysis is running, poll it and skip normal input until it finishes.
    if st.session_state.get("chat_tool_state") == "running":
        _render_tool_run_section()
        return

    user_text = st.chat_input("Type your message...")

    # Image generation fast-path: the explicit "Generate image" toggle only. Every
    # other message goes through the normal chat turn, where the model decides via
    # its generate_image tool — it understands intent/context (a request vs. a
    # question *about* images, a confirming "yes", etc.) far better than a keyword
    # matcher, and crafts a context-aware prompt.
    wants_image = bool(user_text and image_gen_ok and image_mode)
    if wants_image:
        with st.chat_message("user"):
            st.markdown(user_text)
        st.session_state["messages"].append({"role": "user", "content": user_text})
        _start_image_gen_thread(
            user_text, image_width, image_height, image_preset_key, image_seed,
            low_vram=not is_high_memory_gpu(current_gpu),
            active_model=st.session_state.get("selected_model", ""),
        )
        st.rerun()
        return

    if user_text and data_file_path:
        # Router/supervisor: decide whether this message needs the data-analysis tools.
        with st.spinner("Deciding how to answer..."):
            use_tools, instruction = decide_tool_use(
                model_name,
                user_text,
                data_schema or "",
                chat_history=_ollama_messages(st.session_state["messages"]),
            )

        if use_tools:
            with st.chat_message("user"):
                st.markdown(user_text)
            st.session_state["messages"].append({"role": "user", "content": user_text})
            _start_chat_analysis_thread(
                instruction or user_text, data_file_path, data_file_name, model_name
            )
            st.rerun()
            return
        # Otherwise fall through to the normal chat path below.

    if user_text:
        # 1. Process files if they exist
        context_text = ""
        if uploaded_files:
            with st.spinner("Processing files..."):
                context_text, warnings = process_uploaded_files(uploaded_files)
                for warning in warnings:
                    st.warning(warning)
        
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

        # --- DYNAMIC SPINNER LOGIC ---
        if is_model_loaded(model_name):
            spinner_text = "Thinking..."
        else:
            spinner_text = f"🚀 Loading **{model_name}** into GPU memory... This first run may take 1-2 minutes."

        needs_chunking = bool(context_text) and estimate_tokens(context_text) > MAX_CONTEXT_TOKENS

        try:
            if needs_chunking:
                chunks = chunk_text(context_text)
                partial_answers = []
                progress_placeholder = st.empty()

                for i, chunk_content in enumerate(chunks, 1):
                    progress_placeholder.info(
                        f"📄 Analyzing document part {i} of {len(chunks)} "
                        f"(~{estimate_tokens(context_text):,} tokens total)..."
                    )
                    answer = get_chunk_answer(
                        model_name, chunk_content, i, len(chunks),
                        user_text, _ollama_messages(st.session_state["messages"][:-1])
                    )
                    partial_answers.append(answer)

                progress_placeholder.info(f"🔗 Synthesizing responses from {len(chunks)} chunks...")
                with st.chat_message("assistant"):
                    synthesis_stream = get_synthesis_generator(
                        model_name, partial_answers,
                        user_text, _ollama_messages(st.session_state["messages"][:-1])
                    )
                    assistant_reply = st.write_stream(synthesis_stream)
                progress_placeholder.empty()
                st.session_state["messages"].append({"role": "assistant", "content": assistant_reply})
            else:
                last_msg_obj["content"] = full_prompt
                with st.spinner(spinner_text):
                    # Offer the model the real generate_image tool so it can produce
                    # images mid-conversation (e.g. after the user confirms "yes").
                    kind, payload = stream_chat_with_image_tool(
                        model_name,
                        _ollama_messages(st.session_state["messages"]),
                        offer_image=image_gen_ok,
                    )
                if kind == "image":
                    last_msg_obj["content"] = original_content
                    _start_image_gen_thread(
                        payload, image_width, image_height, image_preset_key, image_seed,
                        low_vram=image_unload_chat_model,
                        active_model=model_name,
                    )
                    st.rerun()
                    return
                with st.chat_message("assistant"):
                    assistant_reply = st.write_stream(payload)
                st.session_state["messages"].append({"role": "assistant", "content": assistant_reply})
        except ResponseError as e:
            status = getattr(e, "status_code", "?")
            st.error(f"Ollama ResponseError (status={status})")
            st.code(str(e))
        finally:
            # Always restore display text so chat history shows the friendly version
            last_msg_obj["content"] = original_content

    if len(st.session_state["messages"]) > 0:
        chat_export = format_chat_history(st.session_state["messages"])
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        has_plots = _has_analysis_plots(st.session_state["messages"])

        st.sidebar.markdown("---")
        st.sidebar.download_button(
            label="📥 Download Conversation (.md)",
            data=chat_export,
            file_name=f"text_lab_chat_{timestamp}.md",
            mime="text/markdown"
        )
        if has_plots:
            chat_export_html = format_chat_history_html(st.session_state["messages"])
            st.sidebar.download_button(
                label="🌐 Download with Plots (.html)",
                data=chat_export_html,
                file_name=f"text_lab_chat_{timestamp}.html",
                mime="text/html",
                help="Includes interactive charts. Markdown export can't show interactive plots.",
            )
if __name__ == "__main__":
    main()
