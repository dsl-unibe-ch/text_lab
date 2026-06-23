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
    MAX_CONTEXT_TOKENS,
)

from core.model_config import get_available_models, is_high_memory_gpu

# --- Data-analysis tool integration (reuses the Visualisation MAS, unchanged) ---
from core.visualization.viz_agent import run_analysis
from core.visualization.viz_config import MAX_ROWS, get_tool_label
from core.visualization.viz_utils import save_data_file, get_fast_data_preview
from core.visualization.plot_data import get_all_columns_summary_impl

_SRC_DIR = pathlib.Path(__file__).resolve().parent.parent
MCP_SERVER_SCRIPT = str(_SRC_DIR / "core" / "visualization" / "mcp_server.py")
ARTIFACTS_DIR = str(_SRC_DIR / "mcp_artifacts")
ANALYSIS_TIMEOUT_SECONDS = 600
TABULAR_EXTENSIONS = (".csv", ".tsv", ".xls", ".xlsx", ".json")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

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

    with st.chat_message("assistant"):
        with st.status(
            "Analysing your data...", expanded=(not is_complete),
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
    elif status == "error":
        st.session_state["messages"].append(
            {"role": "assistant", "content": f"An error occurred during analysis: {thread_result.get('error', '')}"}
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

    if st.sidebar.button("🗑️ Start New Chat"):
        st.session_state["messages"] = []
        for key in (
            "chat_data_file_id", "chat_data_path", "chat_data_name", "chat_data_schema",
            "chat_tool_state", "chat_tool_result", "chat_tool_logs", "chat_tool_run_id",
            "chat_tool_instruction", "chat_tool_cancel",
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
                    response_stream = get_response_generator(model_name, _ollama_messages(st.session_state["messages"]))
                    with st.chat_message("assistant"):
                        assistant_reply = st.write_stream(response_stream)
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
